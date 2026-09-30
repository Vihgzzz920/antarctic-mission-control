"""
POLARIS arithmetic and data model -- IMO MSC.1/Circ.1519.

    RIO = sum_i ( C_i x RIV_i )

where C_i is the concentration of ice type i in TENTHS and RIV_i is the Risk
Index Value for that ice type and the vessel's ice class. This module computes
that sum, validates its inputs, classifies the result into the POLARIS
operational categories, and records every number it used so a later layer can
explain why a cell or a leg was rejected or penalised.

THIS MODULE HAS NO POLARIS DATA AND WILL NOT INVENT ANY
  The RIV tables and the RIO band limits are published in MSC.1/Circ.1519.
  They are NOT reproduced here, from memory or otherwise, because a
  mis-transcribed RIV is indistinguishable from a correct one at the call site
  and would silently authorise a ship into ice it is not built for. Instead:

    - load_riv_table(path) reads an authoritative table the project supplies,
      as JSON, carrying its own `source` citation.
    - Every calculation requires such a table. There is no built-in default and
      no fallback; a missing table raises MissingPolarisData.

  The numbers commonly cited for the RIO bands (0 and -10) are deliberately not
  hard-coded either. They live in the same supplied file, so the audit trail can
  name where they came from.

  To wire this up, a person must transcribe the tables from the circular into
  the JSON documented in load_riv_table() and check them against the source.
  Until then this module computes nothing -- which is the intended behaviour.

ICE CLASS CATEGORIES ARE NOT MERGED
  POLARIS treats a Polar Class ship (PC1-PC7) and a ship below PC7 -- including
  one with no assigned ice class -- differently: the band limits are not the
  same. IceClassCategory keeps the three cases separate all the way through, and
  classify_rio() refuses to classify a RIO without knowing which category the
  ship is in. A table that omits bands for the ship's category is an error, not
  an invitation to reuse another category's bands.

WHAT THIS MODULE IS NOT
  It is not a SIC threshold. "SIC > X is blocked" is not POLARIS and is not an
  approximation of it: POLARIS asks which ice TYPES are present and in what
  proportion, and a single concentration number does not carry that. Nothing
  here reads SIC, and nothing here produces a raster. Turning an ice-regime
  product into per-cell regimes is a separate, later step; this module is only
  the arithmetic and the audit trail that step will call.

Usage:
    python -m src.routing.polaris --smoke-test
    python -m src.routing.polaris --table configs/polaris_riv.json --describe
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TENTHS_TOTAL = 10.0          # a complete ice regime is ten tenths
TENTHS_TOL = 1e-6            # how far the sum may drift from 10 and still count


class PolarisError(ValueError):
    """A POLARIS calculation cannot be performed as specified."""


class MissingPolarisData(PolarisError):
    """Authoritative POLARIS data was not supplied. Nothing is substituted."""


# --------------------------------------------------------------- ice class
class IceClassCategory(Enum):
    """Which POLARIS band set applies. These three are never merged."""

    POLAR_CLASS = "polar_class"                  # PC1-PC7
    BELOW_POLAR_CLASS = "below_polar_class"      # e.g. Finnish-Swedish 1A Super..1C
    NO_ICE_CLASS = "no_ice_class"                # not ice strengthened

    @classmethod
    def parse(cls, value: str) -> "IceClassCategory":
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            raise PolarisError(
                f"unknown ice class category {value!r}; expected one of "
                f"{[c.value for c in cls]}")


@dataclass(frozen=True)
class IceClass:
    """One vessel ice class, as named in the supplied table."""

    name: str                                    # e.g. "PC6", "1A", "no ice class"
    category: IceClassCategory

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise PolarisError("ice class name is empty; POLARIS needs the vessel's "
                               "assigned class, and 'unknown' is not a class.")
        if not isinstance(self.category, IceClassCategory):
            raise PolarisError(f"category must be an IceClassCategory, got "
                               f"{type(self.category).__name__}")

    def __str__(self) -> str:
        return f"{self.name} ({self.category.value})"


class OperationalCategory(Enum):
    """The POLARIS outcomes. Names come from the methodology, limits from data."""

    NORMAL_OPERATION = "normal_operation"
    ELEVATED_OPERATIONAL_RISK = "elevated_operational_risk"
    SPECIAL_CONSIDERATION = "operation_subject_to_special_consideration"

    @classmethod
    def parse(cls, value: str) -> "OperationalCategory":
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            raise PolarisError(
                f"unknown operational category {value!r}; expected one of "
                f"{[c.value for c in cls]}")


# ------------------------------------------------------------- ice regime
@dataclass(frozen=True)
class IceTypeConcentration:
    """One ice type and how much of the regime it occupies, in TENTHS."""

    ice_type: str
    tenths: float

    def __post_init__(self) -> None:
        if not str(self.ice_type).strip():
            raise PolarisError("ice type name is empty")
        try:
            value = float(self.tenths)
        except (TypeError, ValueError):
            raise PolarisError(f"{self.ice_type}: concentration must be a number, "
                               f"got {self.tenths!r}")
        if not math.isfinite(value):
            raise PolarisError(f"{self.ice_type}: concentration must be finite, "
                               f"got {value}")
        if value < 0:
            raise PolarisError(f"{self.ice_type}: concentration {value} is negative")
        if value > TENTHS_TOTAL + TENTHS_TOL:
            raise PolarisError(
                f"{self.ice_type}: concentration {value} exceeds {TENTHS_TOTAL:g} "
                f"tenths. Concentrations are in TENTHS, not per cent -- 70% is 7.")
        object.__setattr__(self, "tenths", value)


@dataclass(frozen=True)
class IceRegime:
    """A complete ice regime: the ice types present, summing to ten tenths."""

    components: tuple[IceTypeConcentration, ...]
    label: str = ""

    def __post_init__(self) -> None:
        comps = tuple(self.components)
        if not comps:
            raise PolarisError(
                "the ice regime is empty. Open water is an ice type with its own "
                "RIV, so an ice-free regime is 10 tenths of the open-water type, "
                "not zero components.")
        seen: set[str] = set()
        for c in comps:
            if not isinstance(c, IceTypeConcentration):
                raise PolarisError(f"regime components must be IceTypeConcentration, "
                                   f"got {type(c).__name__}")
            if c.ice_type in seen:
                raise PolarisError(f"ice type {c.ice_type!r} appears twice; combine "
                                   f"the concentrations at the source instead")
            seen.add(c.ice_type)
        total = math.fsum(c.tenths for c in comps)
        if abs(total - TENTHS_TOTAL) > TENTHS_TOL:
            raise PolarisError(
                f"concentrations sum to {total:g} tenths, not {TENTHS_TOTAL:g}. "
                f"A POLARIS regime must be complete: "
                f"{'; '.join(f'{c.ice_type}={c.tenths:g}' for c in comps)}")
        object.__setattr__(self, "components", comps)

    @property
    def total_tenths(self) -> float:
        return math.fsum(c.tenths for c in self.components)

    @property
    def ice_types(self) -> tuple[str, ...]:
        return tuple(c.ice_type for c in self.components)

    @classmethod
    def from_mapping(cls, mapping: dict, label: str = "") -> "IceRegime":
        """IceRegime.from_mapping({"open water": 3, "thin first-year ice": 7})"""
        return cls(tuple(IceTypeConcentration(k, v) for k, v in mapping.items()), label)


# --------------------------------------------------------------- RIV data
@dataclass(frozen=True)
class RIVTable:
    """Risk Index Values, transcribed from an authoritative source by a person.

    `source` is required and is carried into every RIOResult, so a route
    explanation can name which document its numbers came from.
    """

    source: str
    ice_classes: dict[str, IceClassCategory]          # class name -> category
    riv: dict[str, dict[str, float]]                  # class name -> ice type -> RIV
    bands: dict[IceClassCategory, tuple["RIOBand", ...]]
    path: str = ""

    def __post_init__(self) -> None:
        if not str(self.source).strip():
            raise PolarisError(
                "the RIV table carries no `source`. A POLARIS number with no "
                "provenance cannot be audited; cite the document it came from.")
        if not self.riv:
            raise MissingPolarisData("the RIV table is empty")
        for name in self.riv:
            if name not in self.ice_classes:
                raise PolarisError(f"ice class {name!r} has RIVs but no category; "
                                   f"POLARIS bands depend on the category")
        for name, values in self.riv.items():
            if not values:
                raise MissingPolarisData(f"ice class {name!r} has no RIVs")
            for ice_type, value in values.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise PolarisError(f"RIV[{name}][{ice_type}] is {value!r}, "
                                       f"not a number")
                if not math.isfinite(float(value)):
                    raise PolarisError(f"RIV[{name}][{ice_type}] is {value}, "
                                       f"not finite")
        for category, bands in self.bands.items():
            _check_bands(category, bands)

    # ------------------------------------------------------------ lookups
    def ice_class(self, name: str) -> IceClass:
        """The IceClass for a name, with its category taken from the table."""
        if name not in self.ice_classes:
            raise MissingPolarisData(
                f"ice class {name!r} is not in {self.describe_source()}. "
                f"Known: {sorted(self.ice_classes)}. This module does not guess a "
                f"category or reuse another class's RIVs.")
        return IceClass(name=name, category=self.ice_classes[name])

    def riv_for(self, ice_class: IceClass, ice_type: str) -> float:
        """The RIV for one ice type. Missing is an error, never a zero."""
        values = self.riv.get(ice_class.name)
        if values is None:
            raise MissingPolarisData(
                f"no RIVs for ice class {ice_class.name!r} in "
                f"{self.describe_source()}")
        if ice_type not in values:
            raise MissingPolarisData(
                f"no RIV for ice type {ice_type!r} at ice class "
                f"{ice_class.name!r} in {self.describe_source()}. Known ice types: "
                f"{sorted(values)}. A missing RIV is not 0 -- 0 would silently mean "
                f"'this ice adds no risk'.")
        return float(values[ice_type])

    def bands_for(self, category: IceClassCategory) -> tuple["RIOBand", ...]:
        bands = self.bands.get(category)
        if not bands:
            raise MissingPolarisData(
                f"no RIO bands for category {category.value!r} in "
                f"{self.describe_source()}. The bands differ between Polar Class "
                f"and below-Polar-Class ships; this module will not borrow one "
                f"category's limits for another.")
        return bands

    def describe_source(self) -> str:
        return f"{self.source}" + (f" [{self.path}]" if self.path else "")


@dataclass(frozen=True)
class RIOBand:
    """One operational band: RIO >= min_rio maps to this category."""

    min_rio: float
    category: OperationalCategory

    def __post_init__(self) -> None:
        value = float(self.min_rio)
        if math.isnan(value):
            raise PolarisError("band min_rio is NaN")
        object.__setattr__(self, "min_rio", value)


def _check_bands(category: IceClassCategory, bands: tuple[RIOBand, ...]) -> None:
    """Bands must be ordered, non-overlapping and cover every possible RIO."""
    if not bands:
        raise MissingPolarisData(f"no RIO bands supplied for {category.value}")
    lowers = [b.min_rio for b in bands]
    if lowers != sorted(lowers, reverse=True):
        raise PolarisError(
            f"{category.value} bands must be listed from the highest min_rio "
            f"downwards, got {lowers}")
    if len(set(lowers)) != len(lowers):
        raise PolarisError(f"{category.value} bands repeat a min_rio: {lowers}")
    if lowers[-1] != -math.inf:
        raise PolarisError(
            f"{category.value} bands do not cover every RIO: the lowest band must "
            f"start at -inf, got {lowers[-1]}. Otherwise some RIO would fall "
            f"through unclassified.")


# ----------------------------------------------------------------- result
@dataclass(frozen=True)
class RIOTerm:
    """One term of the sum, kept so the total can be explained line by line."""

    ice_type: str
    tenths: float
    riv: float

    @property
    def contribution(self) -> float:
        return self.tenths * self.riv


@dataclass(frozen=True)
class RIOResult:
    """A RIO with everything needed to justify it."""

    ice_class: IceClass
    regime: IceRegime
    terms: tuple[RIOTerm, ...]
    rio: float
    category: OperationalCategory | None
    source: str

    def explain(self) -> str:
        """The audit trail: class, regime, each RIV, the sum, and the outcome."""
        lines = [
            f"vessel ice class : {self.ice_class}",
            f"ice regime       : {self.regime.label or 'unnamed'} "
            f"({self.regime.total_tenths:g}/10 tenths)",
            f"RIV source       : {self.source}",
            "",
            f"  {'ice type':<32}{'tenths':>8}{'RIV':>7}{'C x RIV':>10}",
        ]
        for t in self.terms:
            lines.append(f"  {t.ice_type:<32}{t.tenths:>8g}{t.riv:>7g}"
                         f"{t.contribution:>10g}")
        lines.append(f"  {'RIO':<32}{'':>8}{'':>7}{self.rio:>10g}")
        lines.append("")
        if self.category is None:
            lines.append("operational category: NOT CLASSIFIED (no bands supplied)")
        else:
            lines.append(f"operational category: {self.category.value}")
        return "\n".join(lines)


# ------------------------------------------------------------ calculation
def calculate_rio(regime: IceRegime, ice_class: IceClass,
                  riv_table: RIVTable) -> RIOResult:
    """RIO = sum(C_i x RIV_i). Every RIV must exist; nothing defaults to zero."""
    if riv_table is None:
        raise MissingPolarisData(
            "no RIV table supplied. This module has no built-in POLARIS data; "
            "load an authoritative table with load_riv_table().")
    if not isinstance(regime, IceRegime):
        raise PolarisError(f"regime must be an IceRegime, got {type(regime).__name__}")
    if not isinstance(ice_class, IceClass):
        raise PolarisError(f"ice_class must be an IceClass, got "
                           f"{type(ice_class).__name__}")

    terms: list[RIOTerm] = []
    for component in regime.components:
        riv = riv_table.riv_for(ice_class, component.ice_type)
        if not math.isfinite(riv):
            raise PolarisError(
                f"RIV for {component.ice_type!r} at {ice_class.name} is {riv}, "
                f"not finite")
        terms.append(RIOTerm(component.ice_type, component.tenths, riv))

    rio = math.fsum(t.contribution for t in terms)
    if not math.isfinite(rio):
        raise PolarisError(f"RIO is {rio}, not finite; check the RIVs and "
                           f"concentrations")
    return RIOResult(ice_class=ice_class, regime=regime, terms=tuple(terms),
                     rio=rio, category=None, source=riv_table.describe_source())


def classify_rio(rio: float, ice_class: IceClass,
                 riv_table: RIVTable) -> OperationalCategory:
    """Which operational band a RIO falls in, FOR THIS ICE-CLASS CATEGORY.

    The limits differ between Polar Class and below-Polar-Class ships, so the
    category is a required input, not a convenience. They are read from the
    supplied table rather than hard-coded here.
    """
    try:
        value = float(rio)
    except (TypeError, ValueError):
        raise PolarisError(f"RIO must be a number, got {rio!r}")
    if not math.isfinite(value):
        raise PolarisError(f"RIO must be finite to be classified, got {value}")
    if not isinstance(ice_class, IceClass):
        raise PolarisError(f"ice_class must be an IceClass, got "
                           f"{type(ice_class).__name__}")

    for band in riv_table.bands_for(ice_class.category):
        if value >= band.min_rio:
            return band.category
    # _check_bands guarantees a -inf band, so this is unreachable by construction.
    raise PolarisError(f"RIO {value} fell through every band for "
                       f"{ice_class.category.value}")


def evaluate(regime: IceRegime, ice_class: IceClass,
             riv_table: RIVTable) -> RIOResult:
    """calculate_rio() plus classification, returned as one auditable result."""
    result = calculate_rio(regime, ice_class, riv_table)
    category = classify_rio(result.rio, ice_class, riv_table)
    return RIOResult(ice_class=result.ice_class, regime=result.regime,
                     terms=result.terms, rio=result.rio, category=category,
                     source=result.source)


# ------------------------------------------------- authoritative data I/O
def load_riv_table(path) -> RIVTable:
    """Read an authoritative RIV table transcribed from MSC.1/Circ.1519.

    The file is JSON and a person is responsible for its contents:

        {
          "source": "IMO MSC.1/Circ.1519, Tables 1.1 and 1.4, checked 2026-09-24",
          "ice_classes": {
            "PC6":           "polar_class",
            "1A":            "below_polar_class",
            "no ice class":  "no_ice_class"
          },
          "riv": {
            "PC6": {"ice free": 3, "new ice": 3, "thin first-year ice": 1, ...},
            "1A":  {"ice free": 3, ...}
          },
          "rio_bands": {
            "polar_class": [
              {"min_rio": <from the circular>, "category": "normal_operation"},
              {"min_rio": <from the circular>, "category": "elevated_operational_risk"},
              {"min_rio": "-inf", "category": "operation_subject_to_special_consideration"}
            ],
            "below_polar_class": [ ... ],
            "no_ice_class":      [ ... ]
          }
        }

    Bands are listed from the highest min_rio downwards and the last must be
    "-inf" so every RIO is covered. Ice-type names are matched exactly as
    written, so the regime source and this table must agree on spelling.

    Nothing here is filled in for you. There is no default table, and this
    function does not write one.
    """
    path = Path(path)
    if not path.exists():
        raise MissingPolarisData(
            f"No POLARIS RIV table at {path}.\n"
            f"This project does not ship one: the RIV tables and RIO limits are "
            f"published in IMO MSC.1/Circ.1519 and must be transcribed from that "
            f"document and checked by a person. See load_riv_table.__doc__ for the "
            f"file format. Until then, no RIO can be computed -- which is correct, "
            f"because an invented RIV is worse than no answer.")

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PolarisError(f"{path.name} is not valid JSON: {exc}")
    if not isinstance(raw, dict):
        raise PolarisError(f"{path.name} must hold a JSON object")

    for key in ("source", "ice_classes", "riv", "rio_bands"):
        if key not in raw:
            raise PolarisError(f"{path.name} has no {key!r}; see "
                               f"load_riv_table.__doc__ for the required format")

    ice_classes = {str(k): IceClassCategory.parse(v)
                   for k, v in dict(raw["ice_classes"]).items()}
    riv = {str(k): {str(t): float(val) for t, val in dict(v).items()}
           for k, v in dict(raw["riv"]).items()}

    bands: dict[IceClassCategory, tuple[RIOBand, ...]] = {}
    for key, entries in dict(raw["rio_bands"]).items():
        category = IceClassCategory.parse(key)
        parsed: list[RIOBand] = []
        for entry in entries:
            parsed.append(RIOBand(min_rio=float(entry["min_rio"]),
                                  category=OperationalCategory.parse(entry["category"])))
        bands[category] = tuple(parsed)

    return RIVTable(source=str(raw["source"]), ice_classes=ice_classes, riv=riv,
                    bands=bands, path=str(path))


# ------------------------------------------------------------- smoke test
def _test_table() -> RIVTable:
    """SYNTHETIC TEST DATA -- NOT POLARIS.

    These RIVs and band limits are made up to exercise the arithmetic and the
    validation. They are NOT from MSC.1/Circ.1519, they are NOT correct for any
    real vessel, and nothing outside this smoke test may use them. The names
    below are deliberately not real ice-type names so they cannot be mistaken
    for a transcription.
    """
    return RIVTable(
        source="SYNTHETIC TEST DATA -- NOT POLARIS, NOT MSC.1/Circ.1519",
        ice_classes={"TEST-PC": IceClassCategory.POLAR_CLASS,
                     "TEST-BELOW": IceClassCategory.BELOW_POLAR_CLASS,
                     "TEST-NONE": IceClassCategory.NO_ICE_CLASS},
        riv={"TEST-PC":    {"test-open": 3.0, "test-thin": 1.0, "test-thick": -2.0},
             "TEST-BELOW": {"test-open": 3.0, "test-thin": -1.0, "test-thick": -4.0},
             "TEST-NONE":  {"test-open": 3.0, "test-thin": -3.0, "test-thick": -8.0}},
        bands={
            IceClassCategory.POLAR_CLASS: (
                RIOBand(0.0, OperationalCategory.NORMAL_OPERATION),
                RIOBand(-10.0, OperationalCategory.ELEVATED_OPERATIONAL_RISK),
                RIOBand(-math.inf, OperationalCategory.SPECIAL_CONSIDERATION)),
            IceClassCategory.BELOW_POLAR_CLASS: (
                RIOBand(0.0, OperationalCategory.NORMAL_OPERATION),
                RIOBand(-math.inf, OperationalCategory.SPECIAL_CONSIDERATION)),
            IceClassCategory.NO_ICE_CLASS: (
                RIOBand(0.0, OperationalCategory.NORMAL_OPERATION),
                RIOBand(-math.inf, OperationalCategory.SPECIAL_CONSIDERATION)),
        })


def smoke_test() -> int:
    """Deterministic checks of the arithmetic, the bands and every validation."""
    print("=" * 74)
    print("POLARIS smoke test")
    print("!! SYNTHETIC TEST DATA -- the RIVs and band limits below are INVENTED")
    print("!! to exercise the code. They are NOT POLARIS values. The real tables")
    print("!! come from IMO MSC.1/Circ.1519 via load_riv_table().")
    print("=" * 74)

    table = _test_table()
    failures: list[str] = []

    def check(label: str, got, want) -> None:
        ok = got == want
        shown = got.value if isinstance(got, Enum) else got
        print(f"  {'ok ' if ok else 'FAIL'}  {label:<52} {shown}")
        if not ok:
            failures.append(f"{label}: got {got!r}, expected {want!r}")

    def raises(label: str, exc_type, fn) -> None:
        try:
            fn()
        except exc_type as exc:
            print(f"  ok    {label:<52} {type(exc).__name__}")
            return
        except Exception as exc:                            # wrong exception type
            print(f"  FAIL  {label:<52} {type(exc).__name__}: {exc}")
            failures.append(f"{label}: raised {type(exc).__name__}, "
                            f"expected {exc_type.__name__}")
            return
        print(f"  FAIL  {label:<52} no exception")
        failures.append(f"{label}: no exception raised")

    # --- arithmetic ------------------------------------------------------
    pc = table.ice_class("TEST-PC")
    regime = IceRegime.from_mapping({"test-open": 3, "test-thin": 5, "test-thick": 2},
                                    label="TEST regime A")
    #  3x3 + 5x1 + 2x(-2) = 9 + 5 - 4 = 10
    print("\narithmetic:")
    check("RIO = 3x3 + 5x1 + 2x(-2)", calculate_rio(regime, pc, table).rio, 10.0)

    all_open = IceRegime.from_mapping({"test-open": 10}, label="TEST open water")
    check("RIO = 10x3 (single ice type)",
          calculate_rio(all_open, pc, table).rio, 30.0)

    heavy = IceRegime.from_mapping({"test-thick": 10}, label="TEST heavy")
    check("RIO = 10x(-2) at TEST-PC", calculate_rio(heavy, pc, table).rio, -20.0)
    check("fractional tenths sum exactly",
          calculate_rio(IceRegime.from_mapping({"test-open": 2.5, "test-thin": 7.5}),
                        pc, table).rio, 15.0)

    # --- bands, and the categories NOT being merged ----------------------
    print("\nbands (the same RIO, two ice-class categories):")
    below = table.ice_class("TEST-BELOW")
    check("TEST-PC    RIO -5  -> ", classify_rio(-5.0, pc, table),
          OperationalCategory.ELEVATED_OPERATIONAL_RISK)
    check("TEST-BELOW RIO -5  -> ", classify_rio(-5.0, below, table),
          OperationalCategory.SPECIAL_CONSIDERATION)
    check("TEST-PC    RIO  0  -> ", classify_rio(0.0, pc, table),
          OperationalCategory.NORMAL_OPERATION)
    check("TEST-PC    RIO -10 -> ", classify_rio(-10.0, pc, table),
          OperationalCategory.ELEVATED_OPERATIONAL_RISK)
    check("TEST-PC    RIO -11 -> ", classify_rio(-11.0, pc, table),
          OperationalCategory.SPECIAL_CONSIDERATION)

    # --- validation ------------------------------------------------------
    print("\nvalidation:")
    raises("negative concentration", PolarisError,
           lambda: IceTypeConcentration("test-thin", -1))
    raises("concentration above 10 tenths", PolarisError,
           lambda: IceTypeConcentration("test-thin", 70))     # per cent, not tenths
    raises("NaN concentration", PolarisError,
           lambda: IceTypeConcentration("test-thin", float("nan")))
    raises("concentrations do not sum to 10", PolarisError,
           lambda: IceRegime.from_mapping({"test-open": 3, "test-thin": 3}))
    raises("empty ice regime", PolarisError, lambda: IceRegime(()))
    raises("duplicate ice type", PolarisError,
           lambda: IceRegime((IceTypeConcentration("test-thin", 5),
                              IceTypeConcentration("test-thin", 5))))
    raises("unknown ice type (no RIV)", MissingPolarisData,
           lambda: calculate_rio(
               IceRegime.from_mapping({"test-open": 5, "not-in-the-table": 5}),
               pc, table))
    raises("unknown ice class", MissingPolarisData,
           lambda: table.ice_class("TEST-NOT-A-CLASS"))
    raises("empty ice class name", PolarisError,
           lambda: IceClass("", IceClassCategory.POLAR_CLASS))
    raises("no RIV table supplied", MissingPolarisData,
           lambda: calculate_rio(regime, pc, None))
    raises("non-finite RIO classified", PolarisError,
           lambda: classify_rio(float("inf"), pc, table))
    raises("NaN RIO classified", PolarisError,
           lambda: classify_rio(float("nan"), pc, table))
    raises("missing bands for a category", MissingPolarisData,
           lambda: classify_rio(0.0, IceClass("TEST-PC", IceClassCategory.POLAR_CLASS),
                                RIVTable(source="TEST", ice_classes=table.ice_classes,
                                         riv=table.riv, bands={})))
    raises("bands not covering every RIO", PolarisError,
           lambda: _check_bands(IceClassCategory.POLAR_CLASS,
                                (RIOBand(0.0, OperationalCategory.NORMAL_OPERATION),)))
    raises("bands out of order", PolarisError,
           lambda: _check_bands(IceClassCategory.POLAR_CLASS,
                                (RIOBand(-math.inf, OperationalCategory.SPECIAL_CONSIDERATION),
                                 RIOBand(0.0, OperationalCategory.NORMAL_OPERATION))))
    raises("RIV table with no source citation", PolarisError,
           lambda: RIVTable(source="  ", ice_classes=table.ice_classes,
                            riv=table.riv, bands=table.bands))
    raises("missing authoritative table on disk", MissingPolarisData,
           lambda: load_riv_table(ROOT / "configs" / "no_such_polaris_table.json"))

    # --- the audit trail -------------------------------------------------
    print("\naudit trail produced by evaluate():\n")
    print("  " + evaluate(regime, pc, table).explain().replace("\n", "\n  "))

    print()
    if failures:
        print("SMOKE TEST FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("SMOKE TEST OK: arithmetic, band classification and every validation "
          "behave as specified.")
    print("Reminder: no real POLARIS data was used or produced here.")
    return 0


def describe(table_path) -> int:
    """Print what an authoritative table actually contains, for checking."""
    table = load_riv_table(table_path)
    print(f"source      : {table.describe_source()}")
    print(f"ice classes : {len(table.ice_classes)}")
    for name, category in sorted(table.ice_classes.items()):
        types = table.riv.get(name, {})
        print(f"  {name:<16} {category.value:<20} {len(types)} ice type(s)")
    print(f"bands       :")
    for category, bands in table.bands.items():
        text = ", ".join(f"RIO >= {b.min_rio:g} -> {b.category.value}" for b in bands)
        print(f"  {category.value:<20} {text}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="POLARIS RIO arithmetic (IMO MSC.1/Circ.1519). Ships with no "
                    "RIV data; an authoritative table must be supplied.")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run the deterministic self-test on SYNTHETIC data")
    ap.add_argument("--table", default=None,
                    help="path to an authoritative RIV table (JSON)")
    ap.add_argument("--describe", action="store_true",
                    help="print the contents of --table for checking")
    args = ap.parse_args()

    try:
        if args.smoke_test:
            code = smoke_test()
        elif args.describe:
            if not args.table:
                raise PolarisError("--describe needs --table")
            code = describe(args.table)
        else:
            ap.print_help()
            code = 0
    except PolarisError as exc:
        print(f"\nPOLARIS ERROR\n\n{exc}\n", file=sys.stderr)
        code = 1
    sys.exit(code)
