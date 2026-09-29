"""
An ESTIMATED, relative fuel proxy: how much harder than open water, times how far.

    from src.routing.fuel_cost import FuelParams, load_fuel_params, route_fuel
    params = load_fuel_params(Path("configs/fuel_model.json"))
    report = route_fuel(path, arrival_times, sic_for_bucket=..., ...)
    report.estimated_fuel            # open-water-equivalent metres
    report.provenance()              # what the number means, in full

WHAT THIS IS
  A dimensionless resistance factor per cell, integrated over the route with
  the same trapezoid edge rule time_astar uses:

      factor(sic) = open_water_factor + ice_penalty * (sic/100) ** ice_exponent
      fuel        = SUM  dist * 0.5 * (factor(A, t_depart) + factor(B, t_arrive))

  One unit is the fuel used steaming one metre in ice-free water at the
  reference speed -- an OPEN-WATER-EQUIVALENT METRE. What that quantity is in
  mass or volume is never resolved, because nothing in this project could
  resolve it.

WHAT THIS IS NOT
  Not litres, tonnes or kilograms; not litres/hour or tonnes/day; not an
  operational fuel prediction; not derived from any vessel's engine, hull,
  propeller or trials data, because this repository contains none of those. An
  audit on 2026-09-29 found no installed power, no SFOC curve, no displacement,
  no hull or ice resistance and no propulsive efficiency anywhere in src/,
  configs/, docs/ or tests/. `absolute_rate` is therefore null in the shipped
  config and to_mass() REFUSES rather than defaulting to a plausible number.

WHY IT IS NOT JUST DISTANCE
  Because the factor varies per cell with sea-ice concentration, sampled at the
  vessel's ARRIVAL time at that cell. Two routes of identical length through
  different ice score differently. At ice_penalty = 0 the model would collapse
  to travelled distance, and the loader refuses to call that fuel.

WHY SPEED DOES NOT APPEAR
  A speed term needs a power curve anchored to a known operating point, and
  there is no such anchor here. The speed is also a single constant over every
  route and cell, so any speed factor would scale all candidates equally and
  could not change which route is chosen. It is recorded as an assumption and
  left out of the equation rather than faked into it.

NOT IN THIS MODULE
  No A*, no POLARIS, no iceberg exposure, no temporal resolver, no I/O beyond
  reading its own config, and no ML. Nothing here decides a route; it prices
  one it is handed.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UNITS = "open_water_equivalent_metres"
MODEL_NAME = "relative_ice_resistance_fuel_proxy"
MODEL_VERSION = "1.0.0"
ESTIMATED = "estimated"

#  the audit that justifies the absence of an absolute figure
NO_ABSOLUTE_FIGURE_REASON = (
    "an absolute fuel figure needs installed power, an SFOC curve, "
    "displacement, hull/ice resistance and propulsive efficiency; an audit of "
    "src/, configs/, docs/ and tests/ on 2026-09-29 found none of them in this "
    "repository, so no mass or volume is emitted")


class FuelError(ValueError):
    """The fuel model cannot be built or applied as specified."""


# ----------------------------------------------------------------- params
@dataclass(frozen=True)
class FuelParams:
    """Every number the proxy uses. open_water_factor is a definition; the
    other two are PROJECT ASSUMPTIONS with no measurement behind them."""

    open_water_factor: float = 1.0
    ice_penalty: float = 2.0
    ice_exponent: float = 1.5
    reference_speed_mps: float = 5.0
    absolute_rate: float | None = None          # mass/volume per OWE metre

    def validate(self) -> None:
        for name in ("open_water_factor", "ice_exponent", "reference_speed_mps"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0:
                raise FuelError(f"{name} must be finite and > 0, got {v!r}")
        v = float(self.ice_penalty)
        if not math.isfinite(v) or v < 0:
            raise FuelError(f"ice_penalty must be finite and >= 0, got {v!r}")
        if v == 0.0:
            raise FuelError(
                "ice_penalty = 0 makes the fuel proxy identical to travelled "
                "distance. This project must not present a distance as a fuel "
                "figure, so the degenerate case is refused rather than shipped.")
        if self.absolute_rate is not None:
            r = float(self.absolute_rate)
            if not math.isfinite(r) or r <= 0:
                raise FuelError(
                    f"absolute_rate must be finite and > 0 when set, got {r!r}")

    def as_dict(self) -> dict:
        return {"open_water_factor": float(self.open_water_factor),
                "ice_penalty": float(self.ice_penalty),
                "ice_exponent": float(self.ice_exponent),
                "reference_speed_mps": float(self.reference_speed_mps),
                "absolute_rate": (None if self.absolute_rate is None
                                  else float(self.absolute_rate))}


def load_fuel_params(path) -> FuelParams:
    """Read the shipped config. Every value comes from the file; nothing here
    substitutes a default for a key the file is missing."""
    try:
        cfg = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise FuelError(f"cannot read the fuel model config {path}: {exc}") from exc
    try:
        p = cfg["parameters"]
        params = FuelParams(
            open_water_factor=float(p["open_water_factor"]["value"]),
            ice_penalty=float(p["ice_penalty"]["value"]),
            ice_exponent=float(p["ice_exponent"]["value"]),
            reference_speed_mps=float(p["reference_speed_mps"]["value"]),
            absolute_rate=(None if p["absolute_rate"]["value"] is None
                           else float(p["absolute_rate"]["value"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise FuelError(
            f"{path} is not a fuel model config this module understands: "
            f"{exc}") from exc
    params.validate()
    return params


# ------------------------------------------------------------ the factor
def resistance_factor(sic, params: FuelParams):
    """The dimensionless factor for one cell or a whole array.

    sic is a concentration in PERCENT. A non-finite sic yields a non-finite
    factor: an unpriceable cell stays unpriceable and is never read as ice-free.
    """
    params.validate()
    a = np.asarray(sic, dtype="float64")
    if np.any(a[np.isfinite(a)] < 0.0) or np.any(a[np.isfinite(a)] > 100.0):
        raise FuelError(
            "sea-ice concentration must be a percentage in [0, 100]; values "
            "outside that range are flag codes or a different scale, and this "
            "module will not guess which")
    with np.errstate(invalid="ignore"):
        out = (float(params.open_water_factor)
               + float(params.ice_penalty)
               * (a / 100.0) ** float(params.ice_exponent))
    return out if out.ndim else float(out)


# ------------------------------------------------------------ the report
@dataclass(frozen=True)
class FuelReport:
    """One route's estimated fuel, and everything needed to read the number."""

    estimated_fuel: float                  # open-water-equivalent metres
    distance_m: float
    segments: tuple[float, ...]            # per-segment fuel, same order
    cell_factors: tuple[float, ...]        # the factor used at each cell
    cell_sic: tuple[float, ...]            # the sic each factor came from
    buckets: tuple[int, ...]               # the bucket each cell was sampled in
    params: FuelParams
    units: str = UNITS

    @property
    def fuel_per_km(self) -> float | None:
        """Open-water-equivalent metres per kilometre travelled: the route's
        mean difficulty. 1000.0 would mean the whole route was ice-free."""
        if self.distance_m <= 0:
            return None
        return self.estimated_fuel / (self.distance_m / 1000.0)

    @property
    def mean_factor(self) -> float | None:
        if self.distance_m <= 0:
            return None
        return self.estimated_fuel / self.distance_m

    def to_mass(self):
        """Refuses unless an absolute rate was configured. There is none."""
        if self.params.absolute_rate is None:
            raise FuelError(
                "this is a RELATIVE fuel proxy in open-water-equivalent "
                f"metres, not a mass or volume: {NO_ABSOLUTE_FIGURE_REASON}. "
                "Set parameters.absolute_rate in the fuel model config, with "
                "its source, to obtain an absolute figure.")
        return self.estimated_fuel * float(self.params.absolute_rate)

    def provenance(self) -> dict:
        """What does this fuel number mean? Everything needed to answer it."""
        return {
            "model": MODEL_NAME,
            "version": MODEL_VERSION,
            "status": ESTIMATED,
            "is_a_measured_fuel_consumption": False,
            "is_an_operational_fuel_prediction": False,
            "units": self.units,
            "unit_definition":
                "1 open-water-equivalent metre = the fuel used steaming 1 m in "
                "ice-free water at the reference speed; that quantity is never "
                "resolved into a mass or volume",
            "equation":
                "sum over segments of dist * 0.5 * (factor(A, t_depart) + "
                "factor(B, t_arrive)), with factor(sic) = open_water_factor + "
                "ice_penalty * (sic/100) ** ice_exponent",
            "edge_rule": "the trapezoid rule src/routing/time_astar.py documents",
            "sic_sampled_at": "the vessel's arrival time at each cell",
            "parameters": self.params.as_dict(),
            "parameter_status": {
                "open_water_factor": "definition (the output's unit)",
                "ice_penalty": "project assumption, not measured",
                "ice_exponent": "project assumption, not measured",
                "reference_speed_mps":
                    "project assumption inherited from DEMO.vessel_speed_mps, "
                    "which itself has no documented source",
                "absolute_rate": "unavailable",
            },
            "environmental_terms_used": ["sea_ice_concentration"],
            "environmental_terms_excluded": [
                "currents (isotropic and shipped at weight 0.0)",
                "wind (absent from the routing cost)",
                "iceberg exposure (an index, not a resistance)",
                "POLARIS (a risk preference, not an energy)",
                "ice-breaking, besetting, convoy and escort behaviour",
            ],
            "vessel_data_available": {
                "ice_class": "PC6",
                "speed_mps": float(self.params.reference_speed_mps),
                "installed_power": None, "sfoc": None, "displacement": None,
                "hull_or_ice_resistance": None, "propulsive_efficiency": None,
                "draft": None,
            },
            "no_absolute_figure_reason": NO_ABSOLUTE_FIGURE_REASON,
            "suitable_for_relative_route_comparison": True,
            "suitable_for_operational_fuel_prediction": False,
            "comparable_only_within": "routes priced with these same parameters",
        }


# ------------------------------------------------------------ evaluation
def route_fuel(path, arrival_times, *, sic_for_bucket, bucket_of,
               pixel_size, params: FuelParams) -> FuelReport:
    """Price a route that already exists. Pure lookups plus the edge rule.

    path            [(row, col), ...] as the search returned it
    arrival_times   one arrival time per cell, same order
    sic_for_bucket  bucket -> the sic field (percent) that bucket was priced from
    bucket_of       arrival time -> bucket index, the provider's own rule
    pixel_size      (dx, dy) in metres

    Re-evaluating a returned path through this function is what makes a route's
    reported fuel reproducible: it reads the path it is given and nothing else.
    """
    params.validate()
    path = [tuple(p) for p in path]
    arrival_times = [float(t) for t in arrival_times]
    if len(path) != len(arrival_times):
        raise FuelError(
            f"one arrival time per cell is required, got {len(path)} cells "
            f"and {len(arrival_times)} times")
    if not path:
        raise FuelError("an empty path has no fuel to estimate")

    dx, dy = (abs(float(pixel_size[0])), abs(float(pixel_size[1])))
    factors, sics, buckets = [], [], []
    cache: dict[int, np.ndarray] = {}
    for (r, c), t in zip(path, arrival_times):
        b = int(bucket_of(t))
        if b not in cache:
            field = np.asarray(sic_for_bucket(b), dtype="float64")
            if field.ndim != 2:
                raise FuelError(
                    f"sic_for_bucket({b}) must return a 2-D field, got shape "
                    f"{field.shape}")
            cache[b] = field
        field = cache[b]
        if not (0 <= r < field.shape[0] and 0 <= c < field.shape[1]):
            raise FuelError(
                f"cell ({r}, {c}) is outside the sic field {field.shape}")
        sic = float(field[r, c])
        f = float(resistance_factor(sic, params)) if math.isfinite(sic) else \
            float("nan")
        if not math.isfinite(f):
            raise FuelError(
                f"cell ({r}, {c}) has no sea-ice concentration at bucket {b}, "
                f"so its fuel cannot be estimated. An unpriceable cell is not "
                f"read as ice-free.")
        factors.append(f)
        sics.append(sic)
        buckets.append(b)

    total, dist_total, segments = 0.0, 0.0, []
    for a, b_, fa, fb in zip(path, path[1:], factors, factors[1:]):
        dist = math.hypot((b_[1] - a[1]) * dx, (b_[0] - a[0]) * dy)
        seg = dist * 0.5 * (fa + fb)
        segments.append(seg)
        dist_total += dist
        total += seg

    return FuelReport(
        estimated_fuel=total, distance_m=dist_total,
        segments=tuple(segments), cell_factors=tuple(factors),
        cell_sic=tuple(sics), buckets=tuple(buckets), params=params)
