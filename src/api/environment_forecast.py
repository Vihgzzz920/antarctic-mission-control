"""
The one door between a predicted sea-ice field and the routing layer.

    resolved = resolve_environment_forecast(
        origin_time=datetime(2025, 1, 8), valid_time=datetime(2025, 1, 9))
    sic = read_environment(resolved)        # float32 (H, W), NaN where invalid

WHAT IS AVAILABLE, EXACTLY
  One lead: +24 h, because that is the only lead the trained model has (see
  src/models/forecast_sic_raster.py). A request for any other valid time is
  REFUSED. It is not served the 24 h field under another name, not served the
  observation, not served persistence, and not served the nearest artifact.

  That refusal is the point of this module. A forecast that answers every
  question by handing back whichever field it happens to hold is worse than no
  forecast, because the route would look forecast-driven while being priced
  against something else.

WHAT IT DELIBERATELY DOES NOT DO
  It does not touch the router. Nothing in src/api/world.py or src/routing
  imports it, the shipped demo still prices every bucket from the departure
  analysis through src/api/environment_timeline.py, and this module does not
  change that. The routing buckets are six hours wide and the demonstration
  route is under twelve hours long, so a 24 h field has no bucket to occupy
  yet; forcing one would mean relabelling the lead, which this project will
  not do. That gap is the subject of the next task, not this one.

  It also offers no persistence policy. Carrying an analysis forward is a
  decision about a ROUTING BUCKET, and environment_timeline.py already owns it,
  declares it per bucket and stamps what was carried forward. Duplicating it
  here would give the same fallback two different names.

POLICIES
  exact_forecast_valid_time   the default. valid_time must be exactly one
                              supported lead after origin_time, and an artifact
                              for that origin must already exist and pass the
                              grid validation.

  observed_analysis           the analysis itself: valid_time == origin_time,
                              served from the observed archive. Any non-zero
                              lead under this policy is refused, because
                              serving an observation for a later time is
                              persistence wearing a forecast's name.

THE LEAD REGISTRY, AND WHY IT HAS THE SHAPE IT HAS
  LEAD_SUPPORT below is the project's answer to "which forecast leads can we
  honestly produce?", and it is derived from the archive rather than from
  ambition. Both sea-ice sources in this repository are DAILY: the Bremen
  AMSR2 product is a day-grid swath composite with no time of day, and the
  independent OSI-430-a record declares time_coverage_resolution P1D at 12:00
  UTC. There is no sub-daily sea-ice observation anywhere in the data, so there
  is no 6 h or 12 h training target to fit a model against, and none is
  invented here. A +48 h target does exist in the archive (361 usable pairs),
  but no model has been trained for it, so it is registered as supported by the
  DATA and unsupported by any MODEL -- a different refusal, for a different
  reason, and the registry keeps them apart.

RESOLVING AN ARRIVAL TIME: FOUR ANSWERS, NEVER A SILENT ONE
  resolve_environment() answers the routing layer's real question -- "what SIC
  field is legitimately available at this arrival time?" -- with exactly one of

      observed_analysis     the analysis itself, at its own time
      model_forecast        a real artifact whose validity covers the request
      explicit_persistence  the analysis carried forward, ONLY when the caller
                            asked for that fallback by name
      unavailable           nothing legitimate covers it, and why

  and always with provenance attached.

VALIDITY INTERVALS ARE DERIVED, NOT INVENTED
  A forecast raster carries one valid time. Whether it may serve a routing
  bucket that SPANS time is a question about what the model was trained to
  predict, and there are two defensible readings, both offered, with the
  conservative one shipped as the default:

    exact_valid_time      the field is a point in time. A bucket can use it
                          only if the bucket is that instant, which no 6 h
                          bucket ever is. This is the default.

    target_composite_day  the training target is the DAILY COMPOSITE for the
                          target date, so the prediction describes that
                          calendar day -- the same interval convention this
                          project already applies to observed daily analyses
                          in environment_timeline.py, where one daily field
                          prices every bucket inside its day. Under this
                          reading a bucket may use the field only if the
                          bucket lies ENTIRELY inside the target day.

  Neither reading lets a 24 h forecast cover the 18-24 h bucket, which straddles
  two days. Under the composite-day reading the 24-30 h bucket IS covered,
  because it lies wholly inside the target day. That asymmetry is the point:
  coverage is decided by the interval, not by proximity to the valid time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from src.data.preprocess import RAW_DIR, load_sic
from src.models.forecast_sic_raster import (LEAD_HOURS, MODEL_NAME, OUT_DIR,
                                            TRAINED_LEAD_HOURS,
                                            ForecastRasterError, raster_path,
                                            validate_forecast_raster)

#  every lead this project can serve a PREDICTED field for. One entry, because
#  one model, one trained lead.
SUPPORTED_LEAD_HOURS = TRAINED_LEAD_HOURS

POLICY_EXACT = "exact_forecast_valid_time"
POLICY_OBSERVED_ANALYSIS = "observed_analysis"
POLICIES = (POLICY_EXACT, POLICY_OBSERVED_ANALYSIS)

KIND_PREDICTED = "predicted_sic_forecast"
KIND_OBSERVED = "observed_sic_analysis"


class EnvironmentForecastError(ValueError):
    """The requested environmental field cannot be resolved as specified."""


class UnsupportedForecastTime(EnvironmentForecastError):
    """No model produces a field for that valid time. Nothing is substituted."""


class ForecastArtifactUnavailable(EnvironmentForecastError):
    """The lead is supported but no artifact exists. Nothing is substituted."""


@dataclass(frozen=True)
class ResolvedEnvironment:
    """A field the caller may read, and the grounds on which it was served."""

    origin_time: datetime
    valid_time: datetime
    lead_hours: float
    policy: str
    kind: str
    path: Path

    @property
    def is_predicted(self) -> bool:
        return self.kind == KIND_PREDICTED

    def to_dict(self) -> dict:
        return {"origin_time": self.origin_time.isoformat(),
                "valid_time": self.valid_time.isoformat(),
                "lead_hours": self.lead_hours, "policy": self.policy,
                "kind": self.kind, "path": self.path.name}


def _lead_hours(origin_time: datetime, valid_time: datetime) -> float:
    if not isinstance(origin_time, datetime) or not isinstance(valid_time, datetime):
        raise EnvironmentForecastError(
            "origin_time and valid_time must both be datetimes")
    delta = (valid_time - origin_time).total_seconds() / 3600.0
    if delta < 0:
        raise UnsupportedForecastTime(
            f"valid_time {valid_time.isoformat()} is before origin_time "
            f"{origin_time.isoformat()}; there is no such forecast.")
    return delta


def supported_valid_times(origin_time: datetime) -> list[datetime]:
    """Exactly the valid times routing may ask for at this origin."""
    return [origin_time + timedelta(hours=h) for h in SUPPORTED_LEAD_HOURS]


def resolve_environment_forecast(origin_time: datetime, valid_time: datetime,
                                 policy: str = POLICY_EXACT, *,
                                 forecast_dir: Path = OUT_DIR,
                                 sic_dir: Path = RAW_DIR,
                                 ) -> ResolvedEnvironment:
    """Resolve one environmental field, or refuse and say why."""
    if policy not in POLICIES:
        raise EnvironmentForecastError(
            f"policy must be one of {POLICIES}, got {policy!r}. There is no "
            f"'nearest', 'latest' or 'persistence' policy here: carrying a "
            f"field forward is a routing-bucket decision and belongs to "
            f"src/api/environment_timeline.py.")
    lead = _lead_hours(origin_time, valid_time)

    if policy == POLICY_OBSERVED_ANALYSIS:
        if lead != 0:
            raise UnsupportedForecastTime(
                f"the observed analysis is valid at its own time only. "
                f"{valid_time.isoformat()} is {lead:g} h after the analysis "
                f"{origin_time.isoformat()}; serving the observation for it "
                f"would be persistence, which this resolver does not do "
                f"silently.")
        path = Path(sic_dir) / f"sic_{origin_time:%Y%m%d}.tif"
        if not path.exists():
            raise ForecastArtifactUnavailable(
                f"no observed field for {origin_time.date()}: {path.name} is "
                f"not in the archive. Nothing is substituted for it.")
        return ResolvedEnvironment(origin_time=origin_time, valid_time=valid_time,
                                   lead_hours=0.0, policy=policy,
                                   kind=KIND_OBSERVED, path=path)

    if lead not in SUPPORTED_LEAD_HOURS:
        raise UnsupportedForecastTime(
            f"no sea-ice forecast is produced for +{lead:g} h. The only "
            f"supported lead(s): {', '.join(f'+{h:g}h' for h in SUPPORTED_LEAD_HOURS)}. "
            f"The +{LEAD_HOURS:g} h field is NOT returned for another lead: it "
            f"was fitted for a one-day step and relabelling it would be a "
            f"fabricated forecast. Ask for "
            f"{[t.isoformat() for t in supported_valid_times(origin_time)]}, "
            f"or price this time from an analysis through "
            f"src/api/environment_timeline.py and say so.")

    origin_day = origin_time.date()
    path = raster_path(origin_day, Path(forecast_dir), int(lead))
    if not path.exists():
        raise ForecastArtifactUnavailable(
            f"the +{lead:g} h lead is supported, but no forecast artifact has "
            f"been generated for origin {origin_day}: {path.name} does not "
            f"exist. Generate it with `python -m src.models.forecast_sic_raster "
            f"--origin {origin_day}`. No other origin's forecast, and no "
            f"observation, is substituted for it.")
    #  a mismatched raster is rejected here, never reprojected at runtime
    validate_forecast_raster(path)
    return ResolvedEnvironment(origin_time=origin_time, valid_time=valid_time,
                               lead_hours=float(lead), policy=policy,
                               kind=KIND_PREDICTED, path=path)


# =========================================================== the lead registry
#  What the DATA can support, stated per lead and independent of any model.
STATUS_OBSERVATION = "observation_not_a_forecast"
STATUS_MODELLED = "model_available"
STATUS_DATA_ONLY = "data_supported_but_no_model_trained"
STATUS_NO_DATA = "no_training_target_exists_in_the_archive"

#  the cadence of every sea-ice source in this repository
SOURCE_CADENCE_HOURS = 24.0


@dataclass(frozen=True)
class LeadSupport:
    """One forecast lead, and exactly how far the project can stand behind it."""

    lead_hours: float
    training_target: str
    data_supported: bool
    model: str | None
    status: str
    note: str

    @property
    def routable(self) -> bool:
        """Can the router legitimately consume this lead today?"""
        return self.status in (STATUS_OBSERVATION, STATUS_MODELLED)

    def to_dict(self) -> dict:
        return {"lead_hours": self.lead_hours,
                "training_target": self.training_target,
                "data_supported": self.data_supported, "model": self.model,
                "status": self.status, "routable": self.routable,
                "note": self.note}


LEAD_SUPPORT: dict = {
    0: LeadSupport(
        lead_hours=0, training_target="none needed: it is the observation",
        data_supported=True, model=None, status=STATUS_OBSERVATION,
        note="the analysis itself. Serving it for any later time is "
             "persistence and must be asked for by name."),
    6: LeadSupport(
        lead_hours=6, training_target="none: no sub-daily sea-ice observation "
                                      "exists in this archive",
        data_supported=False, model=None, status=STATUS_NO_DATA,
        note="both sea-ice sources are daily composites, so there is no 6 h "
             "target to fit against. A 6 h field would have to be a relabelled "
             "24 h forecast or an interpolation between daily observations; "
             "neither is a forecast and neither is produced."),
    12: LeadSupport(
        lead_hours=12, training_target="none: no sub-daily sea-ice observation "
                                       "exists in this archive",
        data_supported=False, model=None, status=STATUS_NO_DATA,
        note="as for 6 h. Requires a sub-daily SIC product before any model "
             "can be trained."),
    24: LeadSupport(
        lead_hours=24, training_target="sic(t) -> sic(t+1 day), 362 usable "
                                       "pairs in the archive",
        data_supported=True, model=MODEL_NAME, status=STATUS_MODELLED,
        note="the one trained lead. Artifacts are written by "
             "src/models/forecast_sic_raster.py."),
    48: LeadSupport(
        lead_hours=48, training_target="sic(t) -> sic(t+2 days), 357 usable "
                                       "pairs with --gap-days 1",
        data_supported=True, model=MODEL_NAME, status=STATUS_MODELLED,
        note="its own artifact, forecast_model_hgb_48h.joblib, trained on its "
             "own +48 h dataset. It is never used for another lead, and no "
             "other lead's model is used for it."),
}


def lead_support(lead_hours: float) -> LeadSupport | None:
    """The registry entry for a lead, or None if the project never considered it."""
    for hours, entry in LEAD_SUPPORT.items():
        if float(hours) == float(lead_hours):
            return entry
    return None


def lead_support_table() -> list:
    return [LEAD_SUPPORT[h].to_dict() for h in sorted(LEAD_SUPPORT)]


# ===================================================== validity of a raster
VALIDITY_EXACT = "exact_valid_time"
VALIDITY_TARGET_DAY = "target_composite_day"
VALIDITY_RULES = (VALIDITY_EXACT, VALIDITY_TARGET_DAY)

VALIDITY_JUSTIFICATION = {
    VALIDITY_EXACT:
        "the field is treated as a single instant. Nothing is claimed about "
        "any other time, so a routing bucket can use it only if the bucket is "
        "that instant.",
    VALIDITY_TARGET_DAY:
        "the training target is the daily composite for the target date "
        "(both sea-ice sources are daily; OSI-430-a states "
        "time_coverage_resolution P1D), so the prediction describes that "
        "calendar day. This is the same interval convention the project "
        "already applies to an observed daily analysis, which prices every "
        "bucket inside its own day.",
}


def validity_interval(valid_time: datetime, rule: str = VALIDITY_EXACT):
    """The half-open interval a field may legitimately be used across."""
    if rule not in VALIDITY_RULES:
        raise EnvironmentForecastError(
            f"validity rule must be one of {VALIDITY_RULES}, got {rule!r}. "
            f"No interval is invented for an unknown rule.")
    if rule == VALIDITY_EXACT:
        return valid_time, valid_time
    start = datetime(valid_time.year, valid_time.month, valid_time.day)
    return start, start + timedelta(days=1)


def _covers(window, interval, rule: str) -> bool:
    """Does the field's validity cover the whole of what the caller asked for?"""
    want_from, want_to = window
    have_from, have_to = interval
    if rule == VALIDITY_EXACT:
        #  a point field covers only a request that is itself that instant
        return want_from == want_to == have_from
    #  half-open [have_from, have_to): the request must lie entirely inside
    return have_from <= want_from and want_to <= have_to


# ================================================ the four-answer resolver
SOURCE_OBSERVED = "observed_analysis"
SOURCE_FORECAST = "model_forecast"
SOURCE_PERSISTENCE = "explicit_persistence"
SOURCE_UNAVAILABLE = "unavailable"

RESOLVE_FORECAST_ONLY = "model_forecast_only"
RESOLVE_FORECAST_ELSE_PERSISTENCE = "model_forecast_else_explicit_persistence"
RESOLVE_POLICIES = (RESOLVE_FORECAST_ONLY, RESOLVE_FORECAST_ELSE_PERSISTENCE)


@dataclass(frozen=True)
class EnvironmentResolution:
    """What the routing layer may use for one arrival time, and on what grounds."""

    source_type: str
    forecast_origin: datetime
    valid_time: datetime | None
    lead_hours: float
    model: str | None
    fallback: str | None
    data_available: bool
    path: Path | None
    policy: str
    validity_rule: str
    validity_from: datetime | None
    validity_to: datetime | None
    reason: str

    @property
    def is_forecast(self) -> bool:
        return self.source_type == SOURCE_FORECAST

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "forecast_origin": self.forecast_origin.isoformat(),
            "valid_time": self.valid_time.isoformat() if self.valid_time else None,
            "lead_hours": self.lead_hours,
            "model": self.model,
            "fallback": self.fallback,
            "data_available": self.data_available,
            "path": self.path.name if self.path else None,
            "policy": self.policy,
            "validity_rule": self.validity_rule,
            "validity_from": (self.validity_from.isoformat()
                              if self.validity_from else None),
            "validity_to": (self.validity_to.isoformat()
                            if self.validity_to else None),
            "reason": self.reason,
        }


def resolve_environment(window_from: datetime, window_to: datetime = None, *,
                        origin_time: datetime,
                        policy: str = RESOLVE_FORECAST_ONLY,
                        validity_rule: str = VALIDITY_EXACT,
                        forecast_dir: Path = OUT_DIR,
                        sic_dir: Path = RAW_DIR) -> EnvironmentResolution:
    """What SIC field is legitimately available across [window_from, window_to]?

    Pass a single instant (window_to omitted) or a routing bucket's whole
    window. The answer is one of the four source types and never a silent
    substitution: an unavailable window says so, with the reason, rather than
    returning the nearest field.
    """
    if policy not in RESOLVE_POLICIES:
        raise EnvironmentForecastError(
            f"policy must be one of {RESOLVE_POLICIES}, got {policy!r}")
    if validity_rule not in VALIDITY_RULES:
        raise EnvironmentForecastError(
            f"validity rule must be one of {VALIDITY_RULES}, got "
            f"{validity_rule!r}")
    window_to = window_from if window_to is None else window_to
    if not isinstance(origin_time, datetime) or not isinstance(window_from, datetime) \
            or not isinstance(window_to, datetime):
        raise EnvironmentForecastError("all times must be datetimes")
    if window_to < window_from:
        raise EnvironmentForecastError(
            f"window ends {window_to.isoformat()} before it starts "
            f"{window_from.isoformat()}")

    def _persist(reason: str) -> EnvironmentResolution:
        """Only reachable when the caller named the fallback."""
        analysis = Path(sic_dir) / f"sic_{origin_time:%Y%m%d}.tif"
        available = analysis.exists()
        return EnvironmentResolution(
            source_type=SOURCE_PERSISTENCE if available else SOURCE_UNAVAILABLE,
            forecast_origin=origin_time, valid_time=None,
            lead_hours=(window_from - origin_time).total_seconds() / 3600.0,
            model=None,
            fallback=(f"explicit_persistence_from_{origin_time.date().isoformat()}"
                      if available else None),
            data_available=available, path=analysis if available else None,
            policy=policy, validity_rule=validity_rule,
            validity_from=None, validity_to=None,
            reason=(f"{reason} The caller asked for the explicit persistence "
                    f"fallback, so the {origin_time.date()} analysis is carried "
                    f"forward and labelled as carried forward."
                    if available else
                    f"{reason} The persistence fallback was requested but the "
                    f"{origin_time.date()} analysis is not in the archive."))

    def _unavailable(reason: str) -> EnvironmentResolution:
        if policy == RESOLVE_FORECAST_ELSE_PERSISTENCE:
            return _persist(reason)
        return EnvironmentResolution(
            source_type=SOURCE_UNAVAILABLE, forecast_origin=origin_time,
            valid_time=None,
            lead_hours=(window_from - origin_time).total_seconds() / 3600.0,
            model=None, fallback=None, data_available=False, path=None,
            policy=policy, validity_rule=validity_rule, validity_from=None,
            validity_to=None, reason=reason)

    #  the analysis itself, for its own instant
    if window_from == window_to == origin_time:
        analysis = Path(sic_dir) / f"sic_{origin_time:%Y%m%d}.tif"
        if analysis.exists():
            return EnvironmentResolution(
                source_type=SOURCE_OBSERVED, forecast_origin=origin_time,
                valid_time=origin_time, lead_hours=0.0, model=None,
                fallback=None, data_available=True, path=analysis,
                policy=policy, validity_rule=validity_rule,
                validity_from=origin_time, validity_to=origin_time,
                reason="the observed analysis, at its own time")
        return _unavailable(
            f"no observed analysis for {origin_time.date()} in the archive.")

    #  every lead this project has an artifact for, newest-covering first
    for lead in sorted(SUPPORTED_LEAD_HOURS):
        valid = origin_time + timedelta(hours=lead)
        interval = validity_interval(valid, validity_rule)
        if not _covers((window_from, window_to), interval, validity_rule):
            continue
        path = raster_path(origin_time.date(), Path(forecast_dir), int(lead))
        if not path.exists():
            return _unavailable(
                f"+{lead:g} h is a supported lead and its validity "
                f"({validity_rule}) covers the request, but no artifact has "
                f"been generated for origin {origin_time.date()}: "
                f"{path.name} does not exist. No other origin's forecast and "
                f"no observation is substituted for it.")
        validate_forecast_raster(path)
        return EnvironmentResolution(
            source_type=SOURCE_FORECAST, forecast_origin=origin_time,
            valid_time=valid, lead_hours=float(lead),
            model=LEAD_SUPPORT[int(lead)].model, fallback=None,
            data_available=True, path=path, policy=policy,
            validity_rule=validity_rule, validity_from=interval[0],
            validity_to=interval[1],
            reason=(f"the +{lead:g} h forecast from {origin_time.date()}; its "
                    f"{validity_rule} validity covers the whole request"))

    #  nothing covered it: say which lead would have, and why it cannot
    asked = (window_from - origin_time).total_seconds() / 3600.0
    entry = lead_support(asked)
    if entry is not None and not entry.routable:
        return _unavailable(
            f"+{asked:g} h is registered as {entry.status}: {entry.note}")
    return _unavailable(
        f"no forecast covers {window_from.isoformat()}"
        + ("" if window_to == window_from else f" .. {window_to.isoformat()}")
        + f" from origin {origin_time.isoformat()} under the "
          f"{validity_rule} validity rule. Supported lead(s): "
        + ", ".join(f"+{h:g}h" for h in SUPPORTED_LEAD_HOURS)
        + ". A field is never stretched to a window it does not cover.")


def resolver_for_policy(policy: str, *, origin_time: datetime,
                        forecast_dir: Path = OUT_DIR, sic_dir: Path = RAW_DIR,
                        resolve_policy: str = RESOLVE_FORECAST_ONLY):
    """The resolver a routing policy must be driven with.

    The validity rule is not the caller's to pick: each routing policy is bound
    to one, in environment_timeline.POLICY_VALIDITY_RULE, and this returns a
    resolver already carrying it. `exact_valid_time` remains available for
    strict checking by calling resolve_environment directly with it.
    """
    from functools import partial

    from src.api.environment_timeline import POLICY_VALIDITY_RULE
    if policy not in POLICY_VALIDITY_RULE:
        raise EnvironmentForecastError(
            f"no validity rule is bound to routing policy {policy!r}. Bound "
            f"policies: {sorted(POLICY_VALIDITY_RULE)}.")
    rule = POLICY_VALIDITY_RULE[policy]
    if rule not in VALIDITY_RULES:
        raise EnvironmentForecastError(
            f"policy {policy!r} is bound to validity rule {rule!r}, which this "
            f"module does not implement: {VALIDITY_RULES}.")
    return partial(resolve_environment, origin_time=origin_time,
                   policy=resolve_policy, validity_rule=rule,
                   forecast_dir=forecast_dir, sic_dir=sic_dir)


def read_environment(resolved) -> np.ndarray:
    """The field itself, under the project's own SIC masking rule.

    Predicted rasters already carry NaN where nothing was predicted; observed
    rasters are read through preprocess.load_sic, which is the same loader the
    router's RoutingGrid uses, so the two cannot disagree about what is valid.
    """
    observed = getattr(resolved, "kind", None) == KIND_OBSERVED or \
        getattr(resolved, "source_type", None) in (SOURCE_OBSERVED,
                                                   SOURCE_PERSISTENCE)
    if resolved.path is None:
        raise EnvironmentForecastError(
            "this resolution carries no field to read: "
            f"{getattr(resolved, 'reason', resolved)}")
    if observed:
        return np.asarray(load_sic(resolved.path).sic, dtype="float32")
    try:
        import rasterio
        with rasterio.open(resolved.path) as src:
            return np.asarray(src.read(1), dtype="float32")
    except ForecastRasterError:                                  # pragma: no cover
        raise
