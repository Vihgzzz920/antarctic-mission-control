"""
Which environmental field each routing bucket is entitled to, and why.

    timeline = plan_environment_timeline(
        buckets=[0, 1], bucket_seconds=6 * 3600.0,
        departure_time=datetime(2025, 1, 8),
        policy=POLICY_PERSISTENCE, available_dates=archive_dates())
    for entry in timeline:
        print(entry.explain())

WHY THIS EXISTS
  time_navigation_cost already serves a composed cost BY ARRIVAL TIME: the
  bucket rule, the lookup and the refusal to substitute a missing field all
  live there and are not restated here. What that layer cannot know is WHICH
  environmental raster each bucket should have been built from. Until this
  module existed the answer was implicit -- src/api/world.py composed one field
  from the departure day's observation and registered the SAME object for every
  bucket, so every future arrival time was priced against the departure-day
  environment without anything on the route saying so.

  That is a legitimate modelling choice. It is persistence, the project's own
  published sea-ice baseline (src/models/persistence.py), and for a route that
  finishes inside the analysis day it is also exact. It is not legitimate for
  it to be silent. This module makes the choice explicit, per bucket, with the
  source it resolved to and whether anything was carried forward.

WHAT A POLICY IS
  persistence_from_departure_analysis   every bucket uses the departure day's
                                        analysis. A bucket whose own valid
                                        window falls on a later date is marked
                                        `persisted`, because that is what it is.

  dated_observation_hindcast            each bucket uses the observation for
                                        its OWN valid date when the archive has
                                        one. An observation later than the
                                        departure analysis is PERFECT FORESIGHT,
                                        not a forecast, and every such entry is
                                        stamped uses_future_observation=True.
                                        It exists so a persistence baseline and
                                        a time-varying environment can be run
                                        over the same mission inputs; it must
                                        never be presented as a forecast.

  There is no third policy, and in particular there is no policy that reads a
  predicted sea-ice raster: this project has a trained SIC model but it writes
  no rasters, so there is nothing to read. That gap is documented rather than
  filled with an invented field.

NOTHING IS FABRICATED AND NOTHING IS SILENT
  A bucket whose required date is absent from the archive is either refused
  (on_missing="error") or served the departure analysis with
  `fallback` recording exactly that. The decision is returned as data, so the
  route can carry it and the API can show it.

NO I/O
  This module opens no file. `available_dates` is supplied by the caller, which
  is what makes the rules testable without the data archive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

POLICY_PERSISTENCE = "persistence_from_departure_analysis"
POLICY_DATED_OBSERVATION = "dated_observation_hindcast"
#  resolve each bucket through the forecast registry; see the module docstring
#  and src/api/environment_forecast.py. Requires a `resolver`.
POLICY_MODEL_FORECAST = "model_forecast_when_supported"
#  An EVALUATION policy, not a production one. It resolves buckets exactly as
#  POLICY_MODEL_FORECAST does; what makes it separate is the cost stack it is
#  run with -- see src/api/long_horizon_evaluation.py, where the iceberg
#  exposure term is OMITTED beyond its configured horizon and the omission is
#  recorded on the result. It exists so that an evaluation run can never be
#  mistaken for the production policy in a route's own provenance.
POLICY_LONG_HORIZON_EVALUATION = "long_horizon_forecast_evaluation"
POLICIES = (POLICY_PERSISTENCE, POLICY_DATED_OBSERVATION, POLICY_MODEL_FORECAST,
            POLICY_LONG_HORIZON_EVALUATION)
#  the policies that resolve a bucket through the forecast registry
FORECAST_POLICIES = (POLICY_MODEL_FORECAST, POLICY_LONG_HORIZON_EVALUATION)

#  THE VALIDITY RULE EACH POLICY IS BOUND TO.
#
#  A policy is a statement about which environment a bucket may use; a validity
#  rule is a statement about how far one field's claim extends. They are not
#  independent, so the pairing is declared here rather than left to whoever
#  builds the resolver: the opt-in forecast policy uses `target_composite_day`,
#  and a resolver handed to it that answers under any other rule is REFUSED.
#
#  The string is spelled out rather than imported so that this module stays
#  free of file access; tests assert it equals
#  environment_forecast.VALIDITY_TARGET_DAY.
POLICY_VALIDITY_RULE = {
    POLICY_MODEL_FORECAST: "target_composite_day",
    POLICY_LONG_HORIZON_EVALUATION: "target_composite_day",
}

#  the four answers the forecast registry can give, re-exported as the
#  vocabulary a bucket reports. They are not redefined here.
SOURCE_OBSERVED = "observed_analysis"
SOURCE_FORECAST = "model_forecast"
SOURCE_PERSISTENCE = "explicit_persistence"
SOURCE_UNAVAILABLE = "unavailable"

ON_MISSING_ERROR = "error"
ON_MISSING_PERSIST = "persist_from_departure_analysis"
ON_MISSING = (ON_MISSING_ERROR, ON_MISSING_PERSIST)

HOUR = 3600.0


class EnvironmentTimelineError(ValueError):
    """The environment timeline cannot be planned as specified."""


class EnvironmentUnavailable(EnvironmentTimelineError):
    """No environmental field exists for a bucket. Nothing is substituted."""


@dataclass(frozen=True)
class BucketEnvironment:
    """Which environment one routing bucket gets, and on what grounds."""

    bucket: int
    #  seconds after departure; the same clock time_navigation_cost buckets on
    valid_from_s: float
    valid_to_s: float
    valid_from: datetime
    valid_to: datetime
    #  the analysis this environment is issued from: the departure day
    analysis_date: date
    #  the date of the raster actually read
    environment_date: date
    policy: str
    source: str
    #  None, or a statement of what was carried forward and from when
    fallback: str | None
    #  the environment is older than the window it is pricing
    persisted: bool
    #  the environment is an observation from AFTER the analysis: perfect
    #  foresight, usable for a baseline comparison, never a forecast
    uses_future_observation: bool
    #  which of the registry's four answers this bucket got. The two
    #  observation policies can only ever produce the first or the third.
    source_type: str = SOURCE_OBSERVED
    #  the model behind the field, when it is a forecast rather than an
    #  observation. None is not "unknown": it means no model was involved.
    model: str | None = None
    #  the forecast's own valid time, for a bucket served by a model
    valid_time: datetime | None = None

    @property
    def horizon_hours(self) -> float:
        """Hours after departure at which this bucket's window opens."""
        return self.valid_from_s / HOUR

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket,
            "valid_from_s": self.valid_from_s,
            "valid_to_s": self.valid_to_s,
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat(),
            "horizon_hours": self.horizon_hours,
            "analysis_date": self.analysis_date.isoformat(),
            "environment_date": self.environment_date.isoformat(),
            "policy": self.policy,
            "source": self.source,
            "fallback": self.fallback,
            "persisted": self.persisted,
            "uses_future_observation": self.uses_future_observation,
            "source_type": self.source_type,
            "model": self.model,
            "valid_time": self.valid_time.isoformat() if self.valid_time else None,
        }

    def explain(self) -> str:
        line = (f"bucket {self.bucket}: {self.valid_from.isoformat()} .. "
                f"{self.valid_to.isoformat()} -> {self.source}")
        if self.fallback:
            line += f"   [{self.fallback}]"
        if self.uses_future_observation:
            line += "   [OBSERVATION FROM AFTER THE ANALYSIS: perfect " \
                    "foresight, not a forecast]"
        if self.source_type == SOURCE_FORECAST:
            line += (f"   [{self.model} forecast valid "
                     f"{self.valid_time.isoformat() if self.valid_time else '?'}]")
        return line


def _validate(buckets, bucket_seconds, departure_time, policy, on_missing):
    if policy not in POLICIES:
        raise EnvironmentTimelineError(
            f"policy must be one of {POLICIES}, got {policy!r}")
    if on_missing not in ON_MISSING:
        raise EnvironmentTimelineError(
            f"on_missing must be one of {ON_MISSING}, got {on_missing!r}")
    if not isinstance(departure_time, datetime):
        raise EnvironmentTimelineError(
            f"departure_time must be a datetime, got "
            f"{type(departure_time).__name__}")
    if not (isinstance(bucket_seconds, (int, float))
            and math.isfinite(bucket_seconds) and bucket_seconds > 0):
        raise EnvironmentTimelineError(
            f"bucket_seconds must be finite and > 0, got {bucket_seconds!r}")
    ordered = sorted({int(b) for b in buckets})
    if not ordered:
        raise EnvironmentTimelineError("no buckets supplied")
    if any(b < 0 for b in ordered):
        raise EnvironmentTimelineError(
            f"bucket indices must be >= 0, got {ordered}")
    return ordered


def plan_environment_timeline(*, buckets, bucket_seconds: float,
                              departure_time: datetime,
                              policy: str = POLICY_PERSISTENCE,
                              available_dates=(),
                              analysis_date: date | None = None,
                              on_missing: str = ON_MISSING_ERROR,
                              resolver=None,
                              ) -> list[BucketEnvironment]:
    """Resolve every routing bucket to the environment it will be priced with.

    The bucket's window is [b * bucket_seconds, (b + 1) * bucket_seconds) after
    departure -- the same arithmetic time_navigation_cost.bucket_for inverts,
    written here only to name the window, never to re-decide a lookup.

    A bucket is dated by the START of its window. Sea ice arrives as one field
    per day, so a window that opens on a date is priced with that date's field;
    splitting a bucket across two daily fields would mean interpolating a field
    the archive does not contain.
    """
    ordered = _validate(buckets, bucket_seconds, departure_time, policy,
                        on_missing)
    base = analysis_date or departure_time.date()
    if policy in FORECAST_POLICIES and resolver is None:
        raise EnvironmentTimelineError(
            f"policy {policy!r} needs a resolver: a callable "
            f"(window_from, window_to) -> resolution, normally "
            f"functools.partial(environment_forecast.resolve_environment, "
            f"origin_time=..., validity_rule=...). This module stays free of "
            f"file access so the rules above can be tested without the "
            f"archive; the registry is the one place that reads artifacts.")
    have = {d if isinstance(d, date) else date.fromisoformat(str(d))
            for d in available_dates}

    out: list[BucketEnvironment] = []
    for b in ordered:
        start_s = b * float(bucket_seconds)
        end_s = (b + 1) * float(bucket_seconds)
        valid_from = departure_time + timedelta(seconds=start_s)
        valid_to = departure_time + timedelta(seconds=end_s)
        wanted = valid_from.date()

        if policy in FORECAST_POLICIES:
            #  ask the registry what legitimately covers this bucket's WHOLE
            #  window. A field is never stretched to a window it does not
            #  cover, and a bucket that nothing covers is refused unless the
            #  documented fallback was asked for.
            answer = resolver(valid_from, valid_to)
            #  the policy is bound to one validity rule; a resolver answering
            #  under another would silently change what "supported" means
            expected = POLICY_VALIDITY_RULE[policy]
            got = getattr(answer, "validity_rule", None)
            if got != expected:
                raise EnvironmentTimelineError(
                    f"policy {policy!r} is bound to the {expected!r} validity "
                    f"rule, but the resolver answered bucket {b} under "
                    f"{got!r}. Build the resolver with "
                    f"environment_forecast.resolver_for_policy({policy!r}, "
                    f"origin_time=...), which binds the rule, rather than "
                    f"passing a different one.")
            if getattr(answer, "source_type", None) == SOURCE_FORECAST:
                out.append(BucketEnvironment(
                    bucket=b, valid_from_s=start_s, valid_to_s=end_s,
                    valid_from=valid_from, valid_to=valid_to,
                    analysis_date=base,
                    environment_date=answer.valid_time.date(), policy=policy,
                    source=f"forecast:{answer.path.name}", fallback=None,
                    persisted=False, uses_future_observation=False,
                    source_type=SOURCE_FORECAST, model=answer.model,
                    valid_time=answer.valid_time))
                continue
            reason = getattr(answer, "reason", str(answer))
            if on_missing == ON_MISSING_ERROR:
                raise EnvironmentUnavailable(
                    f"bucket {b} ({valid_from.isoformat()} .. "
                    f"{valid_to.isoformat()}) has no legitimate forecast: "
                    f"{reason} Nothing is substituted: pass "
                    f"on_missing={ON_MISSING_PERSIST!r} to carry the departure "
                    f"analysis forward deliberately.")
            out.append(BucketEnvironment(
                bucket=b, valid_from_s=start_s, valid_to_s=end_s,
                valid_from=valid_from, valid_to=valid_to, analysis_date=base,
                environment_date=base, policy=policy,
                source=f"observation:{base.isoformat()}",
                fallback=f"persisted_from_{base.isoformat()}",
                persisted=True, uses_future_observation=False,
                source_type=SOURCE_PERSISTENCE))
            continue

        if policy == POLICY_PERSISTENCE:
            if base not in have and have:
                raise EnvironmentUnavailable(
                    f"the departure analysis {base} is not in the environment "
                    f"archive; nothing is substituted for it.")
            out.append(BucketEnvironment(
                bucket=b, valid_from_s=start_s, valid_to_s=end_s,
                valid_from=valid_from, valid_to=valid_to,
                analysis_date=base, environment_date=base, policy=policy,
                source=f"observation:{base.isoformat()}",
                fallback=(None if wanted == base
                          else f"persisted_from_{base.isoformat()}"),
                persisted=wanted != base, uses_future_observation=False,
                source_type=(SOURCE_OBSERVED if wanted == base
                             else SOURCE_PERSISTENCE)))
            continue

        #  POLICY_DATED_OBSERVATION
        if wanted in have:
            out.append(BucketEnvironment(
                bucket=b, valid_from_s=start_s, valid_to_s=end_s,
                valid_from=valid_from, valid_to=valid_to,
                analysis_date=base, environment_date=wanted, policy=policy,
                source=f"observation:{wanted.isoformat()}", fallback=None,
                persisted=False, uses_future_observation=wanted > base,
                source_type=SOURCE_OBSERVED))
            continue
        if on_missing == ON_MISSING_ERROR:
            raise EnvironmentUnavailable(
                f"bucket {b} ({valid_from.isoformat()} .. "
                f"{valid_to.isoformat()}) needs the environment for {wanted}, "
                f"which the archive does not have. Nothing is substituted: "
                f"pass on_missing={ON_MISSING_PERSIST!r} to carry the "
                f"departure analysis forward deliberately.")
        if base not in have and have:
            raise EnvironmentUnavailable(
                f"bucket {b} needs {wanted}, which is absent, and the "
                f"departure analysis {base} is absent too; there is nothing "
                f"to carry forward.")
        out.append(BucketEnvironment(
            bucket=b, valid_from_s=start_s, valid_to_s=end_s,
            valid_from=valid_from, valid_to=valid_to,
            analysis_date=base, environment_date=base, policy=policy,
            source=f"observation:{base.isoformat()}",
            fallback=f"persisted_from_{base.isoformat()}",
            persisted=True, uses_future_observation=False,
            source_type=SOURCE_PERSISTENCE))
    return out


def timeline_summary(timeline) -> dict:
    """What the route can say about its own environmental temporality."""
    entries = list(timeline)
    if not entries:
        return {"buckets": [], "environment_is_time_varying": False}
    dates = {e.environment_date for e in entries}
    return {
        "source_types": sorted({e.source_type for e in entries}),
        "buckets_from_model_forecast": [e.bucket for e in entries
                                        if e.source_type == SOURCE_FORECAST],
        "models": sorted({e.model for e in entries if e.model}),
        "policy": entries[0].policy,
        "analysis_date": entries[0].analysis_date.isoformat(),
        "buckets": [e.to_dict() for e in entries],
        "environment_dates": sorted(d.isoformat() for d in dates),
        #  the honest headline: did the environment actually change with the
        #  arrival time, or was one field priced for every bucket?
        "environment_is_time_varying": len(dates) > 1,
        "environment_persisted_buckets": [e.bucket for e in entries
                                          if e.persisted],
        "uses_future_observations": any(e.uses_future_observation
                                        for e in entries),
    }
