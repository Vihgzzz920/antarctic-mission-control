"""
Serialise the iceberg forecast the model has ALREADY produced.

    GET /api/forecast/icebergs

This module runs no model. It calls iceberg_risk.forecast_icebergs() -- the
same function iceberg_risk.run() calls to build the shipped exposure rasters --
once per configured horizon, over the same population, with the same
calibration, and hands back what it returns. There is no drift rule, no radius
rule, no interpolation and no new horizon here.

OBSERVED IS NOT PREDICTED
  Horizon 0 returns each iceberg's own observed start position, which is the
  model's INPUT. Every other horizon is a prediction advected from it. The two
  are serialised under different keys and carry different labels so a screen
  can never present one as the other.
"""
from __future__ import annotations

import json
from functools import lru_cache

import src.models.iceberg_uncertainty as uncertainty
import src.routing.iceberg_risk as risk

from src.api.config import DEMO, ROOT

PHYSICS_CONFIG = ROOT / "configs" / "iceberg_physics.json"
OBSERVED_LABEL = "Observed"


class ForecastError(RuntimeError):
    """The forecast the API was asked for cannot be produced from real data."""


@lru_cache(maxsize=1)
def _calibration():
    """The population, split and uncertainty calibration, built once.

    Exactly the three lines iceberg_risk.run() opens with; nothing is re-fitted
    and nothing is written.
    """
    population, _ = risk.build_population()
    risk.chronological_split(population, risk.load_split_config(risk.SPLIT_CONFIG))
    unc_cfg = uncertainty.load_config(risk.UNC_CONFIG)
    return population, risk.calibrate(population, unc_cfg), unc_cfg


def _physics_description() -> dict:
    """What the drift model actually is, read from its own config."""
    raw = json.loads(PHYSICS_CONFIG.read_text())
    windage = float(raw["windage_coefficient"])
    return {
        "name": raw.get("honest_description",
                        "first-order current + windage drift baseline"),
        "windage_coefficient": windage,
        "wind_turn_deg": float(raw["wind_turn_deg"]),
        #  The windage coefficient ships at 0.0, so at the shipped setting the
        #  drift is current advection alone. Saying "current + windage" without
        #  that would overstate what is running.
        "windage_is_active": windage != 0.0,
        "summary": ("current advection + windage"
                    if windage != 0.0
                    else "current advection only at the shipped windage "
                         "coefficient of 0.0"),
        "config": str(PHYSICS_CONFIG.relative_to(ROOT)),
        "is_validated_science": False,
    }


def _state(forecast, label: str) -> dict:
    return {
        "hours": forecast.horizon_hours,
        "label": label,
        "latitude": forecast.latitude,
        "longitude": forecast.longitude,
        "radius_km": forecast.radius_km,
        "extrapolated_uncertainty": bool(forecast.extrapolated_uncertainty),
        "physics_position_is_extrapolated": bool(
            forecast.physics_position_is_extrapolated),
        "growth_law": forecast.growth_law,
        "calibration_quantile": forecast.calibration_quantile,
        "calibration_n": forecast.calibration_n,
        "interpolation_quality": forecast.interpolation_quality,
    }


@lru_cache(maxsize=4)
def iceberg_forecast(date: str | None = None) -> dict:
    """Observed position plus one state per configured horizon, per iceberg."""
    population, calibration, unc_cfg = _calibration()
    del population
    cfg = risk.load_config(risk.CONFIG)
    try:
        rows, info = risk._population(date or DEMO["environment_date"],
                                      risk.SPLIT_CONFIG)
    except risk.ExposureError as exc:
        raise ForecastError(str(exc)) from exc

    horizons = [0.0] + [float(h) for h in cfg.horizons_hours]
    by_iceberg: dict[str, dict] = {}
    states_by_horizon: dict[float, list] = {}

    for hours in horizons:
        produced = risk.forecast_icebergs(
            rows, hours * risk.SECONDS_PER_HOUR, calibration, unc_cfg, cfg)
        states_by_horizon[hours] = produced
        for forecast in produced:
            entry = by_iceberg.setdefault(forecast.iceberg_id, {
                "iceberg_id": forecast.iceberg_id,
                "position_source": forecast.position_source,
                "forecast_start_date": forecast.forecast_start_date,
                "observed": None,
                "predicted": [],
            })
            if hours == 0.0:
                entry["observed"] = _state(forecast, OBSERVED_LABEL)
            else:
                entry["predicted"].append(
                    _state(forecast, f"Predicted +{hours:g}h"))

    horizon_summary = []
    for hours in horizons[1:]:
        sample = states_by_horizon[hours][0]
        horizon_summary.append({
            "hours": hours,
            "label": f"+{hours:g}h",
            "radius_km": sample.radius_km,
            "extrapolated_uncertainty": bool(sample.extrapolated_uncertainty),
            "calibrated_to_hours": 24.0,
        })

    return {
        "forecast_start_date": info["forecast_start_date"],
        "icebergs": sorted(by_iceberg.values(), key=lambda e: e["iceberg_id"]),
        "iceberg_count": len(by_iceberg),
        "observed_label": OBSERVED_LABEL,
        "observed_meaning": "the iceberg's own observed start position: the "
                            "model's INPUT, not a forecast",
        "predicted_meaning": "advected from the observed position by the drift "
                             "baseline; a MODEL PREDICTION",
        "horizons": horizon_summary,
        "model": _physics_description(),
        "uncertainty": {
            "source": "src/models/iceberg_uncertainty.py",
            "calibrated_to_hours": 24.0,
            "growth_law": horizon_summary[0]["radius_km"] and
                          states_by_horizon[horizons[1]][0].growth_law,
            "beyond_calibration_is_extrapolated": True,
            "square_root_growth_is_a_project_assumption": True,
        },
        "population": {
            "rows_on_this_date": info["rows_on_this_date"],
            "distinct_icebergs_on_this_date":
                info["distinct_icebergs_on_this_date"],
            "population_note": "clean observed-to-observed MOVED rows, ASCAT "
                               "only; any other iceberg is invisible to this "
                               "forecast",
        },
    }
