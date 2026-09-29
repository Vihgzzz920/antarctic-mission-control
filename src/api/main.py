"""
The thinnest FastAPI layer over the existing routing stack.

    uvicorn src.api.main:app --reload --port 8000

Four things happen here and nothing else: a request is validated, the existing
route_profiles.compare_profiles() is called, its own serialisation is returned,
and the paths are rendered as GeoJSON in the project CRS. There is no A*, no
cost model, no POLARIS arithmetic and no fallback data in this package. When
the backend refuses a request -- a historical date mismatch, an unreachable
goal, a missing forecast bucket -- that refusal is what the API returns. It is
never replaced with a route.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

import hashlib
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.routing.end_to_end_route import RouteError
from src.routing.navigation_domain import NavigationDomain
from src.routing.route_profiles import PROFILES, ProfileError, compare_profiles

from src.api import forecast, geo, simulate
from src.api.config import (BEDMACHINE, DEMO, ROOT, USNIC,
                            demo_departure)
from src.api.mission_decision import (DEFAULT_POLICY, DEMONSTRATION_SITES,
                                      LONG_HORIZON_SITE, LONG_HORIZON_START,
                                      NO_FEASIBLE_OPTION, SUPPORTED_POLICIES,
                                      MissionError, Site, evaluate_mission)
from src.api.world import (WorldError, fresh_grid, get_world, grid_info,
                           sic_for_bucket)

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "backend_data_unavailable"

app = FastAPI(
    title="Antarctic Mission Control API",
    version="0.1.0",
    description="A decision-support PROTOTYPE. Not an operational navigation "
                "service, and not a source of navigational advice.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class SimulatedIceberg(BaseModel):
    """Where the operator put the simulated berg, in projected metres."""

    x: float
    y: float


class CandidateSite(BaseModel):
    """One candidate destination. The caller says where it came from and
    whether it is a real operational location; the server never assumes it."""

    site_id: str
    cell: list[int] = Field(..., min_length=2, max_length=2)
    label: str = ""
    is_operational: bool = False
    source: str = "supplied by the caller"


class MissionEvaluateRequest(BaseModel):
    """Everything the mission-decision layer needs, stated explicitly.

    Nothing important is hidden in a server default: the start, the candidate
    destinations, the departure times and the profiles are all required, so a
    caller cannot get a matrix without having said what it is a matrix OF.
    """

    start: list[int] = Field(..., min_length=2, max_length=2)
    sites: list[CandidateSite] = Field(..., min_length=1)
    departure_times: list[str] = Field(..., min_length=1)
    profiles: list[str] = Field(..., min_length=1)
    vessel_speed_mps: float = DEMO["vessel_speed_mps"]
    polaris_ice_class: str = DEMO["polaris_ice_class"]
    polaris_riv_table: str = DEMO["polaris_riv_table"]
    #  OPT-IN and explicit. Omitted -> the shipped demo policy, unchanged.
    #  Named -> that exact policy or a structured refusal; never a substitute.
    environment_policy: str | None = None
    #  supplying a horizon longer than the world's own provider puts the run in
    #  long-horizon EVALUATION mode, where the iceberg term is omitted
    horizon_buckets: int | None = None


class MissionRequest(BaseModel):
    start: list[int] = Field(..., min_length=2, max_length=2)
    goal: list[int] = Field(..., min_length=2, max_length=2)
    departure_time: str = DEMO["departure_time"]
    vessel_speed_mps: float = DEMO["vessel_speed_mps"]
    polaris_ice_class: str = DEMO["polaris_ice_class"]
    polaris_riv_table: str = DEMO["polaris_riv_table"]
    historical_demo_override: bool = False


def _enrich(world, comparison: dict, branch: str = "conservative") -> None:
    """Add the one audit field the route result does not carry: the USNIC
    dominant fraction for the cell. A lookup through the existing composition,
    not a computation."""
    composition = world["composition"]
    for profile in comparison["profiles"].values():
        if not profile["success"]:
            continue
        for cell in profile["cells"]:
            c = composition.cell(cell["row"], cell["col"], branch)
            cell["dominant_fraction"] = c.dominant_fraction
            cell["charted"] = bool(c.charted)
            cell["mixed"] = bool(c.mixed)


def _status(comparison: dict) -> tuple[str, str]:
    """The backend's own outcome, passed through unchanged."""
    for name in PROFILES:
        profile = comparison["profiles"][name]
        if not profile["success"]:
            return profile["outcome"], profile["reason"]
    return STATUS_OK, "all three profiles produced a route"


def _run(request: MissionRequest, provider=None) -> dict:
    try:
        world = get_world()
    except WorldError as exc:
        raise HTTPException(status_code=503, detail={
            "status": STATUS_UNAVAILABLE, "message": str(exc)}) from exc
    try:
        departure = datetime.fromisoformat(request.departure_time)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={
            "status": "bad_request",
            "message": f"departure_time is not an ISO timestamp: {exc}"}) from exc

    try:
        comparison = compare_profiles(
            fresh_grid, risk_provider=provider or world["provider"],
            usnic_dir=USNIC,
            start=tuple(request.start), goal=tuple(request.goal),
            departure_time=departure, vessel_speed_mps=request.vessel_speed_mps,
            constraints_factory=lambda g: NavigationDomain(g).add_geotiff(
                "land_mask", BEDMACHINE),
            polaris_ice_class=request.polaris_ice_class,
            polaris_riv_table=request.polaris_riv_table,
            polaris_penalties=world["polaris"],
            sic_for_bucket=sic_for_bucket(),
            allow_historical_demo_override=request.historical_demo_override)
    except (ProfileError, RouteError) as exc:
        raise HTTPException(status_code=422, detail={
            "status": "rejected", "message": str(exc)}) from exc

    payload = comparison.to_dict()
    _enrich(world, payload)
    status, message = _status(payload)
    grid = world["grid"]
    epsg = grid.crs.to_epsg()
    return {
        "status": status,
        "message": message,
        "request": request.model_dump(),
        "comparison": payload,
        "geojson": geo.route_collections(grid, payload, epsg),
        "endpoints": geo.endpoint_collection(grid, request.start, request.goal,
                                             epsg),
        "grid": grid_info(),
        "historical_demonstration": bool(request.historical_demo_override),
    }


@app.get("/api/health")
def health() -> dict:
    """Is the backend able to serve measured data right now?"""
    try:
        info = grid_info()
    except (WorldError, Exception) as exc:                        # noqa: BLE001
        return {"status": STATUS_UNAVAILABLE, "message": str(exc),
                "prototype": True}
    world = get_world()
    return {
        "status": STATUS_OK,
        "prototype": True,
        "disclaimer": "decision-support prototype; not an operational "
                      "navigation service",
        "grid": info,
        "iceberg_exposure_weight": world["provider"].config.weight,
        "exposure_layers": world["exposure_meta"],
        "polaris": world["polaris"].to_dict(),
        "requires_historical_demo_override": not world["dates_aligned"],
    }


@app.get("/api/mission/defaults")
def defaults() -> dict:
    """The validated demonstration's inputs, for pre-filling the form.

    `sites` and `policies` are published so the frontend never has to invent a
    destination coordinate or a policy name. Every site here declares whether
    it is a real operational location -- none is -- and says where its
    coordinate came from.
    """
    return {
        "demo": DEMO,
        "grid": grid_info(),
        "sites": [s.to_dict() for s in DEMONSTRATION_SITES],
        "long_horizon_site": LONG_HORIZON_SITE.to_dict(),
        "long_horizon_start": list(LONG_HORIZON_START),
        "policies": {
            "default": DEFAULT_POLICY,
            "supported": list(SUPPORTED_POLICIES),
        },
    }


@app.get("/api/demo/routes")
def demo_routes(historical_demo_override: bool = Query(False)) -> dict:
    """The validated 2025-01-08 demonstration.

    The override defaults OFF, so by default this endpoint returns the SAME
    refusal the routing layer gives for a 2020 chart under 2025 fields.
    """
    return _run(MissionRequest(
        start=DEMO["start"], goal=DEMO["goal"],
        departure_time=demo_departure().isoformat(),
        vessel_speed_mps=DEMO["vessel_speed_mps"],
        polaris_ice_class=DEMO["polaris_ice_class"],
        polaris_riv_table=DEMO["polaris_riv_table"],
        historical_demo_override=historical_demo_override))


@app.post("/api/mission/compare")
def mission_compare(request: MissionRequest) -> dict:
    """Run the three profiles through the existing orchestration."""
    return _run(request)


@app.get("/api/forecast/icebergs")
def forecast_icebergs(date: str | None = Query(None)) -> dict:
    """Observed position plus every configured predicted horizon, per iceberg.

    Serialisation only: the positions and radii are what
    iceberg_risk.forecast_icebergs() returns for the same population the
    shipped exposure rasters were built from.
    """
    try:
        return forecast.iceberg_forecast(date)
    except forecast.ForecastError as exc:
        raise HTTPException(status_code=404, detail={
            "status": "no_forecast", "message": str(exc)}) from exc


@app.get("/api/layers/iceberg-exposure")
def iceberg_exposure(bucket: int = Query(1, ge=0)) -> dict:
    """The non-zero cells of one real exposure raster, as GeoJSON points."""
    world = get_world()
    fields = {f.bucket: f for f in world["exposure_fields"]}
    if bucket not in fields:
        raise HTTPException(status_code=404, detail={
            "status": "no_such_bucket",
            "message": f"exposure buckets available: {sorted(fields)}"})
    return geo.exposure_collection(world["grid"], fields[bucket],
                                   world["grid"].crs.to_epsg())


@app.get("/api/evaluation/historical")
def historical_evaluation() -> dict:
    """The held-out historical forecast -> decision backtest, as produced.

    Serialisation only: the file is read and returned. No number is recomputed
    here, and nothing is summarised in a way the artifact does not already say.
    It is EVIDENCE about the origins and geometry it was run on, not a claim
    that forecast-driven routing is better.
    """
    path = (ROOT / "data" / "processed" / "evaluation"
            / "historical_forecast_decision_backtest.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail={
            "status": "evaluation_artifact_unavailable",
            "message": "the historical backtest artifact has not been "
                       "produced; run "
                       "python -m src.evaluation.historical_forecast_decision"})
    try:
        body = json.loads(path.read_text())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail={
            "status": "evaluation_artifact_unreadable",
            "message": str(exc)}) from exc
    return {"status": STATUS_OK,
            "artifact": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "evaluation": body}


@app.post("/api/mission/evaluate")
def mission_evaluate(request: MissionEvaluateRequest) -> dict:
    """Evaluate several destinations at several departure times.

    Returns one result per (site, departure time) pair with its measured
    attributes, or the router's own refusal. It does NOT rank the sites, rank
    the departure times, name a best option or compute a score.
    """
    try:
        world = get_world()
    except WorldError as exc:
        raise HTTPException(status_code=503, detail={
            "status": STATUS_UNAVAILABLE, "message": str(exc)}) from exc

    #  An unsupported policy is REFUSED, never replaced with a supported one.
    if request.environment_policy is not None and \
            request.environment_policy not in SUPPORTED_POLICIES:
        raise HTTPException(status_code=422, detail={
            "status": "rejected",
            "message": f"{request.environment_policy!r} is not a policy this "
                       f"backend routes under. Supported: "
                       f"{list(SUPPORTED_POLICIES)}. The requested policy is "
                       f"refused rather than replaced.",
            "supported_policies": list(SUPPORTED_POLICIES)})
    try:
        departures = [datetime.fromisoformat(t) for t in request.departure_times]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={
            "status": "bad_request",
            "message": f"a departure_time is not an ISO timestamp: {exc}"}) from exc

    sites = tuple(Site(site_id=s.site_id, cell=(int(s.cell[0]), int(s.cell[1])),
                       label=s.label or s.site_id,
                       is_operational=bool(s.is_operational), source=s.source)
                  for s in request.sites)
    try:
        matrix = evaluate_mission(
            sites=sites, departure_times=departures,
            start=(int(request.start[0]), int(request.start[1])),
            vessel_speed_mps=request.vessel_speed_mps,
            polaris_ice_class=request.polaris_ice_class,
            polaris_riv_table=request.polaris_riv_table,
            profiles=tuple(request.profiles),
            environment_policy=request.environment_policy,
            horizon_buckets=request.horizon_buckets, world=world)
    except (MissionError, ProfileError, RouteError) as exc:
        raise HTTPException(status_code=422, detail={
            "status": "rejected", "message": str(exc)}) from exc

    payload = matrix.to_dict()
    summary = payload["summary"]
    return {"status": (STATUS_OK if matrix.any_feasible else NO_FEASIBLE_OPTION),
            "message": ("no candidate destination routed at any requested "
                        "departure time" if not matrix.any_feasible else
                        f"{summary['options_feasible']} of "
                        f"{summary['options_evaluated']} options routed"),
            #  surfaced at the top, not only nested inside the routes
            "environment_policy": matrix.environment_policy,
            "evaluation_mode": matrix.evaluation_mode,
            "is_operational_assessment": matrix.is_operational_assessment,
            "iceberg_exposure": matrix.iceberg_exposure,
            "limitations": list(matrix.limitations),
            "mission": payload,
            "grid": grid_info()}


@app.post("/api/mission/simulate")
def mission_simulate(request: MissionRequest,
                     iceberg: SimulatedIceberg,
                     profile: str = Query("risk_oriented")) -> dict:
    """Inject one simulated iceberg and replan, returning BOTH routes.

    The baseline is recomputed here rather than trusted from the client, so
    the before/after figures are two results of the same backend on the same
    inputs. Nothing is written: see src/api/simulate.py.
    """
    world = get_world()
    baseline = _run(request)
    if baseline["status"] != STATUS_OK:
        return {**baseline, "simulation": None,
                "message": "the baseline mission did not route, so there is "
                           "nothing to replan: " + baseline["message"]}

    grid = world["grid"]
    row, col = grid.xy_to_rowcol(iceberg.x, iceberg.y)
    if not grid.in_bounds(row, col):
        raise HTTPException(status_code=422, detail={
            "status": "outside_the_grid",
            "message": f"({iceberg.x:.0f}, {iceberg.y:.0f}) is outside the "
                       f"{grid.shape[0]}x{grid.shape[1]} routing grid."})

    chosen = baseline["comparison"]["profiles"].get(profile)
    if not chosen or not chosen["success"]:
        raise HTTPException(status_code=422, detail={
            "status": "no_such_route",
            "message": f"profile {profile!r} produced no route to place a "
                       f"simulated iceberg against."})

    try:
        bucket, arrival_s, arrival_iso, cell, distance = simulate.encounter(
            chosen, row, col, float(grid.pixel_size[0]))
        fields = {f.bucket: f for f in world["exposure_fields"]}
        reference = fields.get(bucket) or fields[max(fields)]
        radius_km = float(reference.forecasts[0].radius_km)
        lat = float(simulate._geometry().lat[row, col])
        lon = float(simulate._geometry().lon[row, col])

        forecast = simulate.simulated_forecast(
            lat, lon, radius_km, float(reference.horizon_hours),
            str(reference.forecast_start_date))
        raster = simulate.simulated_exposure_raster(forecast)
        if not (raster > 0).any():
            raise HTTPException(status_code=422, detail={
                "status": "no_modelled_exposure",
                "message": "a simulated iceberg there puts modelled exposure "
                           "on no routing cell."})
        injected = simulate.inject(
            list(world["exposure_fields"]), raster, from_bucket=bucket,
            simulated_forecast_record=forecast)
        provider = simulate.simulated_provider(
            world["provider"], injected, tuple(grid.shape),
            allow_mismatch=True)
    except simulate.SimulationError as exc:
        raise HTTPException(status_code=422, detail={
            "status": "simulation_refused", "message": str(exc)}) from exc

    replanned = _run(request, provider=provider)
    injection = simulate.Injection(
        latitude=lat, longitude=lon, row=int(row), col=int(col),
        radius_km=radius_km, from_bucket=int(bucket),
        encounter_arrival_s=arrival_s, encounter_datetime=arrival_iso,
        encounter_cell=cell, distance_to_route_m=distance,
        horizon_hours=float(reference.horizon_hours))

    return {
        "status": replanned["status"],
        "message": replanned["message"],
        "simulation": injection.to_dict(),
        "simulated_exposure": geo.exposure_collection(
            grid, simulate.ExposureField(
                bucket=int(bucket), exposure=raster,
                dominant=np.where(raster > 0, 0, -1).astype("int32"),
                forecasts=(forecast,),
                forecast_start_date=reference.forecast_start_date,
                horizon_hours=float(reference.horizon_hours),
                label=simulate.SIMULATION_LABEL),
            grid.crs.to_epsg()),
        "baseline": baseline,
        "replanned": replanned,
        "change": simulate.describe_change(
            baseline["comparison"], replanned["comparison"], raster, injection),
    }
