"""
Tests for the thin API layer.

The ones that matter: the API computes nothing, a refusal is returned as a
refusal, and the coordinates it hands the browser are the project's own.

    python tests/test_api.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient                              # noqa: E402

from src.api.config import DEMO                                        # noqa: E402
import src.models.iceberg_uncertainty as uncertainty                    # noqa: E402
import src.routing.iceberg_risk as risk                                 # noqa: E402
from src.api.main import app                                           # noqa: E402
from src.api.world import get_world                                    # noqa: E402
from src.routing.route_profiles import PROFILES                        # noqa: E402

API_DIR = ROOT / "src" / "api"
client = TestClient(app)
_C: dict = {}


def demo(override: bool):
    key = ("demo", override)
    if key not in _C:
        response = client.get(
            f"/api/demo/routes?historical_demo_override={str(override).lower()}")
        assert response.status_code == 200, response.text
        _C[key] = response.json()
    return _C[key]


def test_health_reports_the_real_grid():
    body = client.get("/api/health").json()
    assert body["status"] == "ok", body
    assert body["prototype"] is True and "prototype" in body["disclaimer"]
    grid = body["grid"]
    assert grid["epsg"] == 3976
    assert grid["shape"] == [1328, 1264]
    assert grid["pixel_size"] == [6250.0, 6250.0]
    assert grid["extent"] == [-3950000.0, -3950000.0, 3950000.0, 4350000.0]
    assert grid["environment_date"] == DEMO["environment_date"]
    assert grid["usnic_chart_date"] == "2020-12-24"
    assert grid["dates_aligned"] is False
    assert body["requires_historical_demo_override"] is True
    assert body["iceberg_exposure_weight"] == 5.0
    assert body["polaris"]["selection"]["ice_class"] == DEMO["polaris_ice_class"]
    return (f"EPSG:{grid['epsg']} {grid['shape']}, chart "
            f"{grid['usnic_chart_date']} vs environment "
            f"{grid['environment_date']}, weight {body['iceberg_exposure_weight']}")


def test_the_demo_refuses_the_historical_pairing_by_default():
    body = demo(False)
    assert body["status"] == "historical_date_mismatch", body["status"]
    assert body["historical_demonstration"] is False
    assert body["geojson"] == {}, "a refused request returned geometry"
    for name in PROFILES:
        profile = body["comparison"]["profiles"][name]
        assert profile["success"] is False
        assert profile["path"] == [] and profile["configured_cost"] is None
    return ("the override defaults OFF, so the endpoint returns the routing "
            "layer's own refusal and no geometry at all")


def test_the_explicit_override_routes_and_is_labelled():
    body = demo(True)
    assert body["status"] == "ok", body["message"]
    assert body["historical_demonstration"] is True
    provenance = body["comparison"]["provenance"]
    assert provenance["historical_demo_override"] is True
    assert provenance["chart_dates"] == ["2020-12-24"]
    assert provenance["environment_dates"] == ["2025-01-08"]
    assert provenance["iceberg_exposure_weight"] == 5.0
    assert sorted(body["geojson"]) == sorted(PROFILES)
    risk = body["comparison"]["profiles"]["risk_oriented"]
    assert risk["polaris"]["selection"]["ice_class"] == "PC6"
    assert risk["polaris"]["selection"]["riv_table_requested"] == "1.3"
    assert risk["polaris_contribution"] > 0
    return " | ".join(
        f"{n}: {body['comparison']['profiles'][n]['distance_km']:.2f} km, cost "
        f"{body['comparison']['profiles'][n]['configured_cost']:,.0f}"
        for n in PROFILES)


def test_geojson_is_the_projects_own_coordinates():
    body = demo(True)
    grid = get_world()["grid"]
    for name in PROFILES:
        collection = body["geojson"][name]
        assert collection["crs"]["properties"]["name"].endswith("EPSG::3976")
        line = collection["features"][0]
        assert line["geometry"]["type"] == "LineString"
        path = body["comparison"]["profiles"][name]["path"]
        assert len(line["geometry"]["coordinates"]) == len(path)
        for (row, col), (x, y) in zip(path, line["geometry"]["coordinates"]):
            wx, wy = grid.rowcol_to_xy(row, col)
            assert abs(x - wx) < 1e-9 and abs(y - wy) < 1e-9
            #  a sanity floor: the grid is metres, not degrees
            assert abs(x) > 180.0 or abs(y) > 180.0
    return ("every vertex equals RoutingGrid.rowcol_to_xy for its cell; nothing "
            "is reprojected on the way to the browser")


def test_route_cells_carry_the_chart_fields_the_inspector_needs():
    body = demo(True)
    cells = body["comparison"]["profiles"]["risk_oriented"]["cells"]
    world = get_world()
    for cell in cells:
        for key in ("arrival_datetime", "bucket", "environmental_cost",
                    "polaris_penalty", "iceberg_exposure", "iceberg_id",
                    "radius_km", "chart_state", "dominant_fraction"):
            assert key in cell, key
        truth = world["composition"].cell(cell["row"], cell["col"], "conservative")
        assert cell["dominant_fraction"] == truth.dominant_fraction
        assert cell["chart_state"] == truth.state
    return (f"{len(cells)} cells, each carrying its arrival bucket, cost terms, "
            f"chart state and USNIC dominant fraction read from the composition")


def test_a_blocked_endpoint_is_reported_not_routed():
    world = get_world()
    excluded = np.argwhere(world["excluded"])[0].tolist()
    body = client.post("/api/mission/compare", json={
        "start": DEMO["start"], "goal": excluded,
        "departure_time": DEMO["departure_time"],
        "vessel_speed_mps": DEMO["vessel_speed_mps"],
        "polaris_ice_class": DEMO["polaris_ice_class"],
        "polaris_riv_table": DEMO["polaris_riv_table"],
        "historical_demo_override": True}).json()
    assert body["status"] == "hard_navigation_constraint", body["status"]
    assert body["geojson"] == {}
    return (f"a goal inside the exclusion at {excluded} returns "
            f"hard_navigation_constraint and no route")


def test_a_bad_selection_is_rejected_with_a_reason():
    response = client.post("/api/mission/compare", json={
        "start": DEMO["start"], "goal": DEMO["goal"],
        "departure_time": DEMO["departure_time"],
        "vessel_speed_mps": DEMO["vessel_speed_mps"],
        "polaris_ice_class": "PC99", "polaris_riv_table": "1.3",
        "historical_demo_override": True})
    assert response.status_code == 422, response.status_code
    detail = response.json()["detail"]
    assert detail["status"] == "rejected"
    assert "PC99" in detail["message"]
    bad_speed = client.post("/api/mission/compare", json={
        "start": DEMO["start"], "goal": DEMO["goal"],
        "departure_time": DEMO["departure_time"], "vessel_speed_mps": 0.0,
        "polaris_ice_class": "PC6", "polaris_riv_table": "1.3",
        "historical_demo_override": True})
    assert bad_speed.status_code == 422
    return "an unknown ice class and a zero speed are both refused with 422"


def test_the_exposure_layer_is_the_real_raster():
    body = client.get("/api/layers/iceberg-exposure?bucket=1").json()
    world = get_world()
    field = {f.bucket: f for f in world["exposure_fields"]}[1]
    assert len(body["features"]) == int((field.exposure > 0).sum())
    assert body["properties"]["is_a_probability"] is False
    assert "NOT a statement" in body["properties"]["zero_meaning"]
    values = [f["properties"]["exposure"] for f in body["features"]]
    assert min(values) > 0 and max(values) <= 1.0
    assert client.get("/api/layers/iceberg-exposure?bucket=9").status_code == 404
    return (f"{len(body['features'])} exposed cells from the real 12 h raster, "
            f"max {max(values):.3f}; zero cells are not sent and zero is "
            f"documented as 'no modelled exposure'")


def test_repeated_requests_are_identical():
    a = client.get("/api/demo/routes?historical_demo_override=true").json()
    b = client.get("/api/demo/routes?historical_demo_override=true").json()
    for name in PROFILES:
        pa, pb = a["comparison"]["profiles"][name], b["comparison"]["profiles"][name]
        assert pa["path"] == pb["path"]
        assert pa["configured_cost"] == pb["configured_cost"]
        assert pa["objective_value"] == pb["objective_value"]
    return "two identical requests return identical paths and costs"


def test_the_api_implements_no_routing_of_its_own():
    """No second A*, no second cost model, no second POLARIS calculation."""
    banned = ("astar", "time_astar", "heapq", "calculate_rio", "classify_rio",
              "riv_for", "penalty_for", "compose_grid_from", "cost_fn")
    offenders = []
    for path in sorted(API_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        funcs = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for word in banned:
            if word in names | attrs | funcs:
                offenders.append(f"{path.name}:{word}")
    assert not offenders, offenders
    #  and it only ever imports the existing stack
    imported = set()
    for path in sorted(API_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        imported |= {n.module for n in ast.walk(tree)
                     if isinstance(n, ast.ImportFrom)
                     and (n.module or "").startswith("src.routing")}
    assert "src.routing.route_profiles" in imported
    assert "src.routing.time_astar" not in imported
    return (f"no search, cost or RIO vocabulary in {len(list(API_DIR.glob('*.py')))} "
            f"API modules; routing is reached only through "
            f"{sorted(m.split('.')[-1] for m in imported)}")



def model_states():
    """The domain layer's OWN forecast states, produced here independently.

    Same population, same calibration, same horizons the risk module uses to
    build the shipped rasters. The endpoint is asserted against these, so the
    test fails the moment the API starts producing a state of its own.
    """
    key = "model_states"
    if key not in _C:
        population, _ = risk.build_population()
        risk.chronological_split(
            population, risk.load_split_config(risk.SPLIT_CONFIG))
        unc_cfg = uncertainty.load_config(risk.UNC_CONFIG)
        calibration = risk.calibrate(population, unc_cfg)
        cfg = risk.load_config(risk.CONFIG)
        rows, info = risk._population(DEMO["environment_date"], risk.SPLIT_CONFIG)
        states = {}
        for hours in [0.0] + [float(h) for h in cfg.horizons_hours]:
            produced = risk.forecast_icebergs(
                rows, hours * risk.SECONDS_PER_HOUR, calibration, unc_cfg, cfg)
            states[hours] = {f.iceberg_id: f for f in produced}
        _C[key] = (states, info, [float(h) for h in cfg.horizons_hours])
    return _C[key]


def forecast_body():
    if "forecast" not in _C:
        response = client.get("/api/forecast/icebergs")
        assert response.status_code == 200, response.text
        _C["forecast"] = response.json()
    return _C["forecast"]


def test_the_forecast_endpoint_serialises_the_models_own_states():
    body = forecast_body()
    states, info, horizons = model_states()
    assert body["forecast_start_date"] == info["forecast_start_date"]
    assert body["iceberg_count"] == len(body["icebergs"]) == len(states[0.0])
    for record in body["icebergs"]:
        berg = record["iceberg_id"]
        assert [p["hours"] for p in record["predicted"]] == horizons
        for served in [record["observed"], *record["predicted"]]:
            truth = states[served["hours"]][berg]
            assert served["latitude"] == truth.latitude, (berg, served["hours"])
            assert served["longitude"] == truth.longitude, (berg, served["hours"])
            assert served["radius_km"] == truth.radius_km, (berg, served["hours"])
            assert served["extrapolated_uncertainty"] == bool(
                truth.extrapolated_uncertainty)
            assert served["physics_position_is_extrapolated"] == bool(
                truth.physics_position_is_extrapolated)
    return (f"{body['iceberg_count']} icebergs x {len(horizons)} horizons: every "
            f"position and radius equals iceberg_risk.forecast_icebergs() to the "
            f"last bit")


def test_observed_is_served_apart_from_predicted():
    body = forecast_body()
    states, _, _ = model_states()
    assert body["observed_label"] == "Observed"
    assert "INPUT" in body["observed_meaning"]
    assert "PREDICTION" in body["predicted_meaning"]
    for record in body["icebergs"]:
        observed = record["observed"]
        assert observed["hours"] == 0.0 and observed["label"] == "Observed"
        assert observed["radius_km"] == 0.0, "an observation was given a radius"
        truth = states[0.0][record["iceberg_id"]]
        assert (observed["latitude"], observed["longitude"]) == (
            truth.latitude, truth.longitude)
        for predicted in record["predicted"]:
            assert predicted["label"] == f"Predicted +{predicted['hours']:g}h"
            assert predicted["radius_km"] > 0.0
            assert (predicted["latitude"], predicted["longitude"]) != (
                observed["latitude"], observed["longitude"])
    return ("horizon 0 is the observed start position under its own key, with a "
            "zero radius; every other horizon is labelled a prediction")


def test_the_forecast_states_the_drift_model_it_actually_ran():
    body = forecast_body()
    shipped = json.loads((ROOT / "configs" / "iceberg_physics.json").read_text())
    windage = float(shipped["windage_coefficient"])
    model = body["model"]
    assert model["windage_coefficient"] == windage
    assert model["windage_is_active"] is (windage != 0.0)
    assert model["is_validated_science"] is False
    if windage == 0.0:
        assert "current advection only" in model["summary"]
        assert "+ windage" not in model["summary"]
    horizons = {h["hours"]: h for h in body["horizons"]}
    assert horizons[24.0]["extrapolated_uncertainty"] is False
    assert horizons[48.0]["extrapolated_uncertainty"] is True
    assert body["uncertainty"]["calibrated_to_hours"] == 24.0
    assert body["uncertainty"]["beyond_calibration_is_extrapolated"] is True
    for record in body["icebergs"]:
        beyond = [p for p in record["predicted"] if p["hours"] > 24.0]
        assert beyond and all(
            p["extrapolated_uncertainty"] and p["physics_position_is_extrapolated"]
            for p in beyond)
    return (f"windage coefficient {windage} -> '{model['summary']}'; 24 h is "
            f"calibrated, 48 h is flagged extrapolated in both the uncertainty "
            f"and the physics position")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78)
    print("mission control API")
    print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<54} {fn() or ''}")
        except AssertionError as exc:
            failures.append(fn.__name__)
            print(f"  FAIL  {fn.__name__:<54} {exc}")
        except Exception as exc:                        # noqa: BLE001
            failures.append(fn.__name__)
            print(f"  ERROR {fn.__name__:<54} {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{len(failures)} of {len(TESTS)} FAILED: {failures}", file=sys.stderr)
        return 1
    print(f"all {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
