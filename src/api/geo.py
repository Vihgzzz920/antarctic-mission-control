"""
Turn routing results into GeoJSON in the PROJECT CRS. Presentation only.

Coordinates come from RoutingGrid.rowcol_to_xy, which is the project's own
row/col -> projected-metre conversion. Nothing is reprojected to Web Mercator:
the browser map is told to use EPSG:3976 and receives metres in that CRS, so
no geometry is resampled, rounded through lon/lat, or approximated on its way
to the screen.
"""
from __future__ import annotations

import numpy as np

PROFILE_ORDER = ("fastest", "risk_oriented", "shortest_distance")


def _crs_name(epsg: int) -> dict:
    return {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}


def route_feature(grid, profile: dict) -> dict:
    """One profile's path as a LineString of cell centres in projected metres."""
    coords = [list(grid.rowcol_to_xy(r, c)) for r, c in profile["path"]]
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {
            "profile": profile["profile"],
            "objective": profile["objective"],
            "objective_units": profile["objective_units"],
            "objective_value": profile["objective_value"],
            "distance_m": profile["distance_m"],
            "travel_time_s": profile["travel_time_s"],
            "configured_cost": profile["configured_cost"],
            "cells": len(profile["path"]),
        },
    }


def cell_features(grid, profile: dict) -> list[dict]:
    """Every priced cell of a route, as a point carrying its own audit row."""
    out = []
    for cell in profile["cells"]:
        x, y = grid.rowcol_to_xy(cell["row"], cell["col"])
        out.append({"type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [x, y]},
                    "properties": {"profile": profile["profile"], **cell}})
    return out


def route_collections(grid, comparison: dict, epsg: int) -> dict:
    """{profile: FeatureCollection} for the profiles that produced a route."""
    out = {}
    for name in PROFILE_ORDER:
        profile = comparison["profiles"].get(name)
        if not profile or not profile["success"]:
            continue
        out[name] = {
            "type": "FeatureCollection", "crs": _crs_name(epsg),
            "features": [route_feature(grid, profile)]
                        + cell_features(grid, profile)}
    return out


def endpoint_collection(grid, start, goal, epsg: int) -> dict:
    """Departure and destination, shared by every profile."""
    features = []
    for role, (r, c) in (("departure", start), ("destination", goal)):
        x, y = grid.rowcol_to_xy(r, c)
        features.append({"type": "Feature",
                         "geometry": {"type": "Point", "coordinates": [x, y]},
                         "properties": {"role": role, "row": int(r),
                                        "col": int(c)}})
    return {"type": "FeatureCollection", "crs": _crs_name(epsg),
            "features": features}


def exposure_collection(grid, field, epsg: int, *, minimum: float = 0.0) -> dict:
    """The non-zero cells of one iceberg exposure raster, as points.

    The raster is 1328 x 1264 but only a few hundred cells carry any modelled
    exposure, so the browser receives those cells rather than the raster. A
    zero here means no MODELLED exposure from the forecast population, never a
    statement that the water is clear.
    """
    rows, cols = np.nonzero(field.exposure > minimum)
    features = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        x, y = grid.rowcol_to_xy(r, c)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": {"row": r, "col": c,
                           "exposure": float(field.exposure[r, c]),
                           "bucket": int(field.bucket),
                           "horizon_hours": float(field.horizon_hours),
                           "label": field.label}})
    return {"type": "FeatureCollection", "crs": _crs_name(epsg),
            "features": features,
            "properties": {"zero_meaning": "no MODELLED exposure from this "
                                           "forecast population; NOT a statement "
                                           "that the water is clear",
                           "is_a_probability": False}}
