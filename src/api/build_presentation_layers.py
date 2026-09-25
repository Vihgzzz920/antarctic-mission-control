"""
Derive lightweight PRESENTATION images from the real rasters, for the browser.

    python -m src.api.build_presentation_layers

WHY THIS EXISTS
  The routing rasters are 1328 x 1264 float32 GeoTIFFs on the EPSG:3976 grid.
  Handing those to a browser is several megabytes per layer for a picture that
  is a few hundred pixels wide on screen. This script block-reduces them into
  small RGBA PNGs and writes a manifest recording exactly how.

WHAT IT GUARANTEES
  - The sources are opened read-only and never rewritten.
  - Nothing is interpolated, gap-filled or smoothed. Reduction is a block mean
    over VALID cells only for a continuous field, and a block ANY for a mask.
  - The manifest carries each source's sha256, so a picture can always be tied
    back to the file it came from.
  - These images are for DISPLAY. Routing reads the full-resolution rasters and
    never sees these files. In particular the exclusion image is a generalised
    outline -- a reduced block is drawn as excluded if ANY of its cells is --
    which is deliberately over-inclusive on screen and is NOT the mask any
    route was planned against.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import rasterio
from PIL import Image

from src.api.config import (BEDMACHINE_CLASSES, DEMO, LAYER_DIR,
                            SIC_DIR)

FACTOR = 4                      # 1328 x 1264 -> 332 x 316, 25 km per screen cell


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _blocks(arr: np.ndarray, factor: int) -> np.ndarray:
    """Trim to a whole number of blocks and fold them into two extra axes."""
    h = arr.shape[0] // factor * factor
    w = arr.shape[1] // factor * factor
    return arr[:h, :w].reshape(h // factor, factor, w // factor, factor)


def _block_mean_valid(arr, valid, factor):
    """Mean over the VALID cells of each block; NaN where a block has none."""
    a = np.where(valid, arr, 0.0)
    total = _blocks(a, factor).sum(axis=(1, 3))
    count = _blocks(valid.astype("float64"), factor).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


def _write_png(rgba: np.ndarray, path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)
    return {"file": path.name, "width": int(rgba.shape[1]),
            "height": int(rgba.shape[0]), "bytes": path.stat().st_size}


def build_sic(factor: int = FACTOR) -> dict:
    """Sea-ice concentration as a cyan ramp; transparent where there is no value."""
    src_path = SIC_DIR / f"sic_{DEMO['environment_date'].replace('-', '')}.tif"
    with rasterio.open(src_path) as src:
        band = src.read(1).astype("float64")
        extent = [src.bounds.left, src.bounds.bottom, src.bounds.right,
                  src.bounds.top]
        epsg = src.crs.to_epsg()
    #  the project's single masking rule: 0 is valid open water, and only
    #  >100, <0 and non-finite are dropped
    valid = np.isfinite(band) & (band >= 0) & (band <= 100)
    reduced = _block_mean_valid(band, valid, factor)
    frac = np.clip(np.nan_to_num(reduced, nan=0.0) / 100.0, 0.0, 1.0)
    present = np.isfinite(reduced)

    rgba = np.zeros((*frac.shape, 4), dtype="uint8")
    #  open water stays near the page background; ice lifts towards white
    rgba[..., 0] = (26 + 200 * frac ** 1.15).astype("uint8")
    rgba[..., 1] = (54 + 190 * frac ** 1.05).astype("uint8")
    rgba[..., 2] = (74 + 165 * frac ** 0.95).astype("uint8")
    rgba[..., 3] = np.where(present, (60 + 175 * frac).astype("uint8"), 0)

    info = _write_png(rgba, LAYER_DIR / "sic.png")
    info.update(source=str(src_path), source_sha256=_sha256(src_path),
                epsg=epsg, extent=extent, factor=factor,
                reduction="block mean over valid cells only",
                value="sea-ice concentration, percent",
                valid_rule="0 <= value <= 100 and finite; nothing gap-filled",
                transparent_where="no valid cell in the block")
    return info


def _mask_layer(name, keep, colour, alpha, note, factor):
    """One class group of the BedMachine mask, block-ANY reduced."""
    with rasterio.open(BEDMACHINE_CLASSES) as src:
        band = src.read(1)
        extent = [src.bounds.left, src.bounds.bottom, src.bounds.right,
                  src.bounds.top]
        epsg = src.crs.to_epsg()
        tags = src.tags()
    selected = np.isin(band, keep)
    reduced = _blocks(selected, factor).any(axis=(1, 3))

    rgba = np.zeros((*reduced.shape, 4), dtype="uint8")
    for i, value in enumerate(colour):
        rgba[..., i] = value
    rgba[..., 3] = np.where(reduced, alpha, 0).astype("uint8")

    info = _write_png(rgba, LAYER_DIR / f"{name}.png")
    info.update(source=str(BEDMACHINE_CLASSES),
                source_sha256=_sha256(BEDMACHINE_CLASSES), epsg=epsg,
                extent=extent, factor=factor, classes_kept=list(keep),
                class_meaning=tags.get("classes", ""),
                reduction="block ANY: a screen cell is drawn if any of its grid "
                          "cells belongs to these classes",
                display_only="GENERALISED OUTLINE, deliberately over-inclusive. "
                             "Routing reads the full-resolution exclusion mask "
                             "and never these images.",
                note=note)
    return info


def build_exclusion(factor: int = FACTOR) -> dict:
    """Land and no-coverage drawn APART.

    The routing exclusion collapses both into one boolean, which is correct for
    a router and wrong for a screen: a third of this grid is outside BedMachine
    coverage, and painting it the same colour as the continent would invent a
    landmass the size of the Southern Ocean. They are drawn as two layers.
    """
    return {
        "land": _mask_layer(
            "land", (1, 2, 4), (17, 26, 38), 240,
            "ice-free land, grounded ice and Lake Vostok: the continent",
            factor),
        "no_coverage": _mask_layer(
            "no_coverage", (255,), (10, 15, 22), 120,
            "outside BedMachine coverage. Excluded by the routing mask for want "
            "of data, NOT because it is land.", factor),
    }


CRS_TS = ROOT_FRONTEND = None


def write_crs_module(grid_crs, extent) -> dict:
    """Emit the project CRS into the frontend as a GENERATED file.

    The browser needs the projection before it can draw anything, so the
    definition is copied out of the repository at build time rather than typed
    into the React source. proj4js does not accept PROJ's `+no_defs=True`
    spelling, so that one token is normalised to a bare `+no_defs`; every other
    parameter is passed through untouched, and both strings are emitted so the
    change is visible.
    """
    verbatim = grid_crs.to_proj4()
    for_proj4js = verbatim.replace("+no_defs=True", "+no_defs").strip()
    target = LAYER_DIR.parent.parent / "src" / "generated" / "crs.ts"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "// GENERATED by src/api/build_presentation_layers.py -- do not edit.\n"
        "// The project CRS, copied from the routing rasters themselves.\n"
        f"// verbatim from rasterio: {verbatim}\n"
        "// proj4js cannot parse PROJ's `+no_defs=True`; that single token is\n"
        "// normalised to `+no_defs` and nothing else is changed.\n"
        f"export const PROJECT_EPSG = 'EPSG:{grid_crs.to_epsg()}';\n"
        f"export const PROJECT_PROJ4 = '{for_proj4js}';\n"
        f"export const PROJECT_PROJ4_VERBATIM = '{verbatim}';\n"
        "// [minX, minY, maxX, maxY] in projected metres, the routing grid's "
        "own bounds\n"
        f"export const GRID_EXTENT: [number, number, number, number] = "
        f"[{extent[0]}, {extent[1]}, {extent[2]}, {extent[3]}];\n")
    return {"file": str(target), "epsg": grid_crs.to_epsg(),
            "proj4_verbatim": verbatim, "proj4_for_proj4js": for_proj4js,
            "extent": list(extent),
            "normalisation": "+no_defs=True -> +no_defs, for proj4js only"}


def main() -> int:
    layers = {"sic": build_sic(), **build_exclusion()}
    with rasterio.open(BEDMACHINE_CLASSES) as src:
        crs = write_crs_module(src.crs, (src.bounds.left, src.bounds.bottom,
                                         src.bounds.right, src.bounds.top))
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "src/api/build_presentation_layers.py",
        "purpose": "lightweight DISPLAY images derived from the real rasters; "
                   "the sources are read-only and routing never reads these",
        "environment_date": DEMO["environment_date"],
        "layers": layers,
        "crs": crs,
    }
    LAYER_DIR.mkdir(parents=True, exist_ok=True)
    (LAYER_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, info in layers.items():
        print(f"GOT  {info['file']:<16} {info['width']}x{info['height']}  "
              f"{info['bytes'] / 1024:.1f} KB   from {info['source']}")
    print(f"GOT  manifest.json")
    print(f"GOT  {crs['file']}  ({crs['proj4_for_proj4js']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
