from pathlib import Path
import argparse
import sys

import numpy as np
import rasterio
import xarray as xr


MIZ_LO, MIZ_HI = 15.0, 80.0    # classified on the OSI value; CLI defaults
DEFAULT_MIN_CONTRIB = 8


def pearson(a: np.ndarray, b: np.ndarray) -> float:
   
    if a.size == 0:
        return float("nan")

    a = a - np.mean(a)
    b = b - np.mean(b)

    ssa = float(np.sum(a * a, dtype="float64"))
    ssb = float(np.sum(b * b, dtype="float64"))

    denom = np.sqrt(ssa * ssb)

    if denom <= 0:
        return float("nan")

    cov = float(np.sum(a * b, dtype="float64"))

    return cov / denom


def metrics(bremen: np.ndarray, osi: np.ndarray):
    bremen = np.asarray(bremen, dtype="float64")
    osi = np.asarray(osi, dtype="float64")

    d = bremen - osi

    if d.size == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "r": float("nan"),
        }

    ssd = float(np.sum(d * d, dtype="float64"))

    return {
        "n": int(d.size),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(ssd / d.size)),
        "bias": float(np.mean(d)),
        "r": pearson(bremen, osi),
    }


def map_bremen_to_osi(
    bx: np.ndarray,
    by: np.ndarray,
    xc_m: np.ndarray,
    yc_m: np.ndarray,
):
    """
    Map Bremen cell centres to OSI cells using OSI cell boundaries.

    Returns:
        flat_index : flattened OSI-cell index for each Bremen cell
        ok         : whether the Bremen cell centre falls inside OSI grid
by = np.asarray(by, dtype="float64").reshape(height, width)        shape      : Bremen raster shape
    """

    h, w = bx.shape

    dx = float(xc_m[1] - xc_m[0])
    dy = float(yc_m[1] - yc_m[0])

    x_edge0 = float(xc_m[0]) - dx / 2.0
    y_edge0 = float(yc_m[0]) - dy / 2.0

    j = np.floor((bx - x_edge0) / dx)
    i = np.floor((by - y_edge0) / dy)

    ok = (
        (i >= 0)
        & (i < yc_m.size)
        & (j >= 0)
        & (j < xc_m.size)
    )

    i = np.where(ok, i, 0).astype(np.intp)
    j = np.where(ok, j, 0).astype(np.intp)

    flat_index = i * xc_m.size + j

    return flat_index, ok, (h, w)


def block_average(
    bremen: np.ndarray,
    valid: np.ndarray,
    mapping: np.ndarray,
    mapping_ok: np.ndarray,
    osi_shape: tuple,
    min_contrib: int,
):
    """
    Average valid Bremen cell-centre values inside each OSI cell.

    This is a resolution-matched block average based on cell centres.
    It is not a sensor-footprint convolution.
    """

    usable = valid & mapping_ok

    flat_index = mapping[usable]
    values = bremen[usable].astype("float64")

    n_cells = osi_shape[0] * osi_shape[1]

    sums = np.bincount(
        flat_index,
        weights=values,
        minlength=n_cells,
    )

    counts = np.bincount(
        flat_index,
        minlength=n_cells,
    )

    result = np.full(n_cells, np.nan, dtype="float64")

    enough = counts >= min_contrib

    result[enough] = sums[enough] / counts[enough]

    return (
        result.reshape(osi_shape),
        counts.reshape(osi_shape),
    )


def read_bremen(
    sic_path: Path,
    mapping: np.ndarray,
    mapping_ok: np.ndarray,
    osi_shape: tuple,
    min_contrib: int,
):
    """
    Read one Bremen SIC GeoTIFF.

    Important:
    Bremen value 0 is valid open water and MUST NOT be treated as nodata.
    Values >100 are invalid/flag values.
    """

    with rasterio.open(sic_path) as src:
        raw = src.read(1).astype("float64")

    valid = np.isfinite(raw) & (raw >= 0) & (raw <= 100)

    averaged, counts = block_average(
        raw,
        valid,
        mapping,
        mapping_ok,
        osi_shape,
        min_contrib,
    )

    return averaged, counts


def get_sic_date(path: Path) -> str:
    """
    Extract YYYYMMDD from filenames such as sic_20250115.tif.
    """

    stem = path.stem

    if stem.startswith("sic_") and len(stem) >= 12:
        return stem[4:12]

    return stem


def compare(
    osi_nc: Path,
    sic_dir: Path,
    min_contrib: int,
    miz_lo: float = MIZ_LO,
    miz_hi: float = MIZ_HI,
) -> int:

    print()
    print("=" * 74)
    print()
    print(
        "BREMEN AMSR2 (6.25 km, EPSG:3976)  vs  "
        "OSI-430-a (25 km, LAEA)"
    )
    print()
    print("=" * 74)
    print()

    # ------------------------------------------------------------------
    # Read OSI dataset
    # ------------------------------------------------------------------

    ds = xr.open_dataset(osi_nc)

    try:
        O = ds["ice_conc"].values.astype("float64")
        U = ds["total_standard_uncertainty"].values.astype("float64")
        FLAGS = ds["status_flag"].values

        xc = ds["xc"].values.astype("float64")
        yc = ds["yc"].values.astype("float64")

        times = ds["time"].values

        # --------------------------------------------------------------
        # OSI coordinates are in kilometres.
        # --------------------------------------------------------------

        xc_m = xc * 1000.0
        yc_m = yc * 1000.0

        print(
            f"  OSI grid      : {xc.size} x {yc.size}, "
            f"25 km, xc/yc units 'km' -> metres"
        )

        print(
            "  method        : Bremen cell centres block-averaged "
            "into OSI cells"
        )

        print(
            f"  min contrib   : {min_contrib} Bremen cell centres per "
            f"OSI cell (approximately 16 available)"
        )

        print(
            f"  MIZ           : OSI ice_conc in "
            f"[{miz_lo:g}, {miz_hi:g})"
        )

        print()

        # --------------------------------------------------------------
        # Bremen grid geometry
        # --------------------------------------------------------------

        sic_files = sorted(sic_dir.glob("sic_*.tif"))

        if not sic_files:
            print(f"No Bremen SIC files found in: {sic_dir}")
            return 1

        with rasterio.open(sic_files[0]) as src:
            transform = src.transform
            width = src.width
            height = src.height

            if src.crs is None:
                print("Bremen raster has no CRS.")
                return 1

            if src.crs.to_epsg() != 3976:
                print(
                    f"Unexpected Bremen CRS: {src.crs}. "
                    "Expected EPSG:3976."
                )
                return 1

        rows, cols = np.indices((height, width))

        bx, by = rasterio.transform.xy(
            transform,
            rows,
            cols,
            offset="center",
        )    
        bx = np.asarray(bx, dtype="float64").reshape(height, width)
        by = np.asarray(by, dtype="float64").reshape(height, width)

	# --------------------------------------------------------------
        # Bremen -> OSI mapping
        # --------------------------------------------------------------

        mapping, mapping_ok, _ = map_bremen_to_osi(
            bx,
            by,
            xc_m,
            yc_m,
        )

        osi_shape = (yc.size, xc.size)

        # --------------------------------------------------------------
        # Storage for pooled statistics
        # --------------------------------------------------------------

        pooled_bremen = []
        pooled_osi = []

        pooled_uncertainty = []
        pooled_flags = []

        miz_bremen = []
        miz_osi = []

        print(
            "date          OSI ocean   compared    kept      "
            "MAE    RMSE    bias       r"
        )
        print("-" * 74)

        # --------------------------------------------------------------
        # Match Bremen files to OSI dates
        # --------------------------------------------------------------

        osi_dates = []

        for t in times:
            date_str = np.datetime_as_string(t, unit="D").replace("-", "")
            osi_dates.append(date_str)

        osi_date_to_index = {
            date: idx
            for idx, date in enumerate(osi_dates)
        }

        date_results = []

        for sic_path in sic_files:

            date = get_sic_date(sic_path)

            if date not in osi_date_to_index:
                continue

            osi_idx = osi_date_to_index[date]

            bremen_avg, counts = read_bremen(
                sic_path,
                mapping,
                mapping_ok,
                osi_shape,
                min_contrib,
            )

            osi_day = O[osi_idx]
            uncertainty_day = U[osi_idx]
            flags_day = FLAGS[osi_idx]

            # ----------------------------------------------------------
            # OSI validity
            # ----------------------------------------------------------

            finite_osi = np.isfinite(osi_day)

            # OSI status flag bit 1 = land.
            land = (flags_day.astype(np.uint16) & 1) != 0

            valid_osi = finite_osi & (~land)

            # ----------------------------------------------------------
            # Comparison mask
            # ----------------------------------------------------------

            comparison = (
                valid_osi
                & np.isfinite(bremen_avg)
                & (counts >= min_contrib)
            )

            b = bremen_avg[comparison]
            o = osi_day[comparison]

            result = metrics(b, o)

            ocean_cells = int(np.count_nonzero(valid_osi))
            compared_cells = int(b.size)

            kept_percent = (
                100.0 * compared_cells / ocean_cells
                if ocean_cells > 0
                else float("nan")
            )

            print(
                f"{date[:4]}-{date[4:6]}-{date[6:8]} "
                f"{ocean_cells:11,d} "
                f"{compared_cells:9,d} "
                f"{kept_percent:6.1f}% "
                f"{result['mae']:7.2f} "
                f"{result['rmse']:7.2f} "
                f"{result['bias']:+7.2f} "
                f"{result['r']:.4f}"
            )

            pooled_bremen.append(b)
            pooled_osi.append(o)

            pooled_uncertainty.append(
                uncertainty_day[comparison]
            )

            pooled_flags.append(
                flags_day[comparison]
            )

            # ----------------------------------------------------------
            # MIZ mask
            # ----------------------------------------------------------

            miz = (
                comparison
                & (osi_day >= miz_lo)
                & (osi_day < miz_hi)
            )

            miz_bremen.append(bremen_avg[miz])
            miz_osi.append(osi_day[miz])

            date_results.append(date)

        # ------------------------------------------------------------------
        # Ensure we actually compared something
        # ------------------------------------------------------------------

        if not pooled_bremen:
            print()
            print("No overlapping Bremen/OSI dates were found.")
            return 1

        # ------------------------------------------------------------------
        # Pooled statistics
        # ------------------------------------------------------------------

        pooled_bremen = np.concatenate(pooled_bremen)
        pooled_osi = np.concatenate(pooled_osi)

        pooled_uncertainty = np.concatenate(pooled_uncertainty)
        pooled_flags = np.concatenate(pooled_flags)

        pooled_result = metrics(
            pooled_bremen,
            pooled_osi,
        )

        miz_bremen = np.concatenate(miz_bremen)
        miz_osi = np.concatenate(miz_osi)

        miz_result = metrics(
            miz_bremen,
            miz_osi,
        )

        # ------------------------------------------------------------------
        # Pooled output
        # ------------------------------------------------------------------

        print("-" * 74)

        print(
            f"POOLED{'':13s}"
            f"{pooled_result['n']:9,d}"
            f"{'':7s}"
            f"{pooled_result['mae']:7.2f}"
            f"{pooled_result['rmse']:7.2f}"
            f"{pooled_result['bias']:+8.2f}"
            f"{pooled_result['r']:.4f}"
        )

        print(
            f"  MIZ only{'':18s}"
            f"{miz_result['n']:9,d}"
            f"{'':7s}"
            f"{miz_result['mae']:7.2f}"
            f"{miz_result['rmse']:7.2f}"
            f"{miz_result['bias']:+8.2f}"
            f"{miz_result['r']:.4f}"
        )

        miz_fraction = (
            100.0 * miz_result["n"] / pooled_result["n"]
            if pooled_result["n"] > 0
            else float("nan")
        )

        print()

        print(
            f"Comparison cells: {pooled_result['n']:,} "
            f"pooled over {len(date_results)} date(s); "
            f"MIZ is {miz_fraction:.1f}% of them."
        )

        print()

        print(
            "Units are percent concentration. "
            "Bias is Bremen minus OSI, so positive = Bremen reads more ice."
        )

        # ------------------------------------------------------------------
        # OSI uncertainty
        # ------------------------------------------------------------------

        finite_uncertainty = np.isfinite(pooled_uncertainty)

        if np.any(finite_uncertainty):

            uncertainty_valid = pooled_uncertainty[finite_uncertainty]

            print()
            print(
                "OSI uncertainty among comparison cells "
                "(percentage points):"
            )

            print(
                f"  mean {np.mean(uncertainty_valid):.2f}   "
                f"median {np.median(uncertainty_valid):.2f}   "
                f"max {np.max(uncertainty_valid):.2f}"
            )

        # MIZ uncertainty
        #
        # We reconstruct this using the same OSI dates and masks so that
        # the reported MIZ uncertainty corresponds to the MIZ cells.
        miz_uncertainties = []

        for sic_path in sic_files:

            date = get_sic_date(sic_path)

            if date not in osi_date_to_index:
                continue

            osi_idx = osi_date_to_index[date]

            bremen_avg, counts = read_bremen(
                sic_path,
                mapping,
                mapping_ok,
                osi_shape,
                min_contrib,
            )

            osi_day = O[osi_idx]
            uncertainty_day = U[osi_idx]
            flags_day = FLAGS[osi_idx]

            land = (flags_day.astype(np.uint16) & 1) != 0

            comparison = (
                np.isfinite(osi_day)
                & (~land)
                & np.isfinite(bremen_avg)
                & (counts >= min_contrib)
            )

            miz = (
                comparison
                & (osi_day >= miz_lo)
                & (osi_day < miz_hi)
                & np.isfinite(uncertainty_day)
            )

            miz_uncertainties.append(
                uncertainty_day[miz]
            )

        if miz_uncertainties:

            miz_uncertainties = np.concatenate(
                miz_uncertainties
            )

            if miz_uncertainties.size:

                print(
                    f"  MIZ mean {np.mean(miz_uncertainties):.2f}"
                    "  -- compare against the MIZ MAE above "
                    "before calling any of it error"
                )

        # ------------------------------------------------------------------
        # OSI quality flags
        # ------------------------------------------------------------------

        print()

        print(
            f"OSI quality flags among the "
            f"{pooled_result['n']:,} comparison cells:"
        )

        flag_definitions = [
            (1, "land"),
            (4, "open_water_filtered"),
            (8, "land_spill_over"),
            (16, "high_t2m"),
            (128, "outside_max_ice_climo"),
        ]

        pooled_flags_uint = pooled_flags.astype(np.uint16)

        for bit, name in flag_definitions:

            flagged = (
                pooled_flags_uint & np.uint16(bit)
            ) != 0

            n_flagged = int(np.count_nonzero(flagged))

            percentage = (
                100.0 * n_flagged / pooled_result["n"]
                if pooled_result["n"] > 0
                else float("nan")
            )

            print(
                f"  {name:<23s} bit {bit:3d} "
                f"{n_flagged:9,d} ({percentage:6.2f}%)"
            )

        # ------------------------------------------------------------------
        # Interpolation flags
        # ------------------------------------------------------------------

        spatial_interp = (
            pooled_flags_uint & np.uint16(32)
        ) != 0

        temporal_interp = (
            pooled_flags_uint & np.uint16(64)
        ) != 0

        interpolated = spatial_interp | temporal_interp

        n_interpolated = int(
            np.count_nonzero(interpolated)
        )

        interpolated_percentage = (
            100.0 * n_interpolated / pooled_result["n"]
            if pooled_result["n"] > 0
            else float("nan")
        )

        print(
            f"  -> {n_interpolated:,} cells "
            f"({interpolated_percentage:.2f}%) are spatially "
            f"or temporally interpolated in OSI, not independent "
            f"observations."
        )

        # ------------------------------------------------------------------
        # Interpretation note
        # ------------------------------------------------------------------

        print()

        print(
            "This is a resolution-matched spatial consistency comparison "
            "between two"
        )

        print(
            "retrievals, not a measure of accuracy of either. The actual "
            "OSI sensor"
        )

        print(
            "footprint is not modelled (see the note at the top of this "
            "file), so part of"
        )

        print(
            "the residual is that unmodelled effect and cannot be separated "
            "from genuine"
        )

        print(
            "retrieval differences here."
        )

        print()

        return 0

    finally:
        ds.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare Bremen SIC against OSI-430-a"
    )

    parser.add_argument(
        "--osi",
        type=str,
        default="data/raw/osi430_sample_20250115_20250117.nc",
        help="OSI-430-a NetCDF",
    )

    parser.add_argument(
        "--sic-dir",
        type=str,
        default="data/raw",
        help="directory holding sic_YYYYMMDD.tif",
    )

    parser.add_argument(
        "--min-contrib",
        type=int,
        default=DEFAULT_MIN_CONTRIB,
        help="minimum Bremen cells required per OSI cell",
    )

    parser.add_argument(
        "--miz-low",
        type=float,
        default=MIZ_LO,
        help=f"MIZ lower bound, inclusive (default {MIZ_LO:g})",
    )

    parser.add_argument(
        "--miz-high",
        type=float,
        default=MIZ_HI,
        help=f"MIZ upper bound, exclusive (default {MIZ_HI:g})",
    )

    args = parser.parse_args()

    sys.exit(
        compare(
            Path(args.osi),
            Path(args.sic_dir),
            args.min_contrib,
            args.miz_low,
            args.miz_high,
        )
    )
