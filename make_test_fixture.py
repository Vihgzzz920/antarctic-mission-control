"""Generate a SYNTHETIC SIC GeoTIFF so the inspection code can be developed offline.

!!! THIS IS NOT REAL DATA. It is a concentric-ring pattern with deliberately
!!! awkward properties (uint8 dtype, flag codes 120 and 255, EPSG:3412 rather
!!! than 3031) so that inspect_sic.py is exercised on the messy cases.
!!! NEVER use its output for analysis or in the demo.

Usage:  python scripts/make_test_fixture.py
Writes: data/raw/sic_TESTFIXTURE.tif
"""
import numpy as np, rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

N, RES, HALF = 1216, 6250.0, 3800000.0
yy, xx = np.mgrid[0:N, 0:N]
cy = cx = N / 2
r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

sic = np.clip(100 - (r - 180) * 0.85, 0, 100)   # ice ring around a centre
sic[r < 180] = 120                               # "land" flag code (continent)
sic[r > 560] = 0                                 # open ocean
rng = np.random.default_rng(0)
noise = np.clip(sic + rng.normal(0, 6, sic.shape), 0, 100)
sic = np.where(sic == 120, 120, noise)
sic[r > 585] = 255                               # "missing" flag code outside swath
arr = sic.astype("uint8")

prof = dict(driver="GTiff", height=N, width=N, count=1, dtype="uint8",
            crs=CRS.from_epsg(3412),             # deliberately NOT 3031, to prove we read it
            transform=from_origin(-HALF, HALF, RES, RES), nodata=255)
with rasterio.open("data/raw/sic_TESTFIXTURE.tif", "w", **prof) as dst:
    dst.write(arr, 1)
print("fixture written:", arr.shape, arr.dtype, "unique tail:", np.unique(arr)[-5:])
