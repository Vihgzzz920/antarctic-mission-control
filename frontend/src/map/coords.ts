// Display-only coordinate conversion.
//
// The mission request is, and stays, a pair of grid cells: that is what the
// router takes and what the API is sent. This converts a cell to geographic
// coordinates so an operator reads a position rather than an array index. The
// arithmetic mirrors RoutingGrid.rowcol_to_xy -- cell CENTRE, same +0.5 -- and
// then proj4 takes it from the project CRS to WGS 84.

import proj4 from 'proj4'

import { PROJECT_EPSG, PROJECT_PROJ4 } from '../generated/crs'
import { projectProjection } from './crs'

export interface GridGeometry {
  transform: number[] // [a, b, c, d, e, f] as rasterio orders it
}

export function cellCentreXY(
  grid: GridGeometry,
  row: number,
  col: number,
): [number, number] {
  const [a, b, c, d, e, f] = grid.transform
  return [
    c + (col + 0.5) * a + (row + 0.5) * b,
    f + (col + 0.5) * d + (row + 0.5) * e,
  ]
}

/** [lon, lat] in degrees, or null when the grid geometry is not known yet. */
export function cellLonLat(
  grid: GridGeometry | null | undefined,
  row: number,
  col: number,
): [number, number] | null {
  if (!grid?.transform || grid.transform.length < 6) return null
  projectProjection() // makes sure the definition is registered
  proj4.defs(PROJECT_EPSG, PROJECT_PROJ4)
  const [x, y] = cellCentreXY(grid, row, col)
  const out = proj4(PROJECT_PROJ4, 'EPSG:4326', [x, y]) as [number, number]
  if (!Number.isFinite(out[0]) || !Number.isFinite(out[1])) return null
  return out
}

/** "71.42 S  12.30 E" -- the form navigators read, not a decimal pair. */
export function formatLonLat(pair: [number, number] | null): string | null {
  if (!pair) return null
  const [lon, lat] = pair
  const ns = lat < 0 ? 'S' : 'N'
  const ew = lon < 0 ? 'W' : 'E'
  return `${Math.abs(lat).toFixed(2)}° ${ns}   ${Math.abs(lon).toFixed(2)}° ${ew}`
}
