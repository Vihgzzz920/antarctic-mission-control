// Registering the PROJECT CRS with proj4 and OpenLayers.
//
// The definition is NOT written here. src/generated/crs.ts is emitted by
// src/api/build_presentation_layers.py straight out of the routing rasters'
// own CRS, so the browser draws in EPSG:3976 -- the NSIDC South Polar
// Stereographic grid every layer and every route is already on -- rather than
// in Web Mercator, which cannot represent the pole and distorts badly below
// about 70 degrees south.
//
// Nothing is reprojected on the way to the screen: the API sends projected
// metres in this CRS and the map consumes them directly.

import proj4 from 'proj4'
import { get as getProjection } from 'ol/proj'
import { register } from 'ol/proj/proj4'
import type { Projection } from 'ol/proj'

import {
  GRID_EXTENT,
  PROJECT_EPSG,
  PROJECT_PROJ4,
  PROJECT_PROJ4_VERBATIM,
} from '../generated/crs'

let registered: Projection | null = null

export function projectProjection(): Projection {
  if (registered) return registered
  proj4.defs(PROJECT_EPSG, PROJECT_PROJ4)
  register(proj4)
  const projection = getProjection(PROJECT_EPSG)
  if (!projection) {
    throw new Error(
      `could not register ${PROJECT_EPSG} with proj4: ${PROJECT_PROJ4}`,
    )
  }
  projection.setExtent(GRID_EXTENT)
  registered = projection
  return projection
}

/** Does the CRS the backend reports match the one compiled into the bundle? */
export function crsMatchesBackend(backendProj4: string | undefined): boolean {
  if (!backendProj4) return true
  const normalise = (s: string) =>
    s.replace('+no_defs=True', '+no_defs').trim().split(/\s+/).sort().join(' ')
  return (
    normalise(backendProj4) === normalise(PROJECT_PROJ4) ||
    normalise(backendProj4) === normalise(PROJECT_PROJ4_VERBATIM)
  )
}

export { GRID_EXTENT, PROJECT_EPSG, PROJECT_PROJ4 }
