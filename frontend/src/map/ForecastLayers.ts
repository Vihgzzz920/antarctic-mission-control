// Building the OpenLayers sources for one frame of forecast playback.
//
// Positions come from map/forecast.resolveState, which returns the model's own
// state at a horizon and interpolates only between horizons. Nothing here
// computes a drift or a radius.

import Feature from 'ol/Feature'
import Circle from 'ol/geom/Circle'
import LineString from 'ol/geom/LineString'
import Point from 'ol/geom/Point'
import VectorSource from 'ol/source/Vector'
import proj4 from 'proj4'

import { PROJECT_PROJ4 } from '../generated/crs'
import type { IcebergForecastRecord } from '../api/types'
import { resolveState } from './forecast'

export interface ForecastFrame {
  observed: VectorSource
  predicted: VectorSource
  rings: VectorSource
  tracks: VectorSource
}

/** lon/lat degrees to the project CRS's metres. The one place the frontend
 *  does this conversion; the camera reuses it so a forecast frame is built
 *  from the same coordinates the forecast layer draws. */
export const projectLonLat = (lon: number, lat: number): [number, number] =>
  proj4('EPSG:4326', PROJECT_PROJ4, [lon, lat]) as [number, number]

export function buildForecastFrame(
  records: IcebergForecastRecord[],
  hours: number,
): ForecastFrame {
  const observed: Feature[] = []
  const predicted: Feature[] = []
  const rings: Feature[] = []
  const tracks: Feature[] = []

  for (const record of records) {
    const state = resolveState(record, hours)
    if (!state) continue
    const here = projectLonLat(state.longitude, state.latitude)

    if (record.observed) {
      const start = projectLonLat(
        record.observed.longitude,
        record.observed.latitude,
      )
      observed.push(
        new Feature({
          geometry: new Point(start),
          iceberg_id: record.iceberg_id,
          kind: 'observed',
        }),
      )
      if (hours > 0) {
        tracks.push(new Feature({ geometry: new LineString([start, here]) }))
      }
    }

    if (hours > 0) {
      predicted.push(
        new Feature({
          geometry: new Point(here),
          iceberg_id: record.iceberg_id,
          kind: 'predicted',
          hours: state.hours,
          radius_km: state.radiusKm,
          extrapolated: state.extrapolatedUncertainty,
          exact: state.exact,
        }),
      )
      if (state.radiusKm > 0) {
        rings.push(
          new Feature({
            geometry: new Circle(here, state.radiusKm * 1000),
            extrapolated: state.extrapolatedUncertainty,
          }),
        )
      }
    }
  }

  return {
    observed: new VectorSource({ features: observed }),
    predicted: new VectorSource({ features: predicted }),
    rings: new VectorSource({ features: rings }),
    tracks: new VectorSource({ features: tracks }),
  }
}
