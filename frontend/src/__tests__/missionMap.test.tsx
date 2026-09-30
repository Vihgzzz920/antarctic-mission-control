import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type Map from 'ol/Map'

import MissionMap from '../map/MissionMap'
import { visibleLayers } from '../map/layerContract'
import type { LayerAvailability, MapLayerId } from '../map/layerContract'
import { resolveCameraPlan } from '../map/camera'
import { GRID_EXTENT } from '../map/crs'
import { forecastResponse, missionResponse, simulationResponse } from './fixtures'
import type { GeoJsonCollection } from '../api/types'

// The REAL map, in jsdom. What matters here is the guarantee the legend rests
// on: the OpenLayers map ends up holding exactly the layers the contract asked
// for -- no more, so the legend cannot describe something invisible, and no
// fewer, so it cannot omit something on screen.

const MISSION = missionResponse()
const EXPOSURE: GeoJsonCollection = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [-809375, -1490625] },
      properties: { exposure: 0.5 },
    },
  ],
}

const AVAILABLE: LayerAvailability = {
  stage: 'observe',
  hasForecast: true,
  hasExposure: true,
  routedProfiles: 3,
  hasEndpoints: true,
  hasSimulation: true,
  hasVessel: true,
  pricedHorizonHours: 12,
}

/** Render the map with EVERY piece of data available, and only the given
 *  layer list. Anything absent from the result was gated out, not missing. */
function drawnLayers(layers: MapLayerId[]): MapLayerId[] {
  let instance: Map | null = null
  render(
    <MissionMap
      routes={MISSION.geojson}
      endpoints={MISSION.endpoints}
      exposure={EXPOSURE}
      supersededRoute={MISSION.geojson.fastest}
      simulatedExposure={simulationResponse().simulated_exposure}
      simulatedRadiusKm={simulationResponse().simulation!.radius_km}
      forecastRecords={forecastResponse().icebergs}
      forecastHours={12}
      vessel={[-809375, -1490625]}
      camera={resolveCameraPlan({
        stage: 'plan',
        gridExtent: GRID_EXTENT,
        mission: null,
        subject: null,
        routes: null,
        hazard: null,
      })}
      layers={layers}
      onMapReady={(map) => {
        instance = map
      }}
      placing={false}
      onPlace={() => undefined}
      selected="risk_oriented"
      onCellPick={() => undefined}
    />,
  )
  const map = instance as Map | null
  expect(map).not.toBeNull()
  return map!
    .getLayers()
    .getArray()
    .filter((layer) => layer.getVisible())
    .map((layer) => layer.get('layerId') as MapLayerId)
}

describe('the map draws exactly what the contract lists', () => {
  it.each(['observe', 'forecast', 'plan', 'respond'] as const)(
    '%s',
    (stage) => {
      const wanted = visibleLayers({ ...AVAILABLE, stage })
      const drawn = drawnLayers(wanted)
      //  one contract id can cover several OpenLayers layers -- the two
      //  alternate routes are one legend kind -- so the SETS must match
      expect([...new Set(drawn)].sort()).toEqual([...new Set(wanted)].sort())
      expect(drawn.every((id) => id !== undefined)).toBe(true)
      //  and the kinds that fan out do so only where the contract allows
      for (const id of drawn) expect(wanted).toContain(id)
    },
  )

  it('gates a layer out even though its data was handed over', () => {
    //  every prop above is populated; the list is what decides
    const drawn = drawnLayers(['sic', 'land'])
    expect([...new Set(drawn)].sort()).toEqual(['land', 'sic'])
    for (const absent of [
      'iceberg_exposure',
      'selected_route',
      'alternate_routes',
      'observed_icebergs',
      'predicted_icebergs',
      'forecast_envelope',
      'forecast_track',
      'simulated_iceberg',
      'simulated_envelope',
      'superseded_route',
      'endpoints',
      'vessel',
    ]) {
      expect(drawn).not.toContain(absent)
    }
  })

  it('draws observed and predicted icebergs as two separate layers', () => {
    const drawn = drawnLayers(
      visibleLayers({ ...AVAILABLE, stage: 'forecast' }),
    )
    expect(drawn).toContain('observed_icebergs')
    expect(drawn).toContain('predicted_icebergs')
    expect(drawn).toContain('forecast_track')
    expect(drawn).toContain('forecast_envelope')
  })

  it('advertises its layer list on the element the legend sits beside', () => {
    const wanted = visibleLayers({ ...AVAILABLE, stage: 'plan' })
    drawnLayers(wanted)
    const host = document.querySelectorAll('[data-testid="mission-map"]')
    const last = host[host.length - 1]
    expect(last.getAttribute('data-layers')).toBe(wanted.join(' '))
  })
})
