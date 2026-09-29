import { describe, expect, it } from 'vitest'

import {
  LAYER,
  MAP_LAYERS,
  legendEntries,
  visibleLayers,
} from '../map/layerContract'
import type {
  LayerAvailability,
  LegendContext,
  MapLayerId,
} from '../map/layerContract'
import { CAMERA_STAGES } from '../map/camera'
import { PROFILE_COLOUR } from '../map/layers'
import { PROFILE_ORDER } from '../api/types'

const EVERYTHING: LayerAvailability = {
  stage: 'observe',
  hasForecast: true,
  hasExposure: true,
  routedProfiles: 3,
  hasEndpoints: true,
  hasSimulation: true,
  hasVessel: true,
  pricedHorizonHours: 12,
}

const at = (over: Partial<LayerAvailability>) =>
  visibleLayers({ ...EVERYTHING, ...over })

const CONTEXT: LegendContext = {
  routedProfiles: PROFILE_ORDER,
  selectedProfile: 'risk_oriented',
  profileColour: PROFILE_COLOUR,
  pricedHorizonHours: 12,
  exposureFieldHours: 12,
  exposureIcebergCount: 11,
  calibratedToHours: 24,
}

describe('the layer contract', () => {
  it('describes every layer exactly once', () => {
    expect(Object.keys(LAYER).sort()).toEqual([...MAP_LAYERS].sort())
    for (const id of MAP_LAYERS) expect(LAYER[id].id).toBe(id)
    expect(new Set(MAP_LAYERS).size).toBe(MAP_LAYERS.length)
  })

  it('emits only known layer ids, whatever it is given', () => {
    for (const stage of CAMERA_STAGES) {
      for (const over of [
        {},
        { hasForecast: false },
        { hasExposure: false },
        { routedProfiles: 0 },
        { hasSimulation: false },
        { pricedHorizonHours: null },
      ]) {
        for (const id of at({ stage, ...over })) {
          expect(MAP_LAYERS).toContain(id)
        }
      }
    }
  })
})

describe('OBSERVE draws', () => {
  const drawn = at({ stage: 'observe' })

  it('the environment and the observations, and nothing later', () => {
    expect(drawn).toEqual([
      'sic',
      'no_coverage',
      'land',
      'observed_icebergs',
      'endpoints',
    ])
  })

  it('no forecast product, no exposure, no route, no hazard', () => {
    for (const absent of [
      'predicted_icebergs',
      'forecast_track',
      'forecast_envelope',
      'iceberg_exposure',
      'selected_route',
      'alternate_routes',
      'superseded_route',
      'simulated_iceberg',
      'simulated_envelope',
    ] as MapLayerId[]) {
      expect(drawn).not.toContain(absent)
    }
  })

  it('drops the observations when no forecast has arrived', () => {
    expect(at({ stage: 'observe', hasForecast: false })).not.toContain(
      'observed_icebergs',
    )
  })
})

describe('FORECAST draws', () => {
  const drawn = at({ stage: 'forecast' })

  it('observed, predicted, the track between them and the envelope', () => {
    expect(drawn).toEqual([
      'sic',
      'no_coverage',
      'land',
      'observed_icebergs',
      'predicted_icebergs',
      'forecast_track',
      'forecast_envelope',
      'endpoints',
    ])
  })

  it('no route, no exposure field, no hazard', () => {
    for (const absent of [
      'selected_route',
      'alternate_routes',
      'iceberg_exposure',
      'simulated_iceberg',
    ] as MapLayerId[]) {
      expect(drawn).not.toContain(absent)
    }
  })
})

describe('PLAN draws', () => {
  const drawn = at({ stage: 'plan' })

  it('the routes, the priced exposure field and the states it covers', () => {
    expect(drawn).toEqual([
      'sic',
      'no_coverage',
      'land',
      'iceberg_exposure',
      'observed_icebergs',
      'predicted_icebergs',
      'selected_route',
      'alternate_routes',
      'vessel',
      'endpoints',
    ])
  })

  it('no forecast track or envelope: the exposure field carries that radius', () => {
    expect(drawn).not.toContain('forecast_track')
    expect(drawn).not.toContain('forecast_envelope')
  })

  it('no iceberg states at all when the priced horizon is unknown', () => {
    const unknown = at({ stage: 'plan', pricedHorizonHours: null })
    expect(unknown).not.toContain('predicted_icebergs')
    expect(unknown).not.toContain('observed_icebergs')
  })

  it('no exposure entry when the field has no drawable cells', () => {
    expect(at({ stage: 'plan', hasExposure: false })).not.toContain(
      'iceberg_exposure',
    )
  })

  it('no alternates when only one profile routed', () => {
    const one = at({ stage: 'plan', routedProfiles: 1 })
    expect(one).toContain('selected_route')
    expect(one).not.toContain('alternate_routes')
    expect(at({ stage: 'plan', routedProfiles: 0 })).not.toContain('selected_route')
  })

  it('no hazard: the simulation belongs to RESPOND', () => {
    expect(drawn).not.toContain('simulated_iceberg')
    expect(drawn).not.toContain('superseded_route')
  })
})

describe('RESPOND draws', () => {
  const drawn = at({ stage: 'respond' })

  it('the plan, plus what the injection added', () => {
    expect(drawn).toEqual([
      ...at({ stage: 'plan' }).filter((id) => id !== 'endpoints'),
      'superseded_route',
      'simulated_iceberg',
      'simulated_envelope',
      'endpoints',
    ])
  })

  it('falls back to the plan when nothing has been injected', () => {
    expect(at({ stage: 'respond', hasSimulation: false })).toEqual(
      at({ stage: 'plan' }),
    )
  })
})

describe('the legend follows the layers', () => {
  it('describes every drawn layer and nothing else', () => {
    for (const stage of CAMERA_STAGES) {
      const drawn = at({ stage })
      const entries = legendEntries(drawn, CONTEXT)
      //  every row belongs to a layer that was drawn
      for (const entry of entries) expect(drawn).toContain(entry.layer)
      //  and every drawn layer is accounted for
      expect(new Set(entries.map((e) => e.layer))).toEqual(new Set(drawn))
    }
  })

  it('cannot describe a layer that was not drawn', () => {
    const entries = legendEntries(['sic', 'land'], CONTEXT)
    expect(entries.map((e) => e.layer)).toEqual(['sic', 'land'])
    for (const banned of [
      'iceberg_exposure',
      'forecast_envelope',
      'selected_route',
    ]) {
      expect(entries.some((e) => e.layer === banned)).toBe(false)
    }
  })

  it('gives each route its own row and its own colour', () => {
    const entries = legendEntries(at({ stage: 'plan' }), CONTEXT)
    const routes = entries.filter(
      (e) => e.layer === 'selected_route' || e.layer === 'alternate_routes',
    )
    expect(routes).toHaveLength(PROFILE_ORDER.length)
    expect(routes.filter((e) => e.emphasis)).toHaveLength(1)
    expect(routes[0].colour).toBe(PROFILE_COLOUR.risk_oriented)
    expect(new Set(routes.map((e) => e.colour)).size).toBe(PROFILE_ORDER.length)
  })

  it('tells observed and predicted apart', () => {
    const entries = legendEntries(at({ stage: 'forecast' }), CONTEXT)
    const observed = entries.find((e) => e.layer === 'observed_icebergs')!
    const predicted = entries.find((e) => e.layer === 'predicted_icebergs')!
    expect(observed.label).toMatch(/observed/i)
    expect(predicted.label).toMatch(/predicted/i)
    expect(observed.swatch).not.toBe(predicted.swatch)
    expect(observed.swatch).toBe('berg-solid')
    expect(predicted.swatch).toBe('berg-hollow')
  })

  it('states the priced horizon rather than a remembered one', () => {
    const entries = legendEntries(at({ stage: 'plan' }), CONTEXT)
    const predicted = entries.find((e) => e.layer === 'predicted_icebergs')!
    expect(predicted.note).toContain('+12h')
    expect(predicted.note).toMatch(/priced/i)
    const exposure = entries.find((e) => e.layer === 'iceberg_exposure')!
    expect(exposure.note).toContain('12h field')
    expect(exposure.note).toContain('max over 11 icebergs')
  })

  it('names the DRAWN exposure field, not the route\u2019s last priced one', () => {
    //  the overlay is one field; the route may have been priced against
    //  several. The row describes the field on screen.
    const entries = legendEntries(at({ stage: 'plan' }), {
      ...CONTEXT,
      pricedHorizonHours: 6,
      exposureFieldHours: 12,
    })
    expect(entries.find((e) => e.layer === 'iceberg_exposure')!.note).toContain(
      '12h field',
    )
    expect(entries.find((e) => e.layer === 'predicted_icebergs')!.note).toContain(
      '+6h',
    )
  })

  it('leaves the note off when the value is not known', () => {
    const entries = legendEntries(at({ stage: 'plan' }), {
      ...CONTEXT,
      pricedHorizonHours: null,
      exposureFieldHours: null,
      exposureIcebergCount: null,
      calibratedToHours: null,
    })
    for (const entry of entries) {
      expect(entry.note ?? '').not.toMatch(/undefined|null|NaN/)
    }
    expect(
      entries.find((e) => e.layer === 'iceberg_exposure')!.note,
    ).toBeUndefined()
  })

  it('calls the envelope what it is, and nothing it is not', () => {
    for (const stage of CAMERA_STAGES) {
      for (const entry of legendEntries(at({ stage }), CONTEXT)) {
        const text = `${entry.label} ${entry.note ?? ''}`.toLowerCase()
        for (const banned of [
          'collision probability',
          'probability',
          'safety zone',
          'safe',
          'unsafe',
          'guarantee',
          'avoids',
          'danger',
          'best route',
        ]) {
          expect(text, `${entry.key} says "${banned}"`).not.toContain(banned)
        }
      }
    }
    const envelope = legendEntries(at({ stage: 'forecast' }), CONTEXT).find(
      (e) => e.layer === 'forecast_envelope',
    )!
    expect(envelope.label).toMatch(/uncertainty radius/i)
    expect(envelope.note).toMatch(/calibrated to 24h/i)
  })
})
