// What is on the map, and therefore what the legend may say.
//
// ONE list. `visibleLayers()` decides which layers a stage draws given the
// data that has actually arrived; MissionMap adds a layer only if its id is in
// that list, and MapLegend describes only the ids in that same list. A legend
// row can therefore never announce something the map did not draw -- not
// exposure, not uncertainty, not a route, not a forecast state.
//
// Pure: no React, no OpenLayers. The descriptors carry the WORDS as well as
// the shape, so the phrasing lives beside the rule that shows it.

import type { CameraStage } from './camera'
import type { ProfileName } from '../api/types'
import { PROFILE_LABEL } from '../api/types'

export const MAP_LAYERS = [
  'sic',
  'no_coverage',
  'land',
  'endpoints',
  'observed_icebergs',
  'predicted_icebergs',
  'forecast_track',
  'forecast_envelope',
  'iceberg_exposure',
  'selected_route',
  'alternate_routes',
  'vessel',
  'superseded_route',
  'simulated_iceberg',
  'simulated_envelope',
] as const
export type MapLayerId = (typeof MAP_LAYERS)[number]

/** How a legend row draws its mark. Rendering detail, named once. */
export type LegendSwatch =
  | 'block'
  | 'line'
  | 'dashed'
  | 'berg-solid'
  | 'berg-hollow'
  | 'berg-hazard'
  | 'ring'
  | 'ring-hazard'
  | 'exposure'
  | 'marker'
  | 'dot'

export interface LayerDescriptor {
  id: MapLayerId
  label: string
  swatch: LegendSwatch
}

/** The one place a layer's name is written. */
export const LAYER: Record<MapLayerId, LayerDescriptor> = {
  sic: { id: 'sic', label: 'Sea-ice concentration', swatch: 'block' },
  no_coverage: {
    id: 'no_coverage',
    label: 'Outside bathymetry coverage',
    swatch: 'block',
  },
  land: { id: 'land', label: 'Land and grounded ice', swatch: 'block' },
  endpoints: { id: 'endpoints', label: 'Departure and destination', swatch: 'marker' },
  observed_icebergs: {
    id: 'observed_icebergs',
    label: 'Observed iceberg position',
    swatch: 'berg-solid',
  },
  predicted_icebergs: {
    id: 'predicted_icebergs',
    label: 'Predicted iceberg position',
    swatch: 'berg-hollow',
  },
  forecast_track: {
    id: 'forecast_track',
    label: 'Observed to predicted',
    swatch: 'dashed',
  },
  forecast_envelope: {
    id: 'forecast_envelope',
    label: 'Modelled uncertainty radius',
    swatch: 'ring',
  },
  iceberg_exposure: {
    id: 'iceberg_exposure',
    label: 'Modelled iceberg exposure',
    swatch: 'exposure',
  },
  selected_route: { id: 'selected_route', label: 'Selected route', swatch: 'line' },
  alternate_routes: {
    id: 'alternate_routes',
    label: 'Alternate route',
    swatch: 'line',
  },
  vessel: { id: 'vessel', label: 'Vessel position', swatch: 'dot' },
  superseded_route: {
    id: 'superseded_route',
    label: 'Route before the injection',
    swatch: 'dashed',
  },
  simulated_iceberg: {
    id: 'simulated_iceberg',
    label: 'Simulated iceberg',
    swatch: 'berg-hazard',
  },
  simulated_envelope: {
    id: 'simulated_envelope',
    label: "Simulated iceberg's uncertainty radius",
    swatch: 'ring-hazard',
  },
}

/** What the app has actually got hold of at this moment. */
export interface LayerAvailability {
  stage: CameraStage
  /** the forecast payload has arrived and carries at least one iceberg */
  hasForecast: boolean
  /** the exposure field the router priced with, as drawable features */
  hasExposure: boolean
  /** at least one profile returned a route */
  routedProfiles: number
  hasEndpoints: boolean
  hasSimulation: boolean
  hasVessel: boolean
  /**
   * The forecast horizon the routing configuration actually prices, in hours.
   * PLAN and RESPOND draw iceberg states only when this is known, so the map
   * never shows a horizon the route was not priced at.
   */
  pricedHorizonHours: number | null
}

/**
 * The layers this stage draws, in draw order.
 *
 * A layer appears only when the stage wants it AND the data for it exists.
 * Nothing is listed on the strength of the backend merely having the concept.
 */
export function visibleLayers(state: LayerAvailability): MapLayerId[] {
  const out: MapLayerId[] = ['sic', 'no_coverage', 'land']
  const forecastPriced =
    state.hasForecast && state.pricedHorizonHours !== null

  switch (state.stage) {
    case 'observe': {
      if (state.hasForecast) out.push('observed_icebergs')
      break
    }
    case 'forecast': {
      if (state.hasForecast) {
        out.push(
          'observed_icebergs',
          'predicted_icebergs',
          'forecast_track',
          'forecast_envelope',
        )
      }
      break
    }
    case 'plan':
    case 'respond': {
      //  the field the route was priced against, and the iceberg states at
      //  the horizon that field covers -- never a later one
      if (state.hasExposure) out.push('iceberg_exposure')
      if (forecastPriced) out.push('observed_icebergs', 'predicted_icebergs')
      if (state.routedProfiles > 0) out.push('selected_route')
      if (state.routedProfiles > 1) out.push('alternate_routes')
      if (state.hasVessel) out.push('vessel')
      if (state.stage === 'respond' && state.hasSimulation) {
        out.push('superseded_route', 'simulated_iceberg', 'simulated_envelope')
      }
      break
    }
  }

  if (state.hasEndpoints) out.push('endpoints')
  return out
}

// ───────────────────────────────────────────────────────────────── legend
export interface LegendEntry {
  /** unique within one legend rendering */
  key: string
  layer: MapLayerId
  label: string
  swatch: LegendSwatch
  /** a qualifier read off live data, never a fixed string */
  note?: string
  /** a route's own profile colour, when the row is one route */
  colour?: string
  emphasis?: boolean
}

export interface LegendContext {
  /** routed profiles, in the app's own order */
  routedProfiles: readonly ProfileName[]
  selectedProfile: ProfileName | null
  profileColour: Record<ProfileName, string>
  /** the last horizon THIS ROUTE was priced against, from its own buckets */
  pricedHorizonHours: number | null
  /** the horizon of the exposure field that is actually DRAWN, which is not
   *  always the route's last priced field */
  exposureFieldHours: number | null
  /** how many icebergs that field was built from, from the same metadata */
  exposureIcebergCount: number | null
  /** the last horizon the uncertainty radius is calibrated to */
  calibratedToHours: number | null
}

/**
 * One row per drawn layer, in the order the layers were listed.
 *
 * The routes expand to one row each so an operator can tell which colour is
 * which; everything else is a single row. Notes are built from live values --
 * the priced horizon, the iceberg count, the calibration limit -- so the
 * legend states what is on screen rather than a remembered description of it.
 */
export function legendEntries(
  layers: readonly MapLayerId[],
  context: LegendContext,
): LegendEntry[] {
  const entries: LegendEntry[] = []
  const alternates = context.routedProfiles.filter(
    (name) => name !== context.selectedProfile,
  )

  for (const id of layers) {
    const descriptor = LAYER[id]
    switch (id) {
      case 'selected_route': {
        const profile = context.selectedProfile
        if (!profile) break
        entries.push({
          key: `selected_route:${profile}`,
          layer: id,
          label: `${descriptor.label} — ${PROFILE_LABEL[profile]}`,
          swatch: 'line',
          colour: context.profileColour[profile],
          emphasis: true,
        })
        break
      }
      case 'alternate_routes': {
        for (const profile of alternates) {
          entries.push({
            key: `alternate_routes:${profile}`,
            layer: id,
            label: `${descriptor.label} — ${PROFILE_LABEL[profile]}`,
            swatch: 'line',
            colour: context.profileColour[profile],
          })
        }
        break
      }
      case 'predicted_icebergs': {
        entries.push({
          key: id,
          layer: id,
          label: descriptor.label,
          swatch: descriptor.swatch,
          note:
            context.pricedHorizonHours !== null
              ? `+${context.pricedHorizonHours}h — the last horizon this route was priced against`
              : undefined,
        })
        break
      }
      case 'forecast_envelope': {
        entries.push({
          key: id,
          layer: id,
          label: descriptor.label,
          swatch: descriptor.swatch,
          note:
            context.calibratedToHours !== null
              ? `calibrated to ${context.calibratedToHours}h; beyond that it is extrapolated`
              : undefined,
        })
        break
      }
      case 'iceberg_exposure': {
        const parts: string[] = []
        if (context.exposureFieldHours !== null) {
          parts.push(`${context.exposureFieldHours}h field`)
        }
        if (context.exposureIcebergCount !== null) {
          parts.push(`max over ${context.exposureIcebergCount} icebergs`)
        }
        entries.push({
          key: id,
          layer: id,
          label: descriptor.label,
          swatch: descriptor.swatch,
          note: parts.length > 0 ? parts.join(', ') : undefined,
        })
        break
      }
      default: {
        entries.push({
          key: id,
          layer: id,
          label: descriptor.label,
          swatch: descriptor.swatch,
        })
      }
    }
  }
  return entries
}
