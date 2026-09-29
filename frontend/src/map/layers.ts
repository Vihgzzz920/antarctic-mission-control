// The presentation layers, and the styling of everything drawn over them.
//
// The PNGs are small RGBA reductions of the real rasters, produced by
// src/api/build_presentation_layers.py, which records each source file's
// sha256 in public/layers/manifest.json. The full-resolution rasters stay on
// the server; the browser never downloads them.
//
// Drawing order, back to front: sea ice (held well back so it reads as a
// field, not a subject), outside-coverage, land, uncertainty rings, icebergs,
// unselected routes, selected route, endpoints.

import GeoJSON from 'ol/format/GeoJSON'
import ImageLayer from 'ol/layer/Image'
import Circle from 'ol/geom/Circle'
import Feature from 'ol/Feature'
import Static from 'ol/source/ImageStatic'
import VectorSource from 'ol/source/Vector'
import CircleStyle from 'ol/style/Circle'
import Fill from 'ol/style/Fill'
import RegularShape from 'ol/style/RegularShape'
import Stroke from 'ol/style/Stroke'
import Style from 'ol/style/Style'
import type { FeatureLike } from 'ol/Feature'
import type { Projection } from 'ol/proj'

import { GRID_EXTENT } from './crs'
import type { ProfileName } from '../api/types'

export const PROFILE_COLOUR: Record<ProfileName, string> = {
  fastest: '#4cd7f5',
  fuel_efficient: '#c9a7ff',
  risk_oriented: '#8ef2a6',
  shortest_distance: '#ffc857',
}

/**
 * The sea-ice field is background and nothing more. At anything near full
 * strength its texture competes with a 1-2 px line for the eye, so it is held
 * low enough that route geometry reads first and the ice reads as context.
 */
export const SIC_OPACITY = 0.24

export function backgroundLayers(projection: Projection) {
  const image = (url: string, opacity: number, z: number) => {
    const layer = new ImageLayer({
      opacity,
      zIndex: z,
      source: new Static({ url, projection, imageExtent: GRID_EXTENT }),
    })
    return layer
  }
  const sic = image('/layers/sic.png', SIC_OPACITY, 0)
  const noCoverage = image('/layers/no_coverage.png', 0.7, 1)
  const land = image('/layers/land.png', 1, 2)
  sic.set('title', 'Sea-ice concentration')
  noCoverage.set('title', 'Outside bathymetry coverage')
  land.set('title', 'Land and grounded ice')
  return { sic, noCoverage, land }
}

export function geoJsonSource(collection: unknown, projection: Projection) {
  return new VectorSource({
    features: new GeoJSON().readFeatures(collection, {
      dataProjection: projection,
      featureProjection: projection,
    }),
  })
}

/**
 * The forecast uncertainty radius, drawn in PROJECTED METRES so it is the real
 * cone on the chart rather than a fixed pixel blob. One translucent ring per
 * exposed cell, no fill stacking loud enough to hide the route beneath it.
 */
export function uncertaintySource(
  collection: { features: Array<Record<string, unknown>> },
  radiusKm: number,
): VectorSource<Feature<Circle>> {
  const radius = Math.max(radiusKm, 0) * 1000
  const features: Feature<Circle>[] = []
  for (const raw of collection.features) {
    const geometry = raw.geometry as { coordinates?: [number, number] }
    if (!geometry?.coordinates) continue
    features.push(
      new Feature({ geometry: new Circle(geometry.coordinates, radius) }),
    )
  }
  return new VectorSource({ features })
}

export const uncertaintyStyle = new Style({
  //  A ring the route can be read THROUGH: the outline carries the radius,
  //  the fill is barely there so a line crossing it stays unbroken.
  fill: new Fill({ color: 'rgba(255, 138, 128, 0.04)' }),
  stroke: new Stroke({ color: 'rgba(255, 150, 140, 0.26)', width: 0.9 }),
})

/** Icebergs read as bergs: a pale triangle above a waterline. */
export function icebergStyle(feature: FeatureLike) {
  const value = Number(feature.get('exposure') ?? 0)
  const size = 3.6 + 4 * value
  return [
    new Style({
      image: new RegularShape({
        points: 3,
        radius: size,
        fill: new Fill({ color: `rgba(233, 245, 255, ${0.45 + 0.35 * value})` }),
        stroke: new Stroke({ color: 'rgba(120, 190, 220, 0.75)', width: 0.85 }),
      }),
    }),
    new Style({
      image: new RegularShape({
        points: 4,
        radius: size * 0.95,
        radius2: 0,
        angle: Math.PI / 4,
        stroke: new Stroke({
          color: `rgba(140, 200, 230, ${0.2 + 0.25 * value})`,
          width: 1,
        }),
        displacement: [0, -size * 0.75],
      }),
    }),
  ]
}

/**
 * Selected routes get a dark casing plus a bright core, which is how a line
 * stays readable over both open water and the ice field. Alternates stay
 * visible but recede, and each profile keeps its own dash so the three are
 * distinguishable without relying on colour alone.
 */
const DASH: Record<ProfileName, number[] | undefined> = {
  fastest: undefined,
  fuel_efficient: [9, 4, 2, 4],
  risk_oriented: [14, 6],
  shortest_distance: [3, 5],
}

/** A dash tuned at 2 px reads as dots at 5 px, so it scales with the line. */
const scaleDash = (dash: number[] | undefined, factor: number) =>
  dash?.map((segment) => segment * factor)

export function routeStyle(profile: ProfileName, selected: boolean): Style[] {
  const colour = PROFILE_COLOUR[profile]
  if (!selected) {
    //  An alternate has to stay legible over the ice without ever reading as
    //  the answer: a thin casing for contrast, then a half-strength core.
    return [
      new Style({
        stroke: new Stroke({ color: 'rgba(3, 6, 10, 0.7)', width: 5 }),
        zIndex: 8,
      }),
      new Style({
        stroke: new Stroke({
          color: `${colour}8f`,
          width: 2.2,
          lineDash: DASH[profile],
        }),
        zIndex: 9,
      }),
    ]
  }
  //  The selected route is the subject of the screen. Three passes: a wide
  //  dark casing that separates it from whatever it crosses, a faint wash of
  //  its own colour so the corridor reads even where the line is thin, and a
  //  solid core at full saturation.
  return [
    new Style({
      stroke: new Stroke({ color: 'rgba(3, 6, 10, 0.92)', width: 12 }),
      zIndex: 27,
    }),
    new Style({
      stroke: new Stroke({ color: `${colour}33`, width: 8 }),
      zIndex: 28,
    }),
    new Style({
      stroke: new Stroke({
        color: colour,
        width: 5,
        lineDash: scaleDash(DASH[profile], 2),
      }),
      zIndex: 30,
    }),
  ]
}

export function cellStyle(selected: boolean, profile: ProfileName) {
  if (!selected) return new Style({})
  return new Style({
    image: new CircleStyle({
      radius: 3.1,
      fill: new Fill({ color: '#05090d' }),
      stroke: new Stroke({ color: PROFILE_COLOUR[profile], width: 1.4 }),
    }),
    zIndex: 40,
  })
}

/** Departure is a ringed fix; destination is a target diamond. */
export function endpointStyle(feature: FeatureLike) {
  const departure = String(feature.get('role') ?? '') === 'departure'
  //  Departure and destination must be findable at a glance on a dark chart,
  //  so each sits on a solid disc rather than a translucent one.
  const halo = new Style({
    image: new CircleStyle({
      radius: 14,
      fill: new Fill({ color: 'rgba(4, 8, 12, 0.88)' }),
      stroke: new Stroke({ color: 'rgba(232, 244, 255, 0.28)', width: 1 }),
    }),
    zIndex: 48,
  })
  if (departure) {
    return [
      halo,
      new Style({
        image: new CircleStyle({
          radius: 8,
          fill: new Fill({ color: 'rgba(232, 244, 255, 0.14)' }),
          stroke: new Stroke({ color: '#e8f4ff', width: 2.4 }),
        }),
        zIndex: 50,
      }),
      new Style({
        image: new CircleStyle({
          radius: 2.4,
          fill: new Fill({ color: '#e8f4ff' }),
        }),
        zIndex: 51,
      }),
    ]
  }
  return [
    halo,
    new Style({
      image: new RegularShape({
        points: 4,
        radius: 10.5,
        angle: Math.PI / 4,
        fill: new Fill({ color: 'rgba(5, 9, 13, 0.92)' }),
        stroke: new Stroke({ color: '#e8f4ff', width: 2.4 }),
      }),
      zIndex: 50,
    }),
    new Style({
      image: new CircleStyle({
        radius: 2.8,
        fill: new Fill({ color: '#e8f4ff' }),
      }),
      zIndex: 51,
    }),
  ]
}


/** The one simulated berg: amber, dashed, unmistakably not observed data. */
export const SIMULATED_COLOUR = '#ffb020'

export function simulatedIcebergStyle(feature: FeatureLike) {
  const value = Number(feature.get('exposure') ?? 0)
  const size = 6 + 5 * value
  return [
    new Style({
      image: new RegularShape({
        points: 3,
        radius: size,
        fill: new Fill({ color: `rgba(255, 176, 32, ${0.5 + 0.4 * value})` }),
        stroke: new Stroke({ color: '#ffd489', width: 1.2 }),
      }),
      zIndex: 44,
    }),
  ]
}

export const simulatedUncertaintyStyle = new Style({
  fill: new Fill({ color: 'rgba(255, 176, 32, 0.07)' }),
  stroke: new Stroke({
    color: 'rgba(255, 176, 32, 0.6)',
    width: 1.2,
    lineDash: [5, 4],
  }),
  zIndex: 43,
})

/** The route the simulation superseded: present, clearly past tense. */
export function supersededRouteStyle(): Style[] {
  return [
    new Style({
      stroke: new Stroke({ color: 'rgba(4, 8, 12, 0.6)', width: 5 }),
      zIndex: 18,
    }),
    new Style({
      stroke: new Stroke({
        color: 'rgba(226, 238, 250, 0.5)',
        width: 1.8,
        lineDash: [2, 6],
      }),
      zIndex: 19,
    }),
  ]
}

// ────────────────────────────────────── forecast playback and vessel
/** Observed positions: hollow, cool, clearly the model's INPUT. */
/** Observed positions: SOLID -- a recorded fact, grounded on the chart. */
export const observedIcebergStyle = new Style({
  image: new RegularShape({
    points: 3,
    radius: 5.5,
    fill: new Fill({ color: 'rgba(226, 242, 255, 0.92)' }),
    stroke: new Stroke({ color: 'rgba(120, 190, 220, 0.95)', width: 1 }),
  }),
  zIndex: 34,
})

/**
 * Predicted positions: OUTLINED -- a projection, not an observation, and so
 * deliberately hollow beside the solid observed mark. Amber once the horizon
 * runs past the calibration.
 */
export function predictedIcebergStyle(extrapolated: boolean): Style {
  return new Style({
    image: new RegularShape({
      points: 3,
      radius: 6.5,
      fill: new Fill({ color: 'rgba(5, 9, 13, 0.55)' }),
      stroke: new Stroke({
        color: extrapolated ? '#ffc857' : 'rgba(150, 205, 235, 0.95)',
        width: 1.6,
        lineDash: extrapolated ? [3, 2] : undefined,
      }),
    }),
    zIndex: 36,
  })
}

/** The uncertainty radius at the displayed horizon, in real metres. */
export function forecastRingStyle(extrapolated: boolean): Style {
  return new Style({
    fill: new Fill({
      color: extrapolated
        ? 'rgba(255, 176, 32, 0.045)'
        : 'rgba(150, 205, 235, 0.045)',
    }),
    stroke: new Stroke({
      color: extrapolated
        ? 'rgba(255, 176, 32, 0.5)'
        : 'rgba(150, 205, 235, 0.38)',
      width: 1,
      lineDash: extrapolated ? [5, 4] : undefined,
    }),
    zIndex: 33,
  })
}

/** A thin line from the observed position to the displayed prediction. */
export const driftTrackStyle = new Style({
  stroke: new Stroke({
    color: 'rgba(150, 205, 235, 0.35)',
    width: 1,
    lineDash: [2, 4],
  }),
  zIndex: 32,
})

export const vesselStyle = new Style({
  image: new CircleStyle({
    radius: 6,
    fill: new Fill({ color: '#ffffff' }),
    stroke: new Stroke({ color: '#05090d', width: 2 }),
  }),
  zIndex: 60,
})
