import { useEffect, useMemo, useRef } from 'react'
import Map from 'ol/Map'
import View from 'ol/View'
import VectorLayer from 'ol/layer/Vector'
import VectorSource from 'ol/source/Vector'
import Feature from 'ol/Feature'
import Point from 'ol/geom/Point'
import Style from 'ol/style/Style'
import { defaults as defaultInteractions } from 'ol/interaction'
import { defaults as defaultControls } from 'ol/control'
import type { FeatureLike } from 'ol/Feature'

import { GRID_EXTENT, projectProjection } from './crs'
import { buildForecastFrame } from './ForecastLayers'
import {
  backgroundLayers,
  driftTrackStyle,
  forecastRingStyle,
  observedIcebergStyle,
  predictedIcebergStyle,
  vesselStyle,
  cellStyle,
  endpointStyle,
  geoJsonSource,
  icebergStyle,
  routeStyle,
  simulatedIcebergStyle,
  simulatedUncertaintyStyle,
  supersededRouteStyle,
  uncertaintySource,
  uncertaintyStyle,
} from './layers'
import type {
  GeoJsonCollection,
  IcebergForecastRecord,
  ProfileName,
  RouteCell,
} from '../api/types'
import { PROFILE_ORDER } from '../api/types'

interface Props {
  routes: Record<string, GeoJsonCollection> | null
  endpoints: GeoJsonCollection | null
  exposure: GeoJsonCollection | null
  /** the calibrated forecast radius for the exposure bucket being drawn */
  exposureRadiusKm: number | null
  /** the route the simulation superseded, drawn behind the replanned one */
  supersededRoute: GeoJsonCollection | null
  /** the injected berg's own exposure cells, and its uncertainty radius */
  simulatedExposure: GeoJsonCollection | null
  simulatedRadiusKm: number | null
  /** forecast playback: the model's own records, and the clock position */
  forecastRecords: IcebergForecastRecord[] | null
  forecastHours: number
  /** the vessel's drawn position during transit playback */
  vessel: [number, number] | null
  /** while armed, the next map click places the simulated iceberg */
  placing: boolean
  onPlace: (position: [number, number]) => void
  selected: ProfileName | null
  onCellPick: (cell: RouteCell | null) => void
}

export default function MissionMap({
  routes,
  endpoints,
  exposure,
  exposureRadiusKm,
  supersededRoute,
  simulatedExposure,
  simulatedRadiusKm,
  forecastRecords,
  forecastHours,
  vessel,
  placing,
  onPlace,
  selected,
  onCellPick,
}: Props) {
  const host = useRef<HTMLDivElement | null>(null)
  const map = useRef<Map | null>(null)
  const routeLayers = useRef<Record<string, VectorLayer<VectorSource>>>({})
  const overlayLayers = useRef<{
    exposure?: VectorLayer<VectorSource>
    uncertainty?: VectorLayer<VectorSource>
    endpoints?: VectorLayer<VectorSource>
    superseded?: VectorLayer<VectorSource>
    fcObserved?: VectorLayer<VectorSource>
    fcPredicted?: VectorLayer<VectorSource>
    fcRings?: VectorLayer<VectorSource>
    fcTracks?: VectorLayer<VectorSource>
    vessel?: VectorLayer<VectorSource>
    simBerg?: VectorLayer<VectorSource>
    simRing?: VectorLayer<VectorSource>
  }>({})
  const projection = useMemo(() => projectProjection(), [])

  // --- the map itself, created once -------------------------------------
  useEffect(() => {
    if (!host.current || map.current) return
    const { sic, noCoverage, land } = backgroundLayers(projection)
    const instance = new Map({
      target: host.current,
      layers: [sic, noCoverage, land],
      controls: defaultControls({ attribution: false, rotate: false }),
      interactions: defaultInteractions({ altShiftDragRotate: false, pinchRotate: false }),
      view: new View({
        projection,
        center: [0, 0],
        extent: GRID_EXTENT,
        zoom: 3,
        minZoom: 2,
        maxZoom: 9,
        showFullExtent: true,
      }),
    })
    instance.getView().fit(GRID_EXTENT, { padding: [24, 24, 24, 24] })
    map.current = instance
    return () => {
      instance.setTarget(undefined)
      map.current = null
    }
  }, [projection])

  // --- iceberg exposure, with its forecast uncertainty behind it ---------
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    for (const key of ['uncertainty', 'exposure'] as const) {
      const existing = overlayLayers.current[key]
      if (existing) {
        instance.removeLayer(existing)
        overlayLayers.current[key] = undefined
      }
    }
    if (!exposure) return

    if (exposureRadiusKm && exposureRadiusKm > 0) {
      const rings = new VectorLayer({
        source: uncertaintySource(
          exposure as { features: Array<Record<string, unknown>> },
          exposureRadiusKm,
        ),
        style: uncertaintyStyle,
        zIndex: 4,
      })
      rings.set('title', 'Forecast uncertainty radius')
      instance.addLayer(rings)
      overlayLayers.current.uncertainty = rings
    }

    const layer = new VectorLayer({
      source: geoJsonSource(exposure, projection),
      style: icebergStyle,
      zIndex: 6,
    })
    layer.set('title', 'Modelled iceberg exposure')
    instance.addLayer(layer)
    overlayLayers.current.exposure = layer
  }, [exposure, exposureRadiusKm, projection])

  // --- one vector layer per profile -------------------------------------
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    Object.values(routeLayers.current).forEach((l) => instance.removeLayer(l))
    routeLayers.current = {}
    if (!routes) return
    PROFILE_ORDER.forEach((name) => {
      const collection = routes[name]
      if (!collection) return
      const layer = new VectorLayer({
        source: geoJsonSource(collection, projection),
        style: (feature: FeatureLike) =>
          feature.getGeometry()?.getType() === 'LineString'
            ? routeStyle(name, name === selected)
            : cellStyle(name === selected, name),
      })
      layer.set('profile', name)
      layer.setZIndex(name === selected ? 30 : 9)
      instance.addLayer(layer)
      routeLayers.current[name] = layer
    })
  }, [routes, projection, selected])

  // restyle without rebuilding when only the selection changes
  useEffect(() => {
    Object.entries(routeLayers.current).forEach(([name, layer]) => {
      const profile = name as ProfileName
      layer.setStyle((feature: FeatureLike) =>
        feature.getGeometry()?.getType() === 'LineString'
          ? routeStyle(profile, profile === selected)
          : cellStyle(profile === selected, profile),
      )
      layer.setZIndex(profile === selected ? 30 : 9)
    })
  }, [selected])

  // --- the route the simulation superseded --------------------------------
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    if (overlayLayers.current.superseded) {
      instance.removeLayer(overlayLayers.current.superseded)
      overlayLayers.current.superseded = undefined
    }
    if (!supersededRoute) return
    const layer = new VectorLayer({
      source: geoJsonSource(supersededRoute, projection),
      style: (feature: FeatureLike) =>
        feature.getGeometry()?.getType() === 'LineString'
          ? supersededRouteStyle()
          : new Style({}),
      zIndex: 18,
    })
    layer.set('title', 'Current route, superseded')
    instance.addLayer(layer)
    overlayLayers.current.superseded = layer
  }, [supersededRoute, projection])

  // --- the injected iceberg and its uncertainty ---------------------------
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    for (const key of ['simRing', 'simBerg'] as const) {
      const existing = overlayLayers.current[key]
      if (existing) {
        instance.removeLayer(existing)
        overlayLayers.current[key] = undefined
      }
    }
    if (!simulatedExposure) return

    if (simulatedRadiusKm && simulatedRadiusKm > 0) {
      const ring = new VectorLayer({
        source: uncertaintySource(
          simulatedExposure as { features: Array<Record<string, unknown>> },
          simulatedRadiusKm,
        ),
        style: simulatedUncertaintyStyle,
        zIndex: 43,
      })
      ring.set('title', 'Simulated iceberg uncertainty radius')
      instance.addLayer(ring)
      overlayLayers.current.simRing = ring
    }

    const layer = new VectorLayer({
      source: geoJsonSource(simulatedExposure, projection),
      style: simulatedIcebergStyle,
      zIndex: 44,
    })
    layer.set('title', 'SIMULATED ICEBERG')
    instance.addLayer(layer)
    overlayLayers.current.simBerg = layer
  }, [simulatedExposure, simulatedRadiusKm, projection])

  // --- departure / destination -------------------------------------------
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    if (overlayLayers.current.endpoints) {
      instance.removeLayer(overlayLayers.current.endpoints)
      overlayLayers.current.endpoints = undefined
    }
    if (!endpoints) return
    const layer = new VectorLayer({
      source: geoJsonSource(endpoints, projection),
      style: endpointStyle,
      zIndex: 60,
    })
    instance.addLayer(layer)
    overlayLayers.current.endpoints = layer
  }, [endpoints, projection])

  // --- forecast playback ---------------------------------------------------
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    for (const key of ['fcTracks', 'fcRings', 'fcObserved', 'fcPredicted'] as const) {
      const existing = overlayLayers.current[key]
      if (existing) {
        instance.removeLayer(existing)
        overlayLayers.current[key] = undefined
      }
    }
    if (!forecastRecords || forecastRecords.length === 0) return

    const frame = buildForecastFrame(forecastRecords, forecastHours)
    const tracks = new VectorLayer({
      source: frame.tracks, style: driftTrackStyle, zIndex: 32,
    })
    const rings = new VectorLayer({
      source: frame.rings,
      style: (feature: FeatureLike) =>
        forecastRingStyle(Boolean(feature.get('extrapolated'))),
      zIndex: 33,
    })
    const observed = new VectorLayer({
      source: frame.observed, style: observedIcebergStyle, zIndex: 34,
    })
    const predicted = new VectorLayer({
      source: frame.predicted,
      style: (feature: FeatureLike) =>
        predictedIcebergStyle(Boolean(feature.get('extrapolated'))),
      zIndex: 36,
    })
    tracks.set('title', 'Observed to predicted')
    observed.set('title', 'Observed positions')
    predicted.set('title', 'Predicted positions')
    rings.set('title', 'Forecast uncertainty radius')
    for (const layer of [tracks, rings, observed, predicted]) {
      instance.addLayer(layer)
    }
    overlayLayers.current.fcTracks = tracks
    overlayLayers.current.fcRings = rings
    overlayLayers.current.fcObserved = observed
    overlayLayers.current.fcPredicted = predicted
  }, [forecastRecords, forecastHours])

  // --- the vessel during transit playback ---------------------------------
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    if (overlayLayers.current.vessel) {
      instance.removeLayer(overlayLayers.current.vessel)
      overlayLayers.current.vessel = undefined
    }
    if (!vessel) return
    const layer = new VectorLayer({
      source: new VectorSource({
        features: [new Feature({ geometry: new Point(vessel) })],
      }),
      style: vesselStyle,
      zIndex: 60,
    })
    layer.set('title', 'Vessel')
    instance.addLayer(layer)
    overlayLayers.current.vessel = layer
  }, [vessel])

  // --- clicking a route cell --------------------------------------------
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    const handler = (event: { pixel: number[]; coordinate: number[] }) => {
      if (placing) {
        onPlace([event.coordinate[0], event.coordinate[1]])
        return
      }
      let picked: RouteCell | null = null
      instance.forEachFeatureAtPixel(
        event.pixel as [number, number],
        (feature: FeatureLike) => {
          if (picked) return
          if (feature.getGeometry()?.getType() !== 'Point') return
          const props = (feature as Feature).getProperties()
          if (props.row === undefined || props.arrival_time === undefined) return
          const { geometry, ...rest } = props
          void geometry
          picked = rest as RouteCell
        },
        { hitTolerance: 6 },
      )
      onCellPick(picked)
    }
    instance.on('singleclick', handler)
    return () => {
      instance.un('singleclick', handler)
    }
  }, [onCellPick, onPlace, placing])

  // keep the canvas honest when the panel layout changes
  useEffect(() => {
    const instance = map.current
    if (!instance || !host.current) return
    const observer = new ResizeObserver(() => instance.updateSize())
    observer.observe(host.current)
    return () => observer.disconnect()
  }, [])

  return (
    <div
      className={`map-canvas${placing ? ' is-placing' : ''}`}
      ref={host}
      data-testid="mission-map"
    />
  )
}
