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
import type { CameraPlan } from './camera'
import { LAYER } from './layerContract'
import type { MapLayerId } from './layerContract'
import { useMapCamera } from './useMapCamera'
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
} from './layers'
import type {
  GeoJsonCollection,
  IcebergForecastRecord,
  ProfileName,
  RouteCell,
} from '../api/types'
import { PROFILE_ORDER } from '../api/types'

/** every on-map title comes from the contract, so the words match the legend */
const LAYER_TITLE = Object.fromEntries(
  Object.values(LAYER).map((descriptor) => [descriptor.id, descriptor.label]),
) as Record<MapLayerId, string>

interface Props {
  routes: Record<string, GeoJsonCollection> | null
  /**
   * which profiles' geometry to draw. Objectives that returned the same path
   * are drawn once, under the profile named here; see api/routeEquivalence.ts.
   * Null draws every profile the payload carries.
   */
  drawProfiles?: ProfileName[] | null
  endpoints: GeoJsonCollection | null
  exposure: GeoJsonCollection | null
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
  /** where this stage wants to look; see map/camera.ts */
  camera: CameraPlan | null
  /** exactly the layers this stage draws; see map/layerContract.ts. The
   *  legend is built from this same list, so it cannot describe a layer that
   *  was never added here. */
  layers: MapLayerId[]
  /** handed the map once, for tests and for anything that needs the instance */
  onMapReady?: (map: Map) => void
  /** while armed, the next map click places the simulated iceberg */
  placing: boolean
  onPlace: (position: [number, number]) => void
  selected: ProfileName | null
  onCellPick: (cell: RouteCell | null) => void
}

export default function MissionMap({
  routes,
  drawProfiles,
  endpoints,
  exposure,
  supersededRoute,
  simulatedExposure,
  simulatedRadiusKm,
  forecastRecords,
  forecastHours,
  vessel,
  camera,
  layers,
  onMapReady,
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
  const shows = (id: MapLayerId) => layers.includes(id)

  // --- the map itself, created once -------------------------------------
  useEffect(() => {
    if (!host.current || map.current) return
    const { sic, noCoverage, land } = backgroundLayers(projection)
    sic.set('layerId', 'sic')
    noCoverage.set('layerId', 'no_coverage')
    land.set('layerId', 'land')
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
    onMapReady?.(instance)
    return () => {
      instance.setTarget(undefined)
      map.current = null
    }
    //  eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projection])

  //  the view is driven by the stage's camera plan from here on; the fit above
  //  is only the opening frame while no plan has arrived yet
  useMapCamera(map, camera)

  //  the background is built once, but it still answers to the contract: a
  //  base layer the stage does not list is hidden rather than quietly drawn
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    for (const layer of instance.getLayers().getArray()) {
      const id = layer.get('layerId') as MapLayerId | undefined
      if (id === 'sic' || id === 'no_coverage' || id === 'land') {
        layer.setVisible(layers.includes(id))
      }
    }
  }, [layers])

  // --- the exposure field the router priced against ----------------------
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    const existing = overlayLayers.current.exposure
    if (existing) {
      instance.removeLayer(existing)
      overlayLayers.current.exposure = undefined
    }
    if (!exposure || !shows('iceberg_exposure')) return

    const layer = new VectorLayer({
      source: geoJsonSource(exposure, projection),
      style: icebergStyle,
      zIndex: 6,
    })
    layer.set('layerId', 'iceberg_exposure')
    layer.set('title', LAYER_TITLE.iceberg_exposure)
    instance.addLayer(layer)
    overlayLayers.current.exposure = layer
  }, [exposure, layers, projection])

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
      //  one line per distinct geometry: a profile whose path another profile
      //  already draws is left off rather than stacked on top of it
      if (drawProfiles && !drawProfiles.includes(name)) return
      const isSelected = name === selected
      if (!shows(isSelected ? 'selected_route' : 'alternate_routes')) return
      const layer = new VectorLayer({
        source: geoJsonSource(collection, projection),
        style: (feature: FeatureLike) =>
          feature.getGeometry()?.getType() === 'LineString'
            ? routeStyle(name, name === selected)
            : cellStyle(name === selected, name),
      })
      layer.set('profile', name)
      layer.set('layerId', isSelected ? 'selected_route' : 'alternate_routes')
      layer.setZIndex(isSelected ? 30 : 9)
      instance.addLayer(layer)
      routeLayers.current[name] = layer
    })
  }, [drawProfiles, routes, layers, projection, selected])

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
    if (!supersededRoute || !shows('superseded_route')) return
    const layer = new VectorLayer({
      source: geoJsonSource(supersededRoute, projection),
      style: (feature: FeatureLike) =>
        feature.getGeometry()?.getType() === 'LineString'
          ? supersededRouteStyle()
          : new Style({}),
      zIndex: 18,
    })
    layer.set('layerId', 'superseded_route')
    layer.set('title', LAYER_TITLE.superseded_route)
    instance.addLayer(layer)
    overlayLayers.current.superseded = layer
  }, [supersededRoute, layers, projection])

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

    if (
      simulatedRadiusKm &&
      simulatedRadiusKm > 0 &&
      shows('simulated_envelope')
    ) {
      const ring = new VectorLayer({
        source: uncertaintySource(
          simulatedExposure as { features: Array<Record<string, unknown>> },
          simulatedRadiusKm,
        ),
        style: simulatedUncertaintyStyle,
        zIndex: 43,
      })
      ring.set('layerId', 'simulated_envelope')
      ring.set('title', LAYER_TITLE.simulated_envelope)
      instance.addLayer(ring)
      overlayLayers.current.simRing = ring
    }

    if (!shows('simulated_iceberg')) return
    const layer = new VectorLayer({
      source: geoJsonSource(simulatedExposure, projection),
      style: simulatedIcebergStyle,
      zIndex: 44,
    })
    layer.set('layerId', 'simulated_iceberg')
    layer.set('title', LAYER_TITLE.simulated_iceberg)
    instance.addLayer(layer)
    overlayLayers.current.simBerg = layer
  }, [simulatedExposure, simulatedRadiusKm, layers, projection])

  // --- departure / destination -------------------------------------------
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    if (overlayLayers.current.endpoints) {
      instance.removeLayer(overlayLayers.current.endpoints)
      overlayLayers.current.endpoints = undefined
    }
    if (!endpoints || !shows('endpoints')) return
    const layer = new VectorLayer({
      source: geoJsonSource(endpoints, projection),
      style: endpointStyle,
      zIndex: 60,
    })
    layer.set('layerId', 'endpoints')
    layer.set('title', LAYER_TITLE.endpoints)
    instance.addLayer(layer)
    overlayLayers.current.endpoints = layer
  }, [endpoints, layers, projection])

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
    const add = (
      id: 'forecast_track' | 'forecast_envelope' | 'observed_icebergs' |
          'predicted_icebergs',
      key: 'fcTracks' | 'fcRings' | 'fcObserved' | 'fcPredicted',
      layer: VectorLayer<VectorSource>,
    ) => {
      if (!shows(id)) return
      layer.set('layerId', id)
      layer.set('title', LAYER_TITLE[id])
      instance.addLayer(layer)
      overlayLayers.current[key] = layer
    }

    add('forecast_track', 'fcTracks', new VectorLayer({
      source: frame.tracks, style: driftTrackStyle, zIndex: 32,
    }))
    add('forecast_envelope', 'fcRings', new VectorLayer({
      source: frame.rings,
      style: (feature: FeatureLike) =>
        forecastRingStyle(Boolean(feature.get('extrapolated'))),
      zIndex: 33,
    }))
    add('observed_icebergs', 'fcObserved', new VectorLayer({
      source: frame.observed, style: observedIcebergStyle, zIndex: 34,
    }))
    add('predicted_icebergs', 'fcPredicted', new VectorLayer({
      source: frame.predicted,
      style: (feature: FeatureLike) =>
        predictedIcebergStyle(Boolean(feature.get('extrapolated'))),
      zIndex: 36,
    }))
  }, [forecastRecords, forecastHours, layers])

  // --- the vessel during transit playback ---------------------------------
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    if (overlayLayers.current.vessel) {
      instance.removeLayer(overlayLayers.current.vessel)
      overlayLayers.current.vessel = undefined
    }
    if (!vessel || !shows('vessel')) return
    const layer = new VectorLayer({
      source: new VectorSource({
        features: [new Feature({ geometry: new Point(vessel) })],
      }),
      style: vesselStyle,
      zIndex: 60,
    })
    layer.set('layerId', 'vessel')
    layer.set('title', LAYER_TITLE.vessel)
    instance.addLayer(layer)
    overlayLayers.current.vessel = layer
  }, [vessel, layers])

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
      data-layers={layers.join(' ')}
    />
  )
}
