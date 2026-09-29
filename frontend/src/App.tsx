import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  ApiError,
  compareMission,
  getDefaults,
  getHealth,
  getHistoricalEvaluation,
  getIcebergExposure,
  getIcebergForecast,
  simulateIceberg,
} from './api/client'
import type {
  ForecastResponse,
  GeoJsonCollection,
  HealthResponse,
  HistoricalEvaluationResponse,
  MissionRequest,
  MissionResponse,
  ProfileName,
  RouteCell,
  SimulationResponse,
} from './api/types'
import { PROFILE_ORDER } from './api/types'
import CellInspector from './components/CellInspector'
import { DeckFailure, PlanningState } from './components/DeckStates'
import ForecastPanel from './components/ForecastPanel'
import ForecastTimeline from './components/ForecastTimeline'
import MapLegend from './components/MapLegend'
import MissionDrawer from './components/MissionDrawer'
import PlanDock from './components/PlanDock'
import RoutingGate from './components/RoutingGate'
import SelectedRouteBar from './components/SelectedRouteBar'
import RespondDock from './components/RespondDock'
import RespondDrawer from './components/RespondDrawer'
import TopBar from './components/TopBar'
import type { Stage } from './components/TopBar'
import TransitControl from './components/TransitControl'
import ValidationPanel from './components/ValidationPanel'
import WhyDrawer from './components/WhyDrawer'
import { resolveCameraPlan } from './map/camera'
import type { CameraStage, XY } from './map/camera'
import { buildProvenance, routingReadiness } from './api/provenance'
import { drawnProfiles, groupEquivalentRoutes } from './api/routeEquivalence'
import type { RoutingReadiness } from './api/provenance'
import { visibleLayers, legendEntries } from './map/layerContract'
import {
  UNPRICED,
  exposureFields,
  pricedHorizonSummary,
  resolvePricedHorizon,
} from './map/pricedHorizon'
import type { LegendContext } from './map/layerContract'
import { PROFILE_COLOUR } from './map/layers'
import { cellCentreXY } from './map/coords'
import { injectionPoint } from './map/injection'
import { defaultForecastSubject } from './map/subject'
import type { MissionCorridor } from './map/subject'
import { crsMatchesBackend, GRID_EXTENT } from './map/crs'
import { projectLonLat } from './map/ForecastLayers'
import { transitSeconds, vesselAt } from './map/forecast'
import MissionMap from './map/MissionMap'

// The form's opening values only. Every metric on screen comes from the API.
// The form's opening values only, and no date among them: the demonstration's
// own departure time is a backend fact, fetched from /api/mission/defaults
// below. Until it arrives the field is empty and routing is refused rather
// than run against a date invented here.
const INITIAL_REQUEST: MissionRequest = {
  start: [948, 503],
  goal: [922, 497],
  departure_time: '',
  vessel_speed_mps: 5,
  polaris_ice_class: 'PC6',
  polaris_riv_table: '1.3',
  historical_demo_override: false,
}

const EXPOSURE_BUCKET = 1
/** wall-clock ms per forecast hour at 1x */
const FORECAST_MS_PER_HOUR = 90
/** wall-clock ms per transit second at 1x */
const TRANSIT_MS_PER_SECOND = 0.12

export default function App() {
  const [stage, setStage] = useState<Stage>('observe')
  const [reached, setReached] = useState<Stage[]>(['observe'])

  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<ApiError | null>(null)
  const [request, setRequest] = useState<MissionRequest>(INITIAL_REQUEST)
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [mission, setMission] = useState<MissionResponse | null>(null)
  const [exposure, setExposure] = useState<GeoJsonCollection | null>(null)
  const [selected, setSelected] = useState<ProfileName | null>(null)
  const [expanded, setExpanded] = useState<ProfileName | null>(null)
  const [cell, setCell] = useState<RouteCell | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<{ code: string; message: string } | null>(
    null,
  )

  const [hours, setHours] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  //  null until the operator picks one in the panel; the default below
  //  fills in until then
  const [pickedSubject, setPickedSubject] = useState<string | null>(null)

  const [whyOpen, setWhyOpen] = useState(false)
  const [provenanceOpen, setProvenanceOpen] = useState(false)
  //  the held-out historical evaluation. Fetched only when the operator opens
  //  it, so the main flow never waits on evidence it did not ask for.
  const [evidenceOpen, setEvidenceOpen] = useState(false)
  const [evidence, setEvidence] =
    useState<HistoricalEvaluationResponse | null>(null)
  const [evidenceError, setEvidenceError] = useState<string | null>(null)
  const [changeOpen, setChangeOpen] = useState(false)
  //  plan() is declared before the provenance it must respect, so it reads the
  //  current readiness through this mirror rather than a stale closure
  const readinessRef = useRef<RoutingReadiness>({
    state: 'unknown',
    canRoute: true,
    gated: false,
    headline: 'Provenance unknown',
    detail: null,
    action: 'route',
    actionLabel: 'Use forecast for routing',
  })
  const [transitPlaying, setTransitPlaying] = useState(false)
  const [transitSecond, setTransitSecond] = useState(0)

  const [simulation, setSimulation] = useState<SimulationResponse | null>(null)
  const [placing, setPlacing] = useState(false)
  const [simBusy, setSimBusy] = useState(false)

  // ── load what the backend already has ──────────────────────────────────
  useEffect(() => {
    let live = true
    getHealth()
      .then((response) => {
        if (!live) return
        setHealth(response)
        setHealthError(null)
      })
      .catch((cause: ApiError) => live && setHealthError(cause))
    //  the demonstration's own inputs, so no date is written into this app
    getDefaults()
      .then(({ demo }) => {
        if (!live || !demo) return
        setRequest((current) => ({
          ...current,
          start: Array.isArray(demo.start)
            ? (demo.start as [number, number])
            : current.start,
          goal: Array.isArray(demo.goal)
            ? (demo.goal as [number, number])
            : current.goal,
          departure_time:
            typeof demo.departure_time === 'string'
              ? demo.departure_time
              : current.departure_time,
          vessel_speed_mps:
            typeof demo.vessel_speed_mps === 'number'
              ? demo.vessel_speed_mps
              : current.vessel_speed_mps,
          polaris_ice_class:
            typeof demo.polaris_ice_class === 'string'
              ? demo.polaris_ice_class
              : current.polaris_ice_class,
          polaris_riv_table:
            typeof demo.polaris_riv_table === 'string'
              ? demo.polaris_riv_table
              : current.polaris_riv_table,
        }))
      })
      .catch(() => undefined)
    getIcebergExposure(EXPOSURE_BUCKET)
      .then((collection) => live && setExposure(collection as GeoJsonCollection))
      .catch(() => undefined)
    //  OBSERVE shows the iceberg observations themselves, so the forecast is
    //  fetched with the rest of the environment rather than on the way into
    //  the next stage. Start forecast then has nothing left to wait for.
    getIcebergForecast()
      .then((response) => {
        //  only a payload that actually carries icebergs is a forecast; a
        //  half-answer is left alone rather than half-drawn
        if (live && Array.isArray(response?.icebergs)) setForecast(response)
      })
      .catch(() => undefined)
    return () => {
      live = false
    }
  }, [])

  //  fetched on first open only, and never re-fetched: the artifact is a file
  //  the backend produced, not a live figure
  useEffect(() => {
    if (!evidenceOpen || evidence || evidenceError) return
    let live = true
    getHistoricalEvaluation()
      .then((response) => live && setEvidence(response))
      .catch((cause: ApiError) =>
        live &&
        setEvidenceError(
          cause.message ||
            'The historical evaluation artifact is not available.',
        ),
      )
    return () => {
      live = false
    }
  }, [evidenceOpen, evidence, evidenceError])

  const go = useCallback((next: Stage) => {
    setStage(next)
    setReached((current) =>
      current.includes(next) ? current : [...current, next],
    )
  }, [])

  const startForecast = useCallback(async () => {
    setBusy(true)
    setError(null)
    try {
      const response = forecast ?? (await getIcebergForecast())
      setForecast(response)
      setHours(0)
      go('forecast')
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? { code: cause.code, message: cause.message }
          : { code: 'api_error', message: String(cause) },
      )
    } finally {
      setBusy(false)
    }
  }, [forecast, go])

  const plan = useCallback(async () => {
    if (!readinessRef.current.canRoute) {
      //  the pairing the backend refuses. The gate is already on screen; this
      //  is the guard behind it, so no request is spent discovering it.
      return
    }
    if (!request.departure_time) {
      //  the demonstration's departure time has not arrived from the backend;
      //  routing against a date invented here would be a fabrication
      setError({
        code: 'mission_defaults_unavailable',
        message:
          'The mission defaults have not arrived from the backend, so the ' +
          'departure time is unknown. Nothing is assumed in its place: set a ' +
          'departure time in the mission panel, or retry once the API responds.',
      })
      go('plan')
      return
    }
    setBusy(true)
    setError(null)
    setCell(null)
    setSimulation(null)
    setPlacing(false)
    setPlaying(false)
    try {
      const response = await compareMission(request)
      setMission(response)
      const firstRouted = PROFILE_ORDER.find(
        (name) => response.comparison.profiles[name]?.success,
      )
      setSelected(firstRouted ?? null)
      setExpanded(null)
      setTransitSecond(0)
      setTransitPlaying(false)
      if (!firstRouted) {
        setError({ code: response.status, message: response.message })
      }
      go('plan')
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? { code: cause.code, message: cause.message }
          : { code: 'api_error', message: String(cause) },
      )
      setMission(null)
      setSelected(null)
      go('plan')
    } finally {
      setBusy(false)
    }
  }, [go, request])


  // ── forecast playback ──────────────────────────────────────────────────
  const maxHours = forecast?.horizons?.at(-1)?.hours ?? 48
  const frame = useRef<number>(0)
  useEffect(() => {
    if (!playing) return
    let last = performance.now()
    const tick = (now: number) => {
      const delta = now - last
      last = now
      setHours((current) => {
        const next = current + (delta / FORECAST_MS_PER_HOUR) * speed
        if (next >= maxHours) {
          setPlaying(false)
          return maxHours
        }
        return next
      })
      frame.current = requestAnimationFrame(tick)
    }
    frame.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame.current)
  }, [playing, speed, maxHours])

  // ── vessel transit playback ────────────────────────────────────────────
  const shown = simulation?.replanned ?? mission
  const route = shown && selected ? shown.comparison.profiles[selected] : null
  const total = route ? transitSeconds(route.cells) : 0

  const transitFrame = useRef<number>(0)
  useEffect(() => {
    if (!transitPlaying || total <= 0) return
    let last = performance.now()
    const tick = (now: number) => {
      const delta = now - last
      last = now
      setTransitSecond((current) => {
        const next = current + delta / TRANSIT_MS_PER_SECOND
        if (next >= total) {
          setTransitPlaying(false)
          return total
        }
        return next
      })
      transitFrame.current = requestAnimationFrame(tick)
    }
    transitFrame.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(transitFrame.current)
  }, [transitPlaying, total])

  const vesselPosition = useMemo(
    () => (route && transitSecond > 0 ? vesselAt(route.cells, transitSecond) : null),
    [route, transitSecond],
  )
  const grid = shown?.grid ?? health?.grid ?? null
  const vesselXY = useMemo(() => {
    if (!vesselPosition || !grid) return null
    const here = cellCentreXY(grid, vesselPosition.row, vesselPosition.col)
    const next = cellCentreXY(grid, vesselPosition.nextRow, vesselPosition.nextCol)
    const t = vesselPosition.fraction
    return [
      here[0] + (next[0] - here[0]) * t,
      here[1] + (next[1] - here[1]) * t,
    ] as [number, number]
  }, [vesselPosition, grid])
  const vesselCell = vesselPosition && route ? route.cells[vesselPosition.index] : null

  // ── simulated iceberg ──────────────────────────────────────────────────
  const place = useCallback(
    async (position: [number, number]) => {
      if (!mission || !selected) return
      setPlacing(false)
      setSimBusy(true)
      setError(null)
      setCell(null)
      try {
        const response = await simulateIceberg(
          request,
          { x: position[0], y: position[1] },
          selected,
        )
        if (!response.simulation) {
          setError({ code: response.status, message: response.message })
          return
        }
        setSimulation(response)
        setTransitSecond(0)
        setTransitPlaying(false)
        //  the evidence is the point of the stage: it opens with the result
        setChangeOpen(true)
      } catch (cause) {
        setError(
          cause instanceof ApiError
            ? { code: cause.code, message: cause.message }
            : { code: 'api_error', message: String(cause) },
        )
      } finally {
        setSimBusy(false)
      }
    },
    [mission, request, selected],
  )

  //  where a simulated iceberg goes when the operator asks for one ON the
  //  route: a vertex of the path the search returned, ~40% along it. Pure and
  //  repeatable; see map/injection.ts.
  const injection = useMemo(
    () => injectionPoint(route?.path, grid),
    [grid, route?.path],
  )

  const injectOnRoute = useCallback(() => {
    if (!injection) return
    setPlacing(false)
    void place(injection.position)
  }, [injection, place])

  const clearSimulation = useCallback(() => {
    setSimulation(null)
    setChangeOpen(false)
    setPlacing(false)
    setCell(null)
    setError(null)
    setTransitSecond(0)
    setTransitPlaying(false)
  }, [])

  // ── derived ────────────────────────────────────────────────────────────
  const online = Boolean(health && health.status === 'ok')
  const crsOk = useMemo(() => crsMatchesBackend(grid?.proj4), [grid?.proj4])
  const superseded =
    simulation && selected ? simulation.baseline.geojson[selected] ?? null : null
  // ── the priced horizon ─────────────────────────────────────────────────
  //  Which exposure fields exist at all (the routing configuration), and
  //  which of them THIS route actually met (its own buckets_used). A forecast
  //  horizon with no exposure field priced nothing, whatever the timeline can
  //  display; see map/pricedHorizon.ts.
  const fields = useMemo(
    () => exposureFields(health?.exposure_layers),
    [health],
  )
  const configuredPricedHours = useMemo(
    () => [...new Set(fields.map((field) => field.hours))].sort((a, b) => a - b),
    [fields],
  )
  const forecastHours = useMemo(
    () => forecast?.horizons?.map((horizon) => horizon.hours) ?? [],
    [forecast],
  )
  const priced = useMemo(() => {
    const profile = shown && selected ? shown.comparison.profiles[selected] : null
    if (!profile?.success) return UNPRICED
    return resolvePricedHorizon(profile.buckets_used, fields, forecastHours)
  }, [fields, forecastHours, selected, shown])
  const pricedHorizonHours = priced.lastHours
  const pricedSummary = priced === UNPRICED ? null : pricedHorizonSummary(priced)
  //  the overlay is one field, fetched by bucket. It is drawn only when the
  //  route was actually priced in that bucket -- otherwise it would sit beside
  //  a route it had no part in pricing.
  const drawnExposureField = useMemo(
    () => fields.find((field) => field.bucket === EXPOSURE_BUCKET) ?? null,
    [fields],
  )
  const exposureIsPriced =
    drawnExposureField !== null && priced.buckets.includes(EXPOSURE_BUCKET)

  //  the last horizon the uncertainty radius is calibrated to, from the
  //  forecast payload's own horizon list
  const calibratedToHours = useMemo(() => {
    const calibrated = forecast?.horizons?.filter(
      (horizon) => !horizon.extrapolated_uncertainty,
    )
    const last = calibrated?.at(-1)?.hours
    return typeof last === 'number' ? last : null
  }, [forecast])

  // ── where the data came from ───────────────────────────────────────────
  //  Assembled from /api/health, /api/forecast/icebergs and the mission that
  //  ran, if one has. Available from the first paint; see api/provenance.ts.
  const provenance = useMemo(
    () =>
      buildProvenance({
        health,
        forecast,
        mission: shown,
        configuredPricedHours,
      }),
    [configuredPricedHours, forecast, health, shown],
  )

  //  the routing gate: one switch, read here and edited in the mission panel
  const readiness = useMemo(
    () => routingReadiness(provenance, request.historical_demo_override),
    [provenance, request.historical_demo_override],
  )

  useEffect(() => {
    readinessRef.current = readiness
  }, [readiness])

  //  the pairing this app will not send, and the one the backend sent back
  const backendRefusedPairing =
    error?.code === 'historical_date_mismatch' ||
    shown?.status === 'historical_date_mismatch'
  const routingRefused = !readiness.canRoute || backendRefusedPairing
  const refusalNote = backendRefusedPairing
    ? (error?.message ?? shown?.message ?? null)
    : null

  //  RESPOND is only reachable once a route exists to respond with
  const canRespond = Boolean(route)
  const navigable = useMemo(
    () => reached.filter((name) => name !== 'respond' || canRespond),
    [canRespond, reached],
  )
  const respond = useCallback(() => {
    setCell(null)
    go('respond')
  }, [go])

  //  a route that goes away takes RESPOND with it
  useEffect(() => {
    if (stage === 'respond' && !canRespond) setStage('plan')
  }, [canRespond, stage])

  const enableHistoricalDemonstration = useCallback(() => {
    //  the operator's own act; nothing switches itself
    setRequest((current) => ({ ...current, historical_demo_override: true }))
  }, [])

  // ── the mission corridor ───────────────────────────────────────────────
  //  departure to destination in projected metres, from the request's own
  //  cells through the grid transform the API reported
  const missionCorridor = useMemo<MissionCorridor | null>(() => {
    if (!grid) return null
    return {
      start: cellCentreXY(grid, request.start[0], request.start[1]) as XY,
      goal: cellCentreXY(grid, request.goal[0], request.goal[1]) as XY,
    }
  }, [grid, request.goal, request.start])

  // ── which iceberg the forecast opens on ────────────────────────────────
  //  Relevance to this mission, not the order the backend listed them in. An
  //  explicit pick in the panel is held separately and always wins; clearing
  //  it hands the choice back to the default.
  const defaultSubject = useMemo(
    () => defaultForecastSubject(forecast?.icebergs, missionCorridor),
    [forecast?.icebergs, missionCorridor],
  )
  const activeSubject = pickedSubject ?? defaultSubject
  const subjectRecord =
    forecast?.icebergs.find((berg) => berg.iceberg_id === activeSubject) ?? null

  // ── where the map looks ────────────────────────────────────────────────
  //  The stage asks for a frame; map/camera.ts works out the extent from the
  //  geometry that is actually on hand -- the request's own cells, the paths
  //  the search returned, the forecast's own states, the iceberg the
  //  simulation placed. Nothing here is a fixed coordinate, and the plan is
  //  keyed by its geometry, so re-rendering does not move the view.
  //  one stage machine: the camera and the layer contract read the stage
  //  itself, rather than inferring a fourth state from the simulation
  const cameraStage: CameraStage = stage
  //  PLAN and RESPOND draw iceberg states at the horizon the route was priced
  //  at, never at a later one the routing configuration does not cover
  const forecastClock =
    cameraStage === 'plan' || cameraStage === 'respond'
      ? pricedHorizonHours ?? hours
      : hours
  const profiles = shown?.comparison.profiles ?? null
  const berg = simulation?.simulation ?? null

  const cameraPlan = useMemo(() => {
    const at = (cell: readonly number[]): XY | null =>
      grid && cell.length >= 2
        ? (cellCentreXY(grid, cell[0], cell[1]) as XY)
        : null

    const states = subjectRecord
      ? [subjectRecord.observed, ...subjectRecord.predicted].filter(
          (state): state is NonNullable<typeof state> => state !== null,
        )
      : []

    const paths = profiles
      ? PROFILE_ORDER.map((name) => profiles[name])
          .filter((profile) => profile?.success && profile.path.length > 0)
          .map((profile) =>
            profile.path
              .map((cell) => at(cell))
              .filter((point): point is XY => point !== null),
          )
          .filter((path) => path.length > 0)
      : []

    const hazardCentre = berg ? at([berg.row, berg.col]) : null

    return resolveCameraPlan({
      stage: cameraStage,
      gridExtent: GRID_EXTENT,
      mission: missionCorridor,
      subject:
        subjectRecord && states.length > 0
          ? {
              id: subjectRecord.iceberg_id,
              points: states.map((state) =>
                projectLonLat(state.longitude, state.latitude),
              ),
              maxRadiusM:
                Math.max(...states.map((state) => state.radius_km)) * 1000,
            }
          : null,
      routes: paths.length > 0 ? paths : null,
      hazard:
        berg && hazardCentre
          ? {
              centre: hazardCentre,
              radiusM: berg.radius_km * 1000,
              encounter: at(berg.encounter_cell),
            }
          : null,
    })
  }, [berg, cameraStage, grid, missionCorridor, profiles, subjectRecord])

  // ── what is on the map ─────────────────────────────────────────────────
  //  One list, from map/layerContract.ts. The map adds a layer only if its id
  //  is here, and the legend describes only the ids that are here, so a legend
  //  row cannot announce something that was never drawn.
  const routedProfiles = useMemo(
    () =>
      profiles
        ? PROFILE_ORDER.filter((name) => profiles[name]?.success)
        : [],
    [profiles],
  )

  //  Which profiles came back with the SAME geometry. The three objectives are
  //  three different questions, but under one constant vessel speed two of them
  //  can be answered by the same path; that is measured from the returned cell
  //  sequences rather than assumed. See api/routeEquivalence.ts.
  const routeGroups = useMemo(() => groupEquivalentRoutes(profiles), [profiles])
  //  one drawn line per distinct geometry, so a shared path is not stacked
  //  twice on the map
  const drawn = useMemo(
    () => drawnProfiles(routeGroups, selected),
    [routeGroups, selected],
  )

  const layers = useMemo(
    () =>
      visibleLayers({
        stage: cameraStage,
        hasForecast: (forecast?.icebergs?.length ?? 0) > 0,
        hasExposure:
          (exposure?.features?.length ?? 0) > 0 && exposureIsPriced,
        routedProfiles: routedProfiles.length,
        hasEndpoints: Boolean(shown?.endpoints),
        hasSimulation: Boolean(simulation?.simulation),
        hasVessel: Boolean(vesselXY),
        pricedHorizonHours,
      }),
    [
      cameraStage,
      exposure,
      exposureIsPriced,
      forecast,
      pricedHorizonHours,
      routedProfiles.length,
      shown,
      simulation,
      vesselXY,
    ],
  )

  const legend = useMemo(() => {
    const context: LegendContext = {
      routedProfiles,
      selectedProfile: selected,
      profileColour: PROFILE_COLOUR,
      pricedHorizonHours,
      exposureFieldHours: drawnExposureField?.hours ?? null,
      exposureIcebergCount: drawnExposureField?.icebergCount ?? null,
      calibratedToHours: calibratedToHours,
    }
    return legendEntries(layers, context)
  }, [
    calibratedToHours,
    drawnExposureField,
    layers,
    pricedHorizonHours,
    routedProfiles,
    selected,
  ])


  return (
    <div className="app">
      <MissionMap
        routes={shown?.geojson ?? null}
        drawProfiles={drawn}
        endpoints={shown?.endpoints ?? null}
        exposure={exposure}
        supersededRoute={superseded}
        simulatedExposure={simulation?.simulated_exposure ?? null}
        simulatedRadiusKm={simulation?.simulation?.radius_km ?? null}
        forecastRecords={forecast?.icebergs ?? null}
        forecastHours={forecastClock}
        vessel={vesselXY}
        camera={cameraPlan}
        layers={layers}
        placing={placing}
        onPlace={place}
        selected={selected}
        onCellPick={setCell}
      />

      <TopBar
        stage={stage}
        reached={navigable}
        online={online}
        provenance={provenance}
        provenanceOpen={provenanceOpen}
        onProvenanceToggle={setProvenanceOpen}
        onGo={setStage}
        evidenceOpen={evidenceOpen}
        onEvidenceToggle={setEvidenceOpen}
      />

      {stage !== 'respond' && (
        <MissionDrawer
          value={request}
          onChange={setRequest}
          grid={grid}
          busy={busy}
        />
      )}

      {stage === 'forecast' && forecast && (
        <ForecastPanel
          forecast={forecast}
          hours={hours}
          record={subjectRecord}
          onPick={setPickedSubject}
        />
      )}

      {(stage === 'plan' || stage === 'respond') && (
        <CellInspector cell={cell} onClose={() => setCell(null)} />
      )}

      <MapLegend entries={legend} />

      {(healthError || !crsOk) && (
        <div className="floating-alert" data-testid="status-banner">
          <strong>{healthError ? 'API unavailable' : 'Projection mismatch'}</strong>
          <p>
            {healthError
              ? healthError.message
              : 'The CRS compiled into this build does not match the one the backend reports.'}
          </p>
        </div>
      )}

      <div className="dock" data-stage={stage}>
        {stage === 'observe' && (
          <div className="observe-dock" data-testid="observe-dock">
            <div className="observe-copy">
              <h2>Observe</h2>
              <p>
                Real sea-ice field for {grid?.environment_date ?? 'the mission date'},
                the BedMachine exclusion, and the iceberg observations the drift
                model is built from. Departure and destination are set in the
                mission panel.
              </p>
            </div>
            <button
              type="button"
              className="primary-action"
              onClick={startForecast}
              disabled={!online || busy}
              data-testid="start-forecast"
            >
              {busy ? 'Loading forecast…' : 'Start forecast'}
            </button>
          </div>
        )}

        {stage === 'forecast' && forecast && (
          <ForecastTimeline
            forecast={forecast}
            hours={hours}
            playing={playing}
            speed={speed}
            onScrub={(next) => {
              setPlaying(false)
              setHours(next)
            }}
            onToggle={() => setPlaying((current) => !current)}
            onSpeed={setSpeed}
            onAct={
              readiness.action === 'enable_override'
                ? enableHistoricalDemonstration
                : plan
            }
            onDetails={() => setProvenanceOpen(true)}
            readiness={readiness}
            planning={busy}
            ready={Boolean(request.departure_time)}
            pricedHours={configuredPricedHours}
          />
        )}

        {stage === 'plan' && (
          <>
            {routingRefused ? (
              <RoutingGate
                readiness={readiness}
                onAct={
                  readiness.action === 'enable_override'
                    ? enableHistoricalDemonstration
                    : plan
                }
                onDetails={() => setProvenanceOpen(true)}
                busy={busy}
                variant="panel"
                eyebrow="Routing paused"
                note={refusalNote}
                testId="plan-gate"
              />
            ) : busy || simBusy ? (
              <PlanningState />
            ) : error && !mission ? (
              <DeckFailure code={error.code} message={error.message} />
            ) : shown && selected ? (
              <>
                <SelectedRouteBar
                  comparison={shown.comparison}
                  profile={selected}
                  label={simulation ? 'Replanned route' : 'Selected route'}
                  whyOpen={whyOpen}
                  onWhy={() => setWhyOpen((current) => !current)}
                  transit={
                    <TransitControl
                      playing={transitPlaying}
                      onToggle={() => {
                        if (transitSecond >= total) setTransitSecond(0)
                        setTransitPlaying((current) => !current)
                      }}
                      position={vesselPosition}
                      cell={vesselCell}
                    />
                  }
                />
                <PlanDock
                  comparison={shown.comparison}
                  groups={routeGroups}
                  selected={selected}
                  expanded={expanded}
                  onSelect={setSelected}
                  onExpand={setExpanded}
                  provenance={pricedSummary}
                  vesselSpeedMps={shown.request?.vessel_speed_mps ?? null}
                  canRespond={canRespond}
                  onRespond={respond}
                />
              </>
            ) : (
              <DeckFailure
                code={error?.code ?? shown?.status}
                message={error?.message ?? shown?.message ?? 'No route.'}
              />
            )}
          </>
        )}

        {stage === 'respond' && shown && selected && (
          <>
            <SelectedRouteBar
              comparison={shown.comparison}
              profile={selected}
              label={simulation ? 'Replanned route' : 'Current route'}
              whyOpen={false}
              onWhy={() => go('plan')}
              whyLabel="Back to plan"
              transit={
                <TransitControl
                  playing={transitPlaying}
                  onToggle={() => {
                    if (transitSecond >= total) setTransitSecond(0)
                    setTransitPlaying((current) => !current)
                  }}
                  position={vesselPosition}
                  cell={vesselCell}
                />
              }
            />
            <RespondDock
              profile={selected}
              simulation={simulation}
              armed={placing}
              busy={simBusy}
              disabled={!online}
              onInject={injectOnRoute}
              onArm={() => setPlacing((current) => !current)}
              onClear={clearSimulation}
              onExplain={() => setChangeOpen((current) => !current)}
              explaining={changeOpen}
              onBack={() => go('plan')}
              provenance={pricedSummary}
            />
          </>
        )}
      </div>

      {stage === 'respond' && selected && (
        <RespondDrawer
          simulation={simulation}
          profile={selected}
          open={changeOpen}
          onClose={() => setChangeOpen(false)}
        />
      )}

      {/*  Evidence, not part of the mission flow: available from any stage
           and closed by default.  */}
      {evidenceOpen && (
        <aside className="evidence-drawer" data-testid="evidence-panel">
          <header>
            <h2>Historical evaluation</h2>
            <button
              type="button"
              onClick={() => setEvidenceOpen(false)}
              aria-label="Close"
            >
              ×
            </button>
          </header>
          <ValidationPanel data={evidence} error={evidenceError} />
        </aside>
      )}

      {stage === 'plan' && shown && selected && (
        <WhyDrawer
          comparison={shown.comparison}
          profile={selected}
          pricedSummary={pricedSummary}
          open={whyOpen}
          onClose={() => setWhyOpen(false)}
        />
      )}
    </div>
  )
}
