import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  ApiError,
  compareMission,
  getHealth,
  getIcebergExposure,
  getIcebergForecast,
  simulateIceberg,
} from './api/client'
import type {
  ForecastResponse,
  GeoJsonCollection,
  HealthResponse,
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
import SelectedRouteBar from './components/SelectedRouteBar'
import SimulationControl from './components/SimulationControl'
import TopBar from './components/TopBar'
import type { Stage } from './components/TopBar'
import TransitControl from './components/TransitControl'
import WhyDrawer from './components/WhyDrawer'
import { cellCentreXY } from './map/coords'
import { crsMatchesBackend } from './map/crs'
import { transitSeconds, vesselAt } from './map/forecast'
import MissionMap from './map/MissionMap'

// The form's opening values only. Every metric on screen comes from the API.
const INITIAL_REQUEST: MissionRequest = {
  start: [948, 503],
  goal: [922, 497],
  departure_time: '2025-01-08T00:00:00',
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
  const [subject, setSubject] = useState<string | null>(null)

  const [whyOpen, setWhyOpen] = useState(false)
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
    getIcebergExposure(EXPOSURE_BUCKET)
      .then((collection) => live && setExposure(collection as GeoJsonCollection))
      .catch(() => undefined)
    return () => {
      live = false
    }
  }, [])

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
      setSubject((current) => current ?? response.icebergs[0]?.iceberg_id ?? null)
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
  const maxHours = forecast?.horizons.at(-1)?.hours ?? 48
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

  const clearSimulation = useCallback(() => {
    setSimulation(null)
    setPlacing(false)
    setCell(null)
    setError(null)
    setTransitSecond(0)
    setTransitPlaying(false)
  }, [])

  // ── derived ────────────────────────────────────────────────────────────
  const online = Boolean(health && health.status === 'ok')
  const historical = Boolean(mission?.historical_demonstration)
  const crsOk = useMemo(() => crsMatchesBackend(grid?.proj4), [grid?.proj4])
  const superseded =
    simulation && selected ? simulation.baseline.geojson[selected] ?? null : null
  const exposureRadiusKm = useMemo(() => {
    const layer = health?.exposure_layers?.find(
      (entry) => Number(entry.bucket) === EXPOSURE_BUCKET,
    )
    const radius = layer ? Number(layer.radius_km) : NaN
    return Number.isFinite(radius) ? radius : null
  }, [health])

  const subjectRecord =
    forecast?.icebergs.find((berg) => berg.iceberg_id === subject) ?? null

  return (
    <div className="app">
      <MissionMap
        routes={stage === 'plan' ? shown?.geojson ?? null : null}
        endpoints={shown?.endpoints ?? null}
        exposure={stage === 'observe' ? exposure : null}
        exposureRadiusKm={exposureRadiusKm}
        supersededRoute={superseded}
        simulatedExposure={simulation?.simulated_exposure ?? null}
        simulatedRadiusKm={simulation?.simulation?.radius_km ?? null}
        forecastRecords={stage === 'observe' ? null : forecast?.icebergs ?? null}
        forecastHours={hours}
        vessel={vesselXY}
        placing={placing}
        onPlace={place}
        selected={selected}
        onCellPick={setCell}
      />

      <TopBar
        stage={stage}
        reached={reached}
        grid={grid}
        online={online}
        historical={historical}
        onGo={setStage}
      />

      <MissionDrawer
        value={request}
        onChange={setRequest}
        grid={grid}
        busy={busy}
      />

      {stage === 'forecast' && forecast && (
        <ForecastPanel
          forecast={forecast}
          hours={hours}
          record={subjectRecord}
          onPick={setSubject}
        />
      )}

      {stage === 'plan' && (
        <>
          <CellInspector cell={cell} onClose={() => setCell(null)} />
          {mission && selected && (
            <SimulationControl
              armed={placing}
              active={Boolean(simulation)}
              busy={simBusy}
              disabled={!online}
              onArm={() => setPlacing((current) => !current)}
              onClear={clearSimulation}
            />
          )}
        </>
      )}

      <MapLegend
        selected={selected}
        hasRoutes={stage === 'plan' && Boolean(route)}
        simulated={Boolean(simulation)}
      />

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
            onUseForRouting={plan}
            planning={busy}
          />
        )}

        {stage === 'plan' && (
          <>
            {busy || simBusy ? (
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
                  selected={selected}
                  expanded={expanded}
                  onSelect={setSelected}
                  onExpand={setExpanded}
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
      </div>

      {stage === 'plan' && shown && selected && (
        <WhyDrawer
          comparison={shown.comparison}
          profile={selected}
          simulation={simulation}
          open={whyOpen}
          onClose={() => setWhyOpen(false)}
        />
      )}
    </div>
  )
}
