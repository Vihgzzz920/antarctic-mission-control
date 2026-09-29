import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import App from '../App'
import { ApiError } from '../api/client'
import { PROFILE_ORDER } from '../api/types'
import { formatDate } from '../api/provenance'
import { spanOf } from '../map/camera'
import { projectLonLat } from '../map/ForecastLayers'
import type { Extent, XY } from '../map/camera'
import { cellCentreXY, cellLonLat } from '../map/coords'
import { GRID_EXTENT } from '../generated/crs'
import {
  forecastResponse,
  missionDefaults,
  missionResponse,
  simulationResponse,
} from './fixtures'
import { historicalEvaluation } from './missionEvidence.fixtures'

// The map is an OpenLayers canvas; jsdom has no WebGL or layout. The mock
// records the props the map is actually handed, so the tests can see what the
// map would draw at each stage without drawing it.
vi.mock('../map/MissionMap', () => ({
  default: ({
    selected,
    placing,
    onPlace,
    routes,
    drawProfiles,
    supersededRoute,
    exposure,
    forecastRecords,
    forecastHours,
    vessel,
    layers,
    camera,
  }: {
    selected: string | null
    placing: boolean
    onPlace: (position: [number, number]) => void
    routes: unknown
    drawProfiles: string[] | null | undefined
    supersededRoute: unknown
    exposure: unknown
    forecastRecords: unknown[] | null
    forecastHours: number
    vessel: [number, number] | null
    layers: string[]
    camera: {
      key: string
      stage: string
      moves: Array<{ frame: string; extent: number[]; delayMs: number }>
    } | null
  }) => (
    <div
      data-testid="mission-map"
      data-selected={selected ?? 'none'}
      data-placing={String(placing)}
      data-routes={routes ? 'yes' : 'no'}
      data-drawn={drawProfiles ? drawProfiles.join(' ') : 'all'}
      data-superseded={supersededRoute ? 'yes' : 'no'}
      data-exposure={exposure ? 'yes' : 'no'}
      data-forecast-bergs={forecastRecords ? String(forecastRecords.length) : 'none'}
      data-forecast-hours={forecastHours.toFixed(4)}
      data-vessel={vessel ? vessel.map((n) => n.toFixed(1)).join(',') : 'none'}
      data-layers={layers.join(' ')}
      data-camera-stage={camera?.stage ?? 'none'}
      data-camera-key={camera?.key ?? 'none'}
      data-camera-frames={
        camera ? camera.moves.map((move) => move.frame).join('>') : 'none'
      }
      data-camera-extent={
        camera
          ? camera.moves[camera.moves.length - 1].extent
              .map((n) => Math.round(n))
              .join(',')
          : 'none'
      }
    >
      <button type="button" data-testid="fake-map-click" onClick={() => onPlace([1, 2])}>
        place
      </button>
    </div>
  ),
}))

const client = vi.hoisted(() => ({
  getHealth: vi.fn(),
  getDefaults: vi.fn(),
  compareMission: vi.fn(),
  getIcebergExposure: vi.fn(),
  getIcebergForecast: vi.fn(),
  simulateIceberg: vi.fn(),
  getHistoricalEvaluation: vi.fn(),
}))

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>(
    '../api/client',
  )
  return { ...actual, ...client }
})

//  the exposure layer metadata is the backend's own, recorded from
//  GET /api/health: the legend's priced-horizon wording is built from it
const HEALTH = {
  status: 'ok',
  prototype: true,
  grid: missionResponse().grid,
  iceberg_exposure_weight: 5,
  requires_historical_demo_override: true,
  exposure_layers: [
    {
      label: '06h',
      bucket: 0,
      horizon_hours: 6.0,
      radius_km: 9.05053,
      iceberg_count: 11,
      extrapolated_uncertainty: false,
      non_zero_cells: 76,
    },
    {
      label: '12h',
      bucket: 1,
      horizon_hours: 12.0,
      radius_km: 12.799382,
      iceberg_count: 11,
      extrapolated_uncertainty: false,
      non_zero_cells: 147,
    },
  ],
  //  the POLARIS selection health reports, recorded from the real endpoint
  polaris: {
    selection: {
      ice_class: 'PC6',
      riv_table_requested: '1.3',
      sidecar_chart: 'ANTARC20201224',
      riv_table_source:
        'IMO MSC.1/Circ.1519, Appendix, Table 1.3 (Risk Index Values)',
    },
  },
}
/** the bucket the app draws, as the backend describes it */
const PRICED = HEALTH.exposure_layers.find((layer) => layer.bucket === 1)!

const FORECAST = forecastResponse()

/** The record the panel is actually showing, read from its own selector, so
 *  these tests check what is rendered rather than restate how the default is
 *  chosen -- that is subject.test.ts's job. */
const shownSubject = () => {
  const select = screen.getByLabelText(/^iceberg$/i) as HTMLSelectElement
  const record = FORECAST.icebergs.find(
    (berg) => berg.iceberg_id === select.value,
  )
  expect(record, `no fixture record for ${select.value}`).toBeDefined()
  return record!
}

const at = (hours: number) => {
  const record = shownSubject()
  const state =
    hours === 0
      ? record.observed!
      : record.predicted.find((entry) => entry.hours === hours)!
  expect(state).toBeDefined()
  return state
}

/** The panel's own display form, built from the payload rather than typed in. */
const position = (hours: number) => {
  const state = at(hours)
  const ns = state.latitude < 0 ? 'S' : 'N'
  const ew = state.longitude < 0 ? 'W' : 'E'
  return (
    `${Math.abs(state.latitude).toFixed(2)}° ${ns} ` +
    `${Math.abs(state.longitude).toFixed(2)}° ${ew}`
  )
}

const flat = (element: HTMLElement) =>
  (element.textContent ?? '').replace(/\s+/g, ' ')

beforeEach(() => {
  vi.clearAllMocks()
  client.getHealth.mockResolvedValue(HEALTH)
  client.getDefaults.mockResolvedValue(missionDefaults())
  client.getHistoricalEvaluation.mockResolvedValue(historicalEvaluation())
  client.getIcebergExposure.mockResolvedValue({
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [-809375, -1490625] },
        properties: { exposure: 0.5 },
      },
    ],
  })
  client.getIcebergForecast.mockResolvedValue(FORECAST)
})

/** observe → forecast */
async function toForecast() {
  const user = userEvent.setup()
  render(<App />)
  const start = await screen.findByTestId('start-forecast')
  await waitFor(() => expect(start).toBeEnabled())
  await user.click(start)
  await screen.findByTestId('forecast-timeline')
  return user
}

/** press the dock's action, enabling the historical demonstration first when
 *  the gate is up. This is the operator's own path from FORECAST to PLAN. */
async function routeFromForecast(user: ReturnType<typeof userEvent.setup>) {
  const action = screen.getByTestId('use-forecast')
  if (action.getAttribute('data-action') === 'enable_override') {
    //  the operator's own act: the demo's chart and environment are from
    //  different dates, so routing is gated until this is pressed
    await user.click(action)
    await waitFor(() =>
      expect(screen.getByTestId('use-forecast')).toHaveAttribute(
        'data-action',
        'route',
      ),
    )
  }
  await user.click(screen.getByTestId('use-forecast'))
}

/** observe → forecast → plan */
async function toPlan() {
  const user = await toForecast()
  await routeFromForecast(user)
  return user
}

/** plan → respond, and arm the simulation there */
async function armSimulation(user: ReturnType<typeof userEvent.setup>) {
  if (!screen.queryByTestId('respond-dock')) {
    await user.click(screen.getByTestId('go-respond'))
    await screen.findByTestId('respond-dock')
  }
  await user.click(screen.getByTestId('place-manually'))
}

const scrub = (hours: number) =>
  fireEvent.change(screen.getByTestId('forecast-scrubber'), {
    target: { value: String(hours) },
  })

// ───────────────────────────────────────────────────────────────── observe
describe('observe', () => {
  it('opens on a clean screen: map, mission summary, one action', async () => {
    render(<App />)
    expect(await screen.findByTestId('observe-dock')).toHaveTextContent(
      /start forecast/i,
    )
    expect(screen.getByTestId('mission-drawer')).toBeInTheDocument()
    expect(screen.getByTestId('mission-map')).toBeInTheDocument()

    // nothing from the later stages is on screen yet
    expect(screen.queryByTestId('forecast-timeline')).not.toBeInTheDocument()
    expect(screen.queryByTestId('forecast-panel')).not.toBeInTheDocument()
    expect(screen.queryByTestId('plan-dock')).not.toBeInTheDocument()
    expect(screen.queryByTestId('route-card-fastest')).not.toBeInTheDocument()
    expect(screen.queryByTestId('selected-route-bar')).not.toBeInTheDocument()
    expect(screen.queryByTestId('why-panel')).not.toBeInTheDocument()
  })

  it('draws the observation layers and no route', async () => {
    render(<App />)
    const map = await screen.findByTestId('mission-map')
    await waitFor(() =>
      expect(map.getAttribute('data-layers')).toContain('observed_icebergs'),
    )
    const drawn = (map.getAttribute('data-layers') ?? '').split(' ')
    expect(drawn).toEqual(
      expect.arrayContaining(['sic', 'no_coverage', 'land', 'observed_icebergs']),
    )
    //  nothing from the later stages, however much of it the app already holds
    for (const absent of [
      'predicted_icebergs',
      'forecast_envelope',
      'forecast_track',
      'iceberg_exposure',
      'selected_route',
      'alternate_routes',
      'simulated_iceberg',
    ]) {
      expect(drawn).not.toContain(absent)
    }
  })

  it('keeps mission configuration folded away on the validated demo inputs', async () => {
    const user = userEvent.setup()
    render(<App />)
    const drawer = await screen.findByTestId('mission-drawer')
    expect(within(drawer).queryByLabelText(/departure time/i)).not.toBeInTheDocument()

    await user.click(within(drawer).getByRole('button', { name: /mission/i }))
    expect(screen.getByLabelText(/departure time/i)).toHaveValue('2025-01-08T00:00')
    expect(screen.getByLabelText(/polaris class/i)).toHaveValue('PC6')
    expect(screen.getByLabelText(/riv table/i)).toHaveValue('1.3')
    expect(screen.getByLabelText(/vessel speed/i)).toHaveValue(5)
    expect(
      screen.getByLabelText(/historical demonstration mode/i),
    ).not.toBeChecked()
  })
})

// ──────────────────────────────────────────────────────────────── forecast
describe('forecast', () => {
  it('renders the timeline at the horizons the backend produced', async () => {
    await toForecast()
    const timeline = screen.getByTestId('forecast-timeline')
    expect(timeline).toHaveTextContent('Now')
    for (const horizon of FORECAST.horizons) {
      expect(timeline).toHaveTextContent(`+${horizon.hours}h`)
    }
    // the map now carries every iceberg the forecast returned
    expect(screen.getByTestId('mission-map')).toHaveAttribute(
      'data-forecast-bergs',
      String(FORECAST.icebergs.length),
    )
    expect(screen.queryByTestId('route-card-fastest')).not.toBeInTheDocument()
  })

  it('labels the starting frame as the observation, not a prediction', async () => {
    await toForecast()
    expect(screen.getByTestId('forecast-state-label')).toHaveTextContent(
      FORECAST.observed_label,
    )
    expect(screen.getByTestId('forecast-clock')).toHaveTextContent('Observed')
    expect(screen.queryByTestId('extrapolated-flag')).not.toBeInTheDocument()
    expect(flat(screen.getByTestId('forecast-panel'))).toContain(position(0))
  })

  it.each([6, 12, 24, 48])(
    'shows the backend position and radius at +%sh',
    async (hours) => {
      await toForecast()
      scrub(hours)

      const panel = await screen.findByTestId('forecast-panel')
      const state = at(hours)
      expect(screen.getByTestId('forecast-state-label')).toHaveTextContent(
        state.label,
      )
      expect(state.label).toBe(`Predicted +${hours}h`)
      expect(flat(panel)).toContain(position(hours))
      expect(panel).toHaveTextContent(`${state.radius_km.toFixed(2)} km`)
      // and the map is asked to draw that same clock position
      expect(screen.getByTestId('mission-map')).toHaveAttribute(
        'data-forecast-hours',
        hours.toFixed(4),
      )
    },
  )

  it('marks 48 h extrapolated and 24 h not', async () => {
    await toForecast()
    scrub(24)
    expect(screen.queryByTestId('extrapolated-flag')).not.toBeInTheDocument()
    scrub(48)
    expect(await screen.findByTestId('extrapolated-flag')).toHaveTextContent(
      /extrapolated/i,
    )
    expect(at(48).extrapolated_uncertainty).toBe(true)
    expect(at(24).extrapolated_uncertainty).toBe(false)
  })

  it('names a smoothed frame between two horizons as interpolated', async () => {
    await toForecast()
    scrub(9)
    expect(screen.getByTestId('forecast-state-label')).toHaveTextContent(
      /interpolated for display/i,
    )
    expect(screen.getByTestId('forecast-panel')).toHaveTextContent(
      /the model produces states only at its horizons/i,
    )
  })

  it('opens on an iceberg relevant to this mission, not the first listed', async () => {
    await toForecast()
    const shown = shownSubject()

    //  the API lists them in its own id order
    expect(FORECAST.icebergs.length).toBeGreaterThan(2)
    expect(shown.iceberg_id).not.toBe(FORECAST.icebergs[0].iceberg_id)

    //  and the one on screen really is the nearest the mission, measured here
    //  on the sphere rather than the way the app measures it
    const EARTH_M = 6_371_008.8
    const rad = (deg: number) => (deg * Math.PI) / 180
    const departure = cellLonLat(
      missionResponse().grid,
      missionResponse().request.start[0],
      missionResponse().request.start[1],
    )!
    const away = (berg: (typeof FORECAST.icebergs)[number]) => {
      const { longitude, latitude } = berg.observed!
      const dLat = rad(latitude - departure[1])
      const dLon = rad(longitude - departure[0])
      const a =
        Math.sin(dLat / 2) ** 2 +
        Math.cos(rad(departure[1])) *
          Math.cos(rad(latitude)) *
          Math.sin(dLon / 2) ** 2
      return 2 * EARTH_M * Math.asin(Math.sqrt(a))
    }
    const nearest = [...FORECAST.icebergs].sort((a, b) => away(a) - away(b))[0]
    expect(shown.iceberg_id).toBe(nearest.iceberg_id)
    expect(away(FORECAST.icebergs[0])).toBeGreaterThan(away(nearest) * 5)
  })

  it('lets an explicit pick override the default, and keeps it', async () => {
    const user = await toForecast()
    const before = shownSubject().iceberg_id
    const other = FORECAST.icebergs.find(
      (berg) => berg.iceberg_id !== before,
    )!

    await user.selectOptions(
      screen.getByLabelText(/^iceberg$/i),
      other.iceberg_id,
    )
    await waitFor(() => expect(shownSubject().iceberg_id).toBe(other.iceberg_id))

    //  the panel now reads that iceberg's own recorded position
    const observed = other.observed!
    expect(flat(screen.getByTestId('forecast-panel'))).toContain(
      `${Math.abs(observed.latitude).toFixed(2)}° ${
        observed.latitude < 0 ? 'S' : 'N'
      } ${Math.abs(observed.longitude).toFixed(2)}° ${
        observed.longitude < 0 ? 'W' : 'E'
      }`,
    )
    //  and it survives the clock moving
    scrub(24)
    expect(shownSubject().iceberg_id).toBe(other.iceberg_id)
  })

  it('reports the shipped drift model rather than claiming windage', async () => {
    await toForecast()
    expect(screen.getByTestId('forecast-panel')).toHaveTextContent(
      FORECAST.model.summary,
    )
    expect(FORECAST.model.windage_is_active).toBe(false)
  })

  it('advances the clock while playing and holds it when paused', async () => {
    const user = await toForecast()
    const clock = screen.getByTestId('forecast-clock')
    expect(clock).toHaveTextContent('Observed')

    await user.click(screen.getByTestId('forecast-play'))
    await waitFor(
      () =>
        expect(
          Number(
            screen.getByTestId('mission-map').getAttribute('data-forecast-hours'),
          ),
        ).toBeGreaterThan(0.5),
      { timeout: 3000 },
    )
    expect(screen.getByTestId('forecast-clock')).not.toHaveTextContent('Observed')

    await user.click(screen.getByTestId('forecast-play'))
    const held = screen
      .getByTestId('mission-map')
      .getAttribute('data-forecast-hours')
    await new Promise((resolve) => setTimeout(resolve, 250))
    expect(
      screen.getByTestId('mission-map').getAttribute('data-forecast-hours'),
    ).toBe(held)
  })
})

// ──────────────────────────────────────────────────────────────────── plan
describe('plan', () => {
  it('reveals the three options only once the forecast is used for routing', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    await toPlan()

    for (const name of ['fastest', 'risk_oriented', 'shortest_distance']) {
      expect(await screen.findByTestId(`route-card-${name}`)).toBeInTheDocument()
    }
    expect(screen.queryByTestId('forecast-timeline')).not.toBeInTheDocument()
    expect(screen.getByTestId('mission-map')).toHaveAttribute('data-routes', 'yes')
  })

  it("puts the API's own key metrics on each option", async () => {
    const payload = missionResponse(true)
    client.compareMission.mockResolvedValue(payload)
    await toPlan()

    for (const name of ['fastest', 'risk_oriented', 'shortest_distance'] as const) {
      //  the objective's own control, and the card it sits in: objectives
      //  that returned the same path carry one figure line between them
      const chip = await screen.findByTestId(`route-card-${name}`)
      const card = chip.closest('.option') as HTMLElement
      const profile = payload.comparison.profiles[name]
      expect(card).toHaveTextContent(
        `${profile.distance_km!.toLocaleString('en-GB', {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })} km`,
      )
      // only the key metrics until the option is opened
      expect(
        screen.queryByTestId(`route-detail-${name}`),
      ).not.toBeInTheDocument()
    }
  })

  it('expands one option into its detail', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toPlan()

    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    const detail = await screen.findByTestId('route-detail-risk_oriented')
    expect(detail).toHaveTextContent(/minimises the configured navigation cost/i)
    expect(detail).toHaveTextContent('11,325,825')
  })

  it('updates the selected-route bar and the map when an option is chosen', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toPlan()

    const bar = await screen.findByTestId('selected-route-bar')
    expect(bar).toHaveTextContent(/Selected route/i)
    expect(bar).toHaveTextContent('178.03 km')
    expect(screen.getByTestId('mission-map')).toHaveAttribute(
      'data-selected',
      'fastest',
    )

    await user.click(screen.getByTestId('route-card-risk_oriented'))
    await waitFor(() =>
      expect(screen.getByTestId('selected-route-bar')).toHaveTextContent(
        '186.87 km',
      ),
    )
    expect(screen.getByTestId('mission-map')).toHaveAttribute(
      'data-selected',
      'risk_oriented',
    )
  })

  it('keeps the reasoning behind a drawer that opens and closes', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toPlan()
    expect(screen.queryByTestId('why-panel')).not.toBeInTheDocument()

    await user.click(screen.getByTestId('route-card-risk_oriented'))
    await user.click(screen.getByTestId('why-toggle'))

    const why = await screen.findByTestId('why-panel')
    expect(why).toHaveTextContent(/Lower configured POLARIS contribution/i)
    expect(why).toHaveTextContent(/11,325,825/)
    expect(why).toHaveTextContent(/15,227,476/)
    expect(why).toHaveTextContent(/Fewer special-consideration cells/i)
    expect(why).not.toHaveTextContent(/safest|best route|recommended/i)

    await user.click(screen.getByTestId('why-toggle'))
    await waitFor(() =>
      expect(screen.queryByTestId('why-panel')).not.toBeInTheDocument(),
    )
  })

  it('keeps the provenance chip beside the routes', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    await toPlan()
    await screen.findByTestId('route-card-fastest')
    expect(screen.getByTestId('provenance-chip')).toHaveAttribute(
      'data-status',
      'historical_demonstration',
    )
  })

  it('says the override is off for a mission that ran without it', async () => {
    //  the dates still differ -- that is a property of the DATA, not of the
    //  switch -- so the chip still reports a historical demonstration, and the
    //  detail says the override was not in effect for this mission
    client.compareMission.mockResolvedValue(missionResponse(false))
    const user = await toPlan()
    await screen.findByTestId('route-card-fastest')
    expect(screen.getByTestId('provenance-status')).toHaveTextContent(
      /historical polaris demonstration/i,
    )
    await user.click(screen.getByTestId('provenance-chip'))
    expect(await screen.findByTestId('provenance-detail')).toHaveTextContent(
      /off for this mission/i,
    )
  })
})

// ────────────────────────────────────────── objectives that share a path
describe('route options that returned the same geometry', () => {
  //  the recorded payload is the real demo's shape: fastest and
  //  shortest-distance came back with one cell sequence, risk-oriented with
  //  another. Nothing here asserts that pairing -- it reads the payload.
  const payload = missionResponse(true)
  const shared = ['fastest', 'shortest_distance'] as const
  const groupCard = () => screen.getByTestId('route-option-fastest+shortest_distance')

  it('offers one card for the shared geometry and one for the other', async () => {
    client.compareMission.mockResolvedValue(payload)
    await toPlan()
    await screen.findByTestId('route-card-fastest')

    //  the payload's own geometry decides this
    expect(payload.comparison.profiles.fastest.path).toEqual(
      payload.comparison.profiles.shortest_distance.path,
    )
    expect(payload.comparison.profiles.risk_oriented.path).not.toEqual(
      payload.comparison.profiles.fastest.path,
    )

    //  four objectives, three distinct geometries: the shared pair, plus
    //  fuel_efficient and risk_oriented, each with its own path
    const cards = screen.getByTestId('plan-dock').querySelectorAll('.option')
    expect(cards).toHaveLength(3)
    expect(groupCard()).toBeInTheDocument()
    expect(screen.getByTestId('route-option-fuel_efficient')).toBeInTheDocument()
    expect(screen.getByTestId('route-option-risk_oriented')).toBeInTheDocument()
  })

  it('names both objectives and says only what this model supports', async () => {
    client.compareMission.mockResolvedValue(payload)
    await toPlan()
    const group = await screen.findByTestId('route-group-fastest+shortest_distance')

    expect(flat(group)).toContain('Fastest · Shortest distance')
    expect(flat(group)).toContain('Same path under the current constant vessel speed')
    //  the mission's own speed, from the payload the backend echoed back
    expect(flat(group)).toContain(`(${payload.request.vessel_speed_mps} m/s)`)
    //  the figures are the backend's, once, because both objectives report them
    expect(flat(group)).toContain('178.03 km')
    //  `fuel-efficient` is a real objective name now, so the bare word is not
    //  banned; every UNSUPPORTED fuel claim still is, and so is any ranking.
    for (const banned of [
      /duplicate/i,
      /measured fuel/i,
      /actual fuel/i,
      /fuel saving/i,
      /litres/i,
      /tonnes/i,
      /best route/i,
      /optimal/i,
    ]) {
      expect(flat(screen.getByTestId('plan-dock'))).not.toMatch(banned)
    }
  })

  it('keeps risk-oriented separate with its own figures', async () => {
    client.compareMission.mockResolvedValue(payload)
    await toPlan()
    await screen.findByTestId('route-card-risk_oriented')

    const card = screen.getByTestId('route-option-risk_oriented')
    expect(card).toHaveTextContent('186.87 km')
    expect(
      screen.queryByTestId('route-group-risk_oriented'),
    ).not.toBeInTheDocument()
  })

  it('leaves both grouped objectives separately selectable', async () => {
    client.compareMission.mockResolvedValue(payload)
    const user = await toPlan()

    for (const name of shared) {
      await user.click(await screen.findByTestId(`route-card-${name}`))
      await waitFor(() =>
        expect(screen.getByTestId(`route-card-${name}`)).toHaveAttribute(
          'aria-pressed',
          'true',
        ),
      )
      //  and only that one is pressed
      const other = shared.find((candidate) => candidate !== name)!
      expect(screen.getByTestId(`route-card-${other}`)).toHaveAttribute(
        'aria-pressed',
        'false',
      )
      expect(screen.getByTestId('mission-map')).toHaveAttribute(
        'data-selected',
        name,
      )
    }
  })

  it('keeps each objective its own: label, detail and metrics', async () => {
    client.compareMission.mockResolvedValue(payload)
    const user = await toPlan()

    await user.click(await screen.findByTestId('route-card-shortest_distance'))
    expect(await screen.findByTestId('selected-route-bar')).toHaveTextContent(
      /shortest distance/i,
    )
    const detail = await screen.findByTestId('route-detail-shortest_distance')
    expect(detail).toHaveTextContent(/minimises travelled distance/i)
    //  the longer explanation is the backend's own sentence
    expect(flat(detail)).toContain(
      payload.comparison.provenance.why_they_can_coincide!,
    )
    expect(
      screen.queryByTestId('route-detail-fastest'),
    ).not.toBeInTheDocument()

    await user.click(screen.getByTestId('route-card-fastest'))
    expect(await screen.findByTestId('selected-route-bar')).toHaveTextContent(
      /fastest/i,
    )
    await screen.findByTestId('route-detail-fastest')
  })

  it('keeps the selected-route bar on the selected profile’s own metrics', async () => {
    client.compareMission.mockResolvedValue(payload)
    const user = await toPlan()

    const bar = await screen.findByTestId('selected-route-bar')
    expect(bar).toHaveTextContent(
      `${payload.comparison.profiles.fastest.distance_km} km`,
    )
    await user.click(screen.getByTestId('route-card-risk_oriented'))
    await waitFor(() =>
      expect(screen.getByTestId('selected-route-bar')).toHaveTextContent(
        `${payload.comparison.profiles.risk_oriented.distance_km} km`,
      ),
    )
  })

  it('asks the map for one line per geometry, under the selected objective', async () => {
    client.compareMission.mockResolvedValue(payload)
    const user = await toPlan()
    const map = await screen.findByTestId('mission-map')

    await waitFor(() =>
      expect(map).toHaveAttribute(
        'data-drawn',
        'fastest fuel_efficient risk_oriented',
      ),
    )
    await user.click(screen.getByTestId('route-card-shortest_distance'))
    await waitFor(() =>
      expect(map).toHaveAttribute(
        'data-drawn',
        'shortest_distance fuel_efficient risk_oriented',
      ),
    )
  })

  it('injects on the selected objective’s own route, whichever of the pair it is', async () => {
    client.compareMission.mockResolvedValue(payload)
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-shortest_distance'))
    await user.click(screen.getByTestId('go-respond'))
    await user.click(await screen.findByTestId('inject-on-route'))

    await waitFor(() => expect(client.simulateIceberg).toHaveBeenCalledTimes(1))
    const [, iceberg, profile] = client.simulateIceberg.mock.calls[0]
    expect(profile).toBe('shortest_distance')
    //  a cell of the path THAT profile returned
    const centres = payload.comparison.profiles.shortest_distance.path.map(
      (cellRef) => cellCentreXY(payload.grid, cellRef[0], cellRef[1]),
    )
    expect(centres).toContainEqual([iceberg.x, iceberg.y])
  })

  it('does not group four distinct geometries', async () => {
    //  the same recorded payload with the pair pulled apart: the grouping
    //  follows the geometry, not the profile names
    const distinct = missionResponse(true)
    distinct.comparison.profiles.shortest_distance.path = [
      [948, 503],
      [936, 502],
      [922, 497],
    ]
    client.compareMission.mockResolvedValue(distinct)
    await toPlan()
    await screen.findByTestId('route-card-fastest')

    expect(
      screen.getByTestId('plan-dock').querySelectorAll('.option'),
    ).toHaveLength(4)
    expect(
      screen.queryByTestId('route-group-fastest+shortest_distance'),
    ).not.toBeInTheDocument()
    expect(screen.getByTestId('mission-map')).toHaveAttribute(
      'data-drawn',
      'fastest fuel_efficient risk_oriented shortest_distance',
    )
  })
})

// ───────────────────────────────────────────────────────── vessel transit
describe('vessel transit', () => {
  it('walks the cells the API returned, at their own arrival times', async () => {
    const payload = missionResponse(true)
    client.compareMission.mockResolvedValue(payload)
    const user = await toPlan()

    const cells = payload.comparison.profiles.fastest.cells!
    await user.click(await screen.findByRole('button', { name: /play vessel transit/i }))

    const readout = await screen.findByTestId('transit-readout', {}, { timeout: 3000 })
    // the arrival clock, bucket and exposure are the route cell's own values
    expect(readout).toHaveTextContent(
      cells[0].arrival_datetime.slice(11, 16),
    )
    expect(readout).toHaveTextContent(String(cells[0].bucket))
    expect(readout).toHaveTextContent(cells[0].iceberg_exposure!.toFixed(3))

    await waitFor(() =>
      expect(screen.getByTestId('mission-map')).not.toHaveAttribute(
        'data-vessel',
        'none',
      ),
    )
  })
})

// ─────────────────────────────────────────────────────── simulated iceberg
describe('simulated iceberg', () => {
  it('injects one and shows the replanned route', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))

    await armSimulation(user)
    expect(screen.getByTestId('mission-map')).toHaveAttribute('data-placing', 'true')
    await user.click(screen.getByTestId('fake-map-click'))

    const bar = await screen.findByTestId('selected-route-bar')
    expect(bar).toHaveTextContent(/Replanned route/i)
    expect(bar).toHaveTextContent('201.44 km')
    expect(screen.getByTestId('mission-map')).toHaveAttribute(
      'data-superseded',
      'yes',
    )
    expect(client.simulateIceberg).toHaveBeenCalledWith(
      expect.objectContaining({ polaris_ice_class: 'PC6' }),
      { x: 1, y: 2 },
      'risk_oriented',
    )
  })

  it('explains the change with the real before and after numbers', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    await armSimulation(user)
    await user.click(screen.getByTestId('fake-map-click'))
    //  the evidence opens with the result; there is nothing to hunt for
    const impact = await screen.findByTestId('route-impact')
    const payload = simulationResponse()
    const change = payload.change.profiles.risk_oriented

    expect(impact).toHaveTextContent(payload.simulation!.label)
    expect(screen.getByTestId('impact-affected')).toHaveTextContent(
      String(change.cells_with_simulated_exposure),
    )
    //  the before and after the backend returned, in its own table row
    const row = within(screen.getByTestId('impact-table'))
      .getByText('Iceberg exposure contribution')
      .closest('tr') as HTMLTableRowElement
    expect(row.textContent).toContain('26,476')
    expect(row.textContent).toContain('31,902')
    expect(impact).toHaveTextContent('+14.57 km')
    //  and the backend's own disclaimer, verbatim
    expect(screen.getByTestId('impact-claims').textContent).toBe(
      payload.change.claims.note,
    )
    for (const banned of [/collision/i, /guarantee/i, /\bsafe\b/i, /optimal/i]) {
      expect(impact.textContent ?? '').not.toMatch(banned)
    }
  })

  it('restores the original route when the simulation is cleared', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    const before = screen.getByTestId('route-card-risk_oriented').textContent

    await armSimulation(user)
    await user.click(screen.getByTestId('fake-map-click'))
    await waitFor(() =>
      expect(screen.getByTestId('selected-route-bar')).toHaveTextContent(
        /replanned route/i,
      ),
    )

    await user.click(screen.getByTestId('clear-simulation'))
    await waitFor(() =>
      expect(screen.getByTestId('selected-route-bar')).toHaveTextContent(
        /Current route/i,
      ),
    )

    //  back in PLAN the original option is exactly as it was
    await user.click(screen.getByTestId('back-to-plan'))
    expect(
      (await screen.findByTestId('route-card-risk_oriented')).textContent,
    ).toBe(before)
    expect(screen.getByTestId('mission-map')).toHaveAttribute(
      'data-superseded',
      'no',
    )
    // clearing is local: no extra backend call was needed
    expect(client.compareMission).toHaveBeenCalledTimes(1)
  })
})

// ───────────────────────────────────────────────────────────────── refusals
describe('refusals', () => {
  it('renders an API error instead of routes', async () => {
    client.compareMission.mockRejectedValue(
      new ApiError('boom', 0, 'api_unreachable'),
    )
    await toPlan()

    expect(await screen.findByTestId('status-banner')).toHaveTextContent(/boom/i)
    expect(screen.queryByTestId('route-card-fastest')).not.toBeInTheDocument()
  })

  it('reports a backend refusal without inventing a route', async () => {
    const refused = missionResponse(false)
    refused.status = 'historical_date_mismatch'
    refused.message = 'the cost provider was built with the historical-demo override'
    for (const name of ['fastest', 'risk_oriented', 'shortest_distance'] as const) {
      const profile = refused.comparison.profiles[name]
      profile.success = false
      profile.outcome = 'historical_date_mismatch'
      profile.path = []
      profile.distance_km = null
      profile.configured_cost = null
    }
    client.compareMission.mockResolvedValue(refused)
    await toPlan()

    //  the backend refused the pairing even though the override was set. That
    //  is reported as the same operational condition, not as a server error.
    const gate = await screen.findByTestId('plan-gate')
    expect(gate).toHaveTextContent(/routing paused/i)
    expect(screen.getByTestId('gate-note')).toHaveTextContent(refused.message)
    expect(screen.queryByTestId('route-card-fastest')).not.toBeInTheDocument()
    expect(screen.queryByTestId('selected-route-bar')).not.toBeInTheDocument()
    expect(gate.textContent ?? '').not.toMatch(/invalid|broken|bad data|error/i)
  })

  it('shows the API-unavailable state when health fails', async () => {
    client.getHealth.mockRejectedValue(
      new ApiError('The mission API is not reachable.', 0, 'api_unreachable'),
    )
    render(<App />)
    expect(await screen.findByTestId('status-banner')).toHaveTextContent(
      /not reachable/i,
    )
    expect(screen.getByTestId('start-forecast')).toBeDisabled()
  })
})

// ──────────────────────────────────────────────────────────────── camera
describe('camera', () => {
  const GRID = missionResponse().grid
  const at = (cell: readonly number[]): XY =>
    cellCentreXY(GRID, cell[0], cell[1]) as XY
  const map = () => screen.getByTestId('mission-map')
  const frames = () => map().getAttribute('data-camera-frames')
  const extent = (): Extent =>
    (map().getAttribute('data-camera-extent') ?? '')
      .split(',')
      .map(Number) as Extent
  const holds = (frame: Extent, [x, y]: XY) =>
    x >= frame[0] && x <= frame[2] && y >= frame[1] && y <= frame[3]

  it('OBSERVE opens on the continent and asks for the mission sector', async () => {
    render(<App />)
    await screen.findByTestId('observe-dock')
    //  once the grid is known, the sector is derived from the request's cells
    await waitFor(() => expect(frames()).toBe('continent>sector'))
    expect(map()).toHaveAttribute('data-camera-stage', 'observe')

    const sector = extent()
    expect(holds(sector, at(missionResponse().request.start))).toBe(true)
    expect(holds(sector, at(missionResponse().request.goal))).toBe(true)
    expect(spanOf(sector)).toBeLessThan(spanOf(GRID_EXTENT) / 5)
  })

  it('FORECAST fits the subject and never the whole grid', async () => {
    await toForecast()
    await waitFor(() => expect(frames()).toBe('subject'))
    expect(map()).toHaveAttribute('data-camera-stage', 'forecast')

    const frame = extent()
    expect(frame).not.toEqual(GRID_EXTENT)
    expect(spanOf(frame)).toBeLessThan(spanOf(GRID_EXTENT) * 0.02)
  })

  it('FORECAST now frames an iceberg beside the mission, not one a continent away', async () => {
    await toForecast()
    await waitFor(() => expect(frames()).toBe('subject'))

    const frame = extent()
    const centre: XY = [(frame[0] + frame[2]) / 2, (frame[1] + frame[3]) / 2]
    const corridorMid: XY = [
      (at(missionResponse().request.start)[0] +
        at(missionResponse().request.goal)[0]) /
        2,
      (at(missionResponse().request.start)[1] +
        at(missionResponse().request.goal)[1]) /
        2,
    ]
    const away = (point: XY) =>
      Math.hypot(point[0] - corridorMid[0], point[1] - corridorMid[1])

    //  the frame sits beside the mission
    expect(away(centre)).toBeLessThan(500_000)

    //  where opening on the first iceberg the API listed would have put it
    const first = FORECAST.icebergs[0].observed!
    expect(away(projectLonLat(first.longitude, first.latitude))).toBeGreaterThan(
      2_000_000,
    )
  })

  it('FORECAST playback does not keep resetting the camera', async () => {
    await toForecast()
    await waitFor(() => expect(frames()).toBe('subject'))
    const before = map().getAttribute('data-camera-key')

    for (const hours of [3, 6, 17.5, 24, 48]) scrub(hours)
    expect(map()).toHaveAttribute('data-forecast-hours', (48).toFixed(4))
    //  the clock moved right through the forecast; the camera did not
    expect(map().getAttribute('data-camera-key')).toBe(before)
  })

  it('PLAN fits the geometry the search returned', async () => {
    const payload = missionResponse(true)
    client.compareMission.mockResolvedValue(payload)
    await toPlan()
    await waitFor(() => expect(frames()).toBe('route'))
    expect(map()).toHaveAttribute('data-camera-stage', 'plan')

    const frame = extent()
    for (const cell of payload.comparison.profiles.risk_oriented.path) {
      expect(holds(frame, at(cell))).toBe(true)
    }
    expect(spanOf(frame)).toBeLessThan(spanOf(GRID_EXTENT) * 0.05)
  })

  it('PLAN does not move the camera when another option is selected', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toPlan()
    await waitFor(() => expect(frames()).toBe('route'))
    const before = map().getAttribute('data-camera-key')

    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    await waitFor(() =>
      expect(map()).toHaveAttribute('data-selected', 'risk_oriented'),
    )
    expect(map().getAttribute('data-camera-key')).toBe(before)
  })

  it('RESPOND fits the route with the injected iceberg', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    await armSimulation(user)
    await user.click(screen.getByTestId('fake-map-click'))

    await waitFor(() =>
      expect(map()).toHaveAttribute('data-camera-stage', 'respond'),
    )
    const berg = simulationResponse().simulation!
    expect(frames()?.split('>').every((name) => name === 'route+hazard')).toBe(true)
    //  the last move eases to the encounter cell the backend reported
    expect(holds(extent(), at(berg.encounter_cell))).toBe(true)

    //  and clearing the simulation hands the camera back to PLAN
    await user.click(screen.getByTestId('clear-simulation'))
    await waitFor(() => expect(frames()).toBe('route'))
  })
})

// ──────────────────────────────────────────────────────────────── legend
describe('legend', () => {
  const mapLayers = () =>
    (screen.getByTestId('mission-map').getAttribute('data-layers') ?? '')
      .split(' ')
      .filter(Boolean)

  const legendLayers = () =>
    Array.from(
      screen.getByTestId('map-legend').querySelectorAll('[data-layer]'),
    ).map((row) => row.getAttribute('data-layer')!)

  const legendText = () =>
    (screen.getByTestId('map-legend').textContent ?? '').toLowerCase()

  /** the invariant this whole step exists for */
  const agrees = () => {
    expect(new Set(legendLayers())).toEqual(new Set(mapLayers()))
  }

  it('OBSERVE lists the environment and the observations only', async () => {
    render(<App />)
    await screen.findByTestId('observe-dock')
    await waitFor(() => expect(mapLayers()).toContain('observed_icebergs'))
    agrees()

    expect(legendLayers()).toEqual(
      expect.arrayContaining(['sic', 'land', 'no_coverage', 'observed_icebergs']),
    )
    for (const absent of [
      'iceberg_exposure',
      'forecast_envelope',
      'predicted_icebergs',
      'selected_route',
      'simulated_iceberg',
    ]) {
      expect(legendLayers()).not.toContain(absent)
    }
  })

  it('FORECAST distinguishes observed from predicted, and names the envelope', async () => {
    await toForecast()
    await waitFor(() => expect(mapLayers()).toContain('predicted_icebergs'))
    agrees()

    expect(legendLayers()).toEqual(
      expect.arrayContaining([
        'observed_icebergs',
        'predicted_icebergs',
        'forecast_track',
        'forecast_envelope',
      ]),
    )
    const legend = screen.getByTestId('map-legend')
    expect(legend).toHaveTextContent(/observed iceberg position/i)
    expect(legend).toHaveTextContent(/predicted iceberg position/i)
    expect(legend).toHaveTextContent(/modelled uncertainty radius/i)
    //  the two iceberg rows carry different marks
    const observed = legend.querySelector('[data-layer="observed_icebergs"] .legend-mark')
    const predicted = legend.querySelector('[data-layer="predicted_icebergs"] .legend-mark')
    expect(observed?.className).not.toBe(predicted?.className)

    //  no route or exposure is drawn here, so neither is described
    expect(legendLayers()).not.toContain('selected_route')
    expect(legendLayers()).not.toContain('iceberg_exposure')
  })

  it('PLAN names each route and states the horizon the route was priced at', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    await toPlan()
    await waitFor(() => expect(mapLayers()).toContain('selected_route'))
    agrees()

    const legend = screen.getByTestId('map-legend')
    //  one row per route, the selected one marked as such
    expect(legend).toHaveTextContent(/selected route — fastest/i)
    expect(legend).toHaveTextContent(/alternate route — fuel-efficient/i)
    expect(legend).toHaveTextContent(/alternate route — risk-oriented/i)
    expect(legend).toHaveTextContent(/alternate route — shortest distance/i)
    expect(
      legend.querySelectorAll('[data-layer="alternate_routes"]'),
    ).toHaveLength(3)
    //  the priced horizon comes from the backend's own layer metadata
    expect(legend).toHaveTextContent(
      new RegExp(`\\+${PRICED.horizon_hours}h`),
    )
    expect(legend).toHaveTextContent(
      new RegExp(`max over ${PRICED.iceberg_count} icebergs`, 'i'),
    )
    expect(legend).toHaveTextContent(/priced/i)

    //  the forecast products are not drawn in PLAN, so they are not listed
    expect(legendLayers()).not.toContain('forecast_envelope')
    expect(legendLayers()).not.toContain('forecast_track')
  })

  it('drops the exposure row when the field comes back with no cells', async () => {
    client.getIcebergExposure.mockResolvedValue({
      type: 'FeatureCollection',
      features: [],
    })
    client.compareMission.mockResolvedValue(missionResponse(true))
    await toPlan()
    await waitFor(() => expect(mapLayers()).toContain('selected_route'))
    agrees()
    expect(legendLayers()).not.toContain('iceberg_exposure')
    expect(legendText()).not.toContain('exposure')
  })

  it('RESPOND names the hazard without calling it safe or unsafe', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    await armSimulation(user)
    await user.click(screen.getByTestId('fake-map-click'))

    await waitFor(() => expect(mapLayers()).toContain('simulated_iceberg'))
    agrees()
    expect(legendLayers()).toEqual(
      expect.arrayContaining([
        'superseded_route',
        'simulated_iceberg',
        'simulated_envelope',
      ]),
    )
    const legend = screen.getByTestId('map-legend')
    expect(legend).toHaveTextContent(/simulated iceberg/i)
    expect(legend).toHaveTextContent(/route before the injection/i)

    //  clearing it takes those rows away again
    await user.click(screen.getByTestId('clear-simulation'))
    await waitFor(() => expect(legendLayers()).not.toContain('simulated_iceberg'))
    agrees()
  })

  it('never claims what it cannot claim, at any stage', async () => {
    const user = await toForecast()
    for (const banned of [
      'collision probability',
      'probability',
      'safety zone',
      'safe',
      'unsafe',
      'guarantee',
      'best route',
    ]) {
      expect(legendText(), `forecast legend says "${banned}"`).not.toContain(banned)
    }
    client.compareMission.mockResolvedValue(missionResponse(true))
    await routeFromForecast(user)
    await waitFor(() => expect(mapLayers()).toContain('selected_route'))
    for (const banned of ['collision probability', 'safe', 'unsafe', 'guarantee']) {
      expect(legendText(), `plan legend says "${banned}"`).not.toContain(banned)
    }
  })
})

// ─────────────────────────────────────────────────────── priced horizon
describe('priced horizon', () => {
  const clock = () =>
    Number(screen.getByTestId('mission-map').getAttribute('data-forecast-hours'))
  const mapLayers = () =>
    (screen.getByTestId('mission-map').getAttribute('data-layers') ?? '')
      .split(' ')
      .filter(Boolean)

  /** the exposure fields the backend says exist, and what the route used */
  const FIELDS = HEALTH.exposure_layers
  const LAST_PRICED = Math.max(...FIELDS.map((field) => field.horizon_hours))

  it('PLAN pins the iceberg display to the route’s own last priced field', async () => {
    const payload = missionResponse(true)
    client.compareMission.mockResolvedValue(payload)
    await toPlan()

    //  the route reported these buckets; the fields behind them are what
    //  priced it
    expect(payload.comparison.profiles.fastest.buckets_used).toEqual([0, 1])
    await waitFor(() => expect(clock()).toBe(LAST_PRICED))
    expect(clock()).toBe(12)
  })

  it('follows the route: a route that used only the first bucket pins earlier', async () => {
    //  the same recorded payload with the route reporting one bucket, which
    //  is what a voyage inside the first six hours returns
    const shorter = missionResponse(true)
    for (const name of PROFILE_ORDER) {
      shorter.comparison.profiles[name].buckets_used = [0]
    }
    client.compareMission.mockResolvedValue(shorter)
    await toPlan()

    const firstField = Math.min(...FIELDS.map((field) => field.horizon_hours))
    await waitFor(() => expect(clock()).toBe(firstField))
    expect(clock()).toBe(6)
    expect(screen.getByTestId('priced-horizon')).toHaveTextContent(
      /evaluated through \+6h/i,
    )
    //  the exposure overlay is the 12 h field; this route never entered that
    //  bucket, so it is not drawn beside it and not described
    expect(mapLayers()).not.toContain('iceberg_exposure')
    expect(screen.getByTestId('map-legend').textContent ?? '').not.toContain(
      'Modelled iceberg exposure',
    )
  })

  it('scrubbing to +24h or +48h cannot change what the route was priced against', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toForecast()

    for (const hours of [24, 48]) {
      scrub(hours)
      expect(clock()).toBe(hours) // the forecast stage follows the scrubber
    }

    await routeFromForecast(user)
    await waitFor(() => expect(mapLayers()).toContain('selected_route'))
    //  and PLAN ignores where the scrubber was left
    expect(clock()).toBe(LAST_PRICED)
    expect(clock()).not.toBe(48)
  })

  it('states which fields priced the route and which horizons did not', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    await toPlan()

    const line = await screen.findByTestId('priced-horizon')
    expect(line).toHaveTextContent(/modelled iceberg exposure evaluated through/i)
    expect(line).toHaveTextContent(/\+6h and \+12h/)
    //  the forecast runs further, and the line says those states priced nothing
    const furthest = Math.max(...FORECAST.horizons.map((h) => h.hours))
    expect(furthest).toBe(48)
    expect(line).toHaveTextContent(new RegExp(`\\+${furthest}h`))
    expect(line).toHaveTextContent(/priced nothing on this route/i)
    for (const banned of [/safe/i, /optimal/i, /collision/i, /guarantee/i]) {
      expect(line.textContent ?? '').not.toMatch(banned)
    }
  })

  it('the forecast timeline marks the horizons no route can be priced against', async () => {
    await toForecast()
    const note = await screen.findByTestId('timeline-priced-note')
    expect(note).toHaveTextContent(/\+6h and \+12h/)
    expect(note).toHaveTextContent(/forecast only/i)

    const timeline = screen.getByTestId('forecast-timeline')
    const unpriced = timeline.querySelectorAll('.stop.is-display-only')
    //  24 h and 48 h have no exposure field in the routing configuration
    expect(unpriced).toHaveLength(
      FORECAST.horizons.filter(
        (horizon) =>
          !FIELDS.some((field) => field.horizon_hours === horizon.hours),
      ).length,
    )
    expect(unpriced).toHaveLength(2)
  })

  it('RESPOND keeps the same temporal context', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    await armSimulation(user)
    await user.click(screen.getByTestId('fake-map-click'))

    await waitFor(() => expect(mapLayers()).toContain('simulated_iceberg'))
    expect(clock()).toBe(LAST_PRICED)
    expect(screen.getByTestId('priced-horizon')).toHaveTextContent(
      /evaluated through \+6h and \+12h/i,
    )
  })

  it('says so, and draws nothing, when the priced horizon is unknown', async () => {
    //  a backend that described no exposure field: the route's buckets cannot
    //  be mapped to a horizon, so no iceberg state is drawn beside it
    client.getHealth.mockResolvedValue({
      ...HEALTH,
      exposure_layers: undefined,
    })
    client.compareMission.mockResolvedValue(missionResponse(true))
    await toPlan()

    await waitFor(() => expect(mapLayers()).toContain('selected_route'))
    expect(mapLayers()).not.toContain('predicted_icebergs')
    expect(mapLayers()).not.toContain('observed_icebergs')

    const line = screen.getByTestId('priced-horizon')
    expect(line).toHaveTextContent(/priced horizon unavailable/i)
    expect(line).toHaveTextContent(/no iceberg state is shown/i)
  })

  it('carries the same statement into Why this route', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toPlan()
    await user.click(screen.getByTestId('why-toggle'))
    const why = await screen.findByTestId('why-priced-horizon')
    expect(why).toHaveTextContent(/evaluated through \+6h and \+12h/i)
  })
})

// ─────────────────────────────────────────────────────────── provenance
describe('provenance', () => {
  const GRID = missionResponse().grid
  const chip = () => screen.getByTestId('provenance-chip')

  it('is on screen before anything is asked of the backend', async () => {
    render(<App />)
    await waitFor(() =>
      expect(screen.getByTestId('provenance-environment')).toHaveTextContent(
        formatDate(GRID.environment_date)!,
      ),
    )
    expect(screen.getByTestId('provenance-chart')).toHaveTextContent(
      formatDate(GRID.usnic_chart_date)!,
    )
    expect(chip()).toHaveAttribute('data-status', 'historical_demonstration')
    //  and it took no routing to find that out
    expect(client.compareMission).not.toHaveBeenCalled()
    expect(client.simulateIceberg).not.toHaveBeenCalled()
  })

  it('reads both dates from the API rather than carrying them', async () => {
    //  the same recorded payload with the backend reporting other dates
    const moved = {
      ...HEALTH,
      grid: {
        ...GRID,
        environment_date: '2019-07-04',
        usnic_chart_date: '2011-11-11',
      },
    }
    client.getHealth.mockResolvedValue(moved)
    render(<App />)
    await waitFor(() =>
      expect(screen.getByTestId('provenance-environment')).toHaveTextContent(
        '04 JUL 2019',
      ),
    )
    expect(screen.getByTestId('provenance-chart')).toHaveTextContent(
      '11 NOV 2011',
    )
  })

  it('names the mismatch as a demonstration, not a failure', async () => {
    render(<App />)
    const status = await screen.findByTestId('provenance-status')
    expect(GRID.dates_aligned).toBe(false)
    expect(status).toHaveTextContent(/historical polaris demonstration/i)
    for (const banned of [/error/i, /failure/i, /invalid/i, /current polaris/i]) {
      expect(status.textContent ?? '').not.toMatch(banned)
    }
  })

  it('calls aligned dates contemporaneous', async () => {
    client.getHealth.mockResolvedValue({
      ...HEALTH,
      grid: { ...GRID, usnic_chart_date: GRID.environment_date, dates_aligned: true },
      requires_historical_demo_override: false,
    })
    render(<App />)
    await waitFor(() =>
      expect(chip()).toHaveAttribute('data-status', 'contemporaneous'),
    )
    expect(screen.getByTestId('provenance-status')).not.toHaveTextContent(
      /historical/i,
    )
  })

  it('stays put through OBSERVE, FORECAST, PLAN and RESPOND', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    render(<App />)
    await screen.findByTestId('provenance-chip')

    const user = userEvent.setup()
    await user.click(await screen.findByTestId('start-forecast'))
    await screen.findByTestId('forecast-timeline')
    expect(chip()).toBeInTheDocument()

    await routeFromForecast(user)
    await screen.findByTestId('route-card-fastest')
    expect(chip()).toBeInTheDocument()

    await armSimulation(user)
    await user.click(screen.getByTestId('fake-map-click'))
    await waitFor(() =>
      expect(screen.getByTestId('selected-route-bar')).toHaveTextContent(
        /replanned route/i,
      ),
    )
    expect(chip()).toBeInTheDocument()
    expect(chip()).toHaveAttribute('data-status', 'historical_demonstration')
  })

  it('opens a detail view built from the API metadata', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByTestId('provenance-chip'))

    const detail = await screen.findByTestId('provenance-detail')
    //  the chart the sidecar was built from, as health names it
    expect(detail).toHaveTextContent(
      String(HEALTH.polaris.selection.sidecar_chart),
    )
    expect(detail).toHaveTextContent(/IMO MSC.1/)
    //  forecast horizons available, against what routing prices
    for (const horizon of FORECAST.horizons) {
      expect(detail).toHaveTextContent(`+${horizon.hours}h`)
    }
    expect(detail).toHaveTextContent(/routing prices against/i)
    expect(detail).toHaveTextContent(FORECAST.model.summary)
    //  and it says the chart is not a description of the environment
    expect(detail).toHaveTextContent(/not a description of the environment/i)
  })

  it('says unknown when the backend reported no dates', async () => {
    client.getHealth.mockResolvedValue({ status: 'ok', prototype: true })
    render(<App />)
    await waitFor(() => expect(chip()).toHaveAttribute('data-status', 'unknown'))
    expect(screen.getByTestId('provenance-environment')).toHaveTextContent(
      'unknown',
    )
    expect(screen.getByTestId('provenance-chart')).toHaveTextContent('unknown')
  })

  it('refuses to route on a departure time it does not have', async () => {
    client.getDefaults.mockRejectedValue(new ApiError('down', 0, 'api_unreachable'))
    const user = await toForecast()
    expect(screen.getByTestId('use-forecast')).toBeDisabled()

    //  and the mission panel says why, rather than showing a guessed date
    await user.click(
      within(screen.getByTestId('mission-drawer')).getByRole('button', {
        name: /mission/i,
      }),
    )
    expect(screen.getByTestId('departure-unknown')).toHaveTextContent(
      /waiting for the demonstration's own departure time/i,
    )
    expect(client.compareMission).not.toHaveBeenCalled()
  })

  it('seeds the mission from the backend’s own defaults', async () => {
    const user = userEvent.setup()
    render(<App />)
    const drawer = await screen.findByTestId('mission-drawer')
    await user.click(within(drawer).getByRole('button', { name: /mission/i }))

    const demo = missionDefaults().demo
    await waitFor(() =>
      expect(screen.getByLabelText(/departure time/i)).toHaveValue(
        String(demo.departure_time).slice(0, 16),
      ),
    )
    expect(screen.getByLabelText(/polaris class/i)).toHaveValue(
      String(demo.polaris_ice_class),
    )
    expect(screen.getByLabelText(/vessel speed/i)).toHaveValue(
      Number(demo.vessel_speed_mps),
    )
  })
})

// ──────────────────────────────────────────────────── the routing gate
describe('routing gate', () => {
  const GRID = missionResponse().grid
  const action = () => screen.getByTestId('use-forecast')
  const alignedHealth = () => ({
    ...HEALTH,
    grid: {
      ...GRID,
      usnic_chart_date: GRID.environment_date,
      dates_aligned: true,
    },
    requires_historical_demo_override: false,
  })

  it('shows no gate when the dates are contemporaneous', async () => {
    client.getHealth.mockResolvedValue(alignedHealth())
    //  the mission the backend returns reports the same alignment
    const alignedMission = missionResponse(false)
    alignedMission.grid.usnic_chart_date = alignedMission.grid.environment_date
    alignedMission.grid.dates_aligned = true
    client.compareMission.mockResolvedValue(alignedMission)
    const user = await toForecast()

    expect(screen.queryByTestId('forecast-gate')).not.toBeInTheDocument()
    expect(action()).toHaveAttribute('data-action', 'route')
    expect(action()).toHaveTextContent(/use forecast for routing/i)

    //  and routing runs straight through
    await user.click(action())
    await screen.findByTestId('route-card-fastest')
    expect(client.compareMission).toHaveBeenCalledTimes(1)
  })

  it('gates the mismatch before a request is ever sent', async () => {
    await toForecast()
    expect(GRID.dates_aligned).toBe(false)

    const gate = screen.getByTestId('forecast-gate')
    expect(gate).toHaveAttribute('data-state', 'override_required')
    expect(screen.getByTestId('gate-headline')).toHaveTextContent(
      /historical demonstration required/i,
    )
    expect(gate).toHaveTextContent(/from different dates/i)
    expect(action()).toHaveTextContent(/enable historical demonstration/i)
    expect(action()).toHaveAttribute('data-action', 'enable_override')
    expect(client.compareMission).not.toHaveBeenCalled()
  })

  it('pressing the gate sets the override and offers routing', async () => {
    const user = await toForecast()
    await user.click(action())

    //  no request was spent on the refusal, and nothing switched itself: the
    //  press is what set it
    expect(client.compareMission).not.toHaveBeenCalled()
    await waitFor(() =>
      expect(screen.getByTestId('forecast-gate')).toHaveAttribute(
        'data-state',
        'historical_active',
      ),
    )
    expect(action()).toHaveAttribute('data-action', 'route')
    expect(screen.getByTestId('gate-headline')).toHaveTextContent(
      /historical demonstration active/i,
    )

    client.compareMission.mockResolvedValue(missionResponse(true))
    await user.click(action())
    await screen.findByTestId('route-card-fastest')
    expect(client.compareMission).toHaveBeenCalledWith(
      expect.objectContaining({ historical_demo_override: true }),
    )
  })

  it('shares one switch with the mission panel, in both directions', async () => {
    const user = await toForecast()
    const drawer = () => screen.getByTestId('mission-drawer')
    const toggle = () => screen.getByLabelText(/historical demonstration mode/i)

    //  set from the dock, seen in the panel
    await user.click(action())
    await user.click(within(drawer()).getByRole('button', { name: /mission/i }))
    await waitFor(() => expect(toggle()).toBeChecked())

    //  cleared in the panel, seen by the gate
    await user.click(toggle())
    await waitFor(() =>
      expect(screen.getByTestId('forecast-gate')).toHaveAttribute(
        'data-state',
        'override_required',
      ),
    )
    expect(action()).toHaveAttribute('data-action', 'enable_override')

    //  and set again in the panel, seen by the gate
    await user.click(toggle())
    await waitFor(() =>
      expect(action()).toHaveAttribute('data-action', 'route'),
    )
  })

  it('pauses PLAN rather than showing routes when the override is taken away', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toPlan()
    await screen.findByTestId('route-card-fastest')

    await user.click(
      within(screen.getByTestId('mission-drawer')).getByRole('button', {
        name: /mission/i,
      }),
    )
    await user.click(screen.getByLabelText(/historical demonstration mode/i))

    const gate = await screen.findByTestId('plan-gate')
    expect(gate).toHaveTextContent(/routing paused/i)
    expect(gate).toHaveTextContent(/historical demonstration required/i)
    expect(screen.queryByTestId('route-card-fastest')).not.toBeInTheDocument()
    expect(screen.queryByTestId('selected-route-bar')).not.toBeInTheDocument()

    //  and enabling it again brings the same result back, with no new request
    await user.click(screen.getByTestId('gate-action'))
    await screen.findByTestId('route-card-fastest')
    expect(client.compareMission).toHaveBeenCalledTimes(1)
  })

  it('opens the provenance detail from the gate rather than repeating it', async () => {
    const user = await toForecast()
    expect(screen.queryByTestId('provenance-detail')).not.toBeInTheDocument()
    await user.click(screen.getByTestId('gate-details'))
    expect(await screen.findByTestId('provenance-detail')).toHaveTextContent(
      /ANTARC/,
    )
    //  the gate itself does not duplicate the panel
    expect(screen.getByTestId('forecast-gate')).not.toHaveTextContent(/ANTARC/)
  })

  it('does not gate what it cannot judge', async () => {
    //  a backend that reported no dates: the frontend makes no claim and lets
    //  the backend validate the pairing, as it always does
    client.getHealth.mockResolvedValue({ status: 'ok', prototype: true })
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toForecast()

    expect(screen.queryByTestId('forecast-gate')).not.toBeInTheDocument()
    expect(action()).toHaveAttribute('data-action', 'route')
    await user.click(action())
    await waitFor(() => expect(client.compareMission).toHaveBeenCalledTimes(1))
  })
})

// ──────────────────────────────────────────────────────── respond stage
describe('respond stage', () => {
  const steps = () =>
    Array.from(document.querySelectorAll('[data-testid^="stage-"]')).map(
      (step) => step.getAttribute('data-testid')!.replace('stage-', ''),
    )
  const stepFor = (name: string) => screen.getByTestId(`stage-${name}`)

  it('the stage machine is exactly observe, forecast, plan, respond', async () => {
    render(<App />)
    await screen.findByTestId('observe-dock')
    expect(steps()).toEqual(['observe', 'forecast', 'plan', 'respond'])
  })

  it('cannot be entered before a route exists', async () => {
    const user = await toForecast()
    //  observed, forecast -- but nothing has routed yet
    expect(stepFor('respond')).toBeDisabled()
    expect(screen.queryByTestId('go-respond')).not.toBeInTheDocument()
    expect(screen.queryByTestId('respond-dock')).not.toBeInTheDocument()

    await user.click(stepFor('respond'))
    expect(screen.queryByTestId('respond-dock')).not.toBeInTheDocument()
    expect(screen.getByTestId('mission-map')).toHaveAttribute(
      'data-camera-stage',
      'forecast',
    )
  })

  it('PLAN offers the transition once a route exists', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toPlan()
    await screen.findByTestId('route-card-fastest')

    const enter = screen.getByTestId('go-respond')
    expect(enter).toHaveTextContent(/respond/i)
    await user.click(enter)

    await screen.findByTestId('respond-dock')
    expect(stepFor('respond')).toHaveAttribute('aria-current', 'step')
    expect(screen.getByTestId('mission-map')).toHaveAttribute(
      'data-camera-stage',
      'respond',
    )
  })

  it('carries the selected route into RESPOND and back again', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    await waitFor(() =>
      expect(screen.getByTestId('mission-map')).toHaveAttribute(
        'data-selected',
        'risk_oriented',
      ),
    )

    await user.click(screen.getByTestId('go-respond'))
    const bar = await screen.findByTestId('selected-route-bar')
    expect(bar).toHaveTextContent(/risk-oriented/i)
    expect(bar).toHaveTextContent('186.87 km')
    expect(screen.getByTestId('mission-map')).toHaveAttribute(
      'data-selected',
      'risk_oriented',
    )

    //  back to PLAN: same selection, same options, no second request
    await user.click(screen.getByTestId('back-to-plan'))
    await screen.findByTestId('plan-dock')
    expect(screen.getByTestId('route-card-risk_oriented')).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(screen.getByTestId('priced-horizon')).toBeInTheDocument()
    expect(client.compareMission).toHaveBeenCalledTimes(1)
  })

  it('PLAN no longer carries the simulation panel', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    await toPlan()
    await screen.findByTestId('route-card-fastest')
    expect(screen.queryByTestId('simulation-control')).not.toBeInTheDocument()
    expect(screen.queryByTestId('inject-on-route')).not.toBeInTheDocument()
  })

  it('RESPOND carries it, and shows none of PLAN or FORECAST furniture', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toPlan()
    await user.click(await screen.findByTestId('go-respond'))

    await screen.findByTestId('respond-dock')
    expect(screen.getByTestId('simulation-control')).toBeInTheDocument()
    expect(screen.getByTestId('inject-on-route')).toBeInTheDocument()
    expect(screen.getByTestId('place-manually')).toBeInTheDocument()

    //  and the other stages' surfaces are gone
    expect(screen.queryByTestId('forecast-timeline')).not.toBeInTheDocument()
    expect(screen.queryByTestId('forecast-panel')).not.toBeInTheDocument()
    expect(screen.queryByTestId('plan-dock')).not.toBeInTheDocument()
    expect(screen.queryByTestId('route-card-fastest')).not.toBeInTheDocument()
    expect(screen.queryByTestId('mission-drawer')).not.toBeInTheDocument()
    expect(screen.queryByTestId('provenance-detail')).not.toBeInTheDocument()
  })

  it('states the scenario as a simulated event, before and after', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    await user.click(screen.getByTestId('go-respond'))

    const dock = await screen.findByTestId('respond-dock')
    expect(screen.getByTestId('respond-headline')).toHaveTextContent(
      /no simulated event yet/i,
    )
    expect(dock).toHaveTextContent(/written to no dataset/i)
    expect(screen.queryByTestId('respond-figures')).not.toBeInTheDocument()

    await armSimulation(user)
    await user.click(screen.getByTestId('fake-map-click'))

    await waitFor(() =>
      expect(screen.getByTestId('respond-headline')).toHaveTextContent(
        /route replanned/i,
      ),
    )
    const figures = screen.getByTestId('respond-figures')
    //  every figure is the backend's own before/after
    const change = simulationResponse().change.profiles.risk_oriented
    expect(figures).toHaveTextContent(
      String(change.cells_with_simulated_exposure),
    )
    expect(figures).toHaveTextContent(/affected cells/i)
    expect(figures).toHaveTextContent(/travel-time change/i)
    expect(figures).toHaveTextContent(/exposure contribution/i)
    for (const banned of [/\bsafe\b/i, /unsafe/i, /optimal/i, /collision/i, /guarantee/i]) {
      expect(dock.textContent ?? '').not.toMatch(banned)
    }
  })

  it('opens the change explanation from RESPOND itself', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    await armSimulation(user)
    await user.click(screen.getByTestId('fake-map-click'))

    //  it opened with the result; the toggle closes and reopens it
    const drawer = await screen.findByTestId('respond-drawer')
    expect(drawer).toHaveTextContent(/route impact/i)
    expect(within(drawer).getByTestId('route-impact')).toHaveTextContent(
      /SIMULATED ICEBERG/,
    )
    await user.click(screen.getByTestId('what-changed'))
    await waitFor(() =>
      expect(screen.queryByTestId('respond-drawer')).not.toBeInTheDocument(),
    )
    await user.click(screen.getByTestId('what-changed'))
    expect(await screen.findByTestId('respond-drawer')).toBeInTheDocument()
    //  the route-explanation drawer is not duplicated here
    expect(screen.queryByTestId('why-panel')).not.toBeInTheDocument()
  })

  it('keeps the simulation when stepping back to PLAN and returning', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    await armSimulation(user)
    await user.click(screen.getByTestId('fake-map-click'))
    await screen.findByTestId('respond-figures')

    await user.click(screen.getByTestId('back-to-plan'))
    await screen.findByTestId('plan-dock')
    await user.click(screen.getByTestId('go-respond'))

    //  the same scenario, not a fresh one: no extra backend call
    expect(await screen.findByTestId('respond-figures')).toBeInTheDocument()
    expect(client.simulateIceberg).toHaveBeenCalledTimes(1)
    expect(client.compareMission).toHaveBeenCalledTimes(1)
  })
})

// ──────────────────────────────────────────── deterministic injection
describe('inject on this route', () => {
  const GRID = missionResponse().grid

  const enterRespond = async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('go-respond'))
    await screen.findByTestId('respond-dock')
    return user
  }

  it('is the dock’s primary action, with manual placement beside it', async () => {
    await enterRespond()
    const inject = screen.getByTestId('inject-on-route')
    expect(inject).toHaveTextContent(/inject iceberg on this route/i)
    expect(inject.className).toContain('primary-action')
    expect(screen.getByTestId('place-manually')).toHaveTextContent(
      /place manually/i,
    )
  })

  it('sends a cell of the selected route, through the existing API', async () => {
    const user = await enterRespond()
    await user.click(screen.getByTestId('inject-on-route'))

    await waitFor(() => expect(client.simulateIceberg).toHaveBeenCalledTimes(1))
    const [request, iceberg, profile] = client.simulateIceberg.mock.calls[0]
    expect(profile).toBe('fastest')
    expect(request).toEqual(
      expect.objectContaining({ historical_demo_override: true }),
    )

    //  the position is the centre of a cell on the route the backend returned
    const path = missionResponse(true).comparison.profiles.fastest.path
    const centres = path.map((cell) => cellCentreXY(GRID, cell[0], cell[1]))
    expect(centres).toContainEqual([iceberg.x, iceberg.y])
  })

  it('sends the same position however many times it is asked', async () => {
    for (let run = 0; run < 3; run += 1) {
      const user = await enterRespond()
      await user.click(screen.getByTestId('inject-on-route'))
      await waitFor(() => expect(client.simulateIceberg).toHaveBeenCalled())
      cleanup()
    }
    const positions = client.simulateIceberg.mock.calls.map(
      ([, iceberg]) => `${iceberg.x},${iceberg.y}`,
    )
    expect(positions).toHaveLength(3)
    expect(new Set(positions).size).toBe(1)
  })

  it('follows the selection: another route gives another position', async () => {
    //  the same recorded payload with one profile taking a different corridor
    const payload = missionResponse(true)
    payload.comparison.profiles.risk_oriented.path = [
      payload.comparison.profiles.risk_oriented.path[0],
      [940, 500],
      payload.comparison.profiles.risk_oriented.path[1],
    ]
    client.compareMission.mockResolvedValue(payload)
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    await user.click(screen.getByTestId('go-respond'))
    await user.click(await screen.findByTestId('inject-on-route'))

    await waitFor(() => expect(client.simulateIceberg).toHaveBeenCalledTimes(1))
    const [, iceberg, profile] = client.simulateIceberg.mock.calls[0]
    expect(profile).toBe('risk_oriented')
    const centres = payload.comparison.profiles.risk_oriented.path.map((cell) =>
      cellCentreXY(GRID, cell[0], cell[1]),
    )
    expect(centres).toContainEqual([iceberg.x, iceberg.y])
  })

  it('shows the backend’s own verdict, unchanged', async () => {
    const user = await enterRespond()
    await user.click(screen.getByTestId('inject-on-route'))

    //  the recorded response says the FASTEST route did not change
    const change = simulationResponse().change.profiles.fastest
    expect(change.path_changed).toBe(false)
    await waitFor(() =>
      expect(screen.getByTestId('respond-headline')).toHaveTextContent(
        /route unchanged/i,
      ),
    )
    //  and the figures are still the real ones
    expect(screen.getByTestId('respond-figures')).toHaveTextContent(
      String(change.cells_with_simulated_exposure),
    )
  })

  it('hands the hazard geometry to the RESPOND camera and the map', async () => {
    const user = await enterRespond()
    await user.click(screen.getByTestId('inject-on-route'))

    const map = await screen.findByTestId('mission-map')
    await waitFor(() =>
      expect(map.getAttribute('data-layers')).toContain('simulated_iceberg'),
    )
    expect(map.getAttribute('data-layers')).toContain('simulated_envelope')
    expect(map.getAttribute('data-layers')).toContain('superseded_route')
    expect(map).toHaveAttribute('data-camera-stage', 'respond')
    expect(map).toHaveAttribute('data-superseded', 'yes')

    //  the camera frames the encounter the backend reported
    const berg = simulationResponse().simulation!
    const encounter = cellCentreXY(
      GRID,
      berg.encounter_cell[0],
      berg.encounter_cell[1],
    )
    const frame = (map.getAttribute('data-camera-extent') ?? '')
      .split(',')
      .map(Number)
    expect(encounter[0]).toBeGreaterThanOrEqual(frame[0])
    expect(encounter[0]).toBeLessThanOrEqual(frame[2])
    expect(encounter[1]).toBeGreaterThanOrEqual(frame[1])
    expect(encounter[1]).toBeLessThanOrEqual(frame[3])
  })

  it('keeps manual placement working beside it', async () => {
    const user = await enterRespond()
    await user.click(screen.getByTestId('place-manually'))
    expect(screen.getByTestId('mission-map')).toHaveAttribute(
      'data-placing',
      'true',
    )
    await user.click(screen.getByTestId('fake-map-click'))

    await waitFor(() => expect(client.simulateIceberg).toHaveBeenCalledTimes(1))
    //  the free click sends where it was clicked, not the route cell
    expect(client.simulateIceberg.mock.calls[0][1]).toEqual({ x: 1, y: 2 })
  })

  it('preserves the mission and the priced context across the injection', async () => {
    const user = await enterRespond()
    await user.click(screen.getByTestId('inject-on-route'))
    await screen.findByTestId('respond-figures')

    expect(screen.getByTestId('priced-horizon')).toHaveTextContent(
      /evaluated through \+6h and \+12h/i,
    )
    expect(screen.getByTestId('provenance-chip')).toHaveAttribute(
      'data-status',
      'historical_demonstration',
    )
    //  one route request, one simulation request
    expect(client.compareMission).toHaveBeenCalledTimes(1)
    expect(client.simulateIceberg).toHaveBeenCalledTimes(1)
  })
})

// ─────────────────────────────────────────────── respond evidence surface
describe('respond evidence', () => {
  const inject = async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('go-respond'))
    await user.click(await screen.findByTestId('inject-on-route'))
    await screen.findByTestId('respond-figures')
    return user
  }

  it('puts the encounter, the field and the exposure in the dock', async () => {
    await inject()
    const berg = simulationResponse().change.simulation
    const change = simulationResponse().change.profiles.fastest

    expect(screen.getByTestId('figure-affected')).toHaveTextContent(
      String(change.cells_with_simulated_exposure),
    )
    expect(screen.getByTestId('figure-encounter')).toHaveTextContent(
      berg.encounter_cell.join(', '),
    )
    expect(screen.getByTestId('figure-field')).toHaveTextContent(
      `+${berg.horizon_hours}h`,
    )
    expect(screen.getByTestId('figure-exposure')).toHaveTextContent(
      change.iceberg_exposure_contribution.before!.toLocaleString('en-GB'),
    )
    expect(screen.getByTestId('figure-exposure')).toHaveTextContent(
      change.iceberg_exposure_contribution.after!.toLocaleString('en-GB'),
    )
  })

  it('opens the evidence with the result, and keeps the verdict dominant', async () => {
    await inject()
    expect(screen.getByTestId('respond-headline')).toHaveTextContent(
      /route unchanged/i,
    )
    const impact = within(await screen.findByTestId('respond-drawer')).getByTestId(
      'route-impact',
    )
    expect(within(impact).getByTestId('impact-verdict')).toHaveTextContent(
      /route unchanged/i,
    )
  })

  it("stays inside the route's own priced horizon", async () => {
    await inject()
    const berg = simulationResponse().change.simulation
    //  the field the injection used is one the route was priced against
    expect(screen.getByTestId('figure-field')).toHaveTextContent(
      `+${berg.horizon_hours}h`,
    )
    expect(screen.getByTestId('priced-horizon')).toHaveTextContent(
      new RegExp(`\\+${berg.horizon_hours}h`),
    )
    expect(screen.getByTestId('mission-map')).toHaveAttribute(
      'data-forecast-hours',
      (12).toFixed(4),
    )
  })

  it('names the map layers as the evidence names them', async () => {
    await inject()
    const legend = screen.getByTestId('map-legend')
    expect(legend).toHaveTextContent(/simulated iceberg/i)
    expect(legend).toHaveTextContent(/uncertainty radius/i)
    expect(legend).toHaveTextContent(/route before the injection/i)
    //  and the simulated berg's own row never calls it an observation
    const simulated = legend.querySelector('[data-layer="simulated_iceberg"]')
    expect(simulated?.textContent ?? '').toMatch(/simulated iceberg/i)
    expect(simulated?.textContent ?? '').not.toMatch(/observed/i)
  })

  it('survives the trip back to PLAN and into RESPOND again', async () => {
    const user = await inject()
    const before = screen.getByTestId('respond-figures').textContent

    await user.click(screen.getByTestId('back-to-plan'))
    await screen.findByTestId('plan-dock')
    await user.click(screen.getByTestId('go-respond'))

    expect((await screen.findByTestId('respond-figures')).textContent).toBe(
      before,
    )
    expect(client.simulateIceberg).toHaveBeenCalledTimes(1)
  })

  it('takes the evidence away with the simulation', async () => {
    const user = await inject()
    await user.click(screen.getByTestId('clear-simulation'))

    await waitFor(() =>
      expect(screen.queryByTestId('respond-drawer')).not.toBeInTheDocument(),
    )
    expect(screen.queryByTestId('respond-figures')).not.toBeInTheDocument()
    expect(screen.getByTestId('respond-headline')).toHaveTextContent(
      /no simulated event yet/i,
    )
  })
})

// ── the surfaces mounted for the final integration ──────────────────────

describe('which kind of answer the route is', () => {
  it('names the operational demo from the policy the backend reported', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    await toPlan()
    const bar = await screen.findByTestId('selected-route-bar')
    const badge = within(bar).getByTestId('evaluation-badge')
    //  the demo is persistence-priced, so it is the operational demonstration
    expect(badge).toHaveAttribute('data-mode', 'operational_demo')
    expect(flat(badge)).toContain('Operational demo')
    expect(
      within(bar).queryByTestId('evaluation-badge-note'),
    ).not.toBeInTheDocument()
  })

  it('calls it a scientific evaluation when the backend named that policy', async () => {
    const payload = missionResponse(true)
    for (const name of PROFILE_ORDER) {
      const route = payload.comparison.profiles[name]
      if (!route) continue
      route.temporal_provenance = {
        ...route.temporal_provenance,
        environment_policies: ['long_horizon_forecast_evaluation'],
      }
    }
    client.compareMission.mockResolvedValue(payload)
    await toPlan()
    const badge = within(
      await screen.findByTestId('selected-route-bar'),
    ).getByTestId('evaluation-badge')
    expect(badge).toHaveAttribute(
      'data-mode',
      'long_horizon_scientific_evaluation',
    )
    //  the distinction is in words, not colour alone
    expect(flat(badge)).toContain('not an operational assessment')
  })

  it('shows the estimated fuel the backend sent, in OWE-m', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    await toPlan()
    const fuel = await screen.findByTestId('selected-route-fuel')
    expect(flat(fuel)).toContain('OWE-m')
    expect(flat(fuel)).toContain('estimated fuel')
    const bar = flat(screen.getByTestId('selected-route-bar')).toLowerCase()
    for (const banned of ['litre', 'tonne', 'gallon']) {
      expect(bar).not.toContain(banned)
    }
  })
})

describe('forecast provenance in the route drawer', () => {
  it('says plainly when no bucket was priced from a forecast', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toPlan()
    await user.click(await screen.findByTestId('why-toggle'))
    const section = await screen.findByTestId('why-forecast-provenance')
    //  the recorded demo is persistence-priced throughout
    expect(flat(section)).toContain(
      'No bucket on this route was priced from a model forecast',
    )
  })

  it('renders the lineage of every forecast-priced bucket', async () => {
    const payload = missionResponse(true)
    const route = payload.comparison.profiles.risk_oriented
    route.temporal_provenance = {
      ...route.temporal_provenance,
      environment_by_bucket: {
        '0': { bucket: 0, source_type: 'explicit_persistence', model: null },
        '4': {
          bucket: 4,
          source_type: 'model_forecast',
          model: 'HistGradientBoostingRegressor',
          resolution: {
            path: 'sic_forecast_hgb_origin20250108_plus24h_valid20250109_3976.tif',
            lead_hours: 24,
            validity_rule: 'target_composite_day',
            validity_from: '2025-01-09T00:00:00',
            validity_to: '2025-01-10T00:00:00',
            forecast_origin: '2025-01-08T00:00:00',
            valid_time: '2025-01-09T00:00:00',
          },
          model_artifact: 'data/processed/forecast_model_hgb.joblib',
          model_artifact_sha256: '463917f386b92e6e996506504c3662dfb0b5f828',
          dataset: 'data/processed/forecast_dataset.npz',
          dataset_sha256: '1721e165a765b31fc2866f0f7837ce9d80d2a653',
        },
      },
    }
    client.compareMission.mockResolvedValue(payload)
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    await user.click(await screen.findByTestId('why-toggle'))

    const bucket = await screen.findByTestId('forecast-bucket-4')
    expect(flat(bucket)).toContain('+24h')
    expect(flat(bucket)).toContain('HistGradientBoostingRegressor')
    expect(flat(bucket)).toContain('target_composite_day')
    expect(flat(bucket)).toContain('forecast_model_hgb.joblib')
    expect(flat(bucket)).toContain('463917f386b9')
    //  the persistence bucket is not dressed up as a forecast
    expect(screen.queryByTestId('forecast-bucket-0')).not.toBeInTheDocument()
    //  and the heading names the lead the backend reported
    expect(flat(screen.getByTestId('why-forecast-provenance'))).toContain('+24h')
  })

  it('states what the estimated fuel figure is not', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    const user = await toPlan()
    await user.click(await screen.findByTestId('why-toggle'))
    const block = await screen.findByTestId('why-fuel-assumptions')
    const text = flat(block)
    //  <dt>/<dd> pairs concatenate without a space in textContent
    expect(text).toContain('Estimated relative proxy')
    expect(text).toMatch(/Measured consumption\s*no/i)
    expect(text).toMatch(/Operational prediction\s*no/i)
    expect(text).toMatch(/Absolute volume or mass\s*unavailable/i)
  })
})

describe('the validation evidence view', () => {
  it('is closed until asked for, and is not part of the mission flow', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    render(<App />)
    await screen.findByTestId('start-forecast')
    expect(screen.queryByTestId('evidence-panel')).not.toBeInTheDocument()
    //  it was not fetched either
    expect(client.getHistoricalEvaluation).not.toHaveBeenCalled()
  })

  it('opens from the top bar and reports the held-out result faithfully', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByTestId('evidence-toggle'))

    const panel = await screen.findByTestId('evidence-panel')
    expect(client.getHistoricalEvaluation).toHaveBeenCalledTimes(1)
    const counts = flat(await screen.findByTestId('validation-counts'))
    expect(counts).toContain('cases evaluated')
    expect(counts).toContain('routes changed')

    //  the numbers that do not favour the forecast are shown, not softened
    const outcome = flat(screen.getByTestId('validation-outcome'))
    expect(outcome).toContain('1')
    expect(outcome).toContain('4')
    expect(flat(screen.getByTestId('validation-limitation'))).toContain(
      'does not establish universal superiority',
    )
    const text = flat(panel).toLowerCase()
    for (const banned of [
      'proof the model works',
      'accuracy guaranteed',
      'beats persistence',
      'improves safety',
    ]) {
      expect(text).not.toContain(banned)
    }
  })

  it('reports an unavailable artifact instead of an empty panel', async () => {
    client.getHistoricalEvaluation.mockRejectedValue(
      new ApiError('the historical backtest artifact has not been produced', 404),
    )
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByTestId('evidence-toggle'))
    expect(flat(await screen.findByTestId('validation-error'))).toContain(
      'has not been produced',
    )
  })
})
