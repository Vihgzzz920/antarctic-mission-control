import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeAll, describe, expect, it, vi } from 'vitest'

import App from '../App'
import { formatDate } from '../api/provenance'
import { injectionPoint } from '../map/injection'
import {
  groupEquivalentRoutes,
  pathsEqual,
} from '../api/routeEquivalence'
import type {
  ForecastResponse,
  MissionResponse,
  SimulationResponse,
} from '../api/types'

// An end-to-end check against a RUNNING backend: it renders the real screen
// from the real API responses, so nothing between compare_profiles() /
// forecast_icebergs() and what the operator reads can drift unnoticed. It
// SKIPS when the API is not up, because a unit test suite must not require a
// server.
//
//   uvicorn src.api.main:app --port 8000     (then: npm test)

const API = process.env.MISSION_API ?? 'http://127.0.0.1:8000'

vi.mock('../map/MissionMap', () => ({
  default: ({
    selected,
    forecastHours,
  }: {
    selected: string | null
    forecastHours: number
  }) => (
    <div
      data-testid="mission-map"
      data-selected={selected ?? 'none'}
      data-forecast-hours={forecastHours.toFixed(4)}
    />
  ),
}))

let live = false
let payload: MissionResponse | null = null
let forecast: ForecastResponse | null = null
let backendDefaults: Record<string, unknown> | null = null

beforeAll(async () => {
  try {
    const health = await fetch(`${API}/api/health`)
    if (!health.ok) return
    const routes = await fetch(
      `${API}/api/demo/routes?historical_demo_override=true`,
    )
    if (!routes.ok) return
    payload = (await routes.json()) as MissionResponse
    const bergs = await fetch(`${API}/api/forecast/icebergs`)
    if (bergs.ok) forecast = (await bergs.json()) as ForecastResponse
    const defaults = await fetch(`${API}/api/mission/defaults`)
    if (defaults.ok) {
      backendDefaults = ((await defaults.json()) as { demo: Record<string, unknown> })
        .demo
    }
    live = payload.status === 'ok'
  } catch {
    live = false
  }
}, 240_000)

/** Serves the recorded live payloads back to the app under test. */
function serve(real: MissionResponse, bergs: ForecastResponse | null) {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    const body = url.includes('/health')
      ? { status: 'ok', prototype: true, grid: real.grid }
      : url.includes('/mission/defaults')
        ? { demo: backendDefaults, grid: real.grid }
        : url.includes('/forecast/')
          ? bergs
          : url.includes('/layers/')
            ? { type: 'FeatureCollection', features: [] }
            : real
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  })
}

describe('live backend', () => {
  it('groups the objectives the real backend returned the same path for', async ({
    skip,
  }) => {
    if (!live || !payload) {
      skip()
      return
    }
    const profiles = payload.comparison.profiles
    const groups = groupEquivalentRoutes(profiles)
    const together = (a: keyof typeof profiles, b: keyof typeof profiles) =>
      groups.some(
        (group) => group.profiles.includes(a) && group.profiles.includes(b),
      )

    //  the grouping is exactly what the returned cell sequences say, and it
    //  agrees with the backend's OWN measurement of the same thing. Neither
    //  side is asserted to be true today: they are asserted to agree.
    expect(together('fastest', 'shortest_distance')).toBe(
      pathsEqual(profiles.fastest.path, profiles.shortest_distance.path),
    )
    expect(together('fastest', 'shortest_distance')).toBe(
      payload.comparison.provenance.fastest_and_shortest_distance_coincided,
    )
    expect(together('fastest', 'risk_oriented')).toBe(
      pathsEqual(profiles.fastest.path, profiles.risk_oriented.path),
    )
    //  every objective appears exactly once, whatever the geometry did
    expect(groups.flatMap((group) => group.profiles).sort()).toEqual(
      ['fastest', 'risk_oriented', 'shortest_distance'],
    )
  })

  it('renders the real comparison on the real screen', async ({ skip }) => {
    if (!live || !payload) {
      skip()
      return
    }
    const real = payload
    serve(real, forecast)

    const user = userEvent.setup()
    render(<App />)
    const start = await screen.findByTestId('start-forecast')
    await waitFor(() => expect(start).toBeEnabled())
    await user.click(start)

    //  the real demonstration pairs a 2020 chart with a later environment, so
    //  the dock gates routing until the operator enables it
    const dock = await screen.findByTestId('use-forecast')
    if (dock.getAttribute('data-action') === 'enable_override') {
      await user.click(dock)
      await waitFor(() =>
        expect(screen.getByTestId('use-forecast')).toHaveAttribute(
          'data-action',
          'route',
        ),
      )
    }
    await user.click(screen.getByTestId('use-forecast'))

    for (const name of ['fastest', 'risk_oriented', 'shortest_distance'] as const) {
      const chip = await screen.findByTestId(`route-card-${name}`)
      const card = chip.closest('.option') as HTMLElement
      const profile = real.comparison.profiles[name]
      // the exact number the backend computed, rendered on the card
      expect(card).toHaveTextContent(
        `${profile.distance_km!.toLocaleString('en-GB', {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })} km`,
      )
    }
    //  the provenance chip carries the backend's own chart date, formatted
    expect(await screen.findByTestId('provenance-chart')).toHaveTextContent(
      formatDate(real.comparison.provenance.chart_dates![0])!,
    )
    expect(screen.getByTestId('provenance-environment')).toHaveTextContent(
      formatDate(real.comparison.provenance.environment_dates![0])!,
    )
  })

  it('shows the model’s own positions at every horizon', async ({ skip }) => {
    if (!live || !payload || !forecast) {
      skip()
      return
    }
    const bergs = forecast
    serve(payload, bergs)

    const user = userEvent.setup()
    render(<App />)
    const start = await screen.findByTestId('start-forecast')
    await waitFor(() => expect(start).toBeEnabled())
    await user.click(start)
    await screen.findByTestId('forecast-timeline')

    //  whichever iceberg the app opened on -- read from the panel's own
    //  selector, so this checks the numbers rather than the choice
    const selected = (screen.getByLabelText(/^iceberg$/i) as HTMLSelectElement)
      .value
    const subject = bergs.icebergs.find((berg) => berg.iceberg_id === selected)!
    expect(subject).toBeDefined()
    const panel = () =>
      (screen.getByTestId('forecast-panel').textContent ?? '').replace(/\s+/g, ' ')

    for (const state of [subject.observed!, ...subject.predicted]) {
      fireEvent.change(screen.getByTestId('forecast-scrubber'), {
        target: { value: String(state.hours) },
      })
      expect(screen.getByTestId('forecast-state-label')).toHaveTextContent(
        state.label,
      )
      const ns = state.latitude < 0 ? 'S' : 'N'
      const ew = state.longitude < 0 ? 'W' : 'E'
      expect(panel()).toContain(
        `${Math.abs(state.latitude).toFixed(2)}° ${ns} ` +
          `${Math.abs(state.longitude).toFixed(2)}° ${ew}`,
      )
      expect(panel()).toContain(`${state.radius_km.toFixed(2)} km`)
      // the extrapolated flag follows the backend's own field, not a rule here
      const flagged = Boolean(screen.queryByTestId('extrapolated-flag'))
      expect(flagged).toBe(state.extrapolated_uncertainty)
    }
  })

  it('opens on the same inputs the backend calls its validated demo', async ({
    skip,
  }) => {
    if (!live || !backendDefaults) {
      skip()
      return
    }
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const body = String(input).includes('/mission/defaults')
        ? { demo: backendDefaults }
        : { status: 'ok', prototype: true }
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    })
    const user = userEvent.setup()
    render(<App />)
    const drawer = await screen.findByTestId('mission-drawer')
    await user.click(within(drawer).getByRole('button', { name: /mission/i }))

    expect(screen.getByLabelText(/polaris class/i)).toHaveValue(
      backendDefaults.polaris_ice_class as string,
    )
    expect(screen.getByLabelText(/riv table/i)).toHaveValue(
      backendDefaults.polaris_riv_table as string,
    )
    expect(screen.getByLabelText(/departure time/i)).toHaveValue(
      (backendDefaults.departure_time as string).slice(0, 16),
    )
    expect(screen.getByLabelText(/vessel speed/i)).toHaveValue(
      backendDefaults.vessel_speed_mps as number,
    )
  })
})

describe('deterministic injection, against the running backend', () => {
  it('lands exactly on the route: distance_to_route_m is zero', async ({
    skip,
  }) => {
    if (!live || !payload) {
      skip()
      return
    }
    //  earlier tests in this file stub fetch with recorded payloads; this one
    //  talks to the real API, so the stub goes first
    vi.restoreAllMocks()
    const real = payload
    const profile = 'fastest' as const
    const route = real.comparison.profiles[profile]

    //  the SAME helper the dock uses, on the real route the backend returned
    const point = injectionPoint(route.path, real.grid)
    expect(point).not.toBeNull()
    expect(route.path.map((cell) => cell.join(','))).toContain(
      point!.cell.join(','),
    )
    expect(point!.fraction).toBeGreaterThan(0.3)
    expect(point!.fraction).toBeLessThan(0.5)

    //  and the real simulation endpoint, with the body the app's client sends
    //  (the client itself uses a relative URL, which jsdom cannot resolve)
    const sent = await fetch(`${API}/api/mission/simulate?profile=${profile}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        request: real.request,
        iceberg: { x: point!.position[0], y: point!.position[1] },
      }),
    })
    expect(sent.ok).toBe(true)
    const response = (await sent.json()) as SimulationResponse
    expect(response.simulation).not.toBeNull()
    expect(response.simulation!.distance_to_route_m).toBe(0)
    expect(response.simulation!.encounter_cell).toEqual(point!.cell)

    //  whatever the planner decides is what is reported: nothing is forced
    const change = response.change.profiles[profile]
    expect(typeof change.path_changed).toBe('boolean')
    expect(change.cells_with_simulated_exposure).toBeGreaterThan(0)
    expect(response.change.claims.guarantees_avoidance).toBe(false)
  }, 600_000)
})
