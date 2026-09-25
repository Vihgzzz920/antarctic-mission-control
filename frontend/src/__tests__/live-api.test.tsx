import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeAll, describe, expect, it, vi } from 'vitest'

import App from '../App'
import type { ForecastResponse, MissionResponse } from '../api/types'

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
    await user.click(await screen.findByTestId('use-forecast'))

    for (const name of ['fastest', 'risk_oriented', 'shortest_distance'] as const) {
      const card = await screen.findByTestId(`route-card-${name}`)
      const profile = real.comparison.profiles[name]
      // the exact number the backend computed, rendered on the card
      expect(card).toHaveTextContent(
        `${profile.distance_km!.toLocaleString('en-GB', {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })} km`,
      )
    }
    expect(await screen.findByTestId('historical-banner')).toHaveTextContent(
      real.comparison.provenance.chart_dates![0],
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

    const subject = bergs.icebergs[0]
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
    vi.spyOn(globalThis, 'fetch').mockImplementation(
      async () =>
        new Response(JSON.stringify({ status: 'ok', prototype: true }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
    )
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
