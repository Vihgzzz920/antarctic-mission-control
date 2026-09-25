import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import App from '../App'
import { ApiError } from '../api/client'
import { forecastResponse, missionResponse, simulationResponse } from './fixtures'

// The map is an OpenLayers canvas; jsdom has no WebGL or layout. The mock
// records the props the map is actually handed, so the tests can see what the
// map would draw at each stage without drawing it.
vi.mock('../map/MissionMap', () => ({
  default: ({
    selected,
    placing,
    onPlace,
    routes,
    supersededRoute,
    exposure,
    forecastRecords,
    forecastHours,
    vessel,
  }: {
    selected: string | null
    placing: boolean
    onPlace: (position: [number, number]) => void
    routes: unknown
    supersededRoute: unknown
    exposure: unknown
    forecastRecords: unknown[] | null
    forecastHours: number
    vessel: [number, number] | null
  }) => (
    <div
      data-testid="mission-map"
      data-selected={selected ?? 'none'}
      data-placing={String(placing)}
      data-routes={routes ? 'yes' : 'no'}
      data-superseded={supersededRoute ? 'yes' : 'no'}
      data-exposure={exposure ? 'yes' : 'no'}
      data-forecast-bergs={forecastRecords ? String(forecastRecords.length) : 'none'}
      data-forecast-hours={forecastHours.toFixed(4)}
      data-vessel={vessel ? vessel.map((n) => n.toFixed(1)).join(',') : 'none'}
    >
      <button type="button" data-testid="fake-map-click" onClick={() => onPlace([1, 2])}>
        place
      </button>
    </div>
  ),
}))

const client = vi.hoisted(() => ({
  getHealth: vi.fn(),
  compareMission: vi.fn(),
  getIcebergExposure: vi.fn(),
  getIcebergForecast: vi.fn(),
  simulateIceberg: vi.fn(),
}))

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>(
    '../api/client',
  )
  return { ...actual, ...client }
})

const HEALTH = {
  status: 'ok',
  prototype: true,
  grid: missionResponse().grid,
  iceberg_exposure_weight: 5,
  requires_historical_demo_override: true,
}

const FORECAST = forecastResponse()
const SUBJECT = FORECAST.icebergs[0]

const at = (hours: number) => {
  const state =
    hours === 0
      ? SUBJECT.observed!
      : SUBJECT.predicted.find((entry) => entry.hours === hours)!
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
  client.getIcebergExposure.mockResolvedValue({
    type: 'FeatureCollection',
    features: [],
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

/** observe → forecast → plan */
async function toPlan() {
  const user = await toForecast()
  await user.click(screen.getByTestId('use-forecast'))
  return user
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
    await waitFor(() => expect(map).toHaveAttribute('data-exposure', 'yes'))
    expect(map).toHaveAttribute('data-routes', 'no')
    expect(map).toHaveAttribute('data-forecast-bergs', 'none')
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
      const card = await screen.findByTestId(`route-card-${name}`)
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

  it('states both dates of the historical demonstration', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    await toPlan()
    const chip = await screen.findByTestId('historical-banner')
    expect(chip).toHaveTextContent('2025-01-08')
    expect(chip).toHaveTextContent('2020-12-24')
  })

  it('shows no historical label when the override is off', async () => {
    client.compareMission.mockResolvedValue(missionResponse(false))
    await toPlan()
    await screen.findByTestId('route-card-fastest')
    expect(screen.queryByTestId('historical-banner')).not.toBeInTheDocument()
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

    await user.click(screen.getByRole('button', { name: /simulate iceberg/i }))
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
    await user.click(screen.getByRole('button', { name: /simulate iceberg/i }))
    await user.click(screen.getByTestId('fake-map-click'))
    await user.click(await screen.findByTestId('why-toggle'))

    const why = await screen.findByTestId('why-route-changed')
    expect(why).toHaveTextContent(/SIMULATED ICEBERG/)
    expect(why).toHaveTextContent(/modelled exposure on 3 cells/i)
    expect(why).toHaveTextContent(/26,476/)
    expect(why).toHaveTextContent(/31,902/)
    expect(why).toHaveTextContent(/adds 14.57 km/i)
    // the forbidden CLAIMS, not the panel's own denial of them
    expect(why).not.toHaveTextContent(/safest/i)
    expect(why).not.toHaveTextContent(/collision probability/i)
    expect(why).not.toHaveTextContent(/guaranteed avoidance/i)
    expect(why).not.toHaveTextContent(/best route/i)
    expect(why).toHaveTextContent(/not a statement that any route avoids/i)
  })

  it('restores the original route when the simulation is cleared', async () => {
    client.compareMission.mockResolvedValue(missionResponse(true))
    client.simulateIceberg.mockResolvedValue(simulationResponse())
    const user = await toPlan()
    await user.click(await screen.findByTestId('route-card-risk_oriented'))
    const before = screen.getByTestId('route-card-risk_oriented').textContent

    await user.click(screen.getByRole('button', { name: /simulate iceberg/i }))
    await user.click(screen.getByTestId('fake-map-click'))
    await screen.findByText(/replanned route/i)

    await user.click(screen.getByRole('button', { name: /clear simulation/i }))
    await waitFor(() =>
      expect(screen.getByTestId('selected-route-bar')).toHaveTextContent(
        /Selected route/i,
      ),
    )
    expect(screen.getByTestId('route-card-risk_oriented').textContent).toBe(before)
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

    expect(await screen.findByTestId('status-banner')).toHaveTextContent(
      /Historical date mismatch/i,
    )
    expect(screen.queryByTestId('route-card-fastest')).not.toBeInTheDocument()
    expect(screen.queryByTestId('selected-route-bar')).not.toBeInTheDocument()
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
