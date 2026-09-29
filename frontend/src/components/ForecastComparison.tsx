import type { MissionMatrix, ProfileName } from '../api/types'
import { EMPTY, hours, km, num, owe } from './format'

interface Props {
  persistence: MissionMatrix | null
  forecast: MissionMatrix | null
  profile: ProfileName
}

function row(matrix: MissionMatrix | null, profile: ProfileName) {
  const option = matrix?.summary.options.find((o) => o.feasible)
  return option?.objectives[profile] ?? null
}

const delta = (a: number | null | undefined, b: number | null | undefined) =>
  a === null || a === undefined || b === null || b === undefined
    ? null
    : b - a

const signed = (v: number | null, digits = 0) =>
  v === null ? EMPTY : `${v > 0 ? '+' : ''}${num(v, digits)}`

/**
 * The same scenario under the persistence environment and under the
 * forecast-driven long-horizon evaluation.
 *
 * This is a measurement, not a contest. The heading says whether the route
 * changed, never which environment "won", and a forecast-driven route that
 * costs more is displayed exactly as measured.
 */
export default function ForecastComparison({
  persistence,
  forecast,
  profile,
}: Props) {
  const a = row(persistence, profile)
  const b = row(forecast, profile)
  if (!a || !b || !a.routed || !b.routed) {
    return (
      <section className="fc" data-testid="forecast-comparison">
        <h3>Forecast-driven evaluation</h3>
        <p data-testid="forecast-comparison-unavailable">
          Both plans must route before they can be compared.
          {!a?.routed && persistence
            ? ` Persistence: ${(a?.outcome ?? 'did not route').replace(/_/g, ' ')}.`
            : ''}
          {!b?.routed && forecast
            ? ` Forecast-driven: ${(b?.outcome ?? 'did not route').replace(/_/g, ' ')}.`
            : ''}
        </p>
      </section>
    )
  }

  const samePath =
    delta(a.distance_km, b.distance_km) === 0 &&
    delta(a.configured_cost, b.configured_cost) === 0
  const leads = forecast?.options[0]?.provenance.environment.forecast_leads ?? []
  const buckets =
    forecast?.options[0]?.provenance.environment.buckets.filter(
      (x) => x.source_type === 'model_forecast',
    ).length ?? 0

  return (
    <section className="fc" data-testid="forecast-comparison">
      <h3>Forecast-driven evaluation</h3>
      <p className="fc-verdict" data-testid="forecast-comparison-verdict">
        <b>{samePath ? 'Route unchanged' : 'Route changed'}</b>
        <span>
          {' '}
          — persistence compared with the forecast-driven environment, same
          geometry and routing settings.
        </span>
      </p>
      <table className="fc-table" data-testid="forecast-comparison-table">
        <thead>
          <tr>
            <th scope="col">Measure</th>
            <th scope="col">Persistence</th>
            <th scope="col">Forecast-driven</th>
            <th scope="col">Δ</th>
          </tr>
        </thead>
        <tbody>
          <tr data-testid="fc-distance">
            <th scope="row">Distance</th>
            <td>{km(a.distance_km)}</td>
            <td>{km(b.distance_km)}</td>
            <td>{signed(delta(a.distance_km, b.distance_km), 2)}</td>
          </tr>
          <tr data-testid="fc-time">
            <th scope="row">Travel time</th>
            <td>{hours(a.travel_time_h)}</td>
            <td>{hours(b.travel_time_h)}</td>
            <td>{signed(delta(a.travel_time_h, b.travel_time_h), 2)}</td>
          </tr>
          <tr data-testid="fc-cost">
            <th scope="row">Configured cost</th>
            <td>{num(a.configured_cost)}</td>
            <td>{num(b.configured_cost)}</td>
            <td>{signed(delta(a.configured_cost, b.configured_cost))}</td>
          </tr>
          <tr data-testid="fc-env">
            <th scope="row">Environmental contribution</th>
            <td>{num(a.environmental_contribution)}</td>
            <td>{num(b.environmental_contribution)}</td>
            <td>
              {signed(
                delta(a.environmental_contribution, b.environmental_contribution),
              )}
            </td>
          </tr>
          <tr data-testid="fc-polaris">
            <th scope="row">POLARIS contribution</th>
            <td>{num(a.polaris_contribution)}</td>
            <td>{num(b.polaris_contribution)}</td>
            <td>{signed(delta(a.polaris_contribution, b.polaris_contribution))}</td>
          </tr>
          <tr data-testid="fc-fuel">
            <th scope="row">Estimated fuel</th>
            <td>{owe(a.estimated_fuel)}</td>
            <td>{owe(b.estimated_fuel)}</td>
            <td>{signed(delta(a.estimated_fuel, b.estimated_fuel))}</td>
          </tr>
        </tbody>
      </table>
      <p className="fc-buckets" data-testid="forecast-comparison-buckets">
        {buckets} arrival-time bucket{buckets === 1 ? '' : 's'} priced from a
        model forecast
        {leads.length ? ` at lead${leads.length > 1 ? 's' : ''} ${leads.map((l) => `+${l}h`).join(', ')}` : ''}
        . Iceberg exposure is omitted beyond its configured 12 h horizon in this
        mode, so neither column is an operational assessment.
      </p>
    </section>
  )
}
