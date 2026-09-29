import type { HistoricalEvaluationResponse } from '../api/types'
import { EMPTY, num } from './format'

interface Props {
  data: HistoricalEvaluationResponse | null
  error?: string | null
}

const metric = (v: number | undefined | null, digits = 3) =>
  v === undefined || v === null || !Number.isFinite(v) ? EMPTY : v.toFixed(digits)

/**
 * The held-out historical forecast -> decision backtest, as evidence.
 *
 * Three layers, reported separately and never combined into a score: how close
 * the forecast was, whether the decision changed, and how the already-planned
 * routes compared against the sea ice that actually arrived.
 *
 * The numbers are shown as produced, including the ones that do not favour the
 * forecast. This view makes no claim that forecast-driven routing is better,
 * safer or recommended, because the evidence does not support one.
 */
export default function ValidationPanel({ data, error }: Props) {
  if (error) {
    return (
      <section className="validation" data-testid="validation-panel">
        <h3>Historical evaluation</h3>
        <p className="val-error" data-testid="validation-error">
          {error}
        </p>
      </section>
    )
  }
  if (!data) {
    return (
      <section className="validation" data-testid="validation-panel">
        <h3>Historical evaluation</h3>
        <p data-testid="validation-loading">Loading the held-out backtest…</p>
      </section>
    )
  }
  const e = data.evaluation
  const a = e.aggregate
  const ran = e.cases.filter((c) => c.ran && c.decision?.both_planned)

  return (
    <section className="validation" data-testid="validation-panel">
      <h3>Historical evaluation</h3>
      <p className="val-scope" data-testid="validation-scope">
        Held-out origins {e.held_out_window[0]} to {e.held_out_window[1]} ·{' '}
        {e.candidate_rule}. Evidence about these origins and this geometry only.
      </p>

      <div className="val-counts" data-testid="validation-counts">
        <div>
          <b>{a.cases_included}</b>
          <i>cases evaluated</i>
        </div>
        <div>
          <b>{a.cases_skipped}</b>
          <i>skipped</i>
        </div>
        <div>
          <b>{a.route_changed}</b>
          <i>routes changed</i>
        </div>
        <div>
          <b>{a.route_unchanged}</b>
          <i>routes unchanged</i>
        </div>
        <div>
          <b>{a.cases_realized}</b>
          <i>realized against observation</i>
        </div>
        <div>
          <b>{a.cases_not_realized}</b>
          <i>realization unavailable</i>
        </div>
      </div>

      <h4>Forecast accuracy</h4>
      <table className="val-table" data-testid="validation-accuracy">
        <thead>
          <tr>
            <th scope="col">Origin</th>
            <th scope="col">+24 h MAE</th>
            <th scope="col">+24 h persistence</th>
            <th scope="col">+48 h MAE</th>
            <th scope="col">+48 h persistence</th>
          </tr>
        </thead>
        <tbody>
          {ran.map((c) => (
            <tr key={c.case.origin} data-testid={`val-acc-${c.case.origin}`}>
              <th scope="row">{c.case.origin}</th>
              <td>{metric(c.forecast_accuracy?.['+24h']?.model?.mae)}</td>
              <td>{metric(c.forecast_accuracy?.['+24h']?.persistence?.mae)}</td>
              <td>{metric(c.forecast_accuracy?.['+48h']?.model?.mae)}</td>
              <td>{metric(c.forecast_accuracy?.['+48h']?.persistence?.mae)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>Decision effect and realized outcome</h4>
      <table className="val-table" data-testid="validation-decisions">
        <thead>
          <tr>
            <th scope="col">Origin</th>
            <th scope="col">Route</th>
            <th scope="col">Configured cost Δ</th>
            <th scope="col">Estimated fuel Δ</th>
            <th scope="col">Realized environmental cost</th>
          </tr>
        </thead>
        <tbody>
          {ran.map((c) => {
            const z = c.realized ?? {}
            const lower = z.which_plan_met_lower_realized_environmental_cost
            return (
              <tr key={c.case.origin} data-testid={`val-dec-${c.case.origin}`}>
                <th scope="row">{c.case.origin}</th>
                <td>{c.decision?.route_changed ? 'changed' : 'unchanged'}</td>
                <td>{num(c.decision?.configured_cost_delta)}</td>
                <td>{num(c.decision?.estimated_fuel_delta)}</td>
                <td>
                  {z.evaluated
                    ? `${num(z.realized_environmental_cost_delta)} · lower on ${
                        lower === 'forecast_plan'
                          ? 'forecast plan'
                          : lower === 'persistence_plan'
                            ? 'persistence plan'
                            : (lower ?? EMPTY)
                      }`
                    : 'not evaluated — no observed sea ice on part of the path'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      <p className="val-outcome" data-testid="validation-outcome">
        Across the realized cases, the forecast-driven plan met the lower
        realized environmental cost in{' '}
        <b>{a.forecast_plan_lower_realized_environmental_cost}</b>, and the
        persistence plan in{' '}
        <b>{a.persistence_plan_lower_realized_environmental_cost}</b>.
      </p>
      <p className="val-note" data-testid="validation-limitation">
        The current held-out evaluation does not establish universal superiority
        of forecast-driven routing over persistence. This is a scientific
        limitation, not a failure state.
      </p>
      <p className="val-artifact" data-testid="validation-artifact">
        {data.artifact} · sha256 {data.sha256.slice(0, 16)}…
      </p>
    </section>
  )
}
