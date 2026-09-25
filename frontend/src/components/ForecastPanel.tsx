import type { ForecastResponse, IcebergForecastRecord } from '../api/types'
import { resolveState } from '../map/forecast'
import { formatLonLat } from '../map/coords'
import { IcebergIcon } from './icons'

interface Props {
  forecast: ForecastResponse
  hours: number
  record: IcebergForecastRecord | null
  onPick: (icebergId: string) => void
}

export default function ForecastPanel({
  forecast,
  hours,
  record,
  onPick,
}: Props) {
  const subject = record ?? forecast.icebergs[0]
  const state = subject ? resolveState(subject, hours) : null

  return (
    <section className="forecast-panel" data-testid="forecast-panel">
      <header>
        <IcebergIcon size={14} />
        <h2>Iceberg forecast</h2>
        <select
          value={subject?.iceberg_id ?? ''}
          onChange={(event) => onPick(event.target.value)}
          aria-label="Iceberg"
        >
          {forecast.icebergs.map((berg) => (
            <option key={berg.iceberg_id} value={berg.iceberg_id}>
              {berg.iceberg_id}
            </option>
          ))}
        </select>
      </header>

      {state && subject ? (
        <>
          <div className="forecast-state">
            <span
              className={`state-label${state.exact ? '' : ' is-interpolated'}`}
              data-testid="forecast-state-label"
            >
              {state.exact ? state.label : 'Interpolated for display'}
            </span>
            {state.extrapolatedUncertainty && (
              <span className="extrap-flag" data-testid="extrapolated-flag">
                Extrapolated
              </span>
            )}
          </div>

          {!state.exact && state.to && (
            <p className="forecast-note">
              Between {state.from.label} and {state.to.label}. The model
              produces states only at its horizons; this frame is smoothed for
              display.
            </p>
          )}

          <dl className="forecast-grid">
            <div>
              <dt>Predicted position</dt>
              <dd>
                {formatLonLat([state.longitude, state.latitude]) ?? '—'}
              </dd>
            </div>
            <div>
              <dt>Forecast horizon</dt>
              <dd>{hours === 0 ? 'observed' : `+${state.hours.toFixed(hours % 1 ? 2 : 0)} h`}</dd>
            </div>
            <div>
              <dt>Uncertainty radius</dt>
              <dd>{state.radiusKm.toFixed(2)} km</dd>
            </div>
            <div>
              <dt>Drift model</dt>
              <dd className="forecast-model">{forecast.model.summary}</dd>
            </div>
          </dl>

          <p className="forecast-foot">
            <strong>{forecast.observed_label}</strong> is this iceberg's own
            recorded position — the model's input. Everything after it is a
            prediction advected from that position.
            {state.physicsPositionIsExtrapolated &&
              ' This horizon runs past the observed interval it was derived from.'}
          </p>
        </>
      ) : (
        <p className="forecast-foot">No forecast for this date.</p>
      )}
    </section>
  )
}
