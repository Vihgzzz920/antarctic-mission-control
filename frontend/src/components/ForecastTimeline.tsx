import type { ForecastResponse } from '../api/types'
import type { RoutingReadiness } from '../api/provenance'
import RoutingGate from './RoutingGate'

interface Props {
  forecast: ForecastResponse
  hours: number
  playing: boolean
  speed: number
  onScrub: (hours: number) => void
  onToggle: () => void
  onSpeed: (speed: number) => void
  /** the gate's own action: route, or enable the historical override */
  onAct: () => void
  /** opens the provenance detail in the top bar */
  onDetails: () => void
  readiness: RoutingReadiness
  planning: boolean
  /** false while the mission's own departure time is still unknown */
  ready: boolean
  /** horizons the routing configuration has an exposure field for; every
   *  other stop on this timeline can be displayed but prices nothing */
  pricedHours: number[]
}

const SPEEDS = [1, 2, 4]

export default function ForecastTimeline({
  forecast,
  hours,
  playing,
  speed,
  onScrub,
  onToggle,
  onSpeed,
  onAct,
  onDetails,
  readiness,
  planning,
  ready,
  pricedHours,
}: Props) {
  const stops = [0, ...forecast.horizons.map((h) => h.hours)]
  const max = stops[stops.length - 1]
  const position = (value: number) => `${(value / max) * 100}%`
  //  a horizon the routing configuration has no exposure field for can be
  //  looked at, but no route is ever priced against it
  const priced = new Set(pricedHours)
  const displayOnly = forecast.horizons
    .map((horizon) => horizon.hours)
    .filter((hours) => !priced.has(hours))

  return (
    <div className="timeline" data-testid="forecast-timeline">
      <div className="timeline-transport">
        <button
          type="button"
          className="transport"
          onClick={onToggle}
          aria-label={playing ? 'Pause forecast' : 'Play forecast'}
          data-testid="forecast-play"
        >
          {playing ? (
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
              <rect x="3.5" y="2.5" width="3.2" height="11" fill="currentColor" />
              <rect x="9.3" y="2.5" width="3.2" height="11" fill="currentColor" />
            </svg>
          ) : (
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
              <path d="M4 2.5 13 8 4 13.5z" fill="currentColor" />
            </svg>
          )}
        </button>
        <div className="speeds" role="group" aria-label="Playback speed">
          {SPEEDS.map((value) => (
            <button
              key={value}
              type="button"
              className={`speed${speed === value ? ' is-on' : ''}`}
              onClick={() => onSpeed(value)}
            >
              {value}×
            </button>
          ))}
        </div>
      </div>

      <div className="timeline-track">
        <input
          type="range"
          min={0}
          max={max}
          step={0.25}
          value={hours}
          onChange={(event) => onScrub(Number(event.target.value))}
          aria-label="Forecast horizon"
          data-testid="forecast-scrubber"
        />
        <div className="timeline-stops" aria-hidden="true">
          {stops.map((stop) => {
            const horizon = forecast.horizons.find((h) => h.hours === stop)
            const on = Math.abs(hours - stop) < 1e-6
            const unpriced = stop > 0 && !priced.has(stop)
            return (
              <span
                key={stop}
                className={`stop${on ? ' is-on' : ''}${
                  horizon?.extrapolated_uncertainty ? ' is-extrapolated' : ''
                }${unpriced ? ' is-display-only' : ''}`}
                style={{ left: position(stop) }}
              >
                <span className="stop-tick" />
                <span className="stop-label">
                  {stop === 0 ? 'Now' : `+${stop}h`}
                </span>
              </span>
            )
          })}
        </div>
      </div>

      {readiness.gated && (
        <RoutingGate
          readiness={readiness}
          onAct={onAct}
          onDetails={onDetails}
          variant="inline"
          testId="forecast-gate"
        />
      )}

      {displayOnly.length > 0 && pricedHours.length > 0 && (
        <p className="timeline-note" data-testid="timeline-priced-note">
          Exposure fields exist for{' '}
          {pricedHours.map((hours) => `+${hours}h`).join(' and ')}. The later
          horizons are forecast only — no route is priced against them.
        </p>
      )}

      <div className="timeline-actions">
        <span className="timeline-clock" data-testid="forecast-clock">
          {hours === 0 ? 'Observed' : `+${hours.toFixed(hours % 1 ? 2 : 0)} h`}
        </span>
        <button
          type="button"
          className="primary-action"
          onClick={onAct}
          disabled={planning || !ready}
          data-testid="use-forecast"
          data-action={readiness.action}
          title={
            ready
              ? undefined
              : 'waiting for the mission defaults the backend publishes'
          }
        >
          {planning ? 'Planning…' : readiness.actionLabel}
        </button>
      </div>
    </div>
  )
}
