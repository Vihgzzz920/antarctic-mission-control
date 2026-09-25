import type { ForecastResponse } from '../api/types'

interface Props {
  forecast: ForecastResponse
  hours: number
  playing: boolean
  speed: number
  onScrub: (hours: number) => void
  onToggle: () => void
  onSpeed: (speed: number) => void
  onUseForRouting: () => void
  planning: boolean
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
  onUseForRouting,
  planning,
}: Props) {
  const stops = [0, ...forecast.horizons.map((h) => h.hours)]
  const max = stops[stops.length - 1]
  const position = (value: number) => `${(value / max) * 100}%`

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
            return (
              <span
                key={stop}
                className={`stop${on ? ' is-on' : ''}${
                  horizon?.extrapolated_uncertainty ? ' is-extrapolated' : ''
                }`}
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

      <div className="timeline-actions">
        <span className="timeline-clock" data-testid="forecast-clock">
          {hours === 0 ? 'Observed' : `+${hours.toFixed(hours % 1 ? 2 : 0)} h`}
        </span>
        <button
          type="button"
          className="primary-action"
          onClick={onUseForRouting}
          disabled={planning}
          data-testid="use-forecast"
        >
          {planning ? 'Planning…' : 'Use forecast for routing'}
        </button>
      </div>
    </div>
  )
}
