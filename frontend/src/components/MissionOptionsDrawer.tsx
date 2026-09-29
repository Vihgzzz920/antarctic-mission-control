import { useState } from 'react'

import type {
  CandidateSite,
  MissionDefaults,
  MissionMatrix,
  ProfileName,
} from '../api/types'
import ForecastComparison from './ForecastComparison'
import MissionMatrixView from './MissionMatrixView'

interface Props {
  open: boolean
  onClose: () => void
  defaults: MissionDefaults | null
  /** the operational-policy matrix, and the forecast-driven one when asked for */
  matrix: MissionMatrix | null
  forecastMatrix: MissionMatrix | null
  busy: boolean
  error: string | null
  departureTime: string
  onEvaluate: (options: {
    sites: Required<CandidateSite>[]
    departures: string[]
    scientific: boolean
  }) => void
}

const PROFILES: ProfileName[] = ['fastest', 'fuel_efficient', 'risk_oriented']

/** +0, +6 and +12 hours from the mission's own departure time. */
function departureSlots(departure: string): string[] {
  if (!departure) return []
  const base = new Date(departure)
  if (Number.isNaN(base.getTime())) return []
  return [0, 6, 12].map((h) => {
    const at = new Date(base.getTime() + h * 3_600_000)
    return at.toISOString().replace(/\.\d+Z$/, '')
  })
}

/**
 * Evaluate several destinations at several departure times.
 *
 * Every candidate coordinate comes from the backend's own published list, and
 * the departure times are offsets from the mission's own departure. Nothing
 * here invents a place or a time. The result is listed, never ranked: there is
 * no sort control, no score column and no recommendation.
 */
export default function MissionOptionsDrawer({
  open,
  onClose,
  defaults,
  matrix,
  forecastMatrix,
  busy,
  error,
  departureTime,
  onEvaluate,
}: Props) {
  const [scientific, setScientific] = useState(false)
  if (!open) return null

  const sites = defaults?.sites ?? []
  const longHorizon = defaults?.long_horizon_site ?? null
  const departures = departureSlots(departureTime)
  const candidates = scientific && longHorizon ? [longHorizon] : sites
  const canRun = candidates.length > 0 && departures.length > 0 && !busy

  return (
    <aside className="options-drawer" data-testid="mission-options-panel">
      <header>
        <h2>Mission options</h2>
        <button type="button" onClick={onClose} aria-label="Close">×</button>
      </header>

      <div className="options-body">
        <fieldset className="options-controls">
          <legend>What to evaluate</legend>
          <label className="options-toggle">
            <input
              type="checkbox"
              checked={scientific}
              onChange={(event) => setScientific(event.target.checked)}
              data-testid="mission-scientific-toggle"
            />
            <span>
              Forecast-driven long-horizon evaluation
              <i>
                Opt-in. Uses the +24 h and +48 h sea-ice forecasts on the
                synthetic long-horizon geometry, and omits iceberg exposure
                beyond its configured 12 h horizon. Not an operational
                assessment.
              </i>
            </span>
          </label>

          <p className="options-scope" data-testid="mission-options-scope">
            {candidates.length} destination
            {candidates.length === 1 ? '' : 's'} ×{' '}
            {departures.length} departure
            {departures.length === 1 ? '' : 's'}
            {departures.length > 0 && (
              <span className="options-times">
                {' '}
                ({departures
                  .map((d) => d.slice(11, 16))
                  .join(', ')})
              </span>
            )}
          </p>

          <button
            type="button"
            className="primary-action"
            disabled={!canRun}
            onClick={() =>
              onEvaluate({ sites: candidates, departures, scientific })
            }
            data-testid="mission-evaluate"
          >
            {busy ? 'Evaluating…' : 'Evaluate options'}
          </button>
          <p className="options-note">
            Profiles evaluated: {PROFILES.join(', ').replace(/_/g, '-')}.
            Results are listed as requested and are not ranked.
          </p>
        </fieldset>

        {error && (
          <p className="options-error" data-testid="mission-options-error">
            {error}
          </p>
        )}

        {matrix && <MissionMatrixView matrix={matrix} />}

        {matrix && forecastMatrix && (
          <ForecastComparison
            persistence={matrix}
            forecast={forecastMatrix}
            profile="risk_oriented"
          />
        )}
      </div>
    </aside>
  )
}
