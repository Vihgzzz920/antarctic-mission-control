import type { ProfileName, SimulationResponse } from '../api/types'
import { num } from './format'
import { buildRespondEvidence } from './respondEvidence'
import SimulationControl from './SimulationControl'

interface Props {
  profile: ProfileName
  simulation: SimulationResponse | null
  armed: boolean
  busy: boolean
  disabled: boolean
  onInject: () => void
  onArm: () => void
  onClear: () => void
  onExplain: () => void
  explaining: boolean
  onBack: () => void
  /** the same priced-horizon statement PLAN carries, so the temporal context
   *  survives the stage change */
  provenance: string | null
}

const MINUTES = (seconds: number) => `${Math.round(Math.abs(seconds) / 60)} min`

/**
 * What happens to the current mission when the environment changes.
 *
 * Everything below the controls is read off the two route results the backend
 * returned, before and after one simulated iceberg. The simulated event is
 * labelled as one throughout: it is a scenario the operator asked for, not an
 * observation and not a forecast of a real berg.
 */
export default function RespondDock({
  profile,
  simulation,
  armed,
  busy,
  disabled,
  onInject,
  onArm,
  onClear,
  onExplain,
  explaining,
  onBack,
  provenance,
}: Props) {
  const change = simulation?.change.profiles[profile] ?? null
  const berg = simulation?.change.simulation ?? null
  //  the same evidence the drawer renders, so the two can never disagree
  const evidence = buildRespondEvidence(simulation, profile)

  return (
    <div className="respond-dock" data-testid="respond-dock">
      <div className="respond-copy">
        <span className="respond-eyebrow">Respond</span>
        {evidence ? (
          <>
            <strong
              className="respond-verdict"
              data-testid="respond-headline"
              data-state={evidence.verdict}
            >
              {evidence.headline}
            </strong>
            <p data-testid="respond-summary">{evidence.summary}</p>
          </>
        ) : (
          <>
            <strong data-testid="respond-headline">
              {armed ? 'Click the map to place it' : 'No simulated event yet'}
            </strong>
            <p>
              {armed
                ? 'The simulated iceberg goes where you click, at the time the ' +
                  'current route is predicted to reach that cell.'
                : 'Inject one simulated iceberg on a cell of this route and the ' +
                  'same planner runs again. It is a scenario, written to no ' +
                  'dataset.'}
            </p>
          </>
        )}
        {provenance && (
          <p className="respond-provenance" data-testid="priced-horizon">
            {provenance}
          </p>
        )}
      </div>

      {change && berg && (
        <div className="respond-figures" data-testid="respond-figures">
          <span>
            <b data-testid="figure-affected">
              {change.cells_with_simulated_exposure ?? '—'}
            </b>
            <i>affected cells</i>
          </span>
          <span>
            <b data-testid="figure-encounter">{berg.encounter_cell.join(', ')}</b>
            <i>encounter cell</i>
          </span>
          <span>
            <b data-testid="figure-field">{`+${berg.horizon_hours}h`}</b>
            <i>exposure field · bucket {berg.from_bucket}</i>
          </span>
          <span>
            <b data-testid="figure-exposure">
              {change.iceberg_exposure_contribution.before === null ||
              change.iceberg_exposure_contribution.after === null
                ? '—'
                : `${num(
                    change.iceberg_exposure_contribution.before,
                  )} → ${num(change.iceberg_exposure_contribution.after)}`}
            </b>
            <i>exposure contribution</i>
          </span>
          {change.path_changed && (
            <>
              <span>
                <b>
                  {change.distance_m.delta === null
                    ? '—'
                    : `${change.distance_m.delta > 0 ? '+' : '−'}${num(
                        Math.abs(change.distance_m.delta) / 1000,
                        2,
                      )} km`}
                </b>
                <i>distance change</i>
              </span>
              <span>
                <b>
                  {change.travel_time_s.delta === null
                    ? '—'
                    : `${change.travel_time_s.delta > 0 ? '+' : '−'}${MINUTES(
                        change.travel_time_s.delta,
                      )}`}
                </b>
                <i>travel-time change</i>
              </span>
            </>
          )}
        </div>
      )}

      <div className="respond-actions">
        <button
          type="button"
          className="text-action"
          onClick={onBack}
          data-testid="back-to-plan"
        >
          Back to plan
        </button>
        {change && (
          <button
            type="button"
            className={`why-toggle${explaining ? ' is-open' : ''}`}
            onClick={onExplain}
            aria-expanded={explaining}
            data-testid="what-changed"
          >
            Route impact
          </button>
        )}
        <SimulationControl
          armed={armed}
          active={Boolean(simulation)}
          busy={busy}
          disabled={disabled}
          onInject={onInject}
          onArm={onArm}
          onClear={onClear}
        />
      </div>
    </div>
  )
}
