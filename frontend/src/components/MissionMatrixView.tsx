import type {
  MissionMatrix,
  MissionObjectiveRow,
  ProfileName,
} from '../api/types'
import { PROFILE_LABEL, PROFILE_ORDER } from '../api/types'
import EvaluationBadge from './EvaluationBadge'
import { EMPTY, hours, km, num, owe } from './format'

interface Props {
  matrix: MissionMatrix
  /** open the forecast provenance for one option */
  onInspect?: (scenarioId: string) => void
}

const ROW_ORDER: ProfileName[] = PROFILE_ORDER.filter(
  (p) => p !== 'shortest_distance',
)

function Metrics({ row, name }: { row: MissionObjectiveRow; name: ProfileName }) {
  if (!row.routed) {
    return (
      <span className="mx-refused">
        did not route — {(row.outcome ?? 'unknown').replace(/_/g, ' ')}
      </span>
    )
  }
  return (
    <span className="mx-metrics">
      <span>{km(row.distance_km)}</span>
      <span>{hours(row.travel_time_h)}</span>
      {name === 'fuel_efficient' ? (
        <span title="estimated relative proxy in open-water-equivalent metres">
          {owe(row.estimated_fuel)} <i>est.</i>
        </span>
      ) : (
        <span>
          {num(row.configured_cost)} <i>cost</i>
        </span>
      )}
    </span>
  )
}

/**
 * The mission matrix: every candidate destination at every departure time.
 *
 * Deliberately NOT a league table. The options are listed in the order they
 * were requested, no column is sorted by value, nothing is starred, and there
 * is no score. Which option matters is the operator's judgement, and the
 * backend says so too (`this_is_not_a_ranking`).
 */
export default function MissionMatrixView({ matrix, onInspect }: Props) {
  const s = matrix.summary
  const bySite = new Map<string, typeof s.options>()
  for (const option of s.options) {
    const list = bySite.get(option.site) ?? []
    list.push(option)
    bySite.set(option.site, list)
  }
  return (
    <section className="mission-matrix" data-testid="mission-matrix">
      <header className="mx-head">
        <h3>Mission options</h3>
        <EvaluationBadge
          mode={matrix.evaluation_mode}
          isOperationalAssessment={matrix.is_operational_assessment}
          icebergExposure={matrix.iceberg_exposure}
        />
        <p className="mx-sub" data-testid="mission-matrix-scope">
          {s.options_feasible} of {s.options_evaluated} options routed under{' '}
          {matrix.environment_policy.replace(/_/g, ' ')}. Listed as requested;
          not ranked, and none is preferred over another.
        </p>
      </header>

      {s.no_feasible_option && (
        <p className="mx-none" data-testid="mission-matrix-none">
          <b>No feasible option.</b> No candidate destination routed at any
          requested departure time.
          {s.why?.length ? ` ${s.why[0]}` : ''}
        </p>
      )}

      {[...bySite.entries()].map(([siteId, options]) => {
        const site = matrix.sites.find((x) => x.site_id === siteId)
        return (
          <article
            key={siteId}
            className="mx-site"
            data-testid={`mission-site-${siteId}`}
          >
            <h4>
              {site?.label || siteId}
              {site && !site.is_operational && (
                <span className="mx-synthetic" data-testid={`mx-synthetic-${siteId}`}>
                  synthetic — not a station, port, base or waypoint
                </span>
              )}
            </h4>
            {options.map((option) => (
              <div
                key={option.scenario_id}
                className={`mx-option${option.feasible ? '' : ' is-infeasible'}`}
                data-testid={`mission-option-${option.scenario_id}`}
                data-feasible={option.feasible ? 'yes' : 'no'}
              >
                <div className="mx-when">
                  <b>{option.departure_time.replace('T', ' ').slice(0, 16)}</b>
                  <span className="mx-state">
                    {option.feasible ? 'feasible' : 'infeasible'}
                  </span>
                  {onInspect && option.feasible && (
                    <button
                      type="button"
                      className="mx-inspect"
                      data-testid={`mission-inspect-${option.scenario_id}`}
                      onClick={() => onInspect(option.scenario_id)}
                    >
                      Forecast provenance
                    </button>
                  )}
                </div>
                {option.feasible ? (
                  <dl className="mx-rows">
                    {ROW_ORDER.filter((n) => option.objectives[n]).map((n) => (
                      <div key={n} data-testid={`mx-${option.scenario_id}-${n}`}>
                        <dt>{PROFILE_LABEL[n]}</dt>
                        <dd>
                          <Metrics row={option.objectives[n]!} name={n} />
                        </dd>
                      </div>
                    ))}
                  </dl>
                ) : (
                  <p className="mx-reason" data-testid={`mx-reason-${option.scenario_id}`}>
                    <b>{(option.refusal_outcome ?? EMPTY).replace(/_/g, ' ')}</b>
                    {option.refusal_reason ? ` — ${option.refusal_reason}` : ''}
                  </p>
                )}
              </div>
            ))}
          </article>
        )
      })}

      {matrix.limitations.length > 0 && (
        <ul className="mx-limits" data-testid="mission-matrix-limitations">
          {matrix.limitations.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      )}
    </section>
  )
}
