import type { ProfileName, SimulationResponse } from '../api/types'
import { buildRespondEvidence } from './respondEvidence'
import type { EvidenceFact } from './respondEvidence'
import { IcebergIcon } from './icons'

interface Props {
  simulation: SimulationResponse
  profile: ProfileName
}

const EMPTY = '—'

function Facts({ facts, testId }: { facts: EvidenceFact[]; testId: string }) {
  return (
    <dl className="impact-facts" data-testid={testId}>
      {facts.map((fact) => (
        <div key={fact.label}>
          <dt>{fact.label}</dt>
          <dd>
            <b>{fact.value ?? EMPTY}</b>
            {fact.note && <i>{fact.note}</i>}
          </dd>
        </div>
      ))}
    </dl>
  )
}

/**
 * What the simulated event did to the current mission.
 *
 * Every figure comes from respondEvidence.buildRespondEvidence, which reads
 * the one simulation response and derives nothing the backend did not send.
 * This component only lays it out.
 *
 * It does not assume the path changed: "Route unchanged" is a real answer from
 * a real search and is reported as plainly as the other one.
 */
export default function RouteImpact({ simulation, profile }: Props) {
  const evidence = buildRespondEvidence(simulation, profile)
  if (!evidence) return null
  const affected = evidence.impact.find((fact) => fact.label === 'Affected cells')

  return (
    <section className="impact" data-testid="route-impact">
      <div className="impact-verdict" data-state={evidence.verdict}>
        <span className="impact-eyebrow">Decision</span>
        <strong data-testid="impact-verdict">{evidence.headline}</strong>
        <p data-testid="impact-sentence">{evidence.summary}</p>
      </div>

      <h3>Simulated event</h3>
      <div className="impact-chip">
        <IcebergIcon size={13} />
        <span>{simulation.change.simulation.label}</span>
        <span className="impact-chip-note">
          placed by the operator; temporary, and written to no dataset
        </span>
      </div>
      <Facts facts={evidence.event} testId="impact-event" />

      <h3>Route impact</h3>
      <span hidden data-testid="impact-affected">
        {affected?.value ?? EMPTY}
      </span>
      <Facts facts={evidence.impact} testId="impact-summary" />

      <h3>Before → after</h3>
      <table className="impact-table" data-testid="impact-table">
        <thead>
          <tr>
            <th scope="col">Metric</th>
            <th scope="col">Before</th>
            <th scope="col">After</th>
            <th scope="col">Change</th>
          </tr>
        </thead>
        <tbody>
          {evidence.comparison.map((row) => (
            <tr
              key={row.key}
              data-metric={row.key}
              className={row.changed ? 'is-changed' : undefined}
            >
              <th scope="row">{row.label}</th>
              <td>{row.before ?? EMPTY}</td>
              <td>{row.after ?? EMPTY}</td>
              <td>{row.delta ?? EMPTY}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="impact-foot" data-testid="impact-claims">
        {evidence.claims}
      </p>
    </section>
  )
}
