import { PROFILE_LABEL } from '../api/types'
import type { Comparison, ProfileName, SimulationResponse } from '../api/types'
import { buildFindings, identicalTo } from './findings'
import { count, exposure, num } from './format'
import WhyRouteChanged from './WhyRouteChanged'

interface Props {
  comparison: Comparison
  profile: ProfileName
  simulation: SimulationResponse | null
  open: boolean
  onClose: () => void
}

/** A sliding detail panel. Closed by default; nothing here is permanent. */
export default function WhyDrawer({
  comparison,
  profile,
  simulation,
  open,
  onClose,
}: Props) {
  if (!open) return null
  const route = comparison.profiles[profile]
  if (!route?.success) return null

  const findings = buildFindings(comparison, profile)
  const twins = identicalTo(comparison, profile)

  return (
    <aside className="why-drawer" data-testid="why-panel">
      <header>
        <h2>Why this route</h2>
        <button type="button" onClick={onClose} aria-label="Close">×</button>
      </header>

      <div className="why-body">
        <p className="why-lede">
          <strong>{PROFILE_LABEL[profile]}</strong> minimises{' '}
          <em>{route.objective}</em>, measured in {route.objective_units}.
        </p>

        <dl className="why-grid">
          <div>
            <dt>Environmental</dt>
            <dd>{num(route.environmental_contribution)}</dd>
          </div>
          <div>
            <dt>POLARIS</dt>
            <dd>{num(route.polaris_contribution)}</dd>
          </div>
          <div>
            <dt>Iceberg exposure</dt>
            <dd>{num(route.iceberg_exposure_contribution)}</dd>
          </div>
          <div>
            <dt>Max modelled exposure</dt>
            <dd>{exposure(route.max_iceberg_exposure)}</dd>
          </div>
          <div>
            <dt>Special consideration</dt>
            <dd>{count(route.special_consideration_cells)} cells</dd>
          </div>
          <div>
            <dt>Indeterminate</dt>
            <dd>{count(route.indeterminate_cells)} cells</dd>
          </div>
        </dl>

        {findings.length > 0 ? (
          <ul className="findings">
            {findings.map((finding) => (
              <li key={finding.key} className={`finding is-${finding.direction}`}>
                <span className="finding-rule" aria-hidden="true" />
                {finding.text}
              </li>
            ))}
          </ul>
        ) : (
          <p className="why-note">
            No measured difference from the other objectives on these inputs.
          </p>
        )}

        {twins.length > 0 && (
          <p className="why-note">
            {twins.map((name) => PROFILE_LABEL[name]).join(' and ')} returned
            the same path: with one constant vessel speed, travel time is a
            strictly increasing function of path length.
          </p>
        )}

        {simulation && (
          <WhyRouteChanged simulation={simulation} profile={profile} />
        )}
      </div>
    </aside>
  )
}
