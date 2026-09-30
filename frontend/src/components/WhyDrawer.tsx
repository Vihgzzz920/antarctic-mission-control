import { PROFILE_LABEL } from '../api/types'
import type { Comparison, ProfileName } from '../api/types'
import { environmentBuckets, forecastLeads } from '../api/routeEnvironment'
import ForecastProvenanceList from './ForecastProvenanceList'
import { buildFindings, identicalTo } from './findings'
import { count, exposure, num, owe, owePerKm } from './format'

interface Props {
  comparison: Comparison
  profile: ProfileName
  /** which exposure fields priced this route, and how far the forecast runs
   *  past them; from map/pricedHorizon.ts */
  pricedSummary: string | null
  open: boolean
  onClose: () => void
}

/** A sliding detail panel. Closed by default; nothing here is permanent. */
export default function WhyDrawer({
  comparison,
  profile,
  pricedSummary,
  open,
  onClose,
}: Props) {
  if (!open) return null
  const route = comparison.profiles[profile]
  if (!route?.success) return null

  const findings = buildFindings(comparison, profile)
  const twins = identicalTo(comparison, profile)
  const buckets = environmentBuckets(route)
  const leads = forecastLeads(route)
  const fuel = route.fuel_provenance ?? null

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

        {pricedSummary && (
          <p className="why-note" data-testid="why-priced-horizon">
            {pricedSummary}
          </p>
        )}

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

        {/*  What the estimated fuel figure means, where there is one.  */}
        {route.estimated_fuel !== null &&
          route.estimated_fuel !== undefined && (
            <section className="why-section" data-testid="why-fuel">
              <h3>Estimated fuel</h3>
              <p className="why-note">
                <b>{owe(route.estimated_fuel)}</b>
                {route.estimated_fuel_per_km !== null &&
                  route.estimated_fuel_per_km !== undefined &&
                  ` · ${owePerKm(route.estimated_fuel_per_km)}`}
              </p>
              {fuel && (
                <dl className="why-grid" data-testid="why-fuel-assumptions">
                  <div>
                    <dt>Fuel model</dt>
                    <dd>Estimated relative proxy</dd>
                  </div>
                  <div>
                    <dt>Measured consumption</dt>
                    <dd>{fuel.is_a_measured_fuel_consumption ? 'yes' : 'no'}</dd>
                  </div>
                  <div>
                    <dt>Operational prediction</dt>
                    <dd>
                      {fuel.is_an_operational_fuel_prediction ? 'yes' : 'no'}
                    </dd>
                  </div>
                  <div>
                    <dt>Absolute volume or mass</dt>
                    <dd>unavailable</dd>
                  </div>
                </dl>
              )}
            </section>
          )}

        {/*  "What forecast actually affected this route?"  */}
        <section className="why-section" data-testid="why-forecast-provenance">
          <h3>
            Forecast provenance
            {leads.length > 0 && (
              <span className="why-leads">
                {' '}
                {leads.map((l) => `+${l}h`).join(', ')}
              </span>
            )}
          </h3>
          <ForecastProvenanceList buckets={buckets} />
        </section>
      </div>
    </aside>
  )
}
