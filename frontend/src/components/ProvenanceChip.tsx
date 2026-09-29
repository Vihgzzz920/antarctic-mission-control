import type { Provenance } from '../api/provenance'
import { formatDate } from '../api/provenance'
import { AlertIcon } from './icons'

interface Props {
  provenance: Provenance
  /** controlled by App, so the routing gate can open this same panel */
  open: boolean
  onToggle: (open: boolean) => void
}

const EMPTY = 'unknown'

/**
 * Where the data on screen came from, from the first paint.
 *
 * Two dates and a status, in a chip; everything else is one click away. The
 * status is stated as what it is -- a deliberate historical demonstration --
 * rather than dressed as a fault, and the chart is never described as the
 * current environment.
 */
export default function ProvenanceChip({ provenance, open, onToggle }: Props) {
  const historical = provenance.status === 'historical_demonstration'
  const unknown = provenance.status === 'unknown'

  return (
    <div className="provenance">
      <button
        type="button"
        className={`prov-chip${historical ? ' is-historical' : ''}${
          unknown ? ' is-unknown' : ''
        }`}
        onClick={() => onToggle(!open)}
        aria-expanded={open}
        data-testid="provenance-chip"
        data-status={provenance.status}
      >
        {historical && <AlertIcon size={12} />}
        <span className="prov-pair">
          <i>Environment</i>
          <b data-testid="provenance-environment">
            {formatDate(provenance.environmentDate) ?? EMPTY}
          </b>
        </span>
        <span className="prov-pair">
          <i>POLARIS chart</i>
          <b data-testid="provenance-chart">
            {formatDate(provenance.chartDate) ?? EMPTY}
          </b>
        </span>
        <span className="prov-status" data-testid="provenance-status">
          {provenance.headline}
        </span>
        <span className="prov-caret" aria-hidden="true">
          {open ? '−' : '+'}
        </span>
      </button>

      {open && (
        <div className="prov-detail" data-testid="provenance-detail">
          {historical && (
            <p className="prov-lede">
              The environment fields and the POLARIS chart are from different
              dates. The chart is a historical demonstration of the ice regime,
              not a description of the environment shown on the map.
            </p>
          )}
          {unknown && (
            <p className="prov-lede">
              The backend has not reported the dates behind these layers.
              Nothing here is assumed in their place.
            </p>
          )}
          {provenance.details.map((section) => (
            <section key={section.heading}>
              <h3>{section.heading}</h3>
              <dl>
                {section.facts.map((fact) => (
                  <div key={fact.label}>
                    <dt>{fact.label}</dt>
                    <dd>{fact.value}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
          {provenance.details.length === 0 && (
            <p className="prov-lede">No provenance metadata has arrived yet.</p>
          )}
        </div>
      )}
    </div>
  )
}
