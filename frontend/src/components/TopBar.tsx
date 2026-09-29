import type { Provenance } from '../api/provenance'
import ProvenanceChip from './ProvenanceChip'

export type Stage = 'observe' | 'forecast' | 'plan' | 'respond'

const STEPS: Array<{ id: Stage; label: string }> = [
  { id: 'observe', label: 'Observe' },
  { id: 'forecast', label: 'Forecast' },
  { id: 'plan', label: 'Plan' },
  { id: 'respond', label: 'Respond' },
]

interface Props {
  stage: Stage
  reached: Stage[]
  online: boolean
  /** where the data on screen came from; shown from the first paint */
  provenance: Provenance
  provenanceOpen: boolean
  onProvenanceToggle: (open: boolean) => void
  onGo: (stage: Stage) => void
}

export default function TopBar({
  stage,
  reached,
  online,
  provenance,
  provenanceOpen,
  onProvenanceToggle,
  onGo,
}: Props) {
  return (
    <header className="topbar">
      <div className="topbar-brand">
        <svg viewBox="0 0 24 24" width="19" height="19" aria-hidden="true">
          <circle cx="12" cy="12" r="9.2" fill="none" stroke="currentColor"
                  strokeWidth="1.2" />
          <path d="M12 6.4 15.2 12 12 17.6 8.8 12z" fill="currentColor" />
        </svg>
        <span>Antarctic Mission Control</span>
      </div>

      <nav className="stages" aria-label="Mission flow">
        {STEPS.map((step, index) => {
          const done = reached.includes(step.id)
          const current = stage === step.id
          return (
            <button
              key={step.id}
              type="button"
              className={`stage-step${current ? ' is-current' : ''}${
                done ? ' is-reached' : ''
              }`}
              onClick={() => done && onGo(step.id)}
              disabled={!done}
              aria-current={current ? 'step' : undefined}
              data-testid={`stage-${step.id}`}
            >
              <span className="stage-index">{index + 1}</span>
              {step.label}
            </button>
          )
        })}
      </nav>

      <div className="topbar-meta">
        <ProvenanceChip
          provenance={provenance}
          open={provenanceOpen}
          onToggle={onProvenanceToggle}
        />
        <span className={`live-chip${online ? '' : ' is-down'}`}>
          <span className="live-dot" aria-hidden="true" />
          {online ? 'Decision support prototype' : 'API unavailable'}
        </span>
      </div>
    </header>
  )
}
