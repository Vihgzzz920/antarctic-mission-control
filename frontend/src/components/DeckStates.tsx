import { useEffect, useState } from 'react'

import { OUTCOME_COPY, PROFILE_LABEL, PROFILE_ORDER } from '../api/types'
import { PROFILE_COLOUR } from '../map/layers'
import { AlertIcon, PROFILE_ICON } from './icons'

/** Before Generate: what the three objectives are, and what will appear. */
export function MissionSetup() {
  return (
    <section className="setup" data-testid="mission-setup">
      <div className="deck-head">
        <h2>Mission setup</h2>
        <span className="deck-head-note">
          Set departure, destination and vessel, then generate
        </span>
      </div>
      <p className="setup-lede">
        The planner runs the same departure, destination, clock and hard
        constraints through three objectives. Each returns its own route; the
        map draws all three and this deck fills with their measured metrics.
      </p>
      <div className="setup-row">
        {PROFILE_ORDER.map((name) => {
          const Icon = PROFILE_ICON[name]
          return (
            <div
              key={name}
              className="setup-card"
              style={{ '--accent': PROFILE_COLOUR[name] } as React.CSSProperties}
            >
              <span className="setup-icon"><Icon size={16} /></span>
              <span className="setup-name">{PROFILE_LABEL[name]}</span>
              <span className="setup-pending">awaiting run</span>
            </div>
          )
        })}
      </div>
    </section>
  )
}

const STEPS = [
  'Loading environment and hard constraints',
  'Pricing cells at their arrival time',
  'Searching the time-dependent graph',
  'Measuring POLARIS and iceberg contributions',
]

/** Real work, honestly paced: a step list, no invented percentage. */
export function PlanningState() {
  const [step, setStep] = useState(0)
  useEffect(() => {
    const id = window.setInterval(
      () => setStep((current) => Math.min(current + 1, STEPS.length - 1)),
      900,
    )
    return () => window.clearInterval(id)
  }, [])

  return (
    <section className="planning" data-testid="planning-state">
      <div className="deck-head">
        <h2>Planning</h2>
        <span className="deck-head-note">Three objectives, one search each</span>
      </div>
      <div className="planning-track" aria-hidden="true">
        <span className="planning-sweep" />
      </div>
      <ol className="planning-steps">
        {STEPS.map((label, index) => (
          <li
            key={label}
            className={
              index < step ? 'is-done' : index === step ? 'is-active' : ''
            }
          >
            <span className="planning-dot" aria-hidden="true" />
            {label}
          </li>
        ))}
      </ol>
    </section>
  )
}

/** A refusal or an outage, stated as itself. Never replaced with a route. */
export function DeckFailure({
  code,
  message,
}: {
  code?: string
  message: string
}) {
  const copy = code ? OUTCOME_COPY[code] : undefined
  return (
    <section className="failure" data-testid="status-banner">
      <AlertIcon size={18} className="failure-glyph" />
      <div>
        <h2>{copy?.title ?? 'Routing refused'}</h2>
        <p>{copy?.detail ?? message}</p>
        {copy && message && copy.detail !== message && (
          <p className="failure-detail">{message}</p>
        )}
      </div>
    </section>
  )
}
