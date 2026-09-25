import { PROFILE_LABEL, PROFILE_ORDER } from '../api/types'
import type { Comparison, ProfileName } from '../api/types'
import { PROFILE_COLOUR } from '../map/layers'
import { exposure, hours, km, num } from './format'
import { PROFILE_ICON } from './icons'

const OBJECTIVE: Record<ProfileName, string> = {
  fastest: 'Minimises travel time',
  risk_oriented: 'Minimises the configured navigation cost',
  shortest_distance: 'Minimises travelled distance',
}

interface Props {
  comparison: Comparison
  selected: ProfileName | null
  expanded: ProfileName | null
  onSelect: (profile: ProfileName) => void
  onExpand: (profile: ProfileName | null) => void
}

/** Three compact options. Key metrics only until one is opened. */
export default function PlanDock({
  comparison,
  selected,
  expanded,
  onSelect,
  onExpand,
}: Props) {
  return (
    <div className="plan-dock" data-testid="plan-dock">
      {PROFILE_ORDER.map((name) => {
        const profile = comparison.profiles[name]
        if (!profile) return null
        const Icon = PROFILE_ICON[name]
        const open = expanded === name
        return (
          <div
            key={name}
            className={`option${name === selected ? ' is-selected' : ''}${
              open ? ' is-open' : ''
            }`}
            style={{ '--accent': PROFILE_COLOUR[name] } as React.CSSProperties}
            data-testid={`route-option-${name}`}
          >
            <button
              type="button"
              className="option-main"
              aria-pressed={name === selected}
              data-testid={`route-card-${name}`}
              onClick={() => {
                onSelect(name)
                onExpand(open ? null : name)
              }}
            >
              <span className="option-icon"><Icon size={15} /></span>
              <span className="option-name">{PROFILE_LABEL[name]}</span>
              {profile.success ? (
                <span className="option-keys">
                  <span><b>{km(profile.distance_km)}</b></span>
                  <span><b>{hours(profile.travel_time_h)}</b></span>
                  <span>
                    <b>{num(profile.configured_cost)}</b>
                    <i>cost</i>
                  </span>
                </span>
              ) : (
                <span className="option-failed">
                  {profile.outcome.replace(/_/g, ' ')}
                </span>
              )}
            </button>

            {open && profile.success && (
              <dl className="option-detail" data-testid={`route-detail-${name}`}>
                <div><dt>Objective</dt><dd>{OBJECTIVE[name]}</dd></div>
                <div>
                  <dt>POLARIS</dt>
                  <dd>{num(profile.polaris_contribution)}</dd>
                </div>
                <div>
                  <dt>Iceberg exposure</dt>
                  <dd>{num(profile.iceberg_exposure_contribution)}</dd>
                </div>
                <div>
                  <dt>Max modelled exposure</dt>
                  <dd>{exposure(profile.max_iceberg_exposure)}</dd>
                </div>
              </dl>
            )}
          </div>
        )
      })}
    </div>
  )
}
