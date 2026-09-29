import { PROFILE_LABEL } from '../api/types'
import type { Comparison, ProfileName, RouteProfile } from '../api/types'
import type { RouteGroup } from '../api/routeEquivalence'
import {
  equivalenceDetail,
  equivalenceNote,
  figuresProfile,
  groupLabel,
} from '../api/routeEquivalence'
import { PROFILE_COLOUR } from '../map/layers'
import { exposure, hours, km, num, owe, owePerKm } from './format'
import { PROFILE_ICON } from './icons'

/**
 * Estimated fuel, and the four things an operator must know about it before
 * reading the number. Rendered from the backend's own fuel provenance; nothing
 * here is inferred, and no volume or mass is ever shown.
 */
function FuelDetail({ profile }: { profile: RouteProfile }) {
  if (profile.estimated_fuel === null || profile.estimated_fuel === undefined) {
    return null
  }
  const p = profile.fuel_provenance ?? null
  return (
    <>
      <div data-testid="fuel-estimate">
        <dt>Estimated fuel</dt>
        <dd>
          {owe(profile.estimated_fuel)}
          {profile.estimated_fuel_per_km !== null &&
            profile.estimated_fuel_per_km !== undefined && (
              <span className="detail-sub">
                {' '}
                · {owePerKm(profile.estimated_fuel_per_km)}
              </span>
            )}
        </dd>
      </div>
      {p && (
        <div className="detail-note" data-testid="fuel-assumptions">
          <dt>Fuel model</dt>
          <dd>
            <b>Estimated relative proxy.</b> Measured fuel consumption:{' '}
            {p.is_a_measured_fuel_consumption ? 'yes' : 'no'}. Operational fuel
            prediction: {p.is_an_operational_fuel_prediction ? 'yes' : 'no'}.
            Absolute volume or mass: unavailable. Comparable only between{' '}
            {p.comparable_only_within}.
          </dd>
        </div>
      )}
    </>
  )
}

const OBJECTIVE: Record<ProfileName, string> = {
  fastest: 'Minimises travel time',
  fuel_efficient: 'Minimises the estimated fuel proxy (OWE-m)',
  risk_oriented: 'Minimises the configured navigation cost',
  shortest_distance: 'Minimises travelled distance',
}

interface Props {
  comparison: Comparison
  /**
   * the profiles grouped by the geometry the backend actually returned; see
   * api/routeEquivalence.ts. One card per group, so a path the search returned
   * once is offered once.
   */
  groups: RouteGroup[]
  selected: ProfileName | null
  expanded: ProfileName | null
  onSelect: (profile: ProfileName) => void
  onExpand: (profile: ProfileName | null) => void
  /** the temporal provenance of the selected route's iceberg pricing */
  provenance: string | null
  /** the mission's own vessel speed, named in the equivalence note */
  vesselSpeedMps?: number | null
  /** RESPOND is reachable only once a route exists */
  canRespond: boolean
  onRespond: () => void
}

/**
 * The route options, one card per distinct geometry. Objectives that returned
 * the same path share a card and say so; each objective stays separately
 * selectable and keeps its own backend figures.
 */
export default function PlanDock({
  comparison,
  groups,
  selected,
  expanded,
  onSelect,
  onExpand,
  provenance,
  vesselSpeedMps,
  canRespond,
  onRespond,
}: Props) {
  return (
    <div className="plan-dock" data-testid="plan-dock">
      <div className="dock-head">
        {provenance && (
          <p className="dock-provenance" data-testid="priced-horizon">
            {provenance}
          </p>
        )}
        {canRespond && (
          <button
            type="button"
            className="primary-action"
            onClick={onRespond}
            data-testid="go-respond"
          >
            Respond
          </button>
        )}
      </div>
      {groups.map((group) => {
        const lead = figuresProfile(group, selected)
        const figures = comparison.profiles[lead]
        if (!figures) return null
        const open = expanded !== null && group.profiles.includes(expanded)
        const openProfile = open ? (expanded as ProfileName) : null
        const chosen = selected !== null && group.profiles.includes(selected)
        //  one figure line for the group only when every objective in it
        //  reports the same numbers; otherwise each keeps its own
        const shareFigures = group.shared && group.figuresAgree && figures.success
        const note = equivalenceNote(group, vesselSpeedMps)

        return (
          <div
            key={group.key}
            className={`option${chosen ? ' is-selected' : ''}${
              open ? ' is-open' : ''
            }${group.shared ? ' is-shared' : ''}`}
            style={{ '--accent': PROFILE_COLOUR[lead] } as React.CSSProperties}
            data-testid={`route-option-${group.key}`}
          >
            {group.shared && (
              <div className="option-shared" data-testid={`route-group-${group.key}`}>
                <span className="shared-names">{groupLabel(group)}</span>
                {note && <span className="shared-note">{note}</span>}
                {shareFigures && (
                  <span className="option-keys">
                    <span><b>{km(figures.distance_km)}</b></span>
                    <span><b>{hours(figures.travel_time_h)}</b></span>
                    <span>
                      <b>{num(figures.configured_cost)}</b>
                      <i>cost</i>
                    </span>
                  </span>
                )}
              </div>
            )}

            <div className={group.shared ? 'option-choices' : undefined}>
              {group.profiles.map((name) => {
                const profile = comparison.profiles[name]
                if (!profile) return null
                const Icon = PROFILE_ICON[name]
                return (
                  <button
                    key={name}
                    type="button"
                    className={`option-main${
                      name === selected ? ' is-chosen' : ''
                    }`}
                    aria-pressed={name === selected}
                    data-testid={`route-card-${name}`}
                    onClick={() => {
                      onSelect(name)
                      onExpand(expanded === name ? null : name)
                    }}
                  >
                    <span className="option-icon"><Icon size={15} /></span>
                    <span className="option-name">{PROFILE_LABEL[name]}</span>
                    {profile.success ? (
                      shareFigures ? null : (
                        <span className="option-keys">
                          <span><b>{km(profile.distance_km)}</b></span>
                          <span><b>{hours(profile.travel_time_h)}</b></span>
                          {name === 'fuel_efficient' ? (
                            <span>
                              <b>{num(profile.estimated_fuel)}</b>
                              <i>OWE-m est.</i>
                            </span>
                          ) : (
                            <span>
                              <b>{num(profile.configured_cost)}</b>
                              <i>cost</i>
                            </span>
                          )}
                        </span>
                      )
                    ) : (
                      <span className="option-failed">
                        {profile.outcome.replace(/_/g, ' ')}
                      </span>
                    )}
                  </button>
                )
              })}
            </div>

            {openProfile && comparison.profiles[openProfile]?.success && (
              <dl
                className="option-detail"
                data-testid={`route-detail-${openProfile}`}
              >
                <div><dt>Objective</dt><dd>{OBJECTIVE[openProfile]}</dd></div>
                {group.shared && (
                  <div className="detail-note">
                    <dt>Same geometry</dt>
                    <dd>{equivalenceDetail(group, comparison)}</dd>
                  </div>
                )}
                <div>
                  <dt>POLARIS</dt>
                  <dd>{num(comparison.profiles[openProfile].polaris_contribution)}</dd>
                </div>
                <div>
                  <dt>Iceberg exposure</dt>
                  <dd>
                    {num(
                      comparison.profiles[openProfile]
                        .iceberg_exposure_contribution,
                    )}
                  </dd>
                </div>
                <div>
                  <dt>Max modelled exposure</dt>
                  <dd>
                    {exposure(comparison.profiles[openProfile].max_iceberg_exposure)}
                  </dd>
                </div>
                <FuelDetail profile={comparison.profiles[openProfile]} />
              </dl>
            )}
          </div>
        )
      })}
    </div>
  )
}
