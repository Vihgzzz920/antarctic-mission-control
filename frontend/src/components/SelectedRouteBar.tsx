import { PROFILE_LABEL } from '../api/types'
import type { Comparison, ProfileName } from '../api/types'
import { PROFILE_COLOUR } from '../map/layers'
import { hours, km, num } from './format'
import { PROFILE_ICON } from './icons'

interface Props {
  comparison: Comparison
  profile: ProfileName
  /** "Selected route", or "Replanned route" while a simulation is active */
  label?: string
  whyOpen: boolean
  onWhy: () => void
  transit?: React.ReactNode
}

/** A slim band: name and four figures. Everything else lives behind Why. */
export default function SelectedRouteBar({
  comparison,
  profile,
  label = 'Selected route',
  whyOpen,
  onWhy,
  transit,
}: Props) {
  const route = comparison.profiles[profile]
  if (!route?.success) return null
  const Icon = PROFILE_ICON[profile]

  return (
    <div
      className="selected-bar"
      style={{ '--accent': PROFILE_COLOUR[profile] } as React.CSSProperties}
      data-testid="selected-route-bar"
    >
      <span className="selected-icon"><Icon size={15} /></span>
      <span className="selected-name">
        <i>{label}</i>
        {PROFILE_LABEL[profile]}
      </span>

      <span className="bar-figures">
        <span><b>{km(route.distance_km)}</b><i>distance</i></span>
        <span><b>{hours(route.travel_time_h)}</b><i>travel time</i></span>
        <span><b>{num(route.polaris_contribution)}</b><i>POLARIS</i></span>
        <span>
          <b>{num(route.iceberg_exposure_contribution)}</b>
          <i>iceberg exposure</i>
        </span>
      </span>

      {transit}

      <button
        type="button"
        className={`why-toggle${whyOpen ? ' is-open' : ''}`}
        onClick={onWhy}
        aria-expanded={whyOpen}
        data-testid="why-toggle"
      >
        Why this route
      </button>
    </div>
  )
}
