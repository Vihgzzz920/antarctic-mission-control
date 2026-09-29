import type { ProfileName, SimulationResponse } from '../api/types'
import RouteImpact from './RouteImpact'

interface Props {
  simulation: SimulationResponse | null
  profile: ProfileName
  open: boolean
  onClose: () => void
}

/** The evidence for what the simulated event did, in the same sliding shell
 *  the route drawer uses. Every figure inside is read off the one simulation
 *  response; nothing is recomputed here. */
export default function RespondDrawer({
  simulation,
  profile,
  open,
  onClose,
}: Props) {
  if (!open || !simulation) return null
  return (
    <aside className="why-drawer" data-testid="respond-drawer">
      <header>
        <h2>Route impact</h2>
        <button type="button" onClick={onClose} aria-label="Close">
          ×
        </button>
      </header>
      <div className="why-body">
        <RouteImpact simulation={simulation} profile={profile} />
      </div>
    </aside>
  )
}
