import { PROFILE_LABEL } from '../api/types'
import type { ProfileName, SimulationResponse } from '../api/types'
import { exposure, hours, km, num, timestamp } from './format'
import { IcebergIcon } from './icons'

interface Props {
  simulation: SimulationResponse
  profile: ProfileName
}

const MINUTES = (seconds: number) => `${Math.round(Math.abs(seconds) / 60)} min`

export default function WhyRouteChanged({ simulation, profile }: Props) {
  const change = simulation.change.profiles[profile]
  const berg = simulation.change.simulation
  if (!change) return null

  const lines: string[] = []

  if (change.cells_with_simulated_exposure !== null) {
    lines.push(
      `The simulated iceberg created modelled exposure on ` +
        `${change.cells_with_simulated_exposure} cell` +
        `${change.cells_with_simulated_exposure === 1 ? '' : 's'} of the ` +
        `current route.`,
    )
  }

  const max = change.max_iceberg_exposure
  if (max.before !== null && max.after !== null && max.before !== max.after) {
    lines.push(
      `Maximum modelled exposure on the route changed from ` +
        `${exposure(max.before)} to ${exposure(max.after)}.`,
    )
  }

  const berg_c = change.iceberg_exposure_contribution
  if (berg_c.before !== null && berg_c.after !== null) {
    lines.push(
      `Iceberg exposure contribution changed from ${num(berg_c.before)} to ` +
        `${num(berg_c.after)}.`,
    )
  }

  if (change.path_changed) {
    const dKm = change.distance_m.delta
    const dS = change.travel_time_s.delta
    if (dKm !== null && dS !== null) {
      const longer = dKm > 0
      lines.push(
        `The replanned route ${longer ? 'adds' : 'removes'} ` +
          `${num(Math.abs(dKm) / 1000, 2)} km and ${MINUTES(dS)}.`,
      )
    }
  } else {
    lines.push(
      `The route did not change: the added exposure did not outweigh the cost ` +
        `of leaving this corridor under the configured weights.`,
    )
  }

  return (
    <section className="why-changed" data-testid="why-route-changed">
      <div className="deck-head">
        <h2>Why did the route change?</h2>
        <span className="deck-head-note">{PROFILE_LABEL[profile]}</span>
      </div>

      <div className="sim-chip">
        <IcebergIcon size={13} />
        <span>{berg.label}</span>
        <span className="sim-chip-detail">
          {berg.radius_km.toFixed(2)} km uncertainty radius · placed at bucket{' '}
          {berg.from_bucket}
        </span>
      </div>

      <ul className="findings">
        {lines.map((line) => (
          <li key={line} className="finding is-higher">
            <span className="finding-rule" aria-hidden="true" />
            {line}
          </li>
        ))}
      </ul>

      <dl className="why-grid">
        <div>
          <dt>Current distance</dt>
          <dd>{km((change.distance_m.before ?? 0) / 1000)}</dd>
        </div>
        <div>
          <dt>Replanned distance</dt>
          <dd>{km((change.distance_m.after ?? 0) / 1000)}</dd>
        </div>
        <div>
          <dt>Current travel time</dt>
          <dd>{hours((change.travel_time_s.before ?? 0) / 3600)}</dd>
        </div>
        <div>
          <dt>Replanned travel time</dt>
          <dd>{hours((change.travel_time_s.after ?? 0) / 3600)}</dd>
        </div>
      </dl>

      <p className="why-foot">
        Encounter taken from the current route's own predicted arrival at cell{' '}
        {berg.encounter_cell.join(', ')}, {timestamp(berg.encounter_datetime)}.
        The simulated iceberg is temporary and was written to no dataset. This
        is a scenario, not a statement that any route avoids anything.
      </p>
    </section>
  )
}
