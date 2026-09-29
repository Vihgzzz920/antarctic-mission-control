// What the simulated event did, read off the one response the backend sent.
//
// Every value below is a field of /api/mission/simulate: the injection it
// placed, and its own before/after comparison of the two route results. The
// only arithmetic here is turning seconds into hours and metres into
// kilometres for display -- no metric is recomputed, no delta is invented, and
// a field the backend left null is shown as unknown rather than filled in from
// somewhere else.
//
// It does not assume the route changed. `path_changed` decides that, and both
// answers are stated as the outcome they are: a search that returned the same
// path under one more iceberg is a result, not a failure.

import type { ProfileName, SimulationResponse } from '../api/types'
import { PROFILE_LABEL } from '../api/types'
import { count, exposure, hours, km, num, timestamp } from './format'

export type RespondVerdict = 'route_replanned' | 'route_unchanged' | 'not_routed'

export interface EvidenceFact {
  label: string
  /** null when the backend did not supply it */
  value: string | null
  /** what the number means, when a number alone would be ambiguous */
  note?: string
}

export interface BeforeAfterRow {
  key: string
  label: string
  before: string | null
  after: string | null
  delta: string | null
  changed: boolean
}

export interface RespondEvidence {
  verdict: RespondVerdict
  /** the one line that should dominate the dock */
  headline: string
  /** a factual sentence built from the backend's own figures */
  summary: string
  event: EvidenceFact[]
  impact: EvidenceFact[]
  comparison: BeforeAfterRow[]
  /** the backend's own note about what the result does and does not claim */
  claims: string
}

/** seconds after departure, as an operator reads a clock */
const elapsed = (seconds: number | null | undefined): string | null => {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return null
  }
  return `+${hours(seconds / 3600)}`
}

const signed = (
  value: number | null,
  render: (input: number) => string,
): string | null => {
  if (value === null || !Number.isFinite(value)) return null
  if (value === 0) return 'no change'
  return `${value > 0 ? '+' : '−'}${render(Math.abs(value))}`
}

interface Metric {
  key: string
  label: string
  render: (value: number) => string
}

//  one row per before/after pair the backend publishes, in the order an
//  operator reads them
const METRICS: Metric[] = [
  { key: 'distance_m', label: 'Distance', render: (v) => km(v / 1000) },
  { key: 'travel_time_s', label: 'Travel time', render: (v) => hours(v / 3600) },
  {
    key: 'iceberg_exposure_contribution',
    label: 'Iceberg exposure contribution',
    render: (v) => num(v),
  },
  {
    key: 'max_iceberg_exposure',
    label: 'Maximum modelled exposure',
    render: (v) => exposure(v),
  },
  { key: 'configured_cost', label: 'Configured cost', render: (v) => num(v) },
  {
    key: 'polaris_contribution',
    label: 'POLARIS contribution',
    render: (v) => num(v),
  },
  {
    key: 'special_consideration_cells',
    label: 'Special-consideration cells',
    render: (v) => count(v),
  },
]

export function buildRespondEvidence(
  simulation: SimulationResponse | null,
  profile: ProfileName,
): RespondEvidence | null {
  if (!simulation?.simulation) return null
  const change = simulation.change.profiles[profile]
  if (!change) return null
  const berg = simulation.change.simulation

  const entries = change as unknown as Record<
    string,
    { before: number | null; after: number | null; delta: number | null }
  >

  const comparison: BeforeAfterRow[] = []
  for (const metric of METRICS) {
    const entry = entries[metric.key]
    //  a metric the response does not carry is not invented; one it carries
    //  with null values keeps its row and shows the unknown marker
    if (!entry) continue
    comparison.push({
      key: metric.key,
      label: metric.label,
      before: entry.before === null ? null : metric.render(entry.before),
      after: entry.after === null ? null : metric.render(entry.after),
      //  the delta is the backend's own; it is never derived here
      delta: signed(entry.delta, metric.render),
      changed: typeof entry.delta === 'number' && entry.delta !== 0,
    })
  }

  const touched = change.cells_with_simulated_exposure
  const verdict: RespondVerdict = !change.routed_after
    ? 'not_routed'
    : change.path_changed
      ? 'route_replanned'
      : 'route_unchanged'

  const headline =
    verdict === 'not_routed'
      ? 'No route returned'
      : verdict === 'route_replanned'
        ? 'Route replanned'
        : 'Route unchanged'

  const cells =
    touched === null
      ? 'an unreported number of cells'
      : `${touched} cell${touched === 1 ? '' : 's'}`
  const distance = change.distance_m.delta
  const time = change.travel_time_s.delta
  const summary =
    verdict === 'not_routed'
      ? `The simulated iceberg raised modelled exposure on ${cells} of the ` +
        `current route, and the search returned no route under the same ` +
        `configuration.`
      : verdict === 'route_replanned'
        ? `The simulated iceberg raised modelled exposure on ${cells} of the ` +
          `current route, and the configured route search returned a ` +
          `different path` +
          (typeof distance === 'number' && typeof time === 'number'
            ? `: ${signed(distance, (v) => km(v / 1000))} and ` +
              `${signed(time, (v) => hours(v / 3600))}.`
            : '.')
        : `The simulated iceberg raised modelled exposure on ${cells} of the ` +
          `current route, and the configured route search returned the same ` +
          `path.`

  const event: EvidenceFact[] = [
    {
      label: 'Event',
      value: berg.label,
      note: berg.written_to_any_dataset
        ? undefined
        : 'a scenario; written to no dataset',
    },
    {
      label: 'Encounter cell',
      value: berg.encounter_cell ? berg.encounter_cell.join(', ') : null,
      note: 'the cell of the current route it was placed against',
    },
    {
      label: 'Predicted arrival',
      value: elapsed(berg.encounter_arrival_s),
      note: `after departure — ${timestamp(berg.encounter_datetime)}`,
    },
    {
      label: 'Exposure field',
      value:
        typeof berg.horizon_hours === 'number' ? `+${berg.horizon_hours}h` : null,
      note:
        typeof berg.from_bucket === 'number'
          ? `time bucket ${berg.from_bucket}: the field this cell was priced against`
          : undefined,
    },
    {
      label: 'Modelled uncertainty',
      value: typeof berg.radius_km === 'number' ? km(berg.radius_km) : null,
      note: (berg as { radius_source?: string }).radius_source,
    },
    {
      label: 'Distance to the route',
      value:
        typeof berg.distance_to_route_m !== 'number'
          ? null
          : berg.distance_to_route_m === 0
            ? '0 m — on a route cell'
            : `${num(berg.distance_to_route_m, 0)} m`,
    },
    {
      label: 'Written to any dataset',
      value: berg.written_to_any_dataset ? 'yes' : 'no',
    },
  ]

  const impact: EvidenceFact[] = [
    {
      label: 'Affected cells',
      value: touched === null ? null : String(touched),
      note: 'cells of the current route the simulated exposure reaches',
    },
    {
      label: 'Maximum modelled exposure',
      value:
        change.max_iceberg_exposure.before === null ||
        change.max_iceberg_exposure.after === null
          ? null
          : `${exposure(change.max_iceberg_exposure.before)} → ${exposure(
              change.max_iceberg_exposure.after,
            )}`,
      note: 'a project exposure index in [0, 1]',
    },
    {
      label: 'Exposure contribution',
      value:
        change.iceberg_exposure_contribution.before === null ||
        change.iceberg_exposure_contribution.after === null
          ? null
          : `${num(change.iceberg_exposure_contribution.before)} → ${num(
              change.iceberg_exposure_contribution.after,
            )}`,
      note: 'the part of the configured cost this exposure accounts for',
    },
    {
      label: 'Path',
      value: change.path_changed ? 'changed' : 'unchanged',
    },
    {
      label: 'Routed before and after',
      value: `${change.routed_before ? 'yes' : 'no'} · ${
        change.routed_after ? 'yes' : 'no'
      }`,
    },
  ]

  return {
    verdict,
    headline,
    summary: `${PROFILE_LABEL[profile]}: ${summary}`,
    event,
    impact,
    comparison,
    //  the backend's own words when it sends them; otherwise a sentence that
    //  claims nothing, so the panel is never silent about what this is
    claims:
      typeof simulation.change.claims.note === 'string'
        ? simulation.change.claims.note
        : 'This is a scenario, not a statement that any route avoids ' +
          'anything, and not a prediction that an iceberg will be there.',
  }
}
