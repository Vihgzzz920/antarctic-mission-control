// Where everything on screen came from, and when.
//
// The project deliberately runs environment fields from one date against a
// USNIC chart from another, and must never let that pass unremarked. This
// assembles the answer
// from metadata the API already publishes -- nothing here is a date, a chart
// name or a horizon written into the frontend:
//
//   /api/health          grid.environment_date, grid.usnic_chart_date,
//                        grid.dates_aligned, requires_historical_demo_override,
//                        polaris.selection (chart, RIV table and its source),
//                        exposure_layers (the fields routing can price with)
//   /api/mission/defaults the demonstration's own inputs
//   /api/forecast/icebergs horizons, drift model, uncertainty calibration,
//                        population note
//   route response       comparison.provenance for the mission that ran
//
// A value the API did not supply becomes `null` and is rendered as unknown.
// It is never filled in from memory.

import type { ForecastResponse, HealthResponse, MissionResponse } from './types'

export type ProvenanceStatus =
  | 'historical_demonstration'
  | 'contemporaneous'
  | 'unknown'

export interface ProvenanceFact {
  label: string
  /** null when the API did not supply it */
  value: string | null
}

export interface Provenance {
  status: ProvenanceStatus
  /** the environment fields' own date, as the backend reports it */
  environmentDate: string | null
  /** the USNIC/SIGRID-3 chart's date */
  chartDate: string | null
  /** the chart the POLARIS sidecar was built from, e.g. its published id */
  chartName: string | null
  /** true only when the backend says the dates are aligned */
  datesAligned: boolean | null
  /** whether this pairing needs the explicit historical override */
  requiresOverride: boolean | null
  /** the override the mission actually ran with, once one has run */
  overrideInEffect: boolean | null
  /** one short line an operator can read in a couple of seconds */
  headline: string
  /** the detail panel's rows, already filtered to what the API supplied */
  details: Array<{ heading: string; facts: ProvenanceFact[] }>
}

const text = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() !== '' ? value : null

const number = (value: unknown): number | null =>
  Number.isFinite(Number(value)) && value !== null && value !== ''
    ? Number(value)
    : null

/** An ISO date from the backend, rendered day-month-year in small caps form.
 *  Anything that is not an ISO date is passed through exactly as given. */
export function formatDate(iso: string | null): string | null {
  if (!iso) return null
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso)
  if (!match) return iso
  const months = [
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
  ]
  const month = months[Number(match[2]) - 1]
  if (!month) return iso
  return `${match[3]} ${month} ${match[1]}`
}

export interface ProvenanceInputs {
  health: HealthResponse | null
  forecast: ForecastResponse | null
  /** the mission that ran, if one has */
  mission: MissionResponse | null
  /** the horizons the routing configuration can price against, in hours */
  configuredPricedHours: number[]
}

export function buildProvenance(inputs: ProvenanceInputs): Provenance {
  const { health, forecast, mission, configuredPricedHours } = inputs
  const grid = mission?.grid ?? health?.grid ?? null
  const selection =
    (health?.polaris as { selection?: Record<string, unknown> } | undefined)
      ?.selection ?? null

  const environmentDate = text(grid?.environment_date)
  const chartDate = text(grid?.usnic_chart_date)
  const chartName = text(selection?.sidecar_chart)
  const datesAligned =
    typeof grid?.dates_aligned === 'boolean' ? grid.dates_aligned : null
  const requiresOverride =
    typeof health?.requires_historical_demo_override === 'boolean'
      ? health.requires_historical_demo_override
      : null
  const overrideInEffect =
    typeof mission?.historical_demonstration === 'boolean'
      ? mission.historical_demonstration
      : null

  let status: ProvenanceStatus = 'unknown'
  if (datesAligned === true) status = 'contemporaneous'
  else if (datesAligned === false) status = 'historical_demonstration'
  else if (environmentDate && chartDate) {
    status =
      environmentDate === chartDate
        ? 'contemporaneous'
        : 'historical_demonstration'
  }

  const headline =
    status === 'historical_demonstration'
      ? 'Historical POLARIS demonstration'
      : status === 'contemporaneous'
        ? 'Contemporaneous inputs'
        : 'Provenance unknown'

  const facts = (rows: ProvenanceFact[]) => rows.filter((row) => row.value !== null)

  const horizons = forecast?.horizons ?? []
  const availableHours = [0, ...horizons.map((horizon) => horizon.hours)]
  const extrapolated = horizons
    .filter((horizon) => horizon.extrapolated_uncertainty)
    .map((horizon) => horizon.hours)
  const model = forecast?.model ?? null
  const uncertainty =
    (forecast?.uncertainty as Record<string, unknown> | undefined) ?? null
  const population =
    (forecast?.population as Record<string, unknown> | undefined) ?? null

  const details: Provenance['details'] = [
    {
      heading: 'Environment',
      facts: facts([
        { label: 'Fields dated', value: formatDate(environmentDate) },
        {
          label: 'Grid',
          value: grid
            ? `EPSG:${grid.epsg} · ${grid.shape?.join(' × ')} at ${
                grid.pixel_size?.[0] ?? '?'
              } m`
            : null,
        },
        {
          label: 'Excluded cells',
          value:
            number(grid?.excluded_cells) === null
              ? null
              : `${Number(grid?.excluded_cells).toLocaleString('en-GB')}`,
        },
      ]),
    },
    {
      heading: 'POLARIS chart',
      facts: facts([
        { label: 'Chart dated', value: formatDate(chartDate) },
        { label: 'Chart', value: chartName },
        { label: 'Ice class', value: text(selection?.ice_class) },
        { label: 'RIV table', value: text(selection?.riv_table_requested) },
        { label: 'RIV source', value: text(selection?.riv_table_source) },
        {
          label: 'Contemporaneous with the environment',
          value: datesAligned === null ? null : datesAligned ? 'yes' : 'no',
        },
        {
          label: 'Explicit historical override',
          value:
            overrideInEffect !== null
              ? overrideInEffect
                ? 'in effect for this mission'
                : 'off for this mission'
              : requiresOverride === null
                ? null
                : requiresOverride
                  ? 'required before this pairing will route'
                  : 'not required',
        },
      ]),
    },
    {
      heading: 'Iceberg forecast',
      facts: facts([
        {
          label: 'Horizons available',
          value:
            availableHours.length > 1
              ? availableHours
                  .map((hours) => (hours === 0 ? 'observed' : `+${hours}h`))
                  .join(' · ')
              : null,
        },
        {
          label: 'Routing prices against',
          value:
            configuredPricedHours.length > 0
              ? configuredPricedHours.map((hours) => `+${hours}h`).join(' · ')
              : null,
        },
        {
          label: 'Uncertainty calibrated to',
          value:
            number(uncertainty?.calibrated_to_hours) === null
              ? null
              : `+${Number(uncertainty?.calibrated_to_hours)}h`,
        },
        {
          label: 'Extrapolated beyond that',
          value:
            extrapolated.length > 0
              ? extrapolated.map((hours) => `+${hours}h`).join(' · ')
              : null,
        },
        { label: 'Drift model', value: text(model?.summary) },
        {
          label: 'Validated science',
          value:
            model && typeof model.is_validated_science === 'boolean'
              ? model.is_validated_science
                ? 'yes'
                : 'no — a project baseline'
              : null,
        },
        { label: 'Population', value: text(population?.population_note) },
      ]),
    },
  ].filter((section) => section.facts.length > 0)

  return {
    status,
    environmentDate,
    chartDate,
    chartName,
    datesAligned,
    requiresOverride,
    overrideInEffect,
    headline,
    details,
  }
}


// ───────────────────────────────────────────────── the routing gate
//
// The backend refuses a chart and an environment from different dates unless
// the historical demonstration override is set, and it will go on refusing
// them: this adds nothing to that validation and replaces none of it. It only
// lets the operator SEE the condition before spending a request on it, and
// makes enabling the override a deliberate act rather than a recovery from an
// error message.

export type RoutingState =
  | 'ready'
  | 'override_required'
  | 'historical_active'
  | 'unknown'

export interface RoutingReadiness {
  state: RoutingState
  /** true when a route request may be sent for these inputs */
  canRoute: boolean
  /** true when this is a condition worth showing before routing */
  gated: boolean
  headline: string
  /** one factual sentence, or null when there is nothing to explain */
  detail: string | null
  action: 'route' | 'enable_override'
  actionLabel: string
}

/**
 * Whether these inputs may be routed, and what the operator should be offered.
 *
 * `overrideRequested` is the mission request's OWN flag -- the same one the
 * mission panel edits -- so the dock and the panel can never disagree: there
 * is one switch and this reads it.
 */
export function routingReadiness(
  provenance: Provenance,
  overrideRequested: boolean,
): RoutingReadiness {
  const route = 'Use forecast for routing'

  if (provenance.status === 'unknown') {
    return {
      state: 'unknown',
      canRoute: true,
      gated: false,
      headline: 'Provenance unknown',
      detail:
        'The backend has not reported the dates behind these inputs. The ' +
        'request will be sent and the backend will validate the pairing.',
      action: 'route',
      actionLabel: route,
    }
  }

  if (provenance.status === 'contemporaneous') {
    return {
      state: 'ready',
      canRoute: true,
      gated: false,
      headline: 'Routing ready',
      detail: null,
      action: 'route',
      actionLabel: route,
    }
  }

  if (!overrideRequested) {
    return {
      state: 'override_required',
      canRoute: false,
      gated: true,
      headline: 'Historical demonstration required',
      detail:
        'The environmental fields and the POLARIS chart are from different ' +
        'dates. Routing with this pairing requires the explicit historical ' +
        'demonstration override.',
      action: 'enable_override',
      actionLabel: 'Enable historical demonstration',
    }
  }

  return {
    state: 'historical_active',
    canRoute: true,
    gated: true,
    headline: 'Historical demonstration active',
    detail:
      'Routing is using the historical POLARIS chart as an explicit ' +
      'demonstration input. The chart is not a description of the current ' +
      'environmental date.',
    action: 'route',
    actionLabel: route,
  }
}
