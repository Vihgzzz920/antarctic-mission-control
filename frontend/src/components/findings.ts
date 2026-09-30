// Short factual comparisons between the selected route and the other two.
//
// Every line states BOTH numbers and names the route it compares against. None
// of them says one route is better, safer or recommended, and none asserts a
// cause: a route has a lower POLARIS contribution than another, and that is all
// that is claimed. Lines are emitted only where the backend gave both numbers
// and they actually differ.

import { PROFILE_LABEL, PROFILE_ORDER } from '../api/types'
import type { Comparison, ProfileName, RouteProfile } from '../api/types'
import { hours, num } from './format'

export interface Finding {
  key: string
  direction: 'lower' | 'higher' | 'equal'
  text: string
}

type Reader = (profile: RouteProfile) => number | null

interface Metric {
  key: string
  label: string
  read: Reader
  render: (value: number) => string
  /** how a value below the other route's should be described */
  lower: string
  higher: string
}

const METRICS: Metric[] = [
  {
    key: 'polaris',
    label: 'POLARIS contribution',
    read: (p) => p.polaris_contribution,
    render: (v) => num(v),
    lower: 'Lower configured POLARIS contribution',
    higher: 'Higher configured POLARIS contribution',
  },
  {
    key: 'special',
    label: 'special-consideration cells',
    read: (p) => p.special_consideration_cells,
    render: (v) => `${v}`,
    lower: 'Fewer special-consideration cells',
    higher: 'More special-consideration cells',
  },
  {
    key: 'indeterminate',
    label: 'indeterminate cells',
    read: (p) => p.indeterminate_cells,
    render: (v) => `${v}`,
    lower: 'Fewer cells with an indeterminate RIO interval',
    higher: 'More cells with an indeterminate RIO interval',
  },
  {
    key: 'iceberg',
    label: 'iceberg exposure contribution',
    read: (p) => p.iceberg_exposure_contribution,
    render: (v) => num(v),
    lower: 'Reduced modelled iceberg exposure contribution',
    higher: 'Greater modelled iceberg exposure contribution',
  },
]

function compare(
  metric: Metric,
  route: RouteProfile,
  other: RouteProfile,
): Finding | null {
  const mine = metric.read(route)
  const theirs = metric.read(other)
  if (mine === null || theirs === null) return null
  if (mine === theirs) return null
  const direction = mine < theirs ? 'lower' : 'higher'
  const lead = direction === 'lower' ? metric.lower : metric.higher
  return {
    key: `${metric.key}-${other.profile}`,
    direction,
    text:
      `${lead} than ${PROFILE_LABEL[other.profile].toLowerCase()}: ` +
      `${metric.render(mine)} against ${metric.render(theirs)}.`,
  }
}

/** The trade-off line: what the selected route costs in distance and time. */
function tradeoff(route: RouteProfile, other: RouteProfile): Finding | null {
  const dKm = route.distance_km
  const oKm = other.distance_km
  const dH = route.travel_time_h
  const oH = other.travel_time_h
  if (dKm === null || oKm === null || dH === null || oH === null) return null
  if (dKm === oKm && dH === oH) return null
  const longer = dKm > oKm
  const km = Math.abs(dKm - oKm)
  const hrs = Math.abs(dH - oH)
  return {
    key: `tradeoff-${other.profile}`,
    direction: longer ? 'higher' : 'lower',
    text:
      `${longer ? 'Costs' : 'Saves'} ${num(km, 2)} km and ${hours(hrs)} ` +
      `${longer ? 'more' : 'less'} than ${PROFILE_LABEL[
        other.profile
      ].toLowerCase()}.`,
  }
}

export function buildFindings(
  comparison: Comparison,
  selected: ProfileName,
): Finding[] {
  const route = comparison.profiles[selected]
  if (!route?.success) return []
  // Every other routed profile is compared. Two profiles that returned the
  // same path necessarily have the same metrics, so they fall out on their own
  // at the per-metric equality check below -- filtering on path here would
  // instead hide a real difference whenever the paths happened to coincide.
  const others = PROFILE_ORDER.filter(
    (name) => name !== selected && comparison.profiles[name]?.success,
  ).map((name) => comparison.profiles[name])

  const out: Finding[] = []
  for (const other of others) {
    for (const metric of METRICS) {
      const finding = compare(metric, route, other)
      if (finding) out.push(finding)
    }
    const trade = tradeoff(route, other)
    if (trade) out.push(trade)
  }
  return out
}

/** Names the routes that came out identical, so the panel can say so. */
export function identicalTo(
  comparison: Comparison,
  selected: ProfileName,
): ProfileName[] {
  const route = comparison.profiles[selected]
  if (!route?.success) return []
  return PROFILE_ORDER.filter(
    (name) =>
      name !== selected &&
      comparison.profiles[name]?.success &&
      JSON.stringify(comparison.profiles[name].path) ===
        JSON.stringify(route.path),
  )
}
