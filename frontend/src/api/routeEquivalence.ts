// Geometric identity between the route profiles the backend returned.
//
// The planner asks three different questions -- least travel time, least
// configured navigation cost, least travelled distance -- and gets three
// answers back. Under the vessel model this project ships, two of those
// questions can be answered by the SAME path: time_astar advances the clock by
// distance / vessel_speed_mps, so while the speed is one constant a path's
// travel time rises strictly with its length and the two objectives order
// every path identically. That is a property of the current model, not a
// mathematical identity, and it is measured here rather than assumed:
// identity comes from the returned path cell sequence and from nothing else.
//
// Nothing in this file talks to the backend, touches React, or changes a
// route. It reads what the API already sent and groups it.

import { PROFILE_LABEL, PROFILE_ORDER } from './types'
import type { Comparison, ProfileName, RouteProfile } from './types'

/** A cell as the API serialises it: [row, col]. */
export type PathCell = readonly number[]

/**
 * Are these two returned paths the same geometry?
 *
 * The rule, deliberately narrow:
 *   - the same number of cells, in the same order, with the same row and col;
 *   - direction matters. A reversed path is a different route: this project's
 *     routes carry arrival times from a departure point, so the sequence runs
 *     one way only and nothing here makes direction irrelevant;
 *   - an absent or empty path is not equivalent to anything, including another
 *     empty path -- a profile that returned no geometry has nothing to share;
 *   - only the numbers are compared, so two payloads that were parsed
 *     separately, or the same route object passed twice, both answer alike.
 */
export function pathsEqual(
  a: readonly PathCell[] | null | undefined,
  b: readonly PathCell[] | null | undefined,
): boolean {
  if (!Array.isArray(a) || !Array.isArray(b)) return false
  if (a.length === 0 || b.length === 0) return false
  if (a.length !== b.length) return false
  for (let index = 0; index < a.length; index += 1) {
    const left = a[index]
    const right = b[index]
    if (!Array.isArray(left) || !Array.isArray(right)) return false
    if (left.length < 2 || left.length !== right.length) return false
    for (let axis = 0; axis < left.length; axis += 1) {
      const one = left[axis]
      const other = right[axis]
      if (!Number.isFinite(one) || !Number.isFinite(other)) return false
      if (one !== other) return false
    }
  }
  return true
}

/** The figures the plan surface shows side by side. */
const FIGURE_KEYS = [
  'distance_m',
  'travel_time_s',
  'configured_cost',
] as const

/** A set of profiles whose returned paths are the same geometry. */
export interface RouteGroup {
  /** deterministic: the member names in profile order, joined */
  key: string
  /** the members, in profile order. One member for an ungrouped profile. */
  profiles: ProfileName[]
  /** whose geometry stands for the group when only one line is drawn */
  representative: ProfileName
  /** more than one objective returned this geometry */
  shared: boolean
  /** the group carries a path at all */
  routed: boolean
  /**
   * every member reports the same distance, travel time and configured cost.
   * Shared geometry normally means shared figures, but the figures are checked
   * rather than assumed: if they ever disagree, each objective keeps its own.
   */
  figuresAgree: boolean
}

type Profiles = Partial<Record<ProfileName, RouteProfile>> | null | undefined

function routed(profile: RouteProfile | undefined): boolean {
  return Boolean(profile?.success) && (profile?.path?.length ?? 0) > 0
}

function figuresAgree(members: RouteProfile[]): boolean {
  if (members.length < 2) return true
  const [first, ...rest] = members
  return rest.every((profile) =>
    FIGURE_KEYS.every((key) => Object.is(profile[key], first[key])),
  )
}

/**
 * Group the profiles whose returned paths are the same geometry.
 *
 * Profile order decides everything that could otherwise be arbitrary: which
 * group comes first, the order of members inside a group, and which member is
 * the representative. A profile that did not route keeps a group of its own --
 * it has no geometry to share, and the plan surface still has to show what
 * happened to it.
 */
export function groupEquivalentRoutes(
  profiles: Profiles,
  order: readonly ProfileName[] = PROFILE_ORDER,
): RouteGroup[] {
  if (!profiles) return []
  const groups: Array<{ names: ProfileName[]; members: RouteProfile[] }> = []
  for (const name of order) {
    const profile = profiles[name]
    if (!profile) continue
    if (!routed(profile)) {
      groups.push({ names: [name], members: [profile] })
      continue
    }
    const shares = groups.find(
      (group) =>
        routed(group.members[0]) && pathsEqual(group.members[0].path, profile.path),
    )
    if (shares) {
      shares.names.push(name)
      shares.members.push(profile)
    } else {
      groups.push({ names: [name], members: [profile] })
    }
  }
  return groups.map(({ names, members }) => ({
    key: names.join('+'),
    profiles: names,
    representative: names[0],
    shared: names.length > 1,
    routed: routed(members[0]),
    figuresAgree: figuresAgree(members),
  }))
}

/** The group a profile belongs to, or null if it is not in this comparison. */
export function groupFor(
  groups: readonly RouteGroup[],
  profile: ProfileName | null,
): RouteGroup | null {
  if (!profile) return null
  return groups.find((group) => group.profiles.includes(profile)) ?? null
}

/**
 * Which profiles' geometry is worth drawing: one line per group. The selected
 * profile always draws its own, so the line on the map carries the selected
 * objective's colour and styling; a group the selection is not in draws its
 * representative. Profiles that did not route contribute nothing.
 */
export function drawnProfiles(
  groups: readonly RouteGroup[],
  selected: ProfileName | null,
): ProfileName[] {
  return groups
    .filter((group) => group.routed)
    .map((group) =>
      selected && group.profiles.includes(selected) ? selected : group.representative,
    )
}

/** Whose figures a shared group shows: the selection, else the representative. */
export function figuresProfile(
  group: RouteGroup,
  selected: ProfileName | null,
): ProfileName {
  return selected && group.profiles.includes(selected)
    ? selected
    : group.representative
}

/** "Fastest · Shortest distance" */
export function groupLabel(group: RouteGroup): string {
  return group.profiles.map((name) => PROFILE_LABEL[name]).join(' · ')
}

/**
 * One line for a group that shares its geometry, describing THIS result under
 * THIS model -- never a claim that the objectives always coincide. The speed
 * is named when the mission supplied one, because it is the assumption that
 * makes the two objectives order paths alike.
 */
export function equivalenceNote(
  group: RouteGroup,
  vesselSpeedMps?: number | null,
): string | null {
  if (!group.shared || !group.routed) return null
  const speed =
    typeof vesselSpeedMps === 'number' && Number.isFinite(vesselSpeedMps)
      ? ` (${vesselSpeedMps} m/s)`
      : ''
  return `Same path under the current constant vessel speed${speed}`
}

/**
 * The longer form, shown only when an objective in a shared group is opened.
 * The backend already explains the fastest / shortest-distance case in its own
 * words; that sentence is used verbatim when the comparison reports that pair
 * coincided. Any other pairing gets a statement of what was observed and
 * nothing more.
 */
export function equivalenceDetail(
  group: RouteGroup,
  comparison: Comparison | null | undefined,
): string | null {
  if (!group.shared || !group.routed) return null
  const provenance = comparison?.provenance
  const isFastestAndShortest =
    group.profiles.length === 2 &&
    group.profiles.includes('fastest') &&
    group.profiles.includes('shortest_distance')
  const why = provenance?.why_they_can_coincide
  if (
    isFastestAndShortest &&
    provenance?.fastest_and_shortest_distance_coincided === true &&
    typeof why === 'string' &&
    why.length > 0
  ) {
    return why
  }
  return `${groupLabel(group)} returned the same path cell sequence for this mission.`
}
