import { describe, expect, it } from 'vitest'

import {
  drawnProfiles,
  equivalenceDetail,
  equivalenceNote,
  figuresProfile,
  groupEquivalentRoutes,
  groupFor,
  groupLabel,
  pathsEqual,
} from '../api/routeEquivalence'
import type { Comparison, ProfileName, RouteProfile } from '../api/types'
import { missionResponse } from './fixtures'

//  A profile is only ever read here for `success`, `path` and its three
//  headline figures, so the stand-ins carry those and nothing else.
const route = (
  path: number[][],
  over: Partial<RouteProfile> = {},
): RouteProfile =>
  ({
    success: true,
    path,
    distance_m: 1,
    travel_time_s: 1,
    configured_cost: 1,
    ...over,
  }) as unknown as RouteProfile

const A = [
  [948, 503],
  [947, 503],
  [922, 497],
]

describe('pathsEqual', () => {
  it('calls the same cell sequence equivalent', () => {
    expect(pathsEqual(A, A.map((cell) => [...cell]))).toBe(true)
  })

  it('does not care which object the numbers arrived in', () => {
    //  a second parse of the same payload, and the very same array
    const parsedAgain = JSON.parse(JSON.stringify(A)) as number[][]
    expect(pathsEqual(A, parsedAgain)).toBe(true)
    expect(pathsEqual(A, A)).toBe(true)
  })

  it('separates different sequences', () => {
    expect(pathsEqual(A, [[948, 503], [946, 502], [922, 497]])).toBe(false)
  })

  it('is sensitive to cell ordering', () => {
    const swapped = [A[0], A[2], A[1]]
    expect(pathsEqual(A, swapped)).toBe(false)
  })

  it('is sensitive to direction: a reversed path is another route', () => {
    expect(pathsEqual(A, [...A].reverse())).toBe(false)
  })

  it('separates paths of different length, prefix or not', () => {
    expect(pathsEqual(A, A.slice(0, 2))).toBe(false)
  })

  it('treats an absent or empty path as sharing nothing', () => {
    expect(pathsEqual(null, A)).toBe(false)
    expect(pathsEqual(A, undefined)).toBe(false)
    expect(pathsEqual([], [])).toBe(false)
  })

  it('refuses malformed cells rather than guessing', () => {
    expect(pathsEqual([[948]], [[948]])).toBe(false)
    expect(pathsEqual([[948, Number.NaN]], [[948, Number.NaN]])).toBe(false)
  })
})

describe('groupEquivalentRoutes', () => {
  const profiles = (
    fastest: number[][],
    risk: number[][],
    shortest: number[][],
  ) => ({
    fastest: route(fastest),
    risk_oriented: route(risk),
    shortest_distance: route(shortest),
  })

  it('groups the objectives whose returned paths match', () => {
    const groups = groupEquivalentRoutes(
      profiles(A, [[948, 503], [946, 502], [922, 497]], A.map((c) => [...c])),
    )
    expect(groups).toHaveLength(2)
    expect(groups[0].profiles).toEqual(['fastest', 'shortest_distance'])
    expect(groups[0].shared).toBe(true)
    expect(groups[0].representative).toBe('fastest')
    expect(groups[0].key).toBe('fastest+shortest_distance')
    expect(groups[1].profiles).toEqual(['risk_oriented'])
    expect(groups[1].shared).toBe(false)
  })

  it('leaves three distinct geometries as three options', () => {
    const groups = groupEquivalentRoutes(
      profiles(
        A,
        [[948, 503], [946, 502], [922, 497]],
        [[948, 503], [945, 501], [922, 497]],
      ),
    )
    expect(groups).toHaveLength(3)
    expect(groups.every((group) => !group.shared)).toBe(true)
  })

  it('does not group a reversed path with its original', () => {
    const groups = groupEquivalentRoutes({
      fastest: route(A),
      shortest_distance: route([...A].reverse()),
    })
    expect(groups).toHaveLength(2)
  })

  it('keeps a profile that did not route out of every group', () => {
    const groups = groupEquivalentRoutes({
      fastest: route(A),
      risk_oriented: route([], { success: false, path: [] }),
      shortest_distance: route(A.map((c) => [...c])),
    })
    expect(groups).toHaveLength(2)
    expect(groups[0].profiles).toEqual(['fastest', 'shortest_distance'])
    const failed = groups[1]
    expect(failed.profiles).toEqual(['risk_oriented'])
    expect(failed.routed).toBe(false)
  })

  it('reports whether the grouped objectives also report the same figures', () => {
    const agreeing = groupEquivalentRoutes({
      fastest: route(A),
      shortest_distance: route(A.map((c) => [...c])),
    })
    expect(agreeing[0].figuresAgree).toBe(true)

    const disagreeing = groupEquivalentRoutes({
      fastest: route(A),
      shortest_distance: route(A.map((c) => [...c]), { configured_cost: 2 }),
    })
    expect(disagreeing[0].shared).toBe(true)
    expect(disagreeing[0].figuresAgree).toBe(false)
  })

  it('is deterministic: the same payload gives the same grouping', () => {
    const once = groupEquivalentRoutes(
      profiles(A, [[948, 503], [946, 502], [922, 497]], A.map((c) => [...c])),
    )
    const twice = groupEquivalentRoutes(
      profiles(A, [[948, 503], [946, 502], [922, 497]], A.map((c) => [...c])),
    )
    expect(JSON.stringify(once)).toBe(JSON.stringify(twice))
  })

  it('has nothing to group without a comparison', () => {
    expect(groupEquivalentRoutes(null)).toEqual([])
  })
})

describe('the recorded demo payload', () => {
  const comparison = missionResponse(true).comparison as Comparison
  const groups = groupEquivalentRoutes(comparison.profiles)

  it('groups fastest with shortest distance and leaves the others alone', () => {
    //  four objectives, three distinct geometries: fastest and
    //  shortest_distance coincide at one constant vessel speed, while
    //  fuel_efficient and risk_oriented each returned their own path
    expect(groups.map((group) => group.profiles)).toEqual([
      ['fastest', 'shortest_distance'],
      ['fuel_efficient'],
      ['risk_oriented'],
    ])
  })

  it('draws one line per geometry, under the selected objective', () => {
    expect(drawnProfiles(groups, 'fastest')).toEqual([
      'fastest',
      'fuel_efficient',
      'risk_oriented',
    ])
    //  selecting the other member of the group draws THAT objective's line
    expect(drawnProfiles(groups, 'shortest_distance')).toEqual([
      'shortest_distance',
      'fuel_efficient',
      'risk_oriented',
    ])
    expect(drawnProfiles(groups, 'risk_oriented')).toEqual([
      'fastest',
      'fuel_efficient',
      'risk_oriented',
    ])
    expect(drawnProfiles(groups, null)).toEqual([
      'fastest',
      'fuel_efficient',
      'risk_oriented',
    ])
  })

  it('finds a profile’s group, and whose figures a group shows', () => {
    const shared = groupFor(groups, 'shortest_distance')
    expect(shared?.key).toBe('fastest+shortest_distance')
    expect(figuresProfile(shared!, 'shortest_distance')).toBe('shortest_distance')
    expect(figuresProfile(shared!, 'risk_oriented')).toBe('fastest')
    expect(groupFor(groups, null)).toBeNull()
  })

  it('names both objectives and describes only this model', () => {
    const shared = groups[0]
    expect(groupLabel(shared)).toBe('Fastest · Shortest distance')
    const note = equivalenceNote(shared, 5)
    expect(note).toBe('Same path under the current constant vessel speed (5 m/s)')
    expect(note).not.toMatch(/always|identical|duplicate|fuel|equally safe/i)
    expect(equivalenceNote(shared, null)).toBe(
      'Same path under the current constant vessel speed',
    )
    //  a profile of its own has nothing to explain
    expect(equivalenceNote(groups[1], 5)).toBeNull()
  })

  it('uses the backend’s own sentence for the longer explanation', () => {
    expect(equivalenceDetail(groups[0], comparison)).toBe(
      comparison.provenance.why_they_can_coincide,
    )
    expect(equivalenceDetail(groups[1], comparison)).toBeNull()
  })

  it('states only what it saw when the backend said nothing about the pair', () => {
    const pair = groupEquivalentRoutes({
      risk_oriented: route(A),
      shortest_distance: route(A.map((c) => [...c])),
    })
    const detail = equivalenceDetail(pair[0], {
      provenance: {},
    } as unknown as Comparison)
    expect(detail).toBe(
      'Risk-oriented · Shortest distance returned the same path cell ' +
        'sequence for this mission.',
    )
  })

  it('never claims the objectives are always the same route', () => {
    const words = [
      groupLabel(groups[0]),
      equivalenceNote(groups[0], 5) ?? '',
      equivalenceDetail(groups[0], comparison) ?? '',
    ]
      .join(' ')
      .toLowerCase()
    //  `fuel-efficient` is now a real objective NAME, so the bare word is no
    //  longer forbidden. What stays forbidden is any UNSUPPORTED fuel claim --
    //  a measured consumption, a saving, or a mass or volume this project
    //  cannot produce -- and the ranking vocabulary, as before.
    for (const banned of [
      'duplicate',
      'fuel saving',
      'saves fuel',
      'measured fuel',
      'actual fuel',
      'litre',
      'tonne',
      'best route',
      'optimal',
      'safest',
      'always identical',
    ]) {
      expect(words).not.toContain(banned)
    }
  })
})

describe('profile order decides everything arbitrary', () => {
  it('follows the order it is given', () => {
    const order: ProfileName[] = ['shortest_distance', 'risk_oriented', 'fastest']
    const groups = groupEquivalentRoutes(
      {
        fastest: route(A),
        risk_oriented: route([[948, 503], [946, 502], [922, 497]]),
        shortest_distance: route(A.map((c) => [...c])),
      },
      order,
    )
    expect(groups[0].profiles).toEqual(['shortest_distance', 'fastest'])
    expect(groups[0].representative).toBe('shortest_distance')
  })
})
