import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

import {
  defaultForecastSubject,
  distanceToCorridor,
  rankForecastSubjects,
} from '../map/subject'
import type { MissionCorridor } from '../map/subject'
import type { XY } from '../map/camera'
import { projectLonLat } from '../map/ForecastLayers'
import { cellCentreXY, cellLonLat } from '../map/coords'
import { forecastResponse, missionResponse } from './fixtures'

// The payload here is a RECORDED /api/forecast/icebergs response and the
// corridor is the demo request's own cells: nothing in this file is a
// coordinate anyone typed.

const MISSION = missionResponse()
const GRID = MISSION.grid
const at = (cell: readonly number[]): XY =>
  cellCentreXY(GRID, cell[0], cell[1]) as XY
const CORRIDOR: MissionCorridor = {
  start: at(MISSION.request.start),
  goal: at(MISSION.request.goal),
}
const BERGS = forecastResponse().icebergs

/** An INDEPENDENT oracle.
 *
 *  subject.ts measures in projected metres. This measures on the sphere:
 *  great-circle distance to the nearest of 400 points sampled along the
 *  corridor in geographic coordinates. Two different methods; if they disagree
 *  about the ordering, one of them is wrong and these tests say so. */
const EARTH_M = 6_371_008.8
const toRad = (deg: number) => (deg * Math.PI) / 180
function haversine(
  [lon1, lat1]: [number, number],
  [lon2, lat2]: [number, number],
): number {
  const dLat = toRad(lat2 - lat1)
  const dLon = toRad(lon2 - lon1)
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2
  return 2 * EARTH_M * Math.asin(Math.sqrt(a))
}

const CORRIDOR_LONLAT = (() => {
  const a = cellLonLat(GRID, MISSION.request.start[0], MISSION.request.start[1])!
  const b = cellLonLat(GRID, MISSION.request.goal[0], MISSION.request.goal[1])!
  return Array.from({ length: 401 }, (_, i) => {
    const t = i / 400
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t] as [number, number]
  })
})()

/** nearest great-circle approach of an iceberg's own states to the corridor */
function sphericalDistanceM(record: (typeof BERGS)[number]): number {
  const states = [record.observed!, ...record.predicted]
  let best = Infinity
  for (const state of states) {
    for (const sample of CORRIDOR_LONLAT) {
      const d = haversine([state.longitude, state.latitude], sample)
      if (d < best) best = d
    }
  }
  return best
}

const SPHERICAL_ORDER = [...BERGS]
  .map((record) => ({ id: record.iceberg_id, m: sphericalDistanceM(record) }))
  .sort((a, b) => a.m - b.m || a.id.localeCompare(b.id))

describe('distance to the mission corridor', () => {
  //  a simple corridor in metres, so the geometry is checkable by hand
  const corridor: MissionCorridor = { start: [0, 0], goal: [0, 100_000] }

  it('is zero on the corridor itself', () => {
    expect(distanceToCorridor([0, 50_000], corridor)).toBe(0)
    expect(distanceToCorridor([0, 0], corridor)).toBe(0)
    expect(distanceToCorridor([0, 100_000], corridor)).toBe(0)
  })

  it('measures perpendicular to the segment, not to an endpoint', () => {
    //  beside the middle of the crossing: 1 km from the corridor, 50 km from
    //  the nearest endpoint
    expect(distanceToCorridor([1_000, 50_000], corridor)).toBeCloseTo(1_000, 6)
  })

  it('clamps to an end for a point beyond the segment', () => {
    expect(distanceToCorridor([0, -30_000], corridor)).toBeCloseTo(30_000, 6)
    expect(distanceToCorridor([0, 130_000], corridor)).toBeCloseTo(30_000, 6)
  })

  it('degrades to a point distance when departure equals destination', () => {
    expect(
      distanceToCorridor([3_000, 4_000], { start: [0, 0], goal: [0, 0] }),
    ).toBeCloseTo(5_000, 6)
  })

  it('ranks by the corridor rather than by the nearer endpoint', () => {
    //  beside the middle: far from both ends, but right on the corridor
    const beside: XY = [1_000, 50_000]
    //  off one end: nearer an endpoint than `beside` is, but 30 km off course
    const offTheEnd: XY = [0, -30_000]
    expect(distanceToCorridor(beside, corridor)).toBeLessThan(
      distanceToCorridor(offTheEnd, corridor),
    )
    //  and an endpoint-only measure would have got it the other way round
    const toNearestEnd = (p: XY) =>
      Math.min(
        Math.hypot(p[0] - corridor.start[0], p[1] - corridor.start[1]),
        Math.hypot(p[0] - corridor.goal[0], p[1] - corridor.goal[1]),
      )
    expect(toNearestEnd(offTheEnd)).toBeLessThan(toNearestEnd(beside))
  })
})

describe('default forecast subject', () => {
  it('opens on the iceberg nearest the mission, not the first in the payload', () => {
    const chosen = defaultForecastSubject(BERGS, CORRIDOR)
    expect(chosen).not.toBeNull()
    //  the backend lists them in its own id order; the choice is not the first
    expect(chosen).not.toBe(BERGS[0].iceberg_id)
    //  it is the nearest the corridor, by a measure this module does not use
    expect(chosen).toBe(SPHERICAL_ORDER[0].id)
    //  and the one it passed over really is much further away
    expect(SPHERICAL_ORDER[SPHERICAL_ORDER.length - 1].m).toBeGreaterThan(
      SPHERICAL_ORDER[0].m * 5,
    )
  })

  it('agrees with a spherical measure about the whole ordering', () => {
    expect(rankForecastSubjects(BERGS, CORRIDOR).map((r) => r.icebergId)).toEqual(
      SPHERICAL_ORDER.map((entry) => entry.id),
    )
  })

  it('stays within a few percent of the spherical distance it agrees with', () => {
    for (const entry of rankForecastSubjects(BERGS, CORRIDOR)) {
      const spherical = SPHERICAL_ORDER.find((e) => e.id === entry.icebergId)!.m
      //  EPSG:3976 stretches away from its standard parallel; the ordering is
      //  what matters, but the magnitudes should still be close
      expect(entry.projectedDistanceM / spherical).toBeGreaterThan(0.9)
      expect(entry.projectedDistanceM / spherical).toBeLessThan(1.15)
    }
  })

  it('does not depend on the order the backend listed them in', () => {
    const forward = rankForecastSubjects(BERGS, CORRIDOR).map((r) => r.icebergId)
    const reversed = rankForecastSubjects([...BERGS].reverse(), CORRIDOR).map(
      (r) => r.icebergId,
    )
    const rotated = rankForecastSubjects(
      [...BERGS.slice(2), ...BERGS.slice(0, 2)],
      CORRIDOR,
    ).map((r) => r.icebergId)

    expect(reversed).toEqual(forward)
    expect(rotated).toEqual(forward)
    for (const order of [[...BERGS].reverse(), [...BERGS.slice(1), BERGS[0]]]) {
      expect(defaultForecastSubject(order, CORRIDOR)).toBe(
        defaultForecastSubject(BERGS, CORRIDOR),
      )
    }
  })

  it('is deterministic: the same mission and payload give the same answer', () => {
    const answers = new Set(
      Array.from({ length: 25 }, () =>
        defaultForecastSubject(forecastResponse().icebergs, {
          start: at(MISSION.request.start),
          goal: at(MISSION.request.goal),
        }),
      ),
    )
    expect(answers.size).toBe(1)
  })

  it('follows the mission: another corridor chooses another subject', () => {
    const here = defaultForecastSubject(BERGS, CORRIDOR)
    //  a corridor laid between two of the FAR icebergs' own observed
    //  positions -- still backend data, still nothing typed by hand
    const far = [BERGS[0], BERGS[1]]
    const elsewhere: MissionCorridor = {
      start: projectLonLat(far[0].observed!.longitude, far[0].observed!.latitude),
      goal: projectLonLat(far[1].observed!.longitude, far[1].observed!.latitude),
    }
    const chosenThere = defaultForecastSubject(BERGS, elsewhere)
    expect(chosenThere).not.toBe(here)
    expect(far.map((b) => b.iceberg_id)).toContain(chosenThere)
  })

  it('falls back to the id order when there is no mission yet', () => {
    const ranked = rankForecastSubjects([...BERGS].reverse(), null)
    expect(ranked.map((r) => r.icebergId)).toEqual(
      [...BERGS].map((b) => b.iceberg_id).sort(),
    )
    expect(ranked.every((r) => r.projectedDistanceM === Infinity)).toBe(true)
  })

  it('has nothing to choose from an empty forecast', () => {
    expect(defaultForecastSubject([], CORRIDOR)).toBeNull()
    expect(defaultForecastSubject(null, CORRIDOR)).toBeNull()
    expect(rankForecastSubjects(undefined, CORRIDOR)).toEqual([])
  })

  it('reports which horizon the nearest approach happens at', () => {
    for (const entry of rankForecastSubjects(BERGS, CORRIDOR)) {
      const record = BERGS.find((b) => b.iceberg_id === entry.icebergId)!
      const hours = [record.observed!, ...record.predicted].map((s) => s.hours)
      expect(hours).toContain(entry.atHours)
    }
  })
})

describe('no iceberg identity is written into the application', () => {
  const SRC = join(process.cwd(), 'src')
  const sources = (dir: string, out: string[] = []): string[] => {
    for (const entry of readdirSync(dir)) {
      const path = join(dir, entry)
      if (statSync(path).isDirectory()) {
        if (entry === '__tests__' || entry === 'generated') continue
        sources(path, out)
      } else if (/\.(ts|tsx)$/.test(entry)) {
        out.push(path)
      }
    }
    return out
  }

  it('names no iceberg the backend returns', () => {
    const ids = BERGS.map((berg) => berg.iceberg_id)
    expect(ids.length).toBeGreaterThan(1)
    for (const file of sources(SRC)) {
      const code = readFileSync(file, 'utf8')
      for (const id of ids) {
        expect(
          new RegExp(`['"\`]${id}['"\`]`).test(code),
          `${file} names iceberg ${id}`,
        ).toBe(false)
      }
    }
  })
})
