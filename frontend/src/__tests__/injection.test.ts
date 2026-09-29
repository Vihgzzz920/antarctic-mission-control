import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

import {
  DEFAULT_INJECTION_FRACTION,
  injectionPoint,
} from '../map/injection'
import { cellCentreXY } from '../map/coords'
import { missionResponse } from './fixtures'

// The grid transform is the backend's own, from the recorded response; every
// position below is a cell centre derived through it.
const GRID = missionResponse().grid

/** a longer path than the trimmed fixture carries, built by walking the
 *  recorded start to the recorded goal one cell at a time -- still the
 *  backend's own endpoints, no coordinate typed here */
function walk(): Array<[number, number]> {
  const [r0, c0] = missionResponse().request.start
  const [r1, c1] = missionResponse().request.goal
  const steps = Math.max(Math.abs(r1 - r0), Math.abs(c1 - c0))
  return Array.from({ length: steps + 1 }, (_, i) => [
    Math.round(r0 + ((r1 - r0) * i) / steps),
    Math.round(c0 + ((c1 - c0) * i) / steps),
  ])
}

describe('the deterministic injection point', () => {
  const path = walk()
  const point = injectionPoint(path, GRID)!

  it('lands about two fifths along the route', () => {
    expect(DEFAULT_INJECTION_FRACTION).toBeCloseTo(0.4, 6)
    expect(point).not.toBeNull()
    expect(point.fraction).toBeGreaterThan(0.3)
    expect(point.fraction).toBeLessThan(0.5)
    expect(point.alongM).toBeGreaterThan(0)
    expect(point.alongM).toBeLessThan(point.totalM)
  })

  it('is a vertex of the route, not a point beside it', () => {
    expect(path.map((cell) => cell.join(','))).toContain(point.cell.join(','))
    expect(path[point.index]).toEqual(point.cell)
    //  and the position handed to the API is that cell's own centre
    expect(point.position).toEqual(
      cellCentreXY(GRID, point.cell[0], point.cell[1]),
    )
  })

  it('never lands on the departure or the destination', () => {
    expect(point.index).toBeGreaterThan(0)
    expect(point.index).toBeLessThan(path.length - 1)
    expect(point.cell).not.toEqual(path[0])
    expect(point.cell).not.toEqual(path[path.length - 1])
  })

  it('gives the same answer every time it is asked', () => {
    const answers = new Set(
      Array.from({ length: 50 }, () =>
        JSON.stringify(injectionPoint(walk(), GRID)),
      ),
    )
    expect(answers.size).toBe(1)
  })

  it('follows the fraction it is given', () => {
    const early = injectionPoint(path, GRID, 0.1)!
    const late = injectionPoint(path, GRID, 0.9)!
    expect(early.alongM).toBeLessThan(point.alongM)
    expect(late.alongM).toBeGreaterThan(point.alongM)
    //  and clamps rather than running off the end
    expect(injectionPoint(path, GRID, -5)!.index).toBeGreaterThanOrEqual(0)
    expect(injectionPoint(path, GRID, 5)!.index).toBeLessThan(path.length)
  })

  it('handles the short and the empty path without inventing a position', () => {
    //  departure and destination and nothing between them: the injection has
    //  no interior vertex to choose, and must still not invent one
    const full = missionResponse().comparison.profiles.risk_oriented.path
    const two = [full[0], full[full.length - 1]]
    expect(two).toHaveLength(2)
    const shortPoint = injectionPoint(two, GRID)!
    expect(two.map((cell) => cell.join(','))).toContain(shortPoint.cell.join(','))

    expect(injectionPoint([], GRID)).toBeNull()
    expect(injectionPoint(null, GRID)).toBeNull()
    expect(injectionPoint(path, null)).toBeNull()
  })
})

describe('nothing about the injection is written into the source', () => {
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

  it('uses no randomness, no clock and no screen coordinate', () => {
    const code = readFileSync(join(SRC, 'map', 'injection.ts'), 'utf8')
    for (const banned of ['Math.random', 'Date.now', 'performance.now', 'event']) {
      expect(code, `injection.ts uses ${banned}`).not.toContain(banned)
    }
  })

  it('names no projected coordinate and no iceberg anywhere', () => {
    for (const file of sources(SRC)) {
      const code = readFileSync(file, 'utf8')
      //  a projected-metre pair would be a position typed by hand
      expect(code, `${file} carries a projected coordinate`).not.toMatch(
        /-?\d{6,7}\s*,\s*-?\d{6,7}/,
      )
      expect(code, `${file} names a simulated iceberg`).not.toMatch(
        /simulated_iceberg['"]\s*[:=]/,
      )
    }
  })

  it('names no route cell outside the mission form\u2019s opening values', () => {
    //  App.tsx seeds the form with a departure and destination cell so the
    //  panel is not blank on the first frame; /api/mission/defaults replaces
    //  them as soon as it answers, and routing is refused until it has. That
    //  ONE literal is the documented exception; nothing else may name a cell,
    //  and the injection position is never one of them.
    const cells = [
      missionResponse().request.start.join(', '),
      missionResponse().request.goal.join(', '),
    ]
    for (const file of sources(SRC)) {
      const raw = readFileSync(file, 'utf8')
      const code = file.endsWith('App.tsx')
        ? raw.replace(/const INITIAL_REQUEST[\s\S]*?\n}/, '')
        : raw
      for (const cell of cells) {
        expect(code, `${file} names route cell ${cell}`).not.toContain(cell)
      }
    }
  })
})
