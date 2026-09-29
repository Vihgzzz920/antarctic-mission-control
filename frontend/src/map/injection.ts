// Where a simulated iceberg goes when the operator asks for one on the route.
//
// Free placement is a coin toss: at route zoom a click is kilometres wide, and
// a berg that lands beside the corridor legitimately changes nothing, so the
// demonstration's outcome depends on aim rather than on the planner. This
// picks the position from the route the search returned instead.
//
// Pure: the same mission, the same selected route and the same fraction always
// give the same cell. No clock, no randomness, no screen coordinate, and no
// coordinate written here -- the position is a CELL CENTRE from the project's
// own grid transform, taken from a vertex of the path the backend sent back.

import { cellCentreXY } from './coords'
import type { GridGeometry } from './coords'
import type { XY } from './camera'

/** roughly two fifths along the route: past the departure manoeuvre, well
 *  short of the destination, and inside the first priced time buckets */
export const DEFAULT_INJECTION_FRACTION = 0.4

export interface InjectionPoint {
  /** index of the chosen vertex in the route's own path */
  index: number
  /** the route cell itself, as [row, col] */
  cell: [number, number]
  /** that cell's centre in projected metres: what the API is sent */
  position: XY
  /** how far along the path the cell actually sits, 0..1 */
  fraction: number
  alongM: number
  totalM: number
}

/**
 * The vertex of `path` nearest `fraction` of the way along it, by length.
 *
 * Endpoints are skipped whenever the path has an interior vertex, so the berg
 * lands on the corridor rather than on the departure or the destination. With
 * a two-vertex path there is no interior, and the nearer endpoint is used.
 */
export function injectionPoint(
  path: ReadonlyArray<readonly number[]> | null | undefined,
  grid: GridGeometry | null | undefined,
  fraction: number = DEFAULT_INJECTION_FRACTION,
): InjectionPoint | null {
  if (!grid?.transform || !path || path.length === 0) return null

  const vertices = path
    .filter((cell) => cell.length >= 2)
    .map((cell) => ({
      cell: [cell[0], cell[1]] as [number, number],
      position: cellCentreXY(grid, cell[0], cell[1]) as XY,
    }))
  if (vertices.length === 0) return null
  if (vertices.length === 1) {
    return {
      index: 0,
      cell: vertices[0].cell,
      position: vertices[0].position,
      fraction: 0,
      alongM: 0,
      totalM: 0,
    }
  }

  //  cumulative length along the path, in projected metres
  const along: number[] = [0]
  for (let i = 1; i < vertices.length; i += 1) {
    const [x0, y0] = vertices[i - 1].position
    const [x1, y1] = vertices[i].position
    along.push(along[i - 1] + Math.hypot(x1 - x0, y1 - y0))
  }
  const totalM = along[along.length - 1]
  const wanted = Math.min(Math.max(fraction, 0), 1) * totalM

  //  the interior vertices, when there are any
  const first = vertices.length > 2 ? 1 : 0
  const last = vertices.length > 2 ? vertices.length - 2 : vertices.length - 1

  let index = first
  let best = Infinity
  for (let i = first; i <= last; i += 1) {
    const gap = Math.abs(along[i] - wanted)
    //  strict, so the earliest vertex wins a tie: the result cannot depend on
    //  iteration order
    if (gap < best) {
      best = gap
      index = i
    }
  }

  return {
    index,
    cell: vertices[index].cell,
    position: vertices[index].position,
    fraction: totalM > 0 ? along[index] / totalM : 0,
    alongM: along[index],
    totalM,
  }
}
