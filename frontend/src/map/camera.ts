// Where the map looks, and why.
//
// This module is PURE: no React, no OpenLayers, no network, no clock. It turns
// data the backend already returned into an extent in projected metres, and
// nothing more. Every coordinate it consumes comes from somewhere real --
//
//   grid bounds      src/generated/crs.ts, emitted from the routing rasters
//   mission          the request's own start/goal cells through the grid
//                    transform the API reported
//   forecast subject the model's own observed and predicted states
//   route            the path the search returned
//   hazard           the simulated iceberg the backend placed, and the
//                    encounter cell it reported
//
// -- so no camera here can invent a position. A frame with no data behind it
// is not guessed: the resolver falls back to a wider frame that does have
// data, and in the last resort to the grid itself.

import { GRID_EXTENT } from '../generated/crs'

export type XY = [number, number]
/** [minX, minY, maxX, maxY] in projected metres (the project CRS). */
export type Extent = [number, number, number, number]

/** Every frame the camera can be asked for. There are no others. */
export const CAMERA_FRAMES = [
  'continent',
  'sector',
  'subject',
  'route',
  'route+hazard',
] as const
export type CameraFrameName = (typeof CAMERA_FRAMES)[number]

/** The stages that request a frame. `respond` is the simulation interaction. */
export const CAMERA_STAGES = ['observe', 'forecast', 'plan', 'respond'] as const
export type CameraStage = (typeof CAMERA_STAGES)[number]

export interface FrameSpec {
  /** fraction of the content's own size added on EACH side */
  padding: number
  /** the frame never spans less than this, so a short route cannot ask for
   *  an absurd zoom and a degenerate extent can never reach the view */
  minSpanM: number
  durationMs: number
}

export const FRAME_SPEC: Record<CameraFrameName, FrameSpec> = {
  continent: { padding: 0.02, minSpanM: 0, durationMs: 0 },
  sector: { padding: 0.35, minSpanM: 350_000, durationMs: 800 },
  subject: { padding: 0.25, minSpanM: 60_000, durationMs: 700 },
  route: { padding: 0.15, minSpanM: 120_000, durationMs: 700 },
  'route+hazard': { padding: 0.18, minSpanM: 120_000, durationMs: 700 },
}

/** How long the opening continent shot is held before the sector move. */
export const OBSERVE_SECTOR_DELAY_MS = 1500
/** How long the route+hazard frame is held before easing to the encounter. */
export const ENCOUNTER_DELAY_MS = 900
export const ENCOUNTER_DURATION_MS = 800
/** The encounter move is only worth making when the hazard is this much
 *  smaller than the frame it is sitting in. */
export const ENCOUNTER_SPAN_RATIO = 2.5
/** How much of the hazard's own uncertainty radius the encounter frame shows. */
export const ENCOUNTER_RADII = 2.2

export interface CameraMove {
  frame: CameraFrameName
  extent: Extent
  durationMs: number
  /** milliseconds to wait before this move starts */
  delayMs: number
}

export interface CameraPlan {
  /** identity of this plan. The controller re-runs only when this changes, so
   *  a re-render that produces the same geometry moves no camera. */
  key: string
  stage: CameraStage
  /** at least one; played in order */
  moves: CameraMove[]
}

export interface CameraSubject {
  id: string
  /** the model's own states for this iceberg, projected */
  points: XY[]
  /** the largest uncertainty radius across those states, in metres */
  maxRadiusM: number
}

export interface CameraHazard {
  centre: XY
  radiusM: number
  /** the route cell the backend reported as the encounter, if it gave one */
  encounter: XY | null
}

export interface CameraInput {
  stage: CameraStage
  /** the routing grid's own bounds */
  gridExtent: Extent
  /** departure and destination, projected; null until the grid is known */
  mission: { start: XY; goal: XY } | null
  subject: CameraSubject | null
  /** one entry per routed profile, each the profile's own path */
  routes: XY[][] | null
  hazard: CameraHazard | null
}

// ───────────────────────────────────────────────────────────── geometry
export function boundsOf(points: readonly XY[]): Extent | null {
  if (points.length === 0) return null
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const [x, y] of points) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue
    if (x < minX) minX = x
    if (y < minY) minY = y
    if (x > maxX) maxX = x
    if (y > maxY) maxY = y
  }
  if (!Number.isFinite(minX) || !Number.isFinite(minY)) return null
  return [minX, minY, maxX, maxY]
}

export function unionExtents(a: Extent, b: Extent): Extent {
  return [
    Math.min(a[0], b[0]),
    Math.min(a[1], b[1]),
    Math.max(a[2], b[2]),
    Math.max(a[3], b[3]),
  ]
}

export function circleExtent(centre: XY, radiusM: number): Extent {
  const r = Math.max(0, radiusM)
  return [centre[0] - r, centre[1] - r, centre[0] + r, centre[1] + r]
}

/** width and height of an extent. */
export function sizeOf(extent: Extent): [number, number] {
  return [extent[2] - extent[0], extent[3] - extent[1]]
}

/** the larger of the two dimensions: what a fit is governed by. */
export function spanOf(extent: Extent): number {
  const [width, height] = sizeOf(extent)
  return Math.max(width, height)
}

export function centreOf(extent: Extent): XY {
  return [(extent[0] + extent[2]) / 2, (extent[1] + extent[3]) / 2]
}

/**
 * Grow an extent about its own centre: `padding` of its size on each side,
 * and never narrower than `minSpanM` on either axis.
 */
export function expand(extent: Extent, padding: number, minSpanM = 0): Extent {
  const [cx, cy] = centreOf(extent)
  const [width, height] = sizeOf(extent)
  const halfWidth = Math.max((width * (1 + 2 * padding)) / 2, minSpanM / 2)
  const halfHeight = Math.max((height * (1 + 2 * padding)) / 2, minSpanM / 2)
  return [cx - halfWidth, cy - halfHeight, cx + halfWidth, cy + halfHeight]
}

/** Apply a frame's own padding and floor to the content it is framing. */
export function frameExtent(frame: CameraFrameName, content: Extent): Extent {
  const spec = FRAME_SPEC[frame]
  return expand(content, spec.padding, spec.minSpanM)
}

function move(
  frame: CameraFrameName,
  content: Extent,
  overrides: Partial<Pick<CameraMove, 'durationMs' | 'delayMs'>> = {},
): CameraMove {
  return {
    frame,
    extent: frameExtent(frame, content),
    durationMs: overrides.durationMs ?? FRAME_SPEC[frame].durationMs,
    delayMs: overrides.delayMs ?? 0,
  }
}

// ─────────────────────────────────────────────── the content of each frame
/** The mission sector: the two endpoints the request itself carries. */
export function missionContent(
  mission: { start: XY; goal: XY } | null,
): Extent | null {
  return mission ? boundsOf([mission.start, mission.goal]) : null
}

/**
 * The forecast subject: its whole track plus the largest uncertainty envelope
 * the model gave it.
 *
 * Built from EVERY state, not from the clock, so scrubbing or playing the
 * timeline never moves the camera -- the frame a viewer is watching the drift
 * inside stays still while the drift happens.
 */
export function subjectContent(subject: CameraSubject | null): Extent | null {
  if (!subject || subject.points.length === 0) return null
  let extent: Extent | null = null
  for (const point of subject.points) {
    const ring = circleExtent(point, subject.maxRadiusM)
    extent = extent ? unionExtents(extent, ring) : ring
  }
  return extent
}

/** Every routed profile's own path, so switching profiles moves no camera. */
export function routeContent(routes: XY[][] | null): Extent | null {
  if (!routes || routes.length === 0) return null
  let extent: Extent | null = null
  for (const path of routes) {
    const bounds = boundsOf(path)
    if (!bounds) continue
    extent = extent ? unionExtents(extent, bounds) : bounds
  }
  return extent
}

/** The route the operator is looking at, and the iceberg now on it. */
export function hazardContent(
  routes: XY[][] | null,
  hazard: CameraHazard | null,
): Extent | null {
  if (!hazard) return null
  const berg = circleExtent(hazard.centre, hazard.radiusM)
  const route = routeContent(routes)
  return route ? unionExtents(route, berg) : berg
}

/** The encounter the backend reported, framed by the hazard's own radius. */
export function encounterContent(hazard: CameraHazard | null): Extent | null {
  if (!hazard?.encounter) return null
  return circleExtent(hazard.encounter, hazard.radiusM * ENCOUNTER_RADII)
}

// ────────────────────────────────────────────────────────────── the plan
function keyOf(stage: CameraStage, moves: CameraMove[], tag: string): string {
  const frames = moves
    .map((m) => `${m.frame}@${m.extent.map((n) => Math.round(n)).join(',')}`)
    .join('|')
  return `${stage}${tag ? `:${tag}` : ''}:${frames}`
}

/**
 * The frame sequence for one stage, given the geometry that exists right now.
 *
 * Falls back rather than guesses: a stage whose own content is not available
 * yet asks for the widest frame that is.
 */
export function resolveCameraPlan(input: CameraInput): CameraPlan {
  const grid = input.gridExtent ?? GRID_EXTENT
  const continent = move('continent', grid)
  const mission = missionContent(input.mission)
  const sector = mission ? move('sector', mission) : null
  const route = routeContent(input.routes)
  let tag = ''
  let moves: CameraMove[]

  switch (input.stage) {
    case 'observe': {
      //  the continent first, so the viewer is placed, then in to the sector
      //  the mission is actually in
      moves = sector
        ? [continent, { ...sector, delayMs: OBSERVE_SECTOR_DELAY_MS }]
        : [continent]
      break
    }

    case 'forecast': {
      const subject = subjectContent(input.subject)
      if (subject) {
        tag = input.subject?.id ?? ''
        moves = [move('subject', subject)]
      } else {
        moves = [sector ?? continent]
      }
      break
    }

    case 'plan': {
      moves = route ? [move('route', route)] : [sector ?? continent]
      break
    }

    case 'respond': {
      const union = hazardContent(input.routes, input.hazard)
      if (!union) {
        moves = route ? [move('route', route)] : [sector ?? continent]
        break
      }
      const first = move('route+hazard', union)
      const encounter = encounterContent(input.hazard)
      //  only worth easing in when the hazard would otherwise be small in the
      //  frame it is sitting in
      const worthIt =
        encounter !== null &&
        spanOf(first.extent) > ENCOUNTER_SPAN_RATIO * spanOf(encounter)
      moves =
        worthIt && encounter
          ? [
              first,
              move('route+hazard', encounter, {
                delayMs: ENCOUNTER_DELAY_MS,
                durationMs: ENCOUNTER_DURATION_MS,
              }),
            ]
          : [first]
      break
    }
  }

  return { key: keyOf(input.stage, moves, tag), stage: input.stage, moves }
}
