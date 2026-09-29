import { describe, expect, it, vi } from 'vitest'

import {
  CAMERA_FRAMES,
  CAMERA_STAGES,
  ENCOUNTER_DELAY_MS,
  FRAME_SPEC,
  OBSERVE_SECTOR_DELAY_MS,
  boundsOf,
  circleExtent,
  expand,
  resolveCameraPlan,
  sizeOf,
  spanOf,
  unionExtents,
} from '../map/camera'
import type { CameraFrameName, CameraInput, Extent, XY } from '../map/camera'
import { createCameraController } from '../map/cameraController'
import type { CameraScheduler } from '../map/cameraController'
import { cellCentreXY } from '../map/coords'
import { projectLonLat } from '../map/ForecastLayers'
import { GRID_EXTENT } from '../generated/crs'
import { forecastResponse, missionResponse, simulationResponse } from './fixtures'

// Everything the camera is given here comes out of the RECORDED backend
// payloads: the grid transform the API reported, the cells the request carries,
// the path the search returned, the model's own forecast states and the
// iceberg the simulation placed. No coordinate in this file was typed by hand.

const GRID = missionResponse().grid
const FORECAST = forecastResponse()
const at = (cell: readonly number[]): XY =>
  cellCentreXY(GRID, cell[0], cell[1]) as XY

const contains = (outer: Extent, inner: Extent) =>
  outer[0] <= inner[0] &&
  outer[1] <= inner[1] &&
  outer[2] >= inner[2] &&
  outer[3] >= inner[3]

const holds = (extent: Extent, [x, y]: XY) =>
  x >= extent[0] && x <= extent[2] && y >= extent[1] && y <= extent[3]

const MISSION = {
  start: at(missionResponse().request.start),
  goal: at(missionResponse().request.goal),
}

const ROUTE_PATHS = [missionResponse().comparison.profiles.risk_oriented.path.map(at)]

const SUBJECT_RECORD = FORECAST.icebergs[0]
const SUBJECT_STATES = [
  SUBJECT_RECORD.observed!,
  ...SUBJECT_RECORD.predicted,
]
const SUBJECT = {
  id: SUBJECT_RECORD.iceberg_id,
  points: SUBJECT_STATES.map((state) =>
    projectLonLat(state.longitude, state.latitude),
  ),
  maxRadiusM: Math.max(...SUBJECT_STATES.map((s) => s.radius_km)) * 1000,
}

const BERG = simulationResponse().simulation!
const HAZARD = {
  centre: at([BERG.row, BERG.col]),
  radiusM: BERG.radius_km * 1000,
  encounter: at(BERG.encounter_cell),
}

const base: CameraInput = {
  stage: 'observe',
  gridExtent: GRID_EXTENT,
  mission: null,
  subject: null,
  routes: null,
  hazard: null,
}

const plan = (over: Partial<CameraInput>) =>
  resolveCameraPlan({ ...base, ...over })

// ───────────────────────────────────────────────────────────── frame names
describe('camera frames', () => {
  it('has exactly the five named frames', () => {
    expect([...CAMERA_FRAMES]).toEqual([
      'continent',
      'sector',
      'subject',
      'route',
      'route+hazard',
    ])
    expect([...CAMERA_STAGES]).toEqual([
      'observe',
      'forecast',
      'plan',
      'respond',
    ])
  })

  it('rejects a frame name that is not one of them', () => {
    // @ts-expect-error 'orbit' is not a CameraFrameName
    const notAFrame: CameraFrameName = 'orbit'
    void notAFrame
    // @ts-expect-error and there is no spec for one either
    void FRAME_SPEC.orbit
    expect(Object.keys(FRAME_SPEC).sort()).toEqual([...CAMERA_FRAMES].sort())
  })

  it('emits only named frames, whatever the stage and whatever is missing', () => {
    for (const stage of CAMERA_STAGES) {
      for (const over of [
        {},
        { mission: MISSION },
        { mission: MISSION, subject: SUBJECT },
        { mission: MISSION, routes: ROUTE_PATHS },
        { mission: MISSION, routes: ROUTE_PATHS, hazard: HAZARD },
      ]) {
        const resolved = plan({ stage, ...over })
        expect(resolved.moves.length).toBeGreaterThan(0)
        for (const move of resolved.moves) {
          expect(CAMERA_FRAMES).toContain(move.frame)
          expect(Number.isFinite(spanOf(move.extent))).toBe(true)
          expect(spanOf(move.extent)).toBeGreaterThan(0)
        }
      }
    }
  })

  it('states the padding the audit promised', () => {
    expect(FRAME_SPEC.route.padding).toBeCloseTo(0.15, 5)
  })
})

// ───────────────────────────────────────────────────────────────── geometry
describe('camera geometry', () => {
  it('bounds a set of points and ignores an empty one', () => {
    expect(boundsOf([[1, 2], [5, -3]])).toEqual([1, -3, 5, 2])
    expect(boundsOf([])).toBeNull()
  })

  it('pads by a fraction of the content on each side', () => {
    const padded = expand([0, 0, 100, 200], 0.15)
    expect(sizeOf(padded)).toEqual([130, 260])
    //  about its own centre
    expect(padded[0] + padded[2]).toBeCloseTo(100, 6)
  })

  it('never returns a frame narrower than the floor', () => {
    const padded = expand([0, 0, 10, 10], 0.15, 1000)
    expect(sizeOf(padded)).toEqual([1000, 1000])
    //  and a degenerate extent still comes back usable
    expect(spanOf(expand([5, 5, 5, 5], 0.2, 400))).toBe(400)
  })

  it('unions and builds circle extents', () => {
    expect(unionExtents([0, 0, 1, 1], [-2, 3, 0, 4])).toEqual([-2, 0, 1, 4])
    expect(circleExtent([10, 20], 5)).toEqual([5, 15, 15, 25])
  })
})

// ───────────────────────────────────────────────────────────────── observe
describe('OBSERVE', () => {
  it('holds the continent while no mission geometry exists', () => {
    const resolved = plan({ stage: 'observe' })
    expect(resolved.moves).toHaveLength(1)
    expect(resolved.moves[0].frame).toBe('continent')
    expect(contains(resolved.moves[0].extent, GRID_EXTENT)).toBe(true)
  })

  it('opens on the continent and then moves to the mission sector', () => {
    const resolved = plan({ stage: 'observe', mission: MISSION })
    expect(resolved.moves.map((m) => m.frame)).toEqual(['continent', 'sector'])

    const [continent, sector] = resolved.moves
    expect(continent.delayMs).toBe(0)
    expect(sector.delayMs).toBe(OBSERVE_SECTOR_DELAY_MS)
    expect(sector.durationMs).toBeGreaterThanOrEqual(600)
    expect(sector.durationMs).toBeLessThanOrEqual(800)

    //  the sector is derived from the request's own cells
    expect(holds(sector.extent, MISSION.start)).toBe(true)
    expect(holds(sector.extent, MISSION.goal)).toBe(true)
    //  and it is a sector, not the continent
    expect(spanOf(sector.extent)).toBeLessThan(spanOf(GRID_EXTENT) / 5)
  })
})

// ──────────────────────────────────────────────────────────────── forecast
describe('FORECAST', () => {
  it('fits the subject, not the grid', () => {
    const resolved = plan({
      stage: 'forecast',
      mission: MISSION,
      subject: SUBJECT,
    })
    expect(resolved.moves).toHaveLength(1)
    const [move] = resolved.moves
    expect(move.frame).toBe('subject')
    expect(move.extent).not.toEqual(GRID_EXTENT)
    //  a hard floor on "not the whole continent": under 2% of the grid span
    expect(spanOf(move.extent)).toBeLessThan(spanOf(GRID_EXTENT) * 0.02)
  })

  it('carries the whole track and the largest uncertainty envelope', () => {
    const [move] = plan({ stage: 'forecast', subject: SUBJECT }).moves
    for (const point of SUBJECT.points) {
      expect(holds(move.extent, point)).toBe(true)
      expect(contains(move.extent, circleExtent(point, SUBJECT.maxRadiusM))).toBe(
        true,
      )
    }
    //  the subject fills a meaningful share of what is framed
    const envelope = spanOf(circleExtent(SUBJECT.points[0], SUBJECT.maxRadiusM))
    expect(envelope / spanOf(move.extent)).toBeGreaterThan(0.4)
  })

  it('does not move as the forecast clock moves', () => {
    //  the subject frame is built from every model state, so there is no clock
    //  in the input at all: the same subject always resolves to the same key
    const a = plan({ stage: 'forecast', subject: SUBJECT })
    const b = plan({ stage: 'forecast', subject: { ...SUBJECT } })
    expect(b.key).toBe(a.key)
  })

  it('falls back to the sector when the forecast has no subject yet', () => {
    const resolved = plan({ stage: 'forecast', mission: MISSION })
    expect(resolved.moves.map((m) => m.frame)).toEqual(['sector'])
  })
})

// ──────────────────────────────────────────────────────────────────── plan
describe('PLAN', () => {
  it('fits the geometry the search returned', () => {
    const resolved = plan({
      stage: 'plan',
      mission: MISSION,
      routes: ROUTE_PATHS,
    })
    expect(resolved.moves).toHaveLength(1)
    const [move] = resolved.moves
    expect(move.frame).toBe('route')
    for (const vertex of ROUTE_PATHS[0]) {
      expect(holds(move.extent, vertex)).toBe(true)
    }
    expect(spanOf(move.extent)).toBeLessThan(spanOf(GRID_EXTENT) * 0.05)
  })

  it('pads the route bounds by about 15%', () => {
    const bounds = boundsOf(ROUTE_PATHS[0])!
    const [move] = plan({ stage: 'plan', routes: ROUTE_PATHS }).moves
    //  the governing dimension is the padded one (the other may be on the floor)
    expect(spanOf(move.extent) / spanOf(bounds)).toBeCloseTo(1.3, 2)
  })

  it('frames every routed profile, so changing the selection moves nothing', () => {
    const comparison = missionResponse().comparison.profiles
    const everyPath = [
      comparison.fastest.path.map(at),
      comparison.risk_oriented.path.map(at),
      comparison.shortest_distance.path.map(at),
    ]
    const all = plan({ stage: 'plan', routes: everyPath })
    const reordered = plan({ stage: 'plan', routes: [...everyPath].reverse() })
    expect(reordered.key).toBe(all.key)
    for (const path of everyPath) {
      for (const vertex of path) expect(holds(all.moves[0].extent, vertex)).toBe(true)
    }
  })

  it('falls back to the sector before any route exists', () => {
    expect(plan({ stage: 'plan', mission: MISSION }).moves[0].frame).toBe('sector')
  })
})

// ───────────────────────────────────────────────────────────────── respond
describe('RESPOND', () => {
  const resolved = plan({
    stage: 'respond',
    mission: MISSION,
    routes: ROUTE_PATHS,
    hazard: HAZARD,
  })

  it('frames the route together with the injected iceberg', () => {
    expect(resolved.moves[0].frame).toBe('route+hazard')
    const frame = resolved.moves[0].extent
    for (const vertex of ROUTE_PATHS[0]) expect(holds(frame, vertex)).toBe(true)
    expect(contains(frame, circleExtent(HAZARD.centre, HAZARD.radiusM))).toBe(true)
  })

  it('then eases to the encounter the backend reported', () => {
    expect(resolved.moves).toHaveLength(2)
    const [, encounter] = resolved.moves
    expect(encounter.frame).toBe('route+hazard')
    expect(encounter.delayMs).toBe(ENCOUNTER_DELAY_MS)
    expect(holds(encounter.extent, HAZARD.encounter!)).toBe(true)
    expect(spanOf(encounter.extent)).toBeLessThan(spanOf(resolved.moves[0].extent))
  })

  it('skips the second move when the hazard already fills the frame', () => {
    const huge = plan({
      stage: 'respond',
      routes: ROUTE_PATHS,
      hazard: { ...HAZARD, radiusM: HAZARD.radiusM * 40 },
    })
    expect(huge.moves).toHaveLength(1)
  })

  it('falls back to the route when no iceberg has been injected', () => {
    const none = plan({ stage: 'respond', routes: ROUTE_PATHS })
    expect(none.moves.map((m) => m.frame)).toEqual(['route'])
  })
})

// ────────────────────────────────────────────────────────────── controller
describe('camera controller', () => {
  const fakeTimers = () => {
    const queue = new Map<number, () => void>()
    let next = 1
    const scheduler: CameraScheduler = {
      setTimeout: (handler) => {
        const id = next
        next += 1
        queue.set(id, handler)
        return id
      },
      clearTimeout: (id) => {
        queue.delete(id)
      },
    }
    return {
      scheduler,
      fireAll: () => {
        for (const handler of [...queue.values()]) handler()
        queue.clear()
      },
      size: () => queue.size,
    }
  }

  it('applies a plan once and ignores the same plan again', () => {
    const apply = vi.fn()
    const timers = fakeTimers()
    const controller = createCameraController(apply, timers.scheduler)
    const resolved = plan({ stage: 'plan', routes: ROUTE_PATHS })

    controller.run(resolved)
    expect(apply).toHaveBeenCalledTimes(1)

    //  twenty ordinary re-renders, same geometry: the view is left alone
    for (let i = 0; i < 20; i += 1) {
      controller.run(plan({ stage: 'plan', routes: ROUTE_PATHS }))
    }
    expect(apply).toHaveBeenCalledTimes(1)
    expect(controller.getAppliedKey()).toBe(resolved.key)
  })

  it('applies again when the geometry really changed', () => {
    const apply = vi.fn()
    const controller = createCameraController(apply, fakeTimers().scheduler)
    controller.run(plan({ stage: 'plan', routes: ROUTE_PATHS }))
    controller.run(plan({ stage: 'forecast', subject: SUBJECT }))
    expect(apply).toHaveBeenCalledTimes(2)
    expect(apply.mock.calls[1][0].frame).toBe('subject')
  })

  it('holds a delayed move until its delay elapses', () => {
    const apply = vi.fn()
    const timers = fakeTimers()
    const controller = createCameraController(apply, timers.scheduler)
    controller.run(plan({ stage: 'observe', mission: MISSION }))

    expect(apply).toHaveBeenCalledTimes(1)
    expect(apply.mock.calls[0][0].frame).toBe('continent')
    expect(controller.pendingCount()).toBe(1)

    timers.fireAll()
    expect(apply).toHaveBeenCalledTimes(2)
    expect(apply.mock.calls[1][0].frame).toBe('sector')
  })

  it('drops a queued move when the operator takes the map', () => {
    const apply = vi.fn()
    const timers = fakeTimers()
    const controller = createCameraController(apply, timers.scheduler)
    controller.run(plan({ stage: 'observe', mission: MISSION }))

    controller.cancelPending()
    timers.fireAll()
    expect(apply).toHaveBeenCalledTimes(1) // the continent shot only
    expect(controller.pendingCount()).toBe(0)
  })

  it('does nothing with no plan, and forgets everything on dispose', () => {
    const apply = vi.fn()
    const controller = createCameraController(apply, fakeTimers().scheduler)
    controller.run(null)
    expect(apply).not.toHaveBeenCalled()
    controller.run(plan({ stage: 'plan', routes: ROUTE_PATHS }))
    controller.dispose()
    expect(controller.getAppliedKey()).toBeNull()
  })
})
