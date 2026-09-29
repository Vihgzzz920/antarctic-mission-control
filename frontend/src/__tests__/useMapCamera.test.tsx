import { render } from '@testing-library/react'
import { act, useRef } from 'react'
import Map from 'ol/Map'
import View from 'ol/View'
import ViewHint from 'ol/ViewHint'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { centreOf, resolveCameraPlan } from '../map/camera'
import type { CameraInput, CameraPlan } from '../map/camera'
import { useMapCamera } from '../map/useMapCamera'
import { GRID_EXTENT, projectProjection } from '../map/crs'
import { cellCentreXY } from '../map/coords'
import { missionResponse } from './fixtures'

// The hook against a REAL OpenLayers view. camera.ts is unit-tested on its own,
// so what matters here is that a plan reaches the view, that an ordinary
// re-render does not, and that the operator outranks a queued move.

const GRID = missionResponse().grid
const ROUTE = missionResponse().comparison.profiles.risk_oriented.path.map(
  (cell) => cellCentreXY(GRID, cell[0], cell[1]) as [number, number],
)
const MISSION = { start: ROUTE[0], goal: ROUTE[ROUTE.length - 1] }

const input = (over: Partial<CameraInput>): CameraInput => ({
  stage: 'plan',
  gridExtent: GRID_EXTENT,
  mission: null,
  subject: null,
  routes: null,
  hazard: null,
  ...over,
})

const routePlan = () => resolveCameraPlan(input({ routes: [ROUTE] }))
const observePlan = () =>
  resolveCameraPlan(input({ stage: 'observe', mission: MISSION }))

let view: View | null = null
let instance: Map | null = null

function Harness({ plan }: { plan: CameraPlan }) {
  const map = useRef<Map | null>(null)
  if (!map.current) {
    map.current = new Map({
      view: new View({
        projection: projectProjection(),
        center: [0, 0],
        zoom: 3,
      }),
    })
    map.current.setSize([1600, 900])
    view = map.current.getView()
    instance = map.current
  }
  useMapCamera(map, plan)
  return null
}

/** let every queued move and every view animation run to the end */
const settle = (ms = 4000) => act(() => vi.advanceTimersByTime(ms))

beforeEach(() => {
  vi.useFakeTimers()
  view = null
  instance = null
})
afterEach(() => {
  vi.useRealTimers()
})

describe('useMapCamera', () => {
  it('fits the view to the plan it is given', () => {
    const plan = routePlan()
    render(<Harness plan={plan} />)
    const opening = view!.getResolution()!
    settle()

    const centre = view!.getCenter()!
    const wanted = centreOf(plan.moves[0].extent)
    expect(centre[0]).toBeCloseTo(wanted[0], 0)
    expect(centre[1]).toBeCloseTo(wanted[1], 0)
    //  and it zoomed in on the route rather than staying wide
    expect(view!.getResolution()!).toBeLessThan(opening)
  })

  it('leaves the view alone on a re-render with the same geometry', () => {
    const { rerender } = render(<Harness plan={routePlan()} />)
    settle()

    //  the operator pans somewhere of their own
    act(() => view!.setCenter([0, 0]))

    //  a new plan object every render, same geometry -- as React would produce
    for (let i = 0; i < 5; i += 1) {
      rerender(<Harness plan={routePlan()} />)
      settle(100)
    }
    expect(view!.getCenter()).toEqual([0, 0])
  })

  it('moves again when the geometry really changed', () => {
    const { rerender } = render(<Harness plan={routePlan()} />)
    settle()
    act(() => view!.setCenter([0, 0]))

    rerender(<Harness plan={observePlan()} />)
    settle()
    expect(view!.getCenter()).not.toEqual([0, 0])
  })

  it('holds the sector move for its delay, then makes it', () => {
    const plan = observePlan()
    render(<Harness plan={plan} />)

    //  the opening continent shot has landed; the sector has not
    settle(plan.moves[1].delayMs - 100)
    const opening = view!.getResolution()!

    settle(plan.moves[1].durationMs + 500)
    expect(view!.getResolution()!).toBeLessThan(opening)
  })

  it('drops the queued move the moment the operator takes the map', () => {
    const plan = observePlan()
    render(<Harness plan={plan} />)
    settle(200)
    const opening = view!.getResolution()!

    //  the operator pans before the sector move was due
    act(() => {
      view!.setHint(ViewHint.INTERACTING, 1)
      instance!.dispatchEvent('movestart')
      view!.setHint(ViewHint.INTERACTING, -1)
    })
    settle()

    //  the camera never took the view back
    expect(view!.getResolution()!).toBe(opening)
  })
})
