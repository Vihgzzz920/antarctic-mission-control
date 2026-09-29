// Playing a camera plan exactly once.
//
// Separated from both React and OpenLayers so the part that decides WHETHER to
// move can be tested without a map: it takes an `apply` callback and a
// scheduler, and its whole job is
//
//   * run a plan's moves in order, honouring each move's own delay,
//   * do nothing at all when asked to run a plan it has already run, which is
//     what keeps an ordinary React re-render from yanking the view back, and
//   * drop anything still queued the moment the operator takes the map.

import type { CameraMove, CameraPlan } from './camera'

export type CameraApply = (move: CameraMove) => void

export interface CameraScheduler {
  setTimeout: (handler: () => void, ms: number) => number
  clearTimeout: (id: number) => void
}

export interface CameraController {
  /** Apply this plan, unless its key is the one already applied. */
  run: (plan: CameraPlan | null) => void
  /** Forget the moves still waiting to fire; keep the applied key. */
  cancelPending: () => void
  dispose: () => void
  getAppliedKey: () => string | null
  pendingCount: () => number
}

const wallClock: CameraScheduler = {
  setTimeout: (handler, ms) =>
    globalThis.setTimeout(handler, ms) as unknown as number,
  clearTimeout: (id) => globalThis.clearTimeout(id),
}

export function createCameraController(
  apply: CameraApply,
  scheduler: CameraScheduler = wallClock,
): CameraController {
  let appliedKey: string | null = null
  let pending: number[] = []

  const clearPending = () => {
    for (const id of pending) scheduler.clearTimeout(id)
    pending = []
  }

  return {
    run(plan) {
      if (!plan || plan.moves.length === 0) return
      if (plan.key === appliedKey) return // the same geometry: leave the view alone
      clearPending()
      appliedKey = plan.key
      for (const move of plan.moves) {
        if (move.delayMs <= 0) {
          apply(move)
          continue
        }
        const id = scheduler.setTimeout(() => {
          pending = pending.filter((entry) => entry !== id)
          apply(move)
        }, move.delayMs)
        pending.push(id)
      }
    },
    cancelPending: clearPending,
    dispose() {
      clearPending()
      appliedKey = null
    },
    getAppliedKey: () => appliedKey,
    pendingCount: () => pending.length,
  }
}
