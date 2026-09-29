// The camera, wired to one OpenLayers view.
//
// The hook owns no geometry and makes no decisions: camera.ts decides where to
// look, cameraController.ts decides whether to move, and this only carries a
// move to the view. It also gets out of the way -- as soon as the operator
// pans or zooms, anything the plan still had queued is dropped, and the view
// stays exactly where they left it until the stage or the geometry changes.

import { useEffect, useRef } from 'react'
import { easeOut } from 'ol/easing'
import type Map from 'ol/Map'

import type { CameraMove, CameraPlan } from './camera'
import { createCameraController } from './cameraController'
import type { CameraController, CameraScheduler } from './cameraController'

export function useMapCamera(
  map: React.MutableRefObject<Map | null>,
  plan: CameraPlan | null,
  scheduler?: CameraScheduler,
): void {
  const controller = useRef<CameraController | null>(null)

  if (!controller.current) {
    const applyMove = (move: CameraMove) => {
      const view = map.current?.getView()
      if (!view) return
      view.fit(
        move.extent,
        move.durationMs > 0
          ? { duration: move.durationMs, easing: easeOut }
          : undefined,
      )
    }
    controller.current = scheduler
      ? createCameraController(applyMove, scheduler)
      : createCameraController(applyMove)
  }

  useEffect(() => {
    const current = controller.current
    return () => current?.dispose()
  }, [])

  useEffect(() => {
    controller.current?.run(plan)
  }, [plan])

  //  the operator's hand beats the schedule
  useEffect(() => {
    const instance = map.current
    if (!instance) return
    const onMoveStart = () => {
      if (instance.getView().getInteracting()) {
        controller.current?.cancelPending()
      }
    }
    instance.on('movestart', onMoveStart)
    return () => {
      instance.un('movestart', onMoveStart)
    }
  }, [map])
}
