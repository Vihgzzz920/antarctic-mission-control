// Which iceberg the Forecast stage opens on.
//
// The backend returns every iceberg it has a clean observation for on the
// mission date, in its own id order. Opening on whichever one happens to be
// first is an accident of that ordering: on the shipped demo it is an iceberg
// some 3,500 km from the mission, so the Forecast stage would begin by looking
// somewhere the voyage never goes.
//
// So the default is chosen by RELEVANCE TO THE MISSION instead: how near each
// iceberg comes to the departure-to-destination corridor, measured against the
// whole segment rather than either endpoint, over every state the model
// produced for it. Nothing here judges hazard. An iceberg near the corridor is
// not thereby dangerous, and one far from it is not thereby safe -- exposure,
// and what it costs a route, is decided by the backend's own exposure field
// and nothing in this file. This only decides where the camera and the panel
// start.
//
// MEASUREMENT: distances are Euclidean in the project CRS (EPSG:3976), which
// is polar stereographic and therefore stretches by a few percent away from
// its standard parallel at 70S. That is fine for ordering candidates hundreds
// or thousands of kilometres apart, and it is used for NOTHING else: no
// exposure, no cost, no radius and no route is computed from it.

import type { IcebergForecastRecord } from '../api/types'
import { projectLonLat } from './ForecastLayers'
import type { XY } from './camera'

export interface MissionCorridor {
  start: XY
  goal: XY
}

export interface SubjectRanking {
  icebergId: string
  /** nearest approach to the corridor across the iceberg's own model states */
  projectedDistanceM: number
  /** the horizon at which that nearest approach happens, in hours */
  atHours: number
}

/**
 * Shortest distance from a point to the mission corridor, treated as the
 * segment between departure and destination -- not to whichever endpoint
 * happens to be nearer, which would rank an iceberg beside the middle of a
 * long crossing as irrelevant.
 */
export function distanceToCorridor(point: XY, corridor: MissionCorridor): number {
  const [px, py] = point
  const [ax, ay] = corridor.start
  const [bx, by] = corridor.goal
  const dx = bx - ax
  const dy = by - ay
  const lengthSquared = dx * dx + dy * dy
  if (lengthSquared === 0) return Math.hypot(px - ax, py - ay)
  //  where the foot of the perpendicular falls along the segment, clamped to
  //  its ends so a berg beyond either end measures from that end
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSquared))
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy))
}

/** Observed first, then every predicted horizon: the iceberg's own states. */
function statesOf(record: IcebergForecastRecord) {
  return [record.observed, ...record.predicted].filter(
    (state): state is NonNullable<typeof state> => state != null,
  )
}

/**
 * Every subject, nearest the corridor first.
 *
 * Deterministic for a given mission and payload: ties, and the case where
 * there is no corridor yet, fall back to the iceberg id, so the result never
 * depends on the order the backend listed them in.
 */
export function rankForecastSubjects(
  records: readonly IcebergForecastRecord[] | null | undefined,
  corridor: MissionCorridor | null,
): SubjectRanking[] {
  if (!records || records.length === 0) return []

  const ranked = records.map((record) => {
    let projectedDistanceM = Infinity
    let atHours = 0
    if (corridor) {
      for (const state of statesOf(record)) {
        const distance = distanceToCorridor(
          projectLonLat(state.longitude, state.latitude),
          corridor,
        )
        if (distance < projectedDistanceM) {
          projectedDistanceM = distance
          atHours = state.hours
        }
      }
    }
    return { icebergId: record.iceberg_id, projectedDistanceM, atHours }
  })

  return ranked.sort(
    (a, b) =>
      a.projectedDistanceM - b.projectedDistanceM ||
      a.icebergId.localeCompare(b.icebergId),
  )
}

/**
 * The iceberg the Forecast stage opens on, or null when the forecast is empty.
 * An explicit choice by the operator is not made here: the caller keeps that
 * separately and it always wins.
 */
export function defaultForecastSubject(
  records: readonly IcebergForecastRecord[] | null | undefined,
  corridor: MissionCorridor | null,
): string | null {
  return rankForecastSubjects(records, corridor)[0]?.icebergId ?? null
}
