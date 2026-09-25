// Reading the forecast the backend already produced, at an arbitrary clock
// position, WITHOUT inventing a prediction.
//
// The model produces a state at 0, 6, 12, 24 and 48 hours and nowhere else. At
// any of those the functions below return that state untouched and mark it
// `exact`. Strictly between two of them they interpolate for DISPLAY ONLY and
// mark it `exact: false`, naming the two model states it sits between, so a
// panel can never present a smoothed frame as a model output.

import type { ForecastState, IcebergForecastRecord } from '../api/types'

export interface ResolvedState {
  hours: number
  latitude: number
  longitude: number
  radiusKm: number
  /** true only when the clock is exactly on a model horizon */
  exact: boolean
  /** the model state at or before this clock position */
  from: ForecastState
  /** the model state after it, when interpolating */
  to: ForecastState | null
  label: string
  extrapolatedUncertainty: boolean
  physicsPositionIsExtrapolated: boolean
}

/** Observed first, then every predicted horizon, in order. */
export function statesOf(record: IcebergForecastRecord): ForecastState[] {
  const predicted = [...record.predicted].sort((a, b) => a.hours - b.hours)
  return record.observed ? [record.observed, ...predicted] : predicted
}

/** Shortest-arc longitude step, so a pair either side of 180 does not sweep
 *  the long way round. */
function lonDelta(from: number, to: number): number {
  let delta = to - from
  while (delta > 180) delta -= 360
  while (delta < -180) delta += 360
  return delta
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t

export function resolveState(
  record: IcebergForecastRecord,
  hours: number,
): ResolvedState | null {
  const states = statesOf(record)
  if (states.length === 0) return null

  const first = states[0]
  const last = states[states.length - 1]
  if (hours <= first.hours) return asExact(first)
  if (hours >= last.hours) return asExact(last)

  for (let i = 0; i < states.length - 1; i += 1) {
    const from = states[i]
    const to = states[i + 1]
    if (hours === from.hours) return asExact(from)
    if (hours > from.hours && hours < to.hours) {
      const t = (hours - from.hours) / (to.hours - from.hours)
      return {
        hours,
        latitude: lerp(from.latitude, to.latitude, t),
        longitude:
          from.longitude + lonDelta(from.longitude, to.longitude) * t,
        radiusKm: lerp(from.radius_km, to.radius_km, t),
        exact: false,
        from,
        to,
        label: `between ${from.label} and ${to.label}`,
        //  a frame that reaches into an extrapolated horizon is extrapolated
        extrapolatedUncertainty:
          from.extrapolated_uncertainty || to.extrapolated_uncertainty,
        physicsPositionIsExtrapolated:
          from.physics_position_is_extrapolated ||
          to.physics_position_is_extrapolated,
      }
    }
  }
  return asExact(last)
}

function asExact(state: ForecastState): ResolvedState {
  return {
    hours: state.hours,
    latitude: state.latitude,
    longitude: state.longitude,
    radiusKm: state.radius_km,
    exact: true,
    from: state,
    to: null,
    label: state.label,
    extrapolatedUncertainty: state.extrapolated_uncertainty,
    physicsPositionIsExtrapolated: state.physics_position_is_extrapolated,
  }
}

/** The model horizon the clock is on, or null when it is between two. */
export function exactHorizon(
  record: IcebergForecastRecord | undefined,
  hours: number,
): ForecastState | null {
  if (!record) return null
  const resolved = resolveState(record, hours)
  return resolved?.exact ? resolved.from : null
}

// ─────────────────────────────────────────────── vessel transit playback
export interface VesselPosition {
  /** the route cell the vessel has reached or is leaving */
  index: number
  row: number
  col: number
  /** fractional position between this cell and the next, 0..1 */
  fraction: number
  nextRow: number
  nextCol: number
  /** seconds since departure */
  elapsed: number
  arrivalTime: number
  bucket: number
}

/**
 * Where the vessel is at `elapsed` seconds, read off the route's OWN arrival
 * times. Nothing about the route is recomputed: this walks the cells the
 * search returned and interpolates between two of them for drawing.
 */
export function vesselAt(
  cells: Array<{ row: number; col: number; arrival_time: number; bucket: number }>,
  elapsed: number,
): VesselPosition | null {
  if (cells.length === 0) return null
  const start = cells[0].arrival_time
  const clock = start + Math.max(0, elapsed)

  for (let i = 0; i < cells.length - 1; i += 1) {
    const here = cells[i]
    const next = cells[i + 1]
    if (clock >= here.arrival_time && clock < next.arrival_time) {
      const span = next.arrival_time - here.arrival_time
      return {
        index: i,
        row: here.row,
        col: here.col,
        fraction: span > 0 ? (clock - here.arrival_time) / span : 0,
        nextRow: next.row,
        nextCol: next.col,
        elapsed: clock - start,
        arrivalTime: here.arrival_time,
        bucket: here.bucket,
      }
    }
  }
  const last = cells[cells.length - 1]
  return {
    index: cells.length - 1,
    row: last.row,
    col: last.col,
    fraction: 0,
    nextRow: last.row,
    nextCol: last.col,
    elapsed: last.arrival_time - start,
    arrivalTime: last.arrival_time,
    bucket: last.bucket,
  }
}

/** Total transit seconds, from the route's own first and last arrival. */
export function transitSeconds(
  cells: Array<{ arrival_time: number }>,
): number {
  if (cells.length < 2) return 0
  return cells[cells.length - 1].arrival_time - cells[0].arrival_time
}
