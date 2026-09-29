// What the route was actually priced against, in time.
//
// The forecast can be displayed at 6, 12, 24 and 48 hours. The ROUTER prices
// iceberg exposure from the exposure fields that were loaded into it, one per
// time bucket, and a route only ever meets the fields for the buckets it
// actually travels through. Those are two different sets, and collapsing them
// would let the screen imply that a 48 h forecast state had a hand in a cost
// it never touched.
//
// So this reads the truth off three real things and invents nothing:
//
//   profile.buckets_used        the buckets THIS route's own cells landed in
//   health.exposure_layers      bucket -> horizon hours, from the backend
//   forecast.horizons           the horizons the forecast can display
//
// A bucket with no matching field is not quietly dropped: it is reported, and
// the caller shows nothing rather than guessing. (The backend does the same
// thing: iceberg_navigation_cost refuses a missing bucket rather than reading
// it as zero exposure.)

export interface ExposureFieldMeta {
  bucket: number
  hours: number
  radiusKm: number | null
  label: string | null
  icebergCount: number | null
}

export interface PricedHorizon {
  /** the buckets this route's cells were priced in */
  buckets: number[]
  /** the exposure field behind each of those buckets, in bucket order */
  fields: ExposureFieldMeta[]
  /** every priced horizon, in hours, ascending */
  hours: number[]
  /** the last horizon the route was priced against; what PLAN pins to */
  lastHours: number | null
  /** forecast horizons that exist for display but priced nothing */
  displayOnlyHours: number[]
  /** buckets the route used for which no exposure field was described */
  bucketsWithoutField: number[]
  /** true when every bucket the route used maps to a known field */
  known: boolean
}

export const UNPRICED: PricedHorizon = {
  buckets: [],
  fields: [],
  hours: [],
  lastHours: null,
  displayOnlyHours: [],
  bucketsWithoutField: [],
  known: false,
}

/** The backend's exposure-layer metadata, parsed without assuming a shape. */
export function exposureFields(
  layers: ReadonlyArray<Record<string, unknown>> | null | undefined,
): ExposureFieldMeta[] {
  if (!layers) return []
  const out: ExposureFieldMeta[] = []
  for (const layer of layers) {
    const bucket = Number(layer.bucket)
    const hours = Number(layer.horizon_hours)
    if (!Number.isFinite(bucket) || !Number.isFinite(hours)) continue
    const radius = Number(layer.radius_km)
    const count = Number(layer.iceberg_count)
    out.push({
      bucket,
      hours,
      radiusKm: Number.isFinite(radius) ? radius : null,
      label: typeof layer.label === 'string' ? layer.label : null,
      icebergCount: Number.isFinite(count) ? count : null,
    })
  }
  return out.sort((a, b) => a.bucket - b.bucket)
}

/**
 * The temporal provenance of one route.
 *
 * `bucketsUsed` comes from the route result itself, so a short voyage that
 * never leaves the first bucket reports the first field only -- the answer
 * follows the route, not a fixed assumption about which field is "the" one.
 */
export function resolvePricedHorizon(
  bucketsUsed: readonly number[] | null | undefined,
  fields: readonly ExposureFieldMeta[],
  forecastHorizons: readonly number[] = [],
): PricedHorizon {
  const buckets = [...new Set((bucketsUsed ?? []).map(Number))]
    .filter((bucket) => Number.isFinite(bucket))
    .sort((a, b) => a - b)
  if (buckets.length === 0) return UNPRICED

  const byBucket = new Map(fields.map((field) => [field.bucket, field]))
  const used: ExposureFieldMeta[] = []
  const missing: number[] = []
  for (const bucket of buckets) {
    const field = byBucket.get(bucket)
    if (field) used.push(field)
    else missing.push(bucket)
  }

  const hours = [...new Set(used.map((field) => field.hours))].sort(
    (a, b) => a - b,
  )
  const priced = new Set(hours)
  const displayOnlyHours = [...new Set(forecastHorizons)]
    .filter((hour) => !priced.has(hour))
    .sort((a, b) => a - b)

  //  a route that touched a bucket with no field has no complete answer, so
  //  it reports none: the caller must not draw a horizon on a guess
  const known = missing.length === 0 && hours.length > 0
  return {
    buckets,
    fields: used,
    hours,
    lastHours: known ? hours[hours.length - 1] : null,
    displayOnlyHours,
    bucketsWithoutField: missing,
    known,
  }
}

const list = (values: readonly number[]): string =>
  values.length <= 1
    ? values.map((value) => `+${value}h`).join('')
    : `${values.slice(0, -1).map((value) => `+${value}h`).join(', ')} and +${
        values[values.length - 1]
      }h`

/**
 * One sentence an operator can act on, built from the numbers above.
 *
 * It never says a route is safe, optimal or collision-free; it says which
 * exposure fields priced it and how far the forecast runs past them.
 */
export function pricedHorizonSummary(priced: PricedHorizon): string {
  if (!priced.known) {
    return priced.bucketsWithoutField.length > 0
      ? `Priced horizon unavailable: this route used time ${
          priced.bucketsWithoutField.length === 1 ? 'bucket' : 'buckets'
        } ${priced.bucketsWithoutField.join(', ')}, which the backend described no exposure field for. No iceberg state is shown beside it.`
      : 'Priced horizon unavailable: the backend described no exposure field for this route. No iceberg state is shown beside it.'
  }
  const priced_ = `Modelled iceberg exposure evaluated through ${list(
    priced.hours,
  )}.`
  if (priced.displayOnlyHours.length === 0) return priced_
  return `${priced_} The forecast runs to +${
    priced.displayOnlyHours[priced.displayOnlyHours.length - 1]
  }h; those later states are shown for forecasting and priced nothing on this route.`
}
