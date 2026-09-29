import { describe, expect, it } from 'vitest'

import {
  UNPRICED,
  exposureFields,
  pricedHorizonSummary,
  resolvePricedHorizon,
} from '../map/pricedHorizon'
import { forecastResponse, missionResponse } from './fixtures'

// The exposure-layer metadata below is the backend's own, recorded from
// GET /api/health, and the buckets come from the recorded route result. The
// point of every test here is the same: the priced horizon is read off the
// route and the configuration, never assumed.

const LAYERS = [
  {
    label: '06h',
    bucket: 0,
    horizon_hours: 6.0,
    radius_km: 9.05053,
    iceberg_count: 11,
    extrapolated_uncertainty: false,
    non_zero_cells: 76,
  },
  {
    label: '12h',
    bucket: 1,
    horizon_hours: 12.0,
    radius_km: 12.799382,
    iceberg_count: 11,
    extrapolated_uncertainty: false,
    non_zero_cells: 147,
  },
]

const FIELDS = exposureFields(LAYERS)
const FORECAST_HOURS = forecastResponse().horizons.map((h) => h.hours)
const ROUTE_BUCKETS = missionResponse().comparison.profiles.risk_oriented
  .buckets_used

describe('reading the exposure configuration', () => {
  it('parses the backend metadata', () => {
    expect(FIELDS.map((field) => [field.bucket, field.hours])).toEqual([
      [0, 6],
      [1, 12],
    ])
    expect(FIELDS[1].icebergCount).toBe(11)
    expect(FIELDS[1].radiusKm).toBeCloseTo(12.799382, 6)
  })

  it('ignores an entry it cannot read rather than inventing one', () => {
    expect(exposureFields([{ bucket: 'x', horizon_hours: 6 }])).toEqual([])
    expect(exposureFields([{ bucket: 0 }])).toEqual([])
    expect(exposureFields(null)).toEqual([])
    expect(exposureFields(undefined)).toEqual([])
  })

  it('sorts by bucket, whatever order the backend listed them in', () => {
    expect(
      exposureFields([...LAYERS].reverse()).map((field) => field.bucket),
    ).toEqual([0, 1])
  })
})

describe('the priced horizon of a route', () => {
  it('follows the buckets the route itself reported', () => {
    expect(ROUTE_BUCKETS).toEqual([0, 1])
    const priced = resolvePricedHorizon(ROUTE_BUCKETS, FIELDS, FORECAST_HOURS)
    expect(priced.known).toBe(true)
    expect(priced.hours).toEqual([6, 12])
    expect(priced.lastHours).toBe(12)
    expect(priced.fields.map((field) => field.label)).toEqual(['06h', '12h'])
  })

  it('reports ONE field for a route that never leaves the first bucket', () => {
    //  the answer is not a constant: a shorter voyage prices against less
    const priced = resolvePricedHorizon([0], FIELDS, FORECAST_HOURS)
    expect(priced.hours).toEqual([6])
    expect(priced.lastHours).toBe(6)
    expect(priced.displayOnlyHours).toEqual([12, 24, 48])
  })

  it('names the horizons the forecast shows but priced nothing', () => {
    const priced = resolvePricedHorizon(ROUTE_BUCKETS, FIELDS, FORECAST_HOURS)
    expect(FORECAST_HOURS).toEqual([6, 12, 24, 48])
    expect(priced.displayOnlyHours).toEqual([24, 48])
    for (const hours of priced.displayOnlyHours) {
      expect(priced.hours).not.toContain(hours)
    }
  })

  it('refuses an answer when a bucket has no exposure field', () => {
    //  a longer voyage reaches bucket 2, which the configuration has no field
    //  for -- the backend refuses that too, rather than reading it as zero
    const priced = resolvePricedHorizon([0, 1, 2], FIELDS, FORECAST_HOURS)
    expect(priced.known).toBe(false)
    expect(priced.lastHours).toBeNull()
    expect(priced.bucketsWithoutField).toEqual([2])
  })

  it('has nothing to say about a route that did not run', () => {
    expect(resolvePricedHorizon([], FIELDS, FORECAST_HOURS)).toEqual(UNPRICED)
    expect(resolvePricedHorizon(null, FIELDS)).toEqual(UNPRICED)
    expect(resolvePricedHorizon(ROUTE_BUCKETS, []).known).toBe(false)
    expect(resolvePricedHorizon(ROUTE_BUCKETS, []).lastHours).toBeNull()
  })

  it('is stable whatever order the buckets arrive in', () => {
    const a = resolvePricedHorizon([1, 0, 1], FIELDS, FORECAST_HOURS)
    const b = resolvePricedHorizon(ROUTE_BUCKETS, FIELDS, FORECAST_HOURS)
    expect(a).toEqual(b)
  })
})

describe('what the operator is told', () => {
  const summary = pricedHorizonSummary(
    resolvePricedHorizon(ROUTE_BUCKETS, FIELDS, FORECAST_HOURS),
  )

  it('states the fields that priced it and the horizons that did not', () => {
    expect(summary).toContain('evaluated through +6h and +12h')
    expect(summary).toContain('+48h')
    expect(summary).toMatch(/priced nothing on this route/i)
  })

  it('says so explicitly when the priced horizon is unknown', () => {
    const unknown = pricedHorizonSummary(
      resolvePricedHorizon([0, 1, 2], FIELDS, FORECAST_HOURS),
    )
    expect(unknown).toMatch(/unavailable/i)
    expect(unknown).toContain('2')
    expect(unknown).toMatch(/no iceberg state is shown/i)
  })

  it('claims nothing it cannot claim', () => {
    for (const text of [
      summary,
      pricedHorizonSummary(resolvePricedHorizon([0], FIELDS, FORECAST_HOURS)),
      pricedHorizonSummary(resolvePricedHorizon([0, 1, 2], FIELDS, FORECAST_HOURS)),
      pricedHorizonSummary(UNPRICED),
    ]) {
      const lower = text.toLowerCase()
      for (const banned of [
        'safe',
        'unsafe',
        'guarantee',
        'collision',
        'optimal',
        'probability',
        'risk-free',
        'best route',
      ]) {
        expect(lower, `"${banned}" in: ${text}`).not.toContain(banned)
      }
    }
  })
})
