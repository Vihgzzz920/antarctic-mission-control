import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

import { buildProvenance, formatDate } from '../api/provenance'
import type { ProvenanceInputs } from '../api/provenance'
import { forecastResponse, missionDefaults, missionResponse } from './fixtures'

// The health payload here is the backend's own, recorded from GET /api/health.
const HEALTH = {
  status: 'ok',
  prototype: true,
  grid: missionResponse().grid,
  iceberg_exposure_weight: 5,
  requires_historical_demo_override: true,
  exposure_layers: [
    { label: '06h', bucket: 0, horizon_hours: 6, radius_km: 9.05053, iceberg_count: 11 },
    { label: '12h', bucket: 1, horizon_hours: 12, radius_km: 12.799382, iceberg_count: 11 },
  ],
  polaris: {
    selection: {
      ice_class: 'PC6',
      riv_table_requested: '1.3',
      sidecar_chart: 'ANTARC20201224',
      riv_table_source:
        'IMO MSC.1/Circ.1519, Appendix, Table 1.3 (Risk Index Values)',
    },
  },
}

const base: ProvenanceInputs = {
  health: HEALTH,
  forecast: forecastResponse(),
  mission: null,
  configuredPricedHours: [6, 12],
}

const at = (over: Partial<ProvenanceInputs> = {}) =>
  buildProvenance({ ...base, ...over })

const factOf = (heading: string, label: string, over?: Partial<ProvenanceInputs>) =>
  at(over)
    .details.find((section) => section.heading === heading)
    ?.facts.find((fact) => fact.label === label)?.value ?? null

describe('formatting a date', () => {
  it('reads an ISO date the backend supplied', () => {
    expect(formatDate(missionResponse().grid.environment_date)).toMatch(
      /^\d{2} [A-Z]{3} \d{4}$/,
    )
  })

  it('passes anything else through, and null stays null', () => {
    expect(formatDate(null)).toBeNull()
    expect(formatDate('not a date')).toBe('not a date')
  })
})

describe('provenance before any mission has run', () => {
  const provenance = at()
  const grid = missionResponse().grid

  it('knows both dates from health alone', () => {
    expect(provenance.environmentDate).toBe(grid.environment_date)
    expect(provenance.chartDate).toBe(grid.usnic_chart_date)
    expect(provenance.chartName).toBe(HEALTH.polaris.selection.sidecar_chart)
    expect(provenance.overrideInEffect).toBeNull() // no mission yet
  })

  it('calls a mismatch a historical demonstration, not a fault', () => {
    expect(grid.dates_aligned).toBe(false)
    expect(provenance.status).toBe('historical_demonstration')
    expect(provenance.headline).toMatch(/historical polaris demonstration/i)
    expect(provenance.headline.toLowerCase()).not.toMatch(
      /error|fail|invalid|current polaris conditions/,
    )
  })

  it('reports the override as required, before one is chosen', () => {
    expect(provenance.requiresOverride).toBe(true)
    expect(factOf('POLARIS chart', 'Explicit historical override')).toMatch(
      /required before this pairing will route/i,
    )
  })

  it('separates the forecast horizons from what routing prices', () => {
    const available = factOf('Iceberg forecast', 'Horizons available')!
    const priced = factOf('Iceberg forecast', 'Routing prices against')!
    for (const horizon of forecastResponse().horizons) {
      expect(available).toContain(`+${horizon.hours}h`)
    }
    expect(priced).toBe('+6h · +12h')
    expect(available).not.toBe(priced)
  })

  it("carries the forecast model's own description, not a new claim", () => {
    const model = forecastResponse().model
    expect(factOf('Iceberg forecast', 'Drift model')).toBe(model.summary)
    expect(factOf('Iceberg forecast', 'Validated science')).toMatch(
      /no — a project baseline/,
    )
    expect(model.is_validated_science).toBe(false)
  })
})

describe('provenance once a mission has run', () => {
  it('reports the override that was actually in effect', () => {
    expect(
      factOf('POLARIS chart', 'Explicit historical override', {
        mission: missionResponse(true),
      }),
    ).toMatch(/in effect for this mission/i)
    expect(
      factOf('POLARIS chart', 'Explicit historical override', {
        mission: missionResponse(false),
      }),
    ).toMatch(/off for this mission/i)
  })

  it('still calls the pairing historical whichever way the switch was set', () => {
    for (const historical of [true, false]) {
      expect(at({ mission: missionResponse(historical) }).status).toBe(
        'historical_demonstration',
      )
    }
  })

  it('calls aligned dates contemporaneous', () => {
    //  the same recorded payload with the backend reporting one date for both
    const aligned = missionResponse(false)
    aligned.grid.usnic_chart_date = aligned.grid.environment_date
    aligned.grid.dates_aligned = true
    const provenance = at({ mission: aligned })
    expect(provenance.status).toBe('contemporaneous')
    expect(provenance.headline).not.toMatch(/historical/i)
  })
})

describe('provenance with metadata missing', () => {
  it('says unknown rather than guessing', () => {
    const provenance = at({
      health: null,
      forecast: null,
      configuredPricedHours: [],
    })
    expect(provenance.status).toBe('unknown')
    expect(provenance.environmentDate).toBeNull()
    expect(provenance.chartDate).toBeNull()
    expect(provenance.headline).toMatch(/unknown/i)
    expect(provenance.details).toEqual([])
  })

  it('drops a fact the backend did not supply instead of inventing it', () => {
    const thin = { ...HEALTH, polaris: {} }
    expect(factOf('POLARIS chart', 'Chart', { health: thin })).toBeNull()
    expect(factOf('POLARIS chart', 'RIV source', { health: thin })).toBeNull()
    //  what it does know is still reported
    expect(factOf('POLARIS chart', 'Chart dated', { health: thin })).not.toBeNull()
  })

  it('says nothing about pricing when no field was described', () => {
    expect(
      factOf('Iceberg forecast', 'Routing prices against', {
        configuredPricedHours: [],
      }),
    ).toBeNull()
  })
})

describe('no demonstration date is written into the application', () => {
  const SRC = join(process.cwd(), 'src')
  const sources = (dir: string, out: string[] = []): string[] => {
    for (const entry of readdirSync(dir)) {
      const path = join(dir, entry)
      if (statSync(path).isDirectory()) {
        if (entry === '__tests__' || entry === 'generated') continue
        sources(path, out)
      } else if (/\.(ts|tsx)$/.test(entry)) {
        out.push(path)
      }
    }
    return out
  }

  it('contains no ISO date literal anywhere', () => {
    const files = sources(SRC)
    expect(files.length).toBeGreaterThan(10)
    for (const file of files) {
      const code = readFileSync(file, 'utf8')
      const dates = code.match(/\b(19|20)\d{2}-\d{2}-\d{2}\b/g)
      expect(dates, `${file} hardcodes ${dates?.join(', ')}`).toBeNull()
    }
  })

  it('names no chart and no demonstration date the backend publishes', () => {
    const banned = [
      missionResponse().grid.environment_date,
      missionResponse().grid.usnic_chart_date,
      String(missionDefaults().demo.departure_time),
      HEALTH.polaris.selection.sidecar_chart,
    ]
    for (const file of sources(SRC)) {
      const code = readFileSync(file, 'utf8')
      for (const value of banned) {
        expect(code, `${file} hardcodes ${value}`).not.toContain(value)
      }
    }
  })
})
