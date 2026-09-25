import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

// Route metrics must arrive from the API. This walks the application source --
// everything under src/ except the test folder -- and fails if a component
// carries a numeric distance, cost, exposure or cell count of its own.

const SRC = join(process.cwd(), 'src')

function sources(dir: string, out: string[] = []): string[] {
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

const METRIC_KEYS = [
  'distance_km',
  'distance_m',
  'travel_time_h',
  'travel_time_s',
  'configured_cost',
  'polaris_contribution',
  'iceberg_exposure_contribution',
  'max_iceberg_exposure',
  'environmental_contribution',
  'special_consideration_cells',
  'indeterminate_cells',
]

describe('no hardcoded route metrics', () => {
  const files = sources(SRC)

  it('finds application sources to check', () => {
    expect(files.length).toBeGreaterThan(8)
  })

  it.each(files)('%s assigns no route metric a literal value', (file) => {
    const code = readFileSync(file, 'utf8')
    for (const key of METRIC_KEYS) {
      // `distance_km: 178.03` or `distance_km = 178.03` would be a metric baked
      // into the UI. A type declaration (`distance_km: number | null`) is not.
      const assignment = new RegExp(`${key}\\s*[:=]\\s*-?\\d`, 'g')
      expect(
        code.match(assignment),
        `${file} assigns a literal to ${key}`,
      ).toBeNull()
    }
  })

  it('names no route "safest" and no distance "fuel"', () => {
    for (const file of files) {
      const code = readFileSync(file, 'utf8').toLowerCase()
      expect(code, `${file} calls a route safest`).not.toContain('safest')
      expect(code, `${file} mentions fuel`).not.toContain('fuel')
      expect(code, `${file} ranks the routes`).not.toContain('best route')
    }
  })
})
