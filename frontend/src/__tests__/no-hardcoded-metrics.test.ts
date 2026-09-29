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

/** Every index at which `needle` occurs in `haystack`. */
function occurrences(haystack: string, needle: string): number[] {
  const out: number[] = []
  let at = haystack.indexOf(needle)
  while (at !== -1) {
    out.push(at)
    at = haystack.indexOf(needle, at + needle.length)
  }
  return out
}

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
      //  A `fuel_efficient` objective now exists and is backed by
      //  src/routing/fuel_cost.py, so the bare word is no longer forbidden.
      //  What stays forbidden is every UNSUPPORTED fuel claim: this project
      //  produces an ESTIMATED relative proxy in open-water-equivalent metres
      //  and cannot produce a measured consumption, a saving, or any mass or
      //  volume. Distance is still not fuel.
      for (const claim of [
        'measured fuel',
        'actual fuel',
        'fuel saving',
        'saves fuel',
        'fuel consumption of',
        'litres',
        'tonnes',
        'bunker',
      ]) {
        //  A phrase is a CLAIM only where it is not denied. The UI is required
        //  to state "Measured fuel consumption: no" -- which is the opposite
        //  of claiming one -- so an occurrence sitting next to a negation is
        //  the disclosure, not the claim.
        for (const at of occurrences(code, claim)) {
          //  the disclosure reads "Measured fuel consumption: {...? 'yes' : 'no'}",
          //  so the denial sits AFTER the phrase, across a JSX expression
          const around = code.slice(Math.max(0, at - 40), at + claim.length + 160)
          const denied = /\bno\b|\bnot\b|never|unavailable|false|cannot/.test(
            around,
          )
          expect(
            denied,
            `${file} makes an unsupported fuel claim: ...${around.trim()}...`,
          ).toBe(true)
        }
      }
      expect(code, `${file} ranks the routes`).not.toContain('best route')
      expect(code, `${file} ranks the routes`).not.toContain('optimal route')
      expect(code, `${file} ranks the routes`).not.toContain('recommended route')
    }
  })

  it('calls no route option a duplicate of another', () => {
    //  two objectives can return the same path; one of them is not a copy of
    //  the other, and the operator-facing words never say it is
    for (const file of files) {
      const code = readFileSync(file, 'utf8').toLowerCase()
      expect(code, `${file} calls a route a duplicate`).not.toContain('duplicate')
      expect(code, `${file} claims the objectives always agree`).not.toContain(
        'always identical',
      )
    }
  })
})
