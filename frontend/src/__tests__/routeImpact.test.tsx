import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup } from '@testing-library/react'

import RouteImpact from '../components/RouteImpact'
import { simulationResponse } from './fixtures'

// The payload is the RECORDED /api/mission/simulate response. In it the
// backend reports two different outcomes for the same injection: the fastest
// route kept its path, the risk-oriented one did not. Both are rendered from
// the same component, which is the point -- neither outcome is the assumed one.

const SIM = simulationResponse()
const BERG = SIM.change.simulation

const show = (profile: 'fastest' | 'risk_oriented') =>
  render(<RouteImpact simulation={SIM} profile={profile} />)

afterEach(cleanup)

describe('the simulated event', () => {
  it('is labelled as simulated, temporary and written nowhere', () => {
    show('fastest')
    const event = screen.getByTestId('impact-event')
    expect(event).toHaveTextContent(BERG.label)
    expect(BERG.simulated).toBe(true)
    expect(BERG.written_to_any_dataset).toBe(false)
    expect(event).toHaveTextContent(/written to any dataset\s*no/i)
    expect(screen.getByTestId('route-impact')).toHaveTextContent(
      /temporary, and written to no dataset/i,
    )
  })

  it('reports the encounter the backend returned', () => {
    show('fastest')
    const event = screen.getByTestId('impact-event')
    expect(event).toHaveTextContent(BERG.encounter_cell.join(', '))
    expect(event).toHaveTextContent(BERG.encounter_datetime.replace('T', ' '))
    expect(event).toHaveTextContent(`+${BERG.horizon_hours}h`)
    expect(event).toHaveTextContent(`${BERG.from_bucket}`)
    expect(event).toHaveTextContent(`${BERG.radius_km.toFixed(2)} km`)
  })

  it('says plainly when the berg sits on a route cell', () => {
    expect(BERG.distance_to_route_m).toBe(0)
    show('fastest')
    expect(screen.getByTestId('impact-event')).toHaveTextContent(
      /0 m — on a route cell/,
    )
  })
})

describe('the decision', () => {
  it('reads ROUTE UNCHANGED when the backend kept the path', () => {
    const change = SIM.change.profiles.fastest
    expect(change.path_changed).toBe(false)
    show('fastest')

    expect(screen.getByTestId('impact-verdict')).toHaveTextContent(
      /^Route unchanged$/,
    )
    const sentence = screen.getByTestId('impact-sentence')
    expect(sentence).toHaveTextContent(
      new RegExp(`exposure on ${change.cells_with_simulated_exposure} cells`, 'i'),
    )
    expect(sentence).toHaveTextContent(/returned the same path/i)
    //  stated as a result, not as a fault
    for (const banned of [/fail/i, /harmless/i, /impossible/i, /bad model/i]) {
      expect(sentence.textContent ?? '').not.toMatch(banned)
    }
  })

  it('reads ROUTE REPLANNED when the backend returned a different path', () => {
    const change = SIM.change.profiles.risk_oriented
    expect(change.path_changed).toBe(true)
    show('risk_oriented')

    expect(screen.getByTestId('impact-verdict')).toHaveTextContent(
      /^Route replanned$/,
    )
    expect(screen.getByTestId('impact-sentence')).toHaveTextContent(
      /returned a different path/i,
    )
  })

  it('does not assume either outcome', () => {
    show('fastest')
    expect(screen.getByTestId('impact-summary')).toHaveTextContent(
      /Path\s*unchanged/i,
    )
    cleanup()
    show('risk_oriented')
    expect(screen.getByTestId('impact-summary')).toHaveTextContent(
      /Path\s*changed/i,
    )
  })

  it('takes the affected-cell count from the backend', () => {
    for (const profile of ['fastest', 'risk_oriented'] as const) {
      show(profile)
      expect(screen.getByTestId('impact-affected')).toHaveTextContent(
        String(SIM.change.profiles[profile].cells_with_simulated_exposure),
      )
      cleanup()
    }
  })
})

describe('before → after', () => {
  const rowOf = (label: string) =>
    within(screen.getByTestId('impact-table'))
      .getByText(label)
      .closest('tr') as HTMLTableRowElement

  it('shows the backend numbers on both sides', () => {
    const change = SIM.change.profiles.risk_oriented
    show('risk_oriented')

    const distance = rowOf('Distance')
    expect(distance).toHaveTextContent(
      `${(change.distance_m.before! / 1000).toFixed(2)} km`,
    )
    expect(distance).toHaveTextContent(
      `${(change.distance_m.after! / 1000).toFixed(2)} km`,
    )

    const exposure = rowOf('Iceberg exposure contribution')
    expect(exposure).toHaveTextContent(
      change.iceberg_exposure_contribution.before!.toLocaleString('en-GB'),
    )
    expect(exposure).toHaveTextContent(
      change.iceberg_exposure_contribution.after!.toLocaleString('en-GB'),
    )
  })

  it("shows the backend's own delta, not one computed here", () => {
    const change = SIM.change.profiles.risk_oriented
    show('risk_oriented')
    //  the recorded delta is +14,568 m
    expect(change.distance_m.delta).toBeGreaterThan(0)
    expect(rowOf('Distance')).toHaveTextContent(
      `+${(change.distance_m.delta! / 1000).toFixed(2)} km`,
    )
  })

  it('says "no change" rather than a zero that looks like data', () => {
    const change = SIM.change.profiles.fastest
    expect(change.distance_m.delta).toBe(0)
    show('fastest')
    expect(rowOf('Distance')).toHaveTextContent(/no change/i)
  })

  it('marks a missing value unknown instead of substituting another', () => {
    const thin = simulationResponse()
    thin.change.profiles.fastest.polaris_contribution = {
      before: null,
      after: null,
      delta: null,
    }
    render(<RouteImpact simulation={thin} profile="fastest" />)
    const row = within(screen.getByTestId('impact-table'))
      .getByText('POLARIS contribution')
      .closest('tr') as HTMLTableRowElement
    expect(row.textContent).toContain('—')
    expect(row.textContent).not.toMatch(/\d/)
  })

  it('covers every comparable metric the backend returned', () => {
    show('risk_oriented')
    const table = screen.getByTestId('impact-table')
    for (const label of [
      'Distance',
      'Travel time',
      'Iceberg exposure contribution',
      'Maximum modelled exposure',
      'POLARIS contribution',
      'Configured cost',
      'Special-consideration cells',
    ]) {
      expect(table).toHaveTextContent(label)
    }
  })
})

describe('what the evidence never claims', () => {
  it('makes no safety, collision or optimality claim', () => {
    for (const profile of ['fastest', 'risk_oriented'] as const) {
      show(profile)
      const text = (
        screen.getByTestId('route-impact').textContent ?? ''
      ).toLowerCase()
      for (const banned of [
        'collision',
        'probability',
        'guarantee',
        'optimal',
        'best route',
        'dangerous',
        ' safe',
        'unsafe',
      ]) {
        expect(text, `"${banned}" in the evidence`).not.toContain(banned)
      }
      //  the disclaimer is the BACKEND's own sentence, rendered verbatim
      expect(screen.getByTestId('impact-claims').textContent).toBe(
        simulationResponse().change.claims.note,
      )
      expect(text).toContain('not a claim that any route avoids anything')
      cleanup()
    }
  })

  it('keeps the forbidden words out of the evidence sources', () => {
    const files = ['components/RouteImpact.tsx', 'components/RespondDock.tsx']
    for (const file of files) {
      const code = readFileSync(join(process.cwd(), 'src', file), 'utf8')
        .toLowerCase()
      for (const banned of [
        'collision',
        'guaranteed',
        'safest',
        'dangerous',
        'optimal',
        'best route',
      ]) {
        expect(code, `${file} uses "${banned}"`).not.toContain(banned)
      }
    }
  })

  it('changes no weight and no cost anywhere in the frontend', () => {
    const SRC = join(process.cwd(), 'src')
    const sources = (dir: string, out: string[] = []): string[] => {
      for (const entry of readdirSync(dir)) {
        const path = join(dir, entry)
        if (statSync(path).isDirectory()) {
          if (entry === '__tests__' || entry === 'generated') continue
          sources(path, out)
        } else if (/\.(ts|tsx)$/.test(entry)) out.push(path)
      }
      return out
    }
    for (const file of sources(SRC)) {
      const code = readFileSync(file, 'utf8')
      expect(code, `${file} assigns an exposure weight`).not.toMatch(
        /iceberg_exposure_weight\s*[:=]\s*[\d.]/,
      )
      expect(code, `${file} assigns a POLARIS penalty`).not.toMatch(
        /polaris_penalty\s*[:=]\s*[\d.]/,
      )
    }
  })
})
