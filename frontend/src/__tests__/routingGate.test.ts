import { describe, expect, it } from 'vitest'

import { buildProvenance, routingReadiness } from '../api/provenance'
import type { ProvenanceInputs } from '../api/provenance'
import { forecastResponse, missionResponse } from './fixtures'

// The dates come from the recorded health payload; nothing here is typed.
const HEALTH = {
  status: 'ok',
  prototype: true,
  grid: missionResponse().grid,
  requires_historical_demo_override: true,
  polaris: { selection: { ice_class: 'PC6' } },
}

const inputs = (over: Partial<ProvenanceInputs> = {}): ProvenanceInputs => ({
  health: HEALTH,
  forecast: forecastResponse(),
  mission: null,
  configuredPricedHours: [6, 12],
  ...over,
})

const readiness = (overrideRequested: boolean, over?: Partial<ProvenanceInputs>) =>
  routingReadiness(buildProvenance(inputs(over)), overrideRequested)

/** the same recorded payload with the backend reporting one date for both */
const aligned = () => {
  const grid = { ...missionResponse().grid }
  grid.usnic_chart_date = grid.environment_date
  grid.dates_aligned = true
  return { ...HEALTH, grid, requires_historical_demo_override: false }
}

describe('routing readiness', () => {
  it('is ready, and ungated, when the dates are contemporaneous', () => {
    for (const requested of [false, true]) {
      const state = readiness(requested, { health: aligned() })
      expect(state.state).toBe('ready')
      expect(state.canRoute).toBe(true)
      expect(state.gated).toBe(false)
      expect(state.action).toBe('route')
      expect(state.detail).toBeNull()
    }
  })

  it('gates a mismatch while the override is off', () => {
    expect(missionResponse().grid.dates_aligned).toBe(false)
    const state = readiness(false)
    expect(state.state).toBe('override_required')
    expect(state.canRoute).toBe(false)
    expect(state.gated).toBe(true)
    expect(state.action).toBe('enable_override')
    expect(state.headline).toBe('Historical demonstration required')
    expect(state.detail).toMatch(/from different dates/i)
    expect(state.detail).toMatch(/explicit historical demonstration override/i)
  })

  it('lets routing proceed once the override is on, and says so', () => {
    const state = readiness(true)
    expect(state.state).toBe('historical_active')
    expect(state.canRoute).toBe(true)
    expect(state.gated).toBe(true)
    expect(state.action).toBe('route')
    expect(state.headline).toBe('Historical demonstration active')
    expect(state.detail).toMatch(/explicit demonstration input/i)
    expect(state.detail).toMatch(/not a description of the current/i)
  })

  it('turns the gate back on the moment the override goes off', () => {
    expect(readiness(true).canRoute).toBe(true)
    expect(readiness(false).canRoute).toBe(false)
  })

  it('does not gate what it cannot judge, and says the backend will', () => {
    const state = readiness(false, { health: null, configuredPricedHours: [] })
    expect(state.state).toBe('unknown')
    expect(state.canRoute).toBe(true)
    expect(state.gated).toBe(false)
    expect(state.detail).toMatch(/the backend will validate/i)
  })

  it('states the condition without calling the data bad', () => {
    for (const requested of [false, true]) {
      const state = readiness(requested)
      const text = `${state.headline} ${state.detail ?? ''} ${
        state.actionLabel
      }`.toLowerCase()
      for (const banned of [
        'invalid',
        'broken',
        'unsafe',
        'dangerous',
        'bad data',
        'wrong data',
        'error',
        'failure',
        'guaranteed',
        'collision',
      ]) {
        expect(text, `"${banned}" in: ${text}`).not.toContain(banned)
      }
      //  and it uses the vocabulary the project settled on
      expect(text).toContain('historical demonstration')
    }
  })
})
