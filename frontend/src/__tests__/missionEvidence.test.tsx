import { readFileSync } from 'node:fs'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import EvaluationBadge from '../components/EvaluationBadge'
import ForecastComparison from '../components/ForecastComparison'
import ForecastProvenanceList from '../components/ForecastProvenanceList'
import MissionMatrixView from '../components/MissionMatrixView'
import ValidationPanel from '../components/ValidationPanel'
import { PROFILE_LABEL, PROFILE_ORDER } from '../api/types'
import {
  emptyMatrix,
  forecastBuckets,
  historicalEvaluation,
  missionMatrix,
} from './missionEvidence.fixtures'

afterEach(cleanup)

const flat = (el: HTMLElement) => el.textContent ?? ''

// ------------------------------------------------- 1, 12. four profiles
describe('four route objectives', () => {
  it('orders them fastest, fuel-efficient, risk-oriented, with distance last', () => {
    expect(PROFILE_ORDER).toEqual([
      'fastest',
      'fuel_efficient',
      'risk_oriented',
      'shortest_distance',
    ])
    expect(PROFILE_LABEL.fuel_efficient).toBe('Fuel-efficient')
    //  shortest_distance is kept for compatibility, and is never called fuel
    expect(PROFILE_LABEL.shortest_distance).toBe('Shortest distance')
    expect(PROFILE_LABEL.shortest_distance.toLowerCase()).not.toContain('fuel')
  })

  it('shows the three decision objectives in the matrix, not the compatibility one', () => {
    render(<MissionMatrixView matrix={missionMatrix()} />)
    const option = screen.getByTestId('mission-option-demo_goal@2025-01-08T00:00:00')
    for (const name of ['Fastest', 'Fuel-efficient', 'Risk-oriented']) {
      expect(flat(option)).toContain(name)
    }
    expect(flat(option)).not.toContain('Shortest distance')
  })
})

// ---------------------------------------------------- 2. fuel terminology
describe('fuel terminology', () => {
  it('reports the estimate in OWE-m and never as a volume or a mass', () => {
    render(<MissionMatrixView matrix={missionMatrix()} />)
    const row = screen.getByTestId(
      'mx-demo_goal@2025-01-08T00:00:00-fuel_efficient',
    )
    expect(flat(row)).toContain('OWE-m')
    expect(flat(row)).toContain('est.')
    const all = flat(screen.getByTestId('mission-matrix')).toLowerCase()
    for (const banned of ['litre', 'tonne', 'gallon', 'kilogram', 'bunker']) {
      expect(all).not.toContain(banned)
    }
  })

  it('the components themselves make no unsupported fuel claim', () => {
    for (const file of [
      'src/components/MissionMatrixView.tsx',
      'src/components/ForecastComparison.tsx',
      'src/components/ValidationPanel.tsx',
      'src/components/ForecastProvenanceList.tsx',
      'src/components/EvaluationBadge.tsx',
    ]) {
      const code = readFileSync(file, 'utf8').toLowerCase()
      for (const claim of [
        'fuel saving',
        'saves fuel',
        'measured fuel',
        'actual fuel',
        'litres',
        'tonnes',
        'best route',
        'optimal route',
        'recommended route',
        'safest',
        'guaranteed',
        'collision probability',
      ]) {
        expect(code, `${file} claims ${claim}`).not.toContain(claim)
      }
    }
  })
})

// ------------------------------------------- 3, 11. matrix and infeasibility
describe('the mission matrix', () => {
  it('renders one group per site and one row per departure', () => {
    render(<MissionMatrixView matrix={missionMatrix()} />)
    expect(screen.getByTestId('mission-site-demo_goal')).toBeInTheDocument()
    expect(screen.getByTestId('mission-site-site_b')).toBeInTheDocument()
    expect(
      screen.getByTestId('mission-option-demo_goal@2025-01-08T12:00:00'),
    ).toHaveAttribute('data-feasible', 'no')
  })

  it('gives an infeasible option the router’s own reason in text', () => {
    render(<MissionMatrixView matrix={missionMatrix()} />)
    const reason = screen.getByTestId('mx-reason-demo_goal@2025-01-08T12:00:00')
    //  the state is stated in words, not by colour alone
    expect(flat(reason)).toContain('forecast unavailable')
    expect(flat(reason)).toContain('provider horizon')
  })

  it('represents "no feasible option" explicitly', () => {
    render(<MissionMatrixView matrix={emptyMatrix()} />)
    const none = screen.getByTestId('mission-matrix-none')
    expect(flat(none)).toContain('No feasible option')
    expect(flat(none)).toContain('no_feasible_path')
  })

  it('ranks nothing and recommends nothing', () => {
    const matrix = missionMatrix()
    render(<MissionMatrixView matrix={matrix} />)
    expect(matrix.summary.this_is_not_a_ranking).toBe(true)
    expect(matrix.summary.has_a_best_site).toBe(false)
    expect(matrix.summary.has_a_best_departure_time).toBe(false)
    expect(matrix.summary.has_an_overall_score).toBe(false)
    const text = flat(screen.getByTestId('mission-matrix')).toLowerCase()
    for (const banned of [
      'best site',
      'best departure',
      'best option',
      'recommended',
      'winner',
      'score',
      'ranked first',
    ]) {
      expect(text).not.toContain(banned)
    }
    //  the only mention of ranking is the denial of it
    expect(flat(screen.getByTestId('mission-matrix-scope'))).toContain(
      'not ranked',
    )
    expect(flat(screen.getByTestId('mission-matrix-scope'))).toContain(
      'none is preferred over another',
    )
  })
})

// ------------------------------------- 4, 7. operational vs scientific, synthetic
describe('evaluation mode', () => {
  it('names the operational demo and the scientific evaluation differently', () => {
    const { unmount } = render(
      <EvaluationBadge mode="operational_demo" isOperationalAssessment />,
    )
    expect(flat(screen.getByTestId('evaluation-badge'))).toContain(
      'Operational demo',
    )
    expect(
      screen.queryByTestId('evaluation-badge-note'),
    ).not.toBeInTheDocument()
    unmount()

    render(
      <EvaluationBadge
        mode="long_horizon_scientific_evaluation"
        isOperationalAssessment={false}
        icebergExposure="omitted"
      />,
    )
    const badge = screen.getByTestId('evaluation-badge')
    expect(badge).toHaveAttribute('data-mode', 'long_horizon_scientific_evaluation')
    expect(flat(badge)).toContain('Scientific evaluation')
    //  stated in words, not by colour alone
    expect(flat(screen.getByTestId('evaluation-badge-note'))).toContain(
      'not an operational assessment',
    )
    expect(flat(screen.getByTestId('evaluation-badge-iceberg'))).toContain(
      'omitted beyond 12 h',
    )
  })

  it('carries the scientific mode and its limitations into the matrix', () => {
    render(<MissionMatrixView matrix={missionMatrix(true)} />)
    expect(screen.getByTestId('evaluation-badge')).toHaveAttribute(
      'data-mode',
      'long_horizon_scientific_evaluation',
    )
    expect(
      flat(screen.getByTestId('mission-matrix-limitations')),
    ).toContain('iceberg exposure is OMITTED')
  })

  it('labels every synthetic site as synthetic and as no kind of place', () => {
    render(<MissionMatrixView matrix={missionMatrix(true)} />)
    for (const site of ['demo_goal', 'site_b']) {
      const note = screen.getByTestId(`mx-synthetic-${site}`)
      expect(flat(note)).toContain('synthetic')
      expect(flat(note)).toContain('not a station, port, base or waypoint')
    }
  })
})

// ------------------------------------------------ 5. forecast provenance
describe('forecast provenance', () => {
  it('answers what forecast affected the route, field by field', () => {
    render(<ForecastProvenanceList buckets={forecastBuckets} />)
    //  only the forecast-priced buckets appear; the persistence one does not
    expect(screen.queryByTestId('forecast-bucket-0')).not.toBeInTheDocument()
    const b24 = screen.getByTestId('forecast-bucket-4')
    expect(flat(b24)).toContain('+24h')
    expect(flat(b24)).toContain('HistGradientBoostingRegressor')
    expect(flat(b24)).toContain('target_composite_day')
    expect(flat(b24)).toContain('plus24h')
    expect(flat(b24)).toContain('forecast_model_hgb.joblib')
    expect(flat(b24)).toContain('463917f386b9')
    expect(flat(b24)).toContain('forecast_dataset.npz')

    const b48 = screen.getByTestId('forecast-bucket-11')
    expect(flat(b48)).toContain('+48h')
    expect(flat(b48)).toContain('plus48h')
    expect(flat(b48)).toContain('forecast_model_hgb_48h.joblib')
    //  the two leads never share an artifact, and the UI shows that
    expect(flat(b24)).not.toContain('plus48h')
    expect(flat(b48)).not.toContain('forecast_dataset.npz\n')
  })

  it('says plainly when no bucket was forecast-priced', () => {
    render(<ForecastProvenanceList buckets={[forecastBuckets[0]]} />)
    expect(
      flat(screen.getByTestId('forecast-provenance-empty')),
    ).toContain('No bucket on this route was priced from a model forecast')
  })

  it('infers nothing: every rendered value comes from the payload', () => {
    const code = readFileSync(
      'src/components/ForecastProvenanceList.tsx',
      'utf8',
    )
    //  no filename parsing and no hardcoded lead anywhere
    expect(code).not.toMatch(/plus\d+h/)
    expect(code).not.toMatch(/\bsplit\(|\bmatch\(|RegExp/)
    expect(code).not.toMatch(/lead_hours\s*[|?]{2}\s*\d/)
  })
})

// ------------------------------------------- 10. route changed / unchanged
describe('persistence against forecast-driven', () => {
  it('says the route changed, and shows every delta as measured', () => {
    const persistence = missionMatrix(false)
    const forecast = missionMatrix(true)
    const f = forecast.summary.options[0].objectives.risk_oriented!
    f.configured_cost = 11_708_666.7
    f.environmental_contribution = 356_398
    f.estimated_fuel = 418_617.6
    render(
      <ForecastComparison
        persistence={persistence}
        forecast={forecast}
        profile="risk_oriented"
      />,
    )
    expect(
      flat(screen.getByTestId('forecast-comparison-verdict')),
    ).toContain('Route changed')
    //  a forecast-driven route that costs MORE is displayed as such
    expect(flat(screen.getByTestId('fc-cost'))).toContain('+49,397')
    expect(flat(screen.getByTestId('fc-env'))).toContain('+49,397')
    expect(flat(screen.getByTestId('fc-fuel'))).toContain('OWE-m')
    expect(
      flat(screen.getByTestId('forecast-comparison-buckets')),
    ).toContain('omitted beyond its configured 12 h horizon')
  })

  it('says the route is unchanged when nothing moved', () => {
    render(
      <ForecastComparison
        persistence={missionMatrix(false)}
        forecast={missionMatrix(false)}
        profile="fastest"
      />,
    )
    expect(
      flat(screen.getByTestId('forecast-comparison-verdict')),
    ).toContain('Route unchanged')
  })

  it('never phrases the comparison as one side winning', () => {
    render(
      <ForecastComparison
        persistence={missionMatrix(false)}
        forecast={missionMatrix(true)}
        profile="fastest"
      />,
    )
    const text = flat(screen.getByTestId('forecast-comparison')).toLowerCase()
    for (const banned of ['wins', 'better', 'improves', 'superior', 'beats']) {
      expect(text).not.toContain(banned)
    }
  })
})

// ------------------------------------------ 8. historical validation view
describe('the historical validation view', () => {
  const data = historicalEvaluation()

  it('reports the held-out counts faithfully', () => {
    render(<ValidationPanel data={data} />)
    const counts = flat(screen.getByTestId('validation-counts'))
    expect(counts).toContain('6')
    expect(counts).toContain('cases evaluated')
    expect(counts).toContain('routes changed')
    expect(counts).toContain('realization unavailable')
    expect(flat(screen.getByTestId('validation-outcome'))).toContain('1')
    expect(flat(screen.getByTestId('validation-outcome'))).toContain('4')
  })

  it('shows both leads against persistence, and the unrealized case', () => {
    render(<ValidationPanel data={data} />)
    const row = screen.getByTestId('val-acc-2025-12-01')
    expect(flat(row)).toContain('9.819')
    expect(flat(row)).toContain('9.776')
    expect(flat(row)).toContain('11.113')
    const unrealized = screen.getByTestId('val-dec-2025-12-21')
    expect(flat(unrealized)).toContain('not evaluated')
    expect(flat(unrealized)).toContain('no observed sea ice')
  })

  it('states the limitation and claims no superiority', () => {
    render(<ValidationPanel data={data} />)
    expect(flat(screen.getByTestId('validation-limitation'))).toContain(
      'does not establish universal superiority',
    )
    const text = flat(screen.getByTestId('validation-panel')).toLowerCase()
    for (const banned of [
      'proof the model works',
      'accuracy guaranteed',
      'ai predicts better',
      'beats persistence',
      'improves safety',
    ]) {
      expect(text).not.toContain(banned)
    }
    expect(flat(screen.getByTestId('validation-artifact'))).toContain(
      'historical_forecast_decision_backtest.json',
    )
  })

  it('handles an unavailable artifact explicitly', () => {
    render(<ValidationPanel data={null} error="the artifact has not been produced" />)
    expect(flat(screen.getByTestId('validation-error'))).toContain(
      'has not been produced',
    )
  })
})
