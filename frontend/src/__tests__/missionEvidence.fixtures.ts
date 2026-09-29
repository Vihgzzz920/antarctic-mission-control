// Recorded shapes of the real backend responses for the mission-decision,
// forecast-provenance and historical-evaluation views. Every value below was
// taken from an actual response; nothing here is an invented metric.

import type {
  ForecastBucketProvenance,
  HistoricalEvaluationResponse,
  MissionMatrix,
} from '../api/types'

export const forecastBuckets: ForecastBucketProvenance[] = [
  {
    bucket: 0,
    source_type: 'explicit_persistence',
    model: null,
    environment_date: '2025-01-08',
  },
  {
    bucket: 4,
    source_type: 'model_forecast',
    model: 'HistGradientBoostingRegressor',
    valid_from: '2025-01-09T00:00:00',
    valid_to: '2025-01-09T06:00:00',
    resolution: {
      path: 'sic_forecast_hgb_origin20250108_plus24h_valid20250109_3976.tif',
      lead_hours: 24,
      validity_rule: 'target_composite_day',
      validity_from: '2025-01-09T00:00:00',
      validity_to: '2025-01-10T00:00:00',
      forecast_origin: '2025-01-08T00:00:00',
      valid_time: '2025-01-09T00:00:00',
    },
    model_artifact: 'data/processed/forecast_model_hgb.joblib',
    model_artifact_sha256:
      '463917f386b92e6e996506504c3662dfb0b5f828fec08c0f68e6159ed0f3682a',
    dataset: 'data/processed/forecast_dataset.npz',
    dataset_sha256:
      '1721e165a765b31fc2866f0f7837ce9d80d2a65375d0941e1d86edd482682430',
    dataset_link:
      'forecast_metrics.json records artifact_sha256 == the raster model_artifact_sha256',
  },
  {
    bucket: 11,
    source_type: 'model_forecast',
    model: 'HistGradientBoostingRegressor',
    valid_from: '2025-01-10T18:00:00',
    valid_to: '2025-01-11T00:00:00',
    resolution: {
      path: 'sic_forecast_hgb_origin20250108_plus48h_valid20250110_3976.tif',
      lead_hours: 48,
      validity_rule: 'target_composite_day',
      validity_from: '2025-01-10T00:00:00',
      validity_to: '2025-01-11T00:00:00',
      forecast_origin: '2025-01-08T00:00:00',
      valid_time: '2025-01-10T00:00:00',
    },
    model_artifact: 'data/processed/forecast_model_hgb_48h.joblib',
    model_artifact_sha256:
      '78adb269710a76944ead6e9458db4a29235239a803a79fc10ac2a34026c7ec26',
    dataset: 'data/processed/forecast_dataset_48h.npz',
    dataset_sha256:
      '0aa398aa43d98edeb63ef5c8cb2b93f7141d93b54ebdf433aa814e699e3ad7f5',
    dataset_link:
      'forecast_metrics_48h.json records artifact_sha256 == the raster model_artifact_sha256',
  },
]

const option = (
  scenarioId: string,
  site: string,
  departure: string,
  feasible: boolean,
  extra: Record<string, unknown> = {},
) => ({
  scenario_id: scenarioId,
  site,
  departure_time: departure,
  feasible,
  refusal_outcome: feasible ? null : 'forecast_unavailable',
  refusal_reason: feasible
    ? null
    : 'departure_offset_s=43200 starts the route at or after the provider horizon 43200 s; there is no bucket left to price.',
  environment_source_types: feasible ? ['observed_analysis'] : null,
  environment_models: [],
  buckets_used: feasible ? [0, 1] : [],
  objectives: feasible
    ? {
        fastest: {
          routed: true,
          distance_km: 178.03,
          travel_time_h: 9.89,
          configured_cost: 15554111.8,
          estimated_fuel: 416429.44,
          estimated_fuel_per_km: 2339.09,
          polaris_contribution: 15227476,
          environmental_contribution: 303265,
        },
        fuel_efficient: {
          routed: true,
          distance_km: 202.4,
          travel_time_h: 11.24,
          configured_cost: 15545819.9,
          estimated_fuel: 335683.06,
          estimated_fuel_per_km: 1658.5,
          polaris_contribution: 15227476,
          environmental_contribution: 318412,
        },
        risk_oriented: {
          routed: true,
          distance_km: 186.87,
          travel_time_h: 10.38,
          configured_cost: 11659269.7,
          estimated_fuel: 407540.66,
          estimated_fuel_per_km: 2180.9,
          polaris_contribution: 11325825,
          environmental_contribution: 307001,
        },
      }
    : {},
  ...extra,
})

export function missionMatrix(
  scientific = false,
): MissionMatrix {
  const options = [
    option('demo_goal@2025-01-08T00:00:00', 'demo_goal', '2025-01-08T00:00:00', true),
    option('demo_goal@2025-01-08T12:00:00', 'demo_goal', '2025-01-08T12:00:00', false),
    option('site_b@2025-01-08T00:00:00', 'site_b', '2025-01-08T00:00:00', true),
  ]
  return {
    start: [948, 503],
    sites: [
      {
        site_id: 'demo_goal',
        cell: [922, 497],
        label: 'the documented demonstration goal',
        is_operational: false,
        source: "src/api/config.py DEMO['goal']",
      },
      {
        site_id: 'site_b',
        cell: [922, 483],
        label: 'SYNTHETIC demonstration site B',
        is_operational: false,
        source:
          'the documented DEMO goal offset by a whole number of routing-grid cells along one axis; arithmetic on the project grid, not a station, port, base or waypoint',
      },
    ],
    departure_times: ['2025-01-08T00:00:00', '2025-01-08T12:00:00'],
    profiles_requested: ['fastest', 'fuel_efficient', 'risk_oriented'],
    environment_policy: scientific
      ? 'long_horizon_forecast_evaluation'
      : 'persistence_from_departure_analysis',
    evaluation_mode: scientific
      ? 'long_horizon_scientific_evaluation'
      : 'operational_demo',
    is_operational_assessment: !scientific,
    iceberg_exposure: scientific ? 'omitted' : 'priced',
    limitations: scientific
      ? [
          'iceberg exposure is OMITTED entirely beyond its configured 12 h horizon; this is not an operational assessment',
          'the sea-ice forecast leads available are 24 h and 48 h only',
        ]
      : [],
    options: [
      {
        scenario: {},
        feasible: true,
        refusal_outcome: null,
        refusal_reason: null,
        objectives: options[0].objectives,
        provenance: {
          scenario_id: options[0].scenario_id,
          departure_time: '2025-01-08T00:00:00',
          environment_policy: scientific
            ? 'long_horizon_forecast_evaluation'
            : 'persistence_from_departure_analysis',
          evaluation_mode: scientific
            ? 'long_horizon_scientific_evaluation'
            : 'operational_demo',
          is_operational_mission: false,
          is_operational_assessment: !scientific,
          iceberg_exposure: {
            status: scientific ? 'omitted' : 'priced',
            reason: scientific
              ? 'the routing configuration loads iceberg exposure for buckets 0-1 only (a 12 h horizon)'
              : null,
          },
          estimated_fuel_provenance: null,
          environment: {
            source_types: scientific
              ? ['explicit_persistence', 'model_forecast']
              : ['observed_analysis'],
            models: scientific ? ['HistGradientBoostingRegressor'] : [],
            forecast_origins: scientific ? ['2025-01-08T00:00:00'] : [],
            forecast_leads: scientific ? [24, 48] : [],
            validity_rules: scientific ? ['target_composite_day'] : [],
            artifacts: scientific
              ? [
                  'sic_forecast_hgb_origin20250108_plus24h_valid20250109_3976.tif',
                  'sic_forecast_hgb_origin20250108_plus48h_valid20250110_3976.tif',
                ]
              : [],
            buckets_used: [0, 4, 11],
            buckets: scientific ? forecastBuckets : [forecastBuckets[0]],
          },
        },
      },
    ],
    summary: {
      start: [948, 503],
      environment_policy: scientific
        ? 'long_horizon_forecast_evaluation'
        : 'persistence_from_departure_analysis',
      evaluation_mode: scientific
        ? 'long_horizon_scientific_evaluation'
        : 'operational_demo',
      is_operational_assessment: !scientific,
      iceberg_exposure: scientific ? 'omitted' : 'priced',
      limitations: scientific
        ? ['iceberg exposure is OMITTED entirely beyond its configured 12 h horizon']
        : [],
      profiles_requested: ['fastest', 'fuel_efficient', 'risk_oriented'],
      options_evaluated: 3,
      options_feasible: 2,
      sites_feasible: ['demo_goal', 'site_b'],
      sites_infeasible: [],
      departures_feasible: ['2025-01-08T00:00:00'],
      departures_infeasible: ['2025-01-08T12:00:00'],
      objectives_available: ['fastest', 'fuel_efficient', 'risk_oriented'],
      options,
      outcome: 'options_evaluated',
      no_feasible_option: false,
      this_is_not_a_ranking: true,
      has_a_best_site: false,
      has_a_best_departure_time: false,
      has_a_recommended_route: false,
      has_an_overall_score: false,
      selection_requires_a_caller_supplied_objective: true,
    },
  }
}

export function emptyMatrix(): MissionMatrix {
  const m = missionMatrix(false)
  m.summary.options = m.summary.options.map((o) => ({
    ...o,
    feasible: false,
    refusal_outcome: 'no_feasible_path',
    refusal_reason: 'goal is unreachable: every route from start is closed off',
    objectives: {},
  }))
  m.summary.options_feasible = 0
  m.summary.sites_feasible = []
  m.summary.sites_infeasible = ['demo_goal', 'site_b']
  m.summary.no_feasible_option = true
  m.summary.outcome = 'no_feasible_option'
  m.summary.why = [
    'no_feasible_path: goal is unreachable: every route from start is closed off',
  ]
  return m
}

export function historicalEvaluation(): HistoricalEvaluationResponse {
  const accuracy = (mae24: number, p24: number, mae48: number, p48: number) => ({
    '+24h': {
      valid_time: '2025-12-02T00:00:00',
      scored_cells: 290245,
      model_label: 'HGB +24h',
      origin_is_inside_the_training_period: false,
      model: { mae: mae24, rmse: 16.12, bias: 2.05, miz_mae: 20.0 },
      persistence: { mae: p24, rmse: 17.04, bias: 0.15, miz_mae: 21.34 },
    },
    '+48h': {
      valid_time: '2025-12-03T00:00:00',
      scored_cells: 288000,
      model_label: 'HGB +48h',
      origin_is_inside_the_training_period: false,
      model: { mae: mae48, rmse: 17.76, bias: 0.74, miz_mae: 22.82 },
      persistence: { mae: p48, rmse: 20.06, bias: -1.94, miz_mae: 24.53 },
    },
  })
  return {
    status: 'ok',
    artifact: 'data/processed/evaluation/historical_forecast_decision_backtest.json',
    sha256:
      '90a49c5f05e162d929ecc852cb561bbefc109908f9e1659a59b62bacea96f1b8',
    evaluation: {
      what_this_is:
        'a historical forecast-value / decision-value backtest. It is NOT a claim that either plan is better, safer, optimal or operationally recommended.',
      is_operational_assessment: false,
      candidate_rule: 'every 5th day of 2025-12, from 2025-12-01',
      held_out_window: ['2025-12-01', '2025-12-29'],
      causal_order: [
        '1. forecast produced from the origin day observation only',
        '2. both plans routed',
        '3. only then is the target-day observation opened',
      ],
      aggregate: {
        cases_attempted: 7,
        cases_included: 6,
        cases_skipped: 1,
        skip_reasons: {
          '2025-12-31': 'origin_is_not_in_the_held_out_test_period',
        },
        cases_both_planned: 6,
        route_changed: 6,
        route_unchanged: 0,
        cases_realized: 5,
        cases_not_realized: 1,
        not_realized_reasons: {
          '2025-12-21':
            '5 of 215 cells on this path have no observed sea ice on the day the vessel would reach them',
        },
        forecast_plan_lower_realized_environmental_cost: 1,
        persistence_plan_lower_realized_environmental_cost: 4,
        tied_or_identical_route: 0,
        this_is_not_a_universal_accuracy_claim: true,
        scope: 'these origins, this geometry, this parameter set only',
      },
      cases: [
        {
          ran: true,
          case: { origin: '2025-12-01', included: true, skip_reason: null, targets: {} },
          forecast_accuracy: accuracy(9.819, 9.776, 11.113, 11.697),
          decision: {
            both_planned: true,
            route_changed: true,
            shared_cells: 214,
            first_divergence_index: 111,
            configured_cost_delta: 98863.1,
            estimated_fuel_delta: 202797,
          },
          realized: {
            evaluated: true,
            realized_environmental_cost_delta: -950.1,
            which_plan_met_lower_realized_environmental_cost: 'forecast_plan',
          },
        },
        {
          ran: true,
          case: { origin: '2025-12-06', included: true, skip_reason: null, targets: {} },
          forecast_accuracy: accuracy(10.37, 10.2, 13.209, 13.5),
          decision: {
            both_planned: true,
            route_changed: true,
            configured_cost_delta: 62981.5,
            estimated_fuel_delta: 123883,
          },
          realized: {
            evaluated: true,
            realized_environmental_cost_delta: 37311.3,
            which_plan_met_lower_realized_environmental_cost: 'persistence_plan',
          },
        },
        {
          ran: true,
          case: { origin: '2025-12-21', included: true, skip_reason: null, targets: {} },
          forecast_accuracy: accuracy(12.306, 12.1, 15.401, 15.9),
          decision: {
            both_planned: true,
            route_changed: true,
            configured_cost_delta: 76867,
            estimated_fuel_delta: 267362,
          },
          realized: {
            evaluated: false,
            reason:
              '5 of 215 cells on this path have no observed sea ice on the day the vessel would reach them',
          },
        },
        {
          ran: false,
          case: {
            origin: '2025-12-31',
            included: false,
            skip_reason: 'origin_is_not_in_the_held_out_test_period',
            targets: {},
          },
        },
      ],
    },
  }
}
