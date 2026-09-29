// Shapes of what the backend already serialises. Nothing here invents a field:
// every key below is produced by RouteProfileResult.to_dict(),
// ProfileComparison.to_dict() or the thin API wrapper around them.

export type ProfileName =
  | 'fastest'
  | 'fuel_efficient'
  | 'risk_oriented'
  | 'shortest_distance'

/**
 * The order the operator sees. `shortest_distance` is kept LAST and only for
 * compatibility with the existing payload and the route-equivalence grouping:
 * it is a distance, never a fuel figure, and the backend says so explicitly
 * (`shortest_distance_is_a_fuel_figure: false`).
 *
 * This list is not a ranking. The three objectives are in different units and
 * none of them is "the best".
 */
export const PROFILE_ORDER: ProfileName[] = [
  'fastest',
  'fuel_efficient',
  'risk_oriented',
  'shortest_distance',
]

export const PROFILE_LABEL: Record<ProfileName, string> = {
  fastest: 'Fastest',
  fuel_efficient: 'Fuel-efficient',
  risk_oriented: 'Risk-oriented',
  shortest_distance: 'Shortest distance',
}

export interface RouteCell {
  row: number
  col: number
  arrival_time: number
  arrival_datetime: string
  bucket: number
  state: string
  chart_state: string | null
  environmental_cost: number
  polaris_penalty: number
  coverage_uncertainty_cost: number
  composed_cost: number
  iceberg_exposure: number | null
  iceberg_cost: number
  final_cost: number
  iceberg_id: string | null
  radius_km: number | null
  forecast_horizon_hours: number | null
  extrapolated_uncertainty: boolean | null
  polygon_id: number | null
  polaris_classification: string | null
  polaris_special_consideration: boolean | null
  polaris_indeterminate: boolean | null
  dominant_fraction?: number | null
  charted?: boolean
  mixed?: boolean
}

export interface RouteProfile {
  profile: ProfileName
  objective: string
  objective_units: string
  objective_value: number | null
  objective_is_the_configured_cost: boolean
  success: boolean
  outcome: string
  reason: string
  start: [number, number] | null
  goal: [number, number] | null
  departure_time: string
  arrival_time: string | null
  path: [number, number][]
  distance_m: number | null
  distance_km: number | null
  travel_time_s: number | null
  travel_time_h: number | null
  configured_cost: number | null
  environmental_contribution: number | null
  polaris_contribution: number | null
  coverage_uncertainty_contribution: number | null
  iceberg_exposure_contribution: number | null
  max_iceberg_exposure: number | null
  contributing_iceberg_ids: string[]
  iceberg_provenance: Record<string, unknown>
  special_consideration_cells: number | null
  indeterminate_cells: number | null
  mapping_sensitive_cells: number | null
  buckets_used: number[]
  cells: RouteCell[]
  temporal_provenance: Record<string, unknown>
  constraint_summary: Record<string, unknown>
  polaris: Record<string, unknown>
  /**
   * Additive fuel fields. They are ESTIMATES in open-water-equivalent metres
   * (OWE-m) -- a relative proxy, not a volume and not a mass -- and they are
   * absent when the backend resolved no sea-ice field for the run.
   */
  estimated_fuel?: number | null
  estimated_fuel_units?: string | null
  estimated_fuel_per_km?: number | null
  fuel_provenance?: FuelProvenance | null
}

/** What the backend says an estimated-fuel number means. Rendered, never inferred. */
export interface FuelProvenance {
  model: string
  version: string
  status: string
  is_a_measured_fuel_consumption: boolean
  is_an_operational_fuel_prediction: boolean
  units: string
  unit_definition: string
  equation: string
  edge_rule: string
  sic_sampled_at: string
  parameters: Record<string, number | null>
  parameter_status: Record<string, string>
  environmental_terms_used: string[]
  environmental_terms_excluded: string[]
  vessel_data_available: Record<string, string | number | null>
  no_absolute_figure_reason: string
  suitable_for_relative_route_comparison: boolean
  suitable_for_operational_fuel_prediction: boolean
  comparable_only_within: string
}

/** One arrival-time bucket's environmental lineage, exactly as the backend resolved it. */
export interface ForecastBucketProvenance {
  bucket: number
  source_type: string
  model?: string | null
  valid_from?: string | null
  valid_to?: string | null
  environment_date?: string | null
  policy?: string | null
  fallback?: string | null
  resolution?: {
    path?: string | null
    lead_hours?: number | null
    validity_rule?: string | null
    validity_from?: string | null
    validity_to?: string | null
    forecast_origin?: string | null
    valid_time?: string | null
    source_type?: string | null
    model?: string | null
  } | null
  model_artifact?: string | null
  model_artifact_sha256?: string | null
  dataset?: string | null
  dataset_sha256?: string | null
  dataset_link?: string | null
  sklearn_version_at_prediction?: string | null
}

export type EvaluationMode =
  | 'operational_demo'
  | 'long_horizon_scientific_evaluation'

/** The router's own refusal vocabulary. The frontend never invents one. */
export type FeasibilityOutcome =
  | 'no_feasible_path'
  | 'hard_navigation_constraint'
  | 'forecast_unavailable'
  | 'historical_date_mismatch'
  | 'search_limit_reached'

export interface Comparison {
  profiles: Record<ProfileName, RouteProfile>
  shared_inputs: Record<string, unknown>
  provenance: {
    environment_dates?: string[]
    chart_dates?: string[]
    all_dates_aligned?: boolean
    historical_demo_override?: boolean
    polaris_ice_class?: string | null
    polaris_riv_table?: string | null
    iceberg_exposure_weight?: number
    fastest_and_shortest_distance_coincided?: boolean | null
    why_they_can_coincide?: string
    [key: string]: unknown
  }
  objectives: Record<ProfileName, Record<string, unknown>>
}

export interface GridInfo {
  epsg: number
  proj4: string
  shape: [number, number]
  transform: number[]
  pixel_size: [number, number]
  extent: [number, number, number, number]
  environment_date: string
  usnic_chart_date: string
  dates_aligned: boolean
  excluded_cells: number
}

export interface MissionResponse {
  status: string
  message: string
  request: MissionRequest
  comparison: Comparison
  geojson: Record<string, GeoJsonCollection>
  endpoints: GeoJsonCollection
  grid: GridInfo
  historical_demonstration: boolean
}

export interface ForecastState {
  hours: number
  label: string
  latitude: number
  longitude: number
  radius_km: number
  extrapolated_uncertainty: boolean
  physics_position_is_extrapolated: boolean
  growth_law: string
  calibration_quantile: number
  calibration_n: number
  interpolation_quality: string
}

export interface IcebergForecastRecord {
  iceberg_id: string
  position_source: string
  forecast_start_date: string
  observed: ForecastState | null
  predicted: ForecastState[]
}

export interface ForecastResponse {
  forecast_start_date: string
  icebergs: IcebergForecastRecord[]
  iceberg_count: number
  observed_label: string
  observed_meaning: string
  predicted_meaning: string
  horizons: Array<{
    hours: number
    label: string
    radius_km: number
    extrapolated_uncertainty: boolean
    calibrated_to_hours: number
  }>
  model: {
    name: string
    windage_coefficient: number
    wind_turn_deg: number
    windage_is_active: boolean
    summary: string
    config: string
    is_validated_science: false
  }
  uncertainty: Record<string, unknown>
  population: Record<string, unknown>
}

export interface SimulatedIceberg {
  label: string
  iceberg_id: string
  simulated: true
  temporary: true
  written_to_any_dataset: false
  latitude: number
  longitude: number
  row: number
  col: number
  radius_km: number
  from_bucket: number
  encounter_arrival_s: number
  encounter_datetime: string
  encounter_cell: [number, number]
  distance_to_route_m: number
  horizon_hours: number
}

export interface ChangeEntry {
  before: number | null
  after: number | null
  delta: number | null
}

export interface ProfileChange {
  path_changed: boolean
  routed_before: boolean
  routed_after: boolean
  cells_with_simulated_exposure: number | null
  distance_m: ChangeEntry
  travel_time_s: ChangeEntry
  configured_cost: ChangeEntry
  iceberg_exposure_contribution: ChangeEntry
  max_iceberg_exposure: ChangeEntry
  polaris_contribution: ChangeEntry
  special_consideration_cells: ChangeEntry
}

export interface SimulationResponse {
  status: string
  message: string
  simulation: SimulatedIceberg | null
  simulated_exposure: GeoJsonCollection
  baseline: MissionResponse
  replanned: MissionResponse
  change: {
    simulation: SimulatedIceberg
    profiles: Record<ProfileName, ProfileChange>
    claims: Record<string, unknown>
  }
}

export interface MissionRequest {
  start: [number, number]
  goal: [number, number]
  departure_time: string
  vessel_speed_mps: number
  polaris_ice_class: string
  polaris_riv_table: string
  historical_demo_override: boolean
}

export interface HealthResponse {
  status: string
  prototype?: boolean
  message?: string
  disclaimer?: string
  grid?: GridInfo
  iceberg_exposure_weight?: number
  exposure_layers?: Array<Record<string, unknown>>
  polaris?: Record<string, unknown>
  requires_historical_demo_override?: boolean
}

// deliberately loose: OpenLayers' GeoJSON reader validates the real shape
export type GeoJsonCollection = {
  type: 'FeatureCollection'
  features: Array<Record<string, unknown>>
  [key: string]: unknown
}

export const OUTCOME_COPY: Record<string, { title: string; detail: string }> = {
  historical_date_mismatch: {
    title: 'Historical date mismatch',
    detail:
      'The USNIC chart and the environment fields are from different dates. The ' +
      'routing layer refuses that pairing unless historical demonstration mode ' +
      'is switched on explicitly.',
  },
  no_feasible_path: {
    title: 'No feasible route',
    detail:
      'The search exhausted every option inside the hard constraints. Nothing ' +
      'is substituted when this happens.',
  },
  forecast_unavailable: {
    title: 'Forecast unavailable',
    detail:
      'A time bucket the route needed has no field. A missing forecast is ' +
      'reported, never read as zero.',
  },
  hard_navigation_constraint: {
    title: 'Blocked endpoint',
    detail:
      'Departure or destination sits inside land, grounded ice or an area ' +
      'outside bathymetry coverage.',
  },
  search_limit_reached: {
    title: 'Search limit reached',
    detail: 'The expansion cap stopped the search before it could finish.',
  },
  backend_data_unavailable: {
    title: 'Backend data unavailable',
    detail:
      'A real input the API needs is missing from data/processed. The API ' +
      'serves measured data only and will not substitute anything for it.',
  },
}

// ---------------------------------------------------------------------------
// Mission decision layer: POST /api/mission/evaluate
//
// Several candidate destinations at several departure times, each evaluated
// independently. The backend does NOT rank the sites, rank the departures,
// name a best option or compute a score, and neither does anything below.
// ---------------------------------------------------------------------------

export interface CandidateSite {
  site_id: string
  cell: [number, number]
  label?: string
  /** Whether this coordinate is a real operational location. The repository
   *  contains none, so every shipped site declares false. */
  is_operational?: boolean
  source?: string
}

export interface MissionEvaluateRequest {
  start: [number, number]
  sites: CandidateSite[]
  departure_times: string[]
  profiles: ProfileName[]
  vessel_speed_mps?: number
  polaris_ice_class?: string
  polaris_riv_table?: string
  /** Omitted -> the shipped persistence policy, unchanged. Named -> that exact
   *  policy or a structured refusal; the server never substitutes one. */
  environment_policy?: string
  /** A horizon longer than the world's own provider puts the run in
   *  long-horizon evaluation mode, where the iceberg term is omitted. */
  horizon_buckets?: number
}

/** One profile's measured attributes for one option, or why it did not route. */
export interface MissionObjectiveRow {
  routed: boolean
  outcome?: string
  reason?: string
  distance_km?: number | null
  travel_time_h?: number | null
  configured_cost?: number | null
  polaris_contribution?: number | null
  environmental_contribution?: number | null
  iceberg_exposure_contribution?: number | null
  max_iceberg_exposure?: number | null
  estimated_fuel?: number | null
  estimated_fuel_units?: string | null
  estimated_fuel_per_km?: number | null
  cells?: number
  is_a_fuel_figure?: boolean
}

export interface MissionOptionSummary {
  scenario_id: string
  site: string
  departure_time: string
  feasible: boolean
  refusal_outcome: FeasibilityOutcome | string | null
  refusal_reason: string | null
  environment_source_types: string[] | null
  environment_models: string[] | null
  buckets_used: number[] | null
  objectives: Partial<Record<ProfileName, MissionObjectiveRow>>
}

export interface MissionMatrixSummary {
  start: [number, number]
  environment_policy: string
  evaluation_mode: EvaluationMode
  is_operational_assessment: boolean
  iceberg_exposure: string
  limitations: string[]
  profiles_requested: ProfileName[]
  options_evaluated: number
  options_feasible: number
  sites_feasible: string[]
  sites_infeasible: string[]
  departures_feasible: string[]
  departures_infeasible: string[]
  objectives_available: ProfileName[]
  options: MissionOptionSummary[]
  outcome: string
  no_feasible_option: boolean
  why?: string[]
  //  the claims the matrix explicitly does NOT make
  this_is_not_a_ranking: boolean
  has_a_best_site: boolean
  has_a_best_departure_time: boolean
  has_a_recommended_route: boolean
  has_an_overall_score: boolean
  selection_requires_a_caller_supplied_objective: boolean
}

export interface MissionMatrix {
  start: [number, number]
  sites: Array<Required<CandidateSite>>
  departure_times: string[]
  profiles_requested: ProfileName[]
  environment_policy: string
  evaluation_mode: EvaluationMode
  is_operational_assessment: boolean
  iceberg_exposure: string
  limitations: string[]
  options: Array<{
    scenario: Record<string, unknown>
    feasible: boolean
    refusal_outcome: string | null
    refusal_reason: string | null
    objectives: Partial<Record<ProfileName, MissionObjectiveRow>>
    provenance: {
      scenario_id: string
      departure_time: string
      environment_policy: string
      evaluation_mode: EvaluationMode
      is_operational_mission: boolean
      is_operational_assessment: boolean
      iceberg_exposure: { status: string; reason?: string | null }
      estimated_fuel_provenance: FuelProvenance | null
      environment: {
        source_types: string[] | null
        models: string[]
        forecast_origins: string[]
        forecast_leads: number[]
        validity_rules: string[]
        artifacts?: string[]
        buckets_used: number[]
        buckets: ForecastBucketProvenance[]
      }
      [key: string]: unknown
    }
  }>
  summary: MissionMatrixSummary
}

export interface MissionEvaluateResponse {
  status: string
  message: string
  environment_policy: string
  evaluation_mode: EvaluationMode
  is_operational_assessment: boolean
  iceberg_exposure: string
  limitations: string[]
  mission: MissionMatrix
  grid: GridInfo
}

// ---------------------------------------------------------------------------
// GET /api/evaluation/historical -- the held-out backtest, as produced.
// ---------------------------------------------------------------------------

export interface BacktestAccuracy {
  valid_time: string
  scored_cells: number
  model_label: string
  origin_is_inside_the_training_period: boolean
  model: { mae: number; rmse: number; bias: number; miz_mae: number } | null
  persistence: { mae: number; rmse: number; bias: number; miz_mae: number } | null
}

export interface BacktestCase {
  ran: boolean
  case: {
    origin: string
    included: boolean
    skip_reason: string | null
    targets: Record<string, string>
  }
  forecast_accuracy?: Record<string, BacktestAccuracy>
  decision?: {
    both_planned: boolean
    route_changed?: boolean
    shared_cells?: number
    first_divergence_index?: number | null
    distance_km_delta?: number
    travel_time_h_delta?: number
    configured_cost_delta?: number
    estimated_fuel_delta?: number
  }
  realized?: {
    evaluated?: boolean
    reason?: string
    realized_environmental_cost_delta?: number
    realized_estimated_fuel_delta?: number
    which_plan_met_lower_realized_environmental_cost?: string
  }
}

export interface HistoricalEvaluationResponse {
  status: string
  artifact: string
  sha256: string
  evaluation: {
    what_this_is: string
    is_operational_assessment: boolean
    candidate_rule: string
    held_out_window: [string, string]
    causal_order: string[]
    aggregate: {
      cases_attempted: number
      cases_included: number
      cases_skipped: number
      skip_reasons: Record<string, string>
      cases_both_planned: number
      route_changed: number
      route_unchanged: number
      cases_realized: number
      cases_not_realized: number
      not_realized_reasons: Record<string, string>
      forecast_plan_lower_realized_environmental_cost: number
      persistence_plan_lower_realized_environmental_cost: number
      tied_or_identical_route: number
      this_is_not_a_universal_accuracy_claim: boolean
      scope: string
    }
    cases: BacktestCase[]
  }
}

/** GET /api/mission/defaults -- the backend's own inputs, sites and policies. */
export interface MissionDefaults {
  demo: Record<string, unknown>
  grid: GridInfo
  /** candidate destinations the BACKEND defines; the UI invents none */
  sites?: Array<Required<CandidateSite>>
  long_horizon_site?: Required<CandidateSite>
  long_horizon_start?: [number, number]
  policies?: { default: string; supported: string[] }
}
