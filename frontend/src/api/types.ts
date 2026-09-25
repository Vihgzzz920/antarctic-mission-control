// Shapes of what the backend already serialises. Nothing here invents a field:
// every key below is produced by RouteProfileResult.to_dict(),
// ProfileComparison.to_dict() or the thin API wrapper around them.

export type ProfileName = 'fastest' | 'risk_oriented' | 'shortest_distance'

export const PROFILE_ORDER: ProfileName[] = [
  'fastest',
  'risk_oriented',
  'shortest_distance',
]

export const PROFILE_LABEL: Record<ProfileName, string> = {
  fastest: 'Fastest',
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
}

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
