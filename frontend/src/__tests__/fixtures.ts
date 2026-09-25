// A RECORDED backend response, captured from a real
// GET /api/demo/routes?historical_demo_override=true run and trimmed to two
// cells per profile. It is test input standing in for the network, never a
// source of numbers for the application: nothing imports it outside __tests__.
import type {
  ForecastResponse,
  MissionResponse,
  RouteProfile,
  SimulationResponse,
} from '../api/types'

const cell = (row: number, col: number, over: Record<string, unknown> = {}) => ({
  row,
  col,
  arrival_time: 0,
  arrival_datetime: '2025-01-08T00:00:00',
  bucket: 0,
  state: 'iceberg_exposure_priced',
  chart_state: 'charted_ice',
  environmental_cost: 1.5,
  polaris_penalty: 100,
  coverage_uncertainty_cost: 0,
  composed_cost: 101.5,
  iceberg_exposure: 0.25,
  iceberg_cost: 1.25,
  final_cost: 102.75,
  iceberg_id: 'exposure_raster_12h_max_over_11_icebergs',
  radius_km: 12.799382,
  forecast_horizon_hours: 12,
  extrapolated_uncertainty: false,
  polygon_id: 291,
  polaris_classification: 'operation_subject_to_special_consideration',
  polaris_special_consideration: true,
  polaris_indeterminate: true,
  dominant_fraction: 0.82,
  ...over,
})

const profile = (
  name: string,
  over: Record<string, unknown> = {},
): RouteProfile => ({
  profile: name,
  objective: 'travel time',
  objective_units: 'seconds',
  objective_value: 35606.6,
  objective_is_the_configured_cost: false,
  success: true,
  outcome: 'route_found',
  reason: 'reached goal',
  start: [948, 503],
  goal: [922, 497],
  departure_time: '2025-01-08T00:00:00',
  arrival_time: '2025-01-08T09:53:26',
  path: [[948, 503], [922, 497]],
  distance_m: 178033.01,
  distance_km: 178.03,
  travel_time_s: 35606.6,
  travel_time_h: 9.89,
  configured_cost: 15554112,
  environmental_contribution: 303265,
  polaris_contribution: 15227476,
  coverage_uncertainty_contribution: 0,
  iceberg_exposure_contribution: 23371,
  max_iceberg_exposure: 0.749926,
  contributing_iceberg_ids: ['exposure_raster_12h_max_over_11_icebergs'],
  iceberg_provenance: {},
  special_consideration_cells: 23,
  indeterminate_cells: 19,
  mapping_sensitive_cells: 28,
  buckets_used: [0, 1],
  cells: [
    cell(948, 503),
    cell(922, 497, {
      bucket: 1,
      arrival_time: 35606.6,
      arrival_datetime: '2025-01-08T09:53:26',
    }),
  ],
  temporal_provenance: {},
  constraint_summary: {},
  polaris: { selected: false },
  ...over,
}) as unknown as RouteProfile

const routeCollection = () => ({
  type: 'FeatureCollection' as const,
  features: [
    {
      type: 'Feature',
      geometry: {
        type: 'LineString',
        coordinates: [
          [-809375, -1490625],
          [-815625, -1540625],
        ],
      },
      properties: { profile: 'risk_oriented' },
    },
  ],
})

export function missionResponse(
  historical = true,
): MissionResponse {
  return {
    status: 'ok',
    message: 'all three profiles produced a route',
    request: {
      start: [948, 503],
      goal: [922, 497],
      departure_time: '2025-01-08T00:00:00',
      vessel_speed_mps: 5,
      polaris_ice_class: 'PC6',
      polaris_riv_table: '1.3',
      historical_demo_override: historical,
    },
    comparison: {
      profiles: {
        fastest: profile('fastest'),
        risk_oriented: profile('risk_oriented', {
          objective: 'the configured navigation cost',
          objective_units: 'configured cost units',
          objective_is_the_configured_cost: true,
          objective_value: 11659269.74,
          distance_km: 186.87,
          travel_time_h: 10.38,
          configured_cost: 11659269.74,
          polaris_contribution: 11325825,
          iceberg_exposure_contribution: 26476,
          special_consideration_cells: 18,
          indeterminate_cells: 14,
        }),
        shortest_distance: profile('shortest_distance', {
          objective: 'travelled distance',
          objective_units: 'metres',
          objective_value: 178033.01,
        }),
      },
      shared_inputs: {},
      provenance: {
        environment_dates: ['2025-01-08'],
        chart_dates: ['2020-12-24'],
        all_dates_aligned: false,
        historical_demo_override: historical,
        polaris_ice_class: 'PC6',
        polaris_riv_table: '1.3',
        iceberg_exposure_weight: 5,
        fastest_and_shortest_distance_coincided: true,
      },
      objectives: {
        fastest: {},
        risk_oriented: {},
        shortest_distance: {},
      },
    },
    geojson: {
      fastest: routeCollection(),
      risk_oriented: routeCollection(),
      shortest_distance: routeCollection(),
    },
    endpoints: { type: 'FeatureCollection', features: [] },
    grid: {
      epsg: 3976,
      proj4: '+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs=True',
      shape: [1328, 1264],
      transform: [6250, 0, -3950000, 0, -6250, 4350000],
      pixel_size: [6250, 6250],
      extent: [-3950000, -3950000, 3950000, 4350000],
      environment_date: '2025-01-08',
      usnic_chart_date: '2020-12-24',
      dates_aligned: false,
      excluded_cells: 854049,
    },
    historical_demonstration: historical,
  }
}


/** A recorded /api/mission/simulate response, trimmed the same way. */
export function simulationResponse(): SimulationResponse {
  const baseline = missionResponse(true)
  const replanned = missionResponse(true)
  const risk = replanned.comparison.profiles.risk_oriented
  risk.path = [
    [948, 503],
    [940, 500],
    [922, 497],
  ]
  risk.distance_km = 201.44
  risk.distance_m = 201_440
  risk.travel_time_h = 11.19
  risk.travel_time_s = 40_284
  risk.iceberg_exposure_contribution = 31_902

  const entry = (before: number, after: number) => ({
    before,
    after,
    delta: after - before,
  })

  return {
    status: 'ok',
    message: 'all three profiles produced a route',
    simulation: {
      label: 'SIMULATED ICEBERG',
      iceberg_id: 'simulated_iceberg',
      simulated: true,
      temporary: true,
      written_to_any_dataset: false,
      latitude: -74.43,
      longitude: -151.5,
      row: 934,
      col: 502,
      radius_km: 9.05053,
      from_bucket: 0,
      encounter_arrival_s: 18_017.77,
      encounter_datetime: '2025-01-08T05:00:17',
      encounter_cell: [934, 502],
      distance_to_route_m: 0,
      horizon_hours: 6,
    },
    simulated_exposure: { type: 'FeatureCollection', features: [] },
    baseline,
    replanned,
    change: {
      simulation: {
        label: 'SIMULATED ICEBERG',
        iceberg_id: 'simulated_iceberg',
        simulated: true,
        temporary: true,
        written_to_any_dataset: false,
        latitude: -74.43,
        longitude: -151.5,
        row: 934,
        col: 502,
        radius_km: 9.05053,
        from_bucket: 0,
        encounter_arrival_s: 18_017.77,
        encounter_datetime: '2025-01-08T05:00:17',
        encounter_cell: [934, 502],
        distance_to_route_m: 0,
        horizon_hours: 6,
      },
      profiles: {
        fastest: {
          path_changed: false,
          routed_before: true,
          routed_after: true,
          cells_with_simulated_exposure: 2,
          distance_m: entry(178_033, 178_033),
          travel_time_s: entry(35_606, 35_606),
          configured_cost: entry(15_554_112, 15_580_000),
          iceberg_exposure_contribution: entry(23_371, 49_259),
          max_iceberg_exposure: entry(0.75, 1),
          polaris_contribution: entry(15_227_476, 15_227_476),
          special_consideration_cells: entry(23, 23),
        },
        risk_oriented: {
          path_changed: true,
          routed_before: true,
          routed_after: true,
          cells_with_simulated_exposure: 3,
          distance_m: entry(186_872, 201_440),
          travel_time_s: entry(37_374, 40_284),
          configured_cost: entry(11_659_270, 11_921_000),
          iceberg_exposure_contribution: entry(26_476, 31_902),
          max_iceberg_exposure: entry(0.75, 0.75),
          polaris_contribution: entry(11_325_825, 11_480_000),
          special_consideration_cells: entry(18, 19),
        },
        shortest_distance: {
          path_changed: false,
          routed_before: true,
          routed_after: true,
          cells_with_simulated_exposure: 2,
          distance_m: entry(178_033, 178_033),
          travel_time_s: entry(35_606, 35_606),
          configured_cost: entry(15_554_112, 15_580_000),
          iceberg_exposure_contribution: entry(23_371, 49_259),
          max_iceberg_exposure: entry(0.75, 1),
          polaris_contribution: entry(15_227_476, 15_227_476),
          special_consideration_cells: entry(23, 23),
        },
      },
      claims: {
        is_a_collision_probability: false,
        guarantees_avoidance: false,
        ranks_the_routes: false,
      },
    },
  }
}


// A RECORDED GET /api/forecast/icebergs response, captured from a real run and
// trimmed to two of the eleven icebergs. Every horizon, position and radius
// below is the backend's own output; nothing here was computed in the frontend.
export function forecastResponse(): ForecastResponse {
  return {
    "forecast_start_date": "2025-01-08",
    "icebergs": [
      {
        "iceberg_id": "a76c",
        "position_source": "ascat",
        "forecast_start_date": "2025-01-08",
        "observed": {
          "hours": 0.0,
          "label": "Observed",
          "latitude": -64.24,
          "longitude": -55.48,
          "radius_km": 0.0,
          "extrapolated_uncertainty": false,
          "physics_position_is_extrapolated": false,
          "growth_law": "power",
          "calibration_quantile": 0.9,
          "calibration_n": 1197,
          "interpolation_quality": "observed_to_observed"
        },
        "predicted": [
          {
            "hours": 6.0,
            "label": "Predicted +6h",
            "latitude": -64.23456860321562,
            "longitude": -55.46545837115913,
            "radius_km": 9.05052989739592,
            "extrapolated_uncertainty": false,
            "physics_position_is_extrapolated": false,
            "growth_law": "power",
            "calibration_quantile": 0.9,
            "calibration_n": 1197,
            "interpolation_quality": "observed_to_observed"
          },
          {
            "hours": 12.0,
            "label": "Predicted +12h",
            "latitude": -64.22913720643125,
            "longitude": -55.45091959827958,
            "radius_km": 12.799382127560488,
            "extrapolated_uncertainty": false,
            "physics_position_is_extrapolated": false,
            "growth_law": "power",
            "calibration_quantile": 0.9,
            "calibration_n": 1197,
            "interpolation_quality": "observed_to_observed"
          },
          {
            "hours": 24.0,
            "label": "Predicted +24h",
            "latitude": -64.2182744128625,
            "longitude": -55.421850616647795,
            "radius_km": 18.10105979479184,
            "extrapolated_uncertainty": false,
            "physics_position_is_extrapolated": false,
            "growth_law": "power",
            "calibration_quantile": 0.9,
            "calibration_n": 1197,
            "interpolation_quality": "observed_to_observed"
          },
          {
            "hours": 48.0,
            "label": "Predicted +48h",
            "latitude": -64.19654882572502,
            "longitude": -55.36374688361914,
            "radius_km": 25.598764255120976,
            "extrapolated_uncertainty": true,
            "physics_position_is_extrapolated": true,
            "growth_law": "power",
            "calibration_quantile": 0.9,
            "calibration_n": 1197,
            "interpolation_quality": "observed_to_observed"
          }
        ]
      },
      {
        "iceberg_id": "a77",
        "position_source": "ascat",
        "forecast_start_date": "2025-01-08",
        "observed": {
          "hours": 0.0,
          "label": "Observed",
          "latitude": -65.6395,
          "longitude": -58.3452,
          "radius_km": 0.0,
          "extrapolated_uncertainty": false,
          "physics_position_is_extrapolated": false,
          "growth_law": "power",
          "calibration_quantile": 0.9,
          "calibration_n": 1197,
          "interpolation_quality": "observed_to_observed"
        },
        "predicted": [
          {
            "hours": 6.0,
            "label": "Predicted +6h",
            "latitude": -65.64301422291145,
            "longitude": -58.34368405207293,
            "radius_km": 9.05052989739592,
            "extrapolated_uncertainty": false,
            "physics_position_is_extrapolated": false,
            "growth_law": "power",
            "calibration_quantile": 0.9,
            "calibration_n": 1197,
            "interpolation_quality": "observed_to_observed"
          },
          {
            "hours": 12.0,
            "label": "Predicted +12h",
            "latitude": -65.64652844582291,
            "longitude": -58.34216789876364,
            "radius_km": 12.799382127560488,
            "extrapolated_uncertainty": false,
            "physics_position_is_extrapolated": false,
            "growth_law": "power",
            "calibration_quantile": 0.9,
            "calibration_n": 1197,
            "interpolation_quality": "observed_to_observed"
          },
          {
            "hours": 24.0,
            "label": "Predicted +24h",
            "latitude": -65.65355689164582,
            "longitude": -58.33913497581428,
            "radius_km": 18.10105979479184,
            "extrapolated_uncertainty": false,
            "physics_position_is_extrapolated": false,
            "growth_law": "power",
            "calibration_quantile": 0.9,
            "calibration_n": 1197,
            "interpolation_quality": "observed_to_observed"
          },
          {
            "hours": 48.0,
            "label": "Predicted +48h",
            "latitude": -65.66761378329164,
            "longitude": -58.33306666330311,
            "radius_km": 25.598764255120976,
            "extrapolated_uncertainty": true,
            "physics_position_is_extrapolated": true,
            "growth_law": "power",
            "calibration_quantile": 0.9,
            "calibration_n": 1197,
            "interpolation_quality": "observed_to_observed"
          }
        ]
      }
    ],
    "iceberg_count": 11,
    "observed_label": "Observed",
    "observed_meaning": "the iceberg's own observed start position: the model's INPUT, not a forecast",
    "predicted_meaning": "advected from the observed position by the drift baseline; a MODEL PREDICTION",
    "horizons": [
      {
        "hours": 6.0,
        "label": "+6h",
        "radius_km": 9.05052989739592,
        "extrapolated_uncertainty": false,
        "calibrated_to_hours": 24.0
      },
      {
        "hours": 12.0,
        "label": "+12h",
        "radius_km": 12.799382127560488,
        "extrapolated_uncertainty": false,
        "calibrated_to_hours": 24.0
      },
      {
        "hours": 24.0,
        "label": "+24h",
        "radius_km": 18.10105979479184,
        "extrapolated_uncertainty": false,
        "calibrated_to_hours": 24.0
      },
      {
        "hours": 48.0,
        "label": "+48h",
        "radius_km": 25.598764255120976,
        "extrapolated_uncertainty": true,
        "calibrated_to_hours": 24.0
      }
    ],
    "model": {
      "name": "first-order current + windage drift baseline",
      "windage_coefficient": 0.0,
      "wind_turn_deg": 0.0,
      "windage_is_active": false,
      "summary": "current advection only at the shipped windage coefficient of 0.0",
      "config": "configs/iceberg_physics.json",
      "is_validated_science": false
    },
    "uncertainty": {
      "source": "src/models/iceberg_uncertainty.py",
      "calibrated_to_hours": 24.0,
      "growth_law": "power",
      "beyond_calibration_is_extrapolated": true,
      "square_root_growth_is_a_project_assumption": true
    },
    "population": {
      "rows_on_this_date": 11,
      "distinct_icebergs_on_this_date": 11,
      "population_note": "clean observed-to-observed MOVED rows, ASCAT only; any other iceberg is invisible to this forecast"
    }
  } as unknown as ForecastResponse
}
