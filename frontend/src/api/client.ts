// The only place the browser talks to the backend. Every route metric on
// screen arrives through one of these calls; none is computed here.

import type {
  ForecastResponse,
  HealthResponse,
  MissionDefaults,
  MissionEvaluateRequest,
  MissionEvaluateResponse,
  HistoricalEvaluationResponse,
  MissionRequest,
  MissionResponse,
  ProfileName,
  SimulationResponse,
} from './types'

export class ApiError extends Error {
  readonly status: number
  readonly code: string

  constructor(message: string, status: number, code = 'api_error') {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

const BASE = '/api'

async function request_<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
  } catch (cause) {
    throw new ApiError(
      'The mission API is not reachable. Start it with: uvicorn src.api.main:app --port 8000',
      0,
      'api_unreachable',
    )
  }
  const text = await response.text()
  let body: unknown = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = null
  }
  if (!response.ok) {
    const detail = (body as { detail?: { message?: string; status?: string } })
      ?.detail
    throw new ApiError(
      detail?.message ?? `The API returned ${response.status}.`,
      response.status,
      detail?.status ?? 'api_error',
    )
  }
  return body as T
}

export function getHealth(): Promise<HealthResponse> {
  return request_<HealthResponse>('/health')
}

export function getDefaults(): Promise<MissionDefaults> {
  return request_<MissionDefaults>('/mission/defaults')
}

export function compareMission(
  body: MissionRequest,
): Promise<MissionResponse> {
  return request_<MissionResponse>('/mission/compare', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

/** Inject one simulated iceberg and replan. Nothing is persisted server-side. */
export function simulateIceberg(
  mission: MissionRequest,
  iceberg: { x: number; y: number },
  profile: ProfileName,
): Promise<SimulationResponse> {
  return request_<SimulationResponse>(
    `/mission/simulate?profile=${profile}`,
    { method: 'POST', body: JSON.stringify({ request: mission, iceberg }) },
  )
}

/**
 * Evaluate several candidate destinations at several departure times.
 *
 * Every decision the backend needs is stated in the body: omitting
 * `environment_policy` keeps the shipped persistence policy, and naming an
 * unsupported one is refused rather than silently replaced.
 */
export function evaluateMission(
  body: MissionEvaluateRequest,
): Promise<MissionEvaluateResponse> {
  return request_<MissionEvaluateResponse>('/mission/evaluate', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

/** The held-out historical backtest artifact, exactly as it was produced. */
export function getHistoricalEvaluation(): Promise<HistoricalEvaluationResponse> {
  return request_<HistoricalEvaluationResponse>('/evaluation/historical')
}

/** The observed and predicted iceberg states the model already produced. */
export function getIcebergForecast(): Promise<ForecastResponse> {
  return request_<ForecastResponse>('/forecast/icebergs')
}

export function getIcebergExposure(bucket = 1) {
  return request_<MissionResponse['endpoints']>(
    `/layers/iceberg-exposure?bucket=${bucket}`,
  )
}
