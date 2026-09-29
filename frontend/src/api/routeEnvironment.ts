// Reading the environmental provenance a route already carries.
//
// `RouteProfile.temporal_provenance` is an open record because the backend
// publishes a great deal in it. These readers narrow the few fields the UI
// renders, and narrow them DEFENSIVELY: a field the backend did not send comes
// back null, never a default that would look like an answer.
//
// Nothing here parses a filename, infers a lead, or decides a mode from
// anything but the policy the backend named.

import type {
  EvaluationMode,
  ForecastBucketProvenance,
  RouteProfile,
} from './types'

const POLICY_LONG_HORIZON = 'long_horizon_forecast_evaluation'

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v)

/** The environment policies this route was priced under, as the backend named them. */
export function environmentPolicies(profile: RouteProfile | null): string[] {
  const raw = profile?.temporal_provenance?.environment_policies
  if (!Array.isArray(raw)) return []
  return raw.filter((x): x is string => typeof x === 'string')
}

/**
 * Which kind of answer this route is.
 *
 * Decided ONLY by the policy the backend reported. A route priced under the
 * long-horizon evaluation policy is a scientific evaluation; anything else is
 * the operational demonstration. The frontend never infers this from a horizon,
 * a bucket count or a route length.
 */
export function evaluationModeOf(profile: RouteProfile | null): EvaluationMode {
  return environmentPolicies(profile).includes(POLICY_LONG_HORIZON)
    ? 'long_horizon_scientific_evaluation'
    : 'operational_demo'
}

/** True when the backend priced this route as an operational assessment. */
export function isOperationalAssessment(profile: RouteProfile | null): boolean {
  return evaluationModeOf(profile) === 'operational_demo'
}

/**
 * One record per arrival-time bucket, in bucket order.
 *
 * The backend publishes these as `environment_by_bucket`, keyed by the bucket
 * number as a string. A payload without it yields an empty list rather than a
 * fabricated one.
 */
export function environmentBuckets(
  profile: RouteProfile | null,
): ForecastBucketProvenance[] {
  const raw = profile?.temporal_provenance?.environment_by_bucket
  if (!isRecord(raw)) return []
  return Object.values(raw)
    .filter(isRecord)
    .map((entry) => entry as unknown as ForecastBucketProvenance)
    .filter((entry) => typeof entry.bucket === 'number')
    .sort((a, b) => a.bucket - b.bucket)
}

/** Only the buckets a model forecast actually priced. */
export function forecastBuckets(
  profile: RouteProfile | null,
): ForecastBucketProvenance[] {
  return environmentBuckets(profile).filter(
    (b) => b.source_type === 'model_forecast',
  )
}

/** The distinct forecast leads this route used, as the backend reported them. */
export function forecastLeads(profile: RouteProfile | null): number[] {
  const leads = new Set<number>()
  for (const b of forecastBuckets(profile)) {
    const lead = b.resolution?.lead_hours
    if (typeof lead === 'number' && Number.isFinite(lead)) leads.add(lead)
  }
  return [...leads].sort((a, b) => a - b)
}
