import type { ForecastBucketProvenance } from '../api/types'
import { EMPTY, timestamp } from './format'

interface Props {
  buckets: ForecastBucketProvenance[]
}

/**
 * "What forecast actually affected this route?"
 *
 * One row per arrival-time bucket that was priced from a model forecast, with
 * the full lineage the backend resolved: lead, origin, valid time, validity
 * rule and window, the raster, the fitted artifact and the dataset behind it.
 *
 * Nothing here is inferred. No lead is read off a filename, no artifact name is
 * constructed, and a field the backend did not supply is shown as missing
 * rather than filled in.
 */
export default function ForecastProvenanceList({ buckets }: Props) {
  const forecast = buckets.filter((b) => b.source_type === 'model_forecast')
  if (!forecast.length) {
    return (
      <p className="prov-empty" data-testid="forecast-provenance-empty">
        No bucket on this route was priced from a model forecast.
      </p>
    )
  }
  return (
    <ul className="prov-list" data-testid="forecast-provenance">
      {forecast.map((b) => {
        const r = b.resolution ?? {}
        const lead = r.lead_hours
        return (
          <li key={b.bucket} data-testid={`forecast-bucket-${b.bucket}`}>
            <div className="prov-head">
              <b>{lead === null || lead === undefined ? EMPTY : `+${lead}h`}</b>
              <span>{b.model ?? r.model ?? EMPTY}</span>
              <span className="prov-bucket">bucket {b.bucket}</span>
            </div>
            <dl className="prov-body">
              <div>
                <dt>Source</dt>
                <dd>{b.source_type}</dd>
              </div>
              <div>
                <dt>Forecast origin</dt>
                <dd>{timestamp(r.forecast_origin)}</dd>
              </div>
              <div>
                <dt>Valid time</dt>
                <dd>{timestamp(r.valid_time ?? b.valid_from)}</dd>
              </div>
              <div>
                <dt>Validity rule</dt>
                <dd>{r.validity_rule ?? EMPTY}</dd>
              </div>
              <div>
                <dt>Valid interval</dt>
                <dd>
                  {timestamp(r.validity_from)} → {timestamp(r.validity_to)}
                </dd>
              </div>
              <div>
                <dt>Artifact</dt>
                <dd className="prov-mono">{r.path ?? EMPTY}</dd>
              </div>
              <div>
                <dt>Model artifact</dt>
                <dd className="prov-mono">
                  {b.model_artifact ?? EMPTY}
                  {b.model_artifact_sha256 && (
                    <span className="prov-sha">
                      {' '}
                      sha256 {b.model_artifact_sha256.slice(0, 12)}…
                    </span>
                  )}
                </dd>
              </div>
              <div>
                <dt>Dataset</dt>
                <dd className="prov-mono">
                  {b.dataset ?? EMPTY}
                  {b.dataset_sha256 && (
                    <span className="prov-sha">
                      {' '}
                      sha256 {b.dataset_sha256.slice(0, 12)}…
                    </span>
                  )}
                </dd>
              </div>
            </dl>
          </li>
        )
      })}
    </ul>
  )
}
