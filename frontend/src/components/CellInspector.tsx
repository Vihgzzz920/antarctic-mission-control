import type { RouteCell } from '../api/types'
import { EMPTY, exposure, num, timestamp } from './format'

interface Props {
  cell: RouteCell | null
  onClose: () => void
}

export default function CellInspector({ cell, onClose }: Props) {
  if (!cell) return null

  const primary: Array<[string, string]> = [
    ['Arrival', timestamp(cell.arrival_datetime)],
    ['Cell cost', num(cell.final_cost, 2)],
    ['POLARIS', num(cell.polaris_penalty, 2)],
    [
      'Iceberg exposure',
      cell.iceberg_exposure === null
        ? 'unavailable'
        : exposure(cell.iceberg_exposure),
    ],
    [
      'Uncertainty radius',
      cell.radius_km === null ? EMPTY : `${num(cell.radius_km, 1)} km`,
    ],
  ]

  const secondary: Array<[string, string]> = [
    ['Time bucket', String(cell.bucket)],
    ['Dominant source', cell.iceberg_id ?? 'none'],
    [
      'Exposure field',
      cell.forecast_horizon_hours === null ||
      cell.forecast_horizon_hours === undefined
        ? 'no modelled exposure here'
        : `+${cell.forecast_horizon_hours}h`,
    ],
    ['Environmental', num(cell.environmental_cost, 2)],
    ['Chart state', (cell.chart_state ?? EMPTY).replace(/_/g, ' ')],
    [
      'Dominant fraction',
      cell.dominant_fraction === null || cell.dominant_fraction === undefined
        ? 'not applicable'
        : cell.dominant_fraction.toFixed(3),
    ],
    [
      'POLARIS class',
      (cell.polaris_classification ?? EMPTY).replace(/_/g, ' '),
    ],
    ['Grid cell', `${cell.row}, ${cell.col}`],
  ]

  return (
    <aside className="inspector" data-testid="cell-inspector">
      <header className="inspector-head">
        <div>
          <span className="inspector-eyebrow">Route cell</span>
          <h3>{timestamp(cell.arrival_datetime).slice(11, 16)} arrival</h3>
        </div>
        <button type="button" onClick={onClose} aria-label="Close inspector">
          ×
        </button>
      </header>

      <dl className="inspector-primary">
        {primary.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>

      <details className="inspector-more">
        <summary>Cell metadata</summary>
        <dl className="inspector-secondary">
          {secondary.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </details>

      {cell.extrapolated_uncertainty && (
        <p className="inspector-flag">
          Uncertainty radius extrapolated beyond the 24 h calibration.
        </p>
      )}
    </aside>
  )
}
