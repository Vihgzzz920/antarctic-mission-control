import { PROFILE_LABEL, PROFILE_ORDER } from '../api/types'
import type { ProfileName } from '../api/types'
import { PROFILE_COLOUR } from '../map/layers'

interface Props {
  selected: ProfileName | null
  hasRoutes: boolean
  simulated?: boolean
}

export default function MapLegend({ selected, hasRoutes, simulated }: Props) {
  return (
    <div className="legend" data-testid="map-legend">
      {hasRoutes && (
        <div className="legend-group">
          {PROFILE_ORDER.map((name) => (
            <span
              key={name}
              className={`legend-row${name === selected ? ' is-active' : ''}`}
            >
              <span
                className="legend-line"
                style={{ background: PROFILE_COLOUR[name] }}
                aria-hidden="true"
              />
              {PROFILE_LABEL[name]}
            </span>
          ))}
        </div>
      )}
      {simulated && (
        <div className="legend-group">
          <span className="legend-row is-active">
            <span className="legend-swatch legend-sim" aria-hidden="true" />
            Simulated iceberg
          </span>
          <span className="legend-row">
            <span className="legend-line legend-superseded" aria-hidden="true" />
            Current route, superseded
          </span>
        </div>
      )}
      <div className="legend-group">
        <span className="legend-row">
          <span className="legend-swatch legend-berg" aria-hidden="true" />
          Modelled iceberg exposure
        </span>
        <span className="legend-row">
          <span className="legend-swatch legend-cone" aria-hidden="true" />
          Forecast uncertainty radius
        </span>
        <span className="legend-row">
          <span className="legend-swatch legend-land" aria-hidden="true" />
          Land and grounded ice
        </span>
        <span className="legend-row">
          <span className="legend-swatch legend-sic" aria-hidden="true" />
          Sea-ice concentration
        </span>
      </div>
    </div>
  )
}
