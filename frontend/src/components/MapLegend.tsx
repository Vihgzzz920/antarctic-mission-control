import type { LegendEntry } from '../map/layerContract'

interface Props {
  /** built by map/layerContract.legendEntries from the SAME list of layer ids
   *  the map was given, so every row here is a layer that was drawn */
  entries: LegendEntry[]
}

/** The mark for a row, matching how that layer is drawn on the map. */
function Swatch({ entry }: { entry: LegendEntry }) {
  if (entry.swatch === 'line' || entry.swatch === 'dashed') {
    return (
      <span
        className={`legend-mark legend-${entry.swatch}`}
        style={entry.colour ? { background: entry.colour } : undefined}
        aria-hidden="true"
      />
    )
  }
  return <span className={`legend-mark legend-${entry.swatch}`} aria-hidden="true" />
}

export default function MapLegend({ entries }: Props) {
  if (entries.length === 0) return null
  return (
    <div className="legend" data-testid="map-legend">
      {entries.map((entry) => (
        <span
          key={entry.key}
          className={`legend-row${entry.emphasis ? ' is-active' : ''}`}
          data-layer={entry.layer}
        >
          <Swatch entry={entry} />
          <span className="legend-text">
            {entry.label}
            {entry.note && <i>{entry.note}</i>}
          </span>
        </span>
      ))}
    </div>
  )
}
