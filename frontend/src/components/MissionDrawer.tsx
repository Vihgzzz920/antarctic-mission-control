import { useState } from 'react'
import type { FormEvent } from 'react'

import type { GridInfo, MissionRequest } from '../api/types'
import { cellLonLat, formatLonLat } from '../map/coords'
import { DepartureIcon, DestinationIcon } from './icons'

interface Props {
  value: MissionRequest
  onChange: (next: MissionRequest) => void
  grid: GridInfo | null
  busy: boolean
}

const ICE_CLASSES = [
  'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7',
  'IA Super', 'IA', 'IB', 'IC', 'Not Ice Strengthened',
]

/** Mission configuration, collapsed to a summary until it is needed. */
export default function MissionDrawer({ value, onChange, grid, busy }: Props) {
  const [open, setOpen] = useState(false)
  const set = <K extends keyof MissionRequest>(
    key: K,
    next: MissionRequest[K],
  ) => onChange({ ...value, [key]: next })

  const position = (key: 'start' | 'goal') =>
    formatLonLat(cellLonLat(grid, value[key][0], value[key][1]))

  const cellInput = (key: 'start' | 'goal', index: 0 | 1) => (
    <input
      type="number"
      value={value[key][index]}
      aria-label={`${key === 'start' ? 'Departure' : 'Destination'} ${
        index === 0 ? 'row' : 'column'
      }`}
      onChange={(event) => {
        const next: [number, number] = [...value[key]] as [number, number]
        next[index] = Number(event.target.value)
        set(key, next)
      }}
    />
  )

  return (
    <form
      className={`mission-drawer${open ? ' is-open' : ''}`}
      onSubmit={(event: FormEvent) => event.preventDefault()}
      autoComplete="off"
      data-testid="mission-drawer"
    >
      <button
        type="button"
        className="drawer-summary"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
      >
        <span className="drawer-title">Mission</span>
        <span className="drawer-line">
          <DepartureIcon size={12} />
          {position('start') ?? `${value.start[0]}, ${value.start[1]}`}
        </span>
        <span className="drawer-line">
          <DestinationIcon size={12} />
          {position('goal') ?? `${value.goal[0]}, ${value.goal[1]}`}
        </span>
        <span className="drawer-line drawer-dim">
          {value.polaris_ice_class} · table {value.polaris_riv_table} ·{' '}
          {value.vessel_speed_mps} m/s
        </span>
        <span className="drawer-caret" aria-hidden="true">
          {open ? '−' : '+'}
        </span>
      </button>

      {open && (
        <div className="drawer-body">
          <label className="field">
            <span className="field-label">Departure time</span>
            <input
              type="datetime-local"
              value={value.departure_time.slice(0, 16)}
              disabled={busy}
              onChange={(event) =>
                set('departure_time', `${event.target.value}:00`)
              }
            />
          </label>

          <div className="field-pair">
            <label className="field">
              <span className="field-label">Vessel speed</span>
              <input
                type="number"
                step="0.1"
                min="0.1"
                value={value.vessel_speed_mps}
                disabled={busy}
                onChange={(event) =>
                  set('vessel_speed_mps', Number(event.target.value))
                }
              />
            </label>
            <label className="field">
              <span className="field-label">POLARIS class</span>
              <select
                value={value.polaris_ice_class}
                disabled={busy}
                onChange={(event) =>
                  set('polaris_ice_class', event.target.value)
                }
              >
                {ICE_CLASSES.map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </label>
          </div>

          <label className="field">
            <span className="field-label">RIV table</span>
            <select
              value={value.polaris_riv_table}
              disabled={busy}
              onChange={(event) => set('polaris_riv_table', event.target.value)}
            >
              <option value="1.3">1.3 — standard</option>
              <option value="1.4">1.4 — decayed ice</option>
            </select>
          </label>

          <details className="technical">
            <summary>Technical coordinates</summary>
            <div className="technical-body">
              <div className="technical-row">
                <span>Departure</span>
                {cellInput('start', 0)}
                {cellInput('start', 1)}
              </div>
              <div className="technical-row">
                <span>Destination</span>
                {cellInput('goal', 0)}
                {cellInput('goal', 1)}
              </div>
            </div>
          </details>

          <label className="historical-toggle">
            <input
              type="checkbox"
              checked={value.historical_demo_override}
              disabled={busy}
              onChange={(event) =>
                set('historical_demo_override', event.target.checked)
              }
            />
            <span>
              Historical demonstration mode
              <em>
                Off, the backend refuses to combine a chart and environment
                fields from different dates.
              </em>
            </span>
          </label>
        </div>
      )}
    </form>
  )
}
