// Formatting only. If a value is missing the display says so; it is never
// replaced with a zero or a guess.

export const EMPTY = '—'

export function num(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY
  return value.toLocaleString('en-GB', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

export function km(value: number | null | undefined): string {
  return value === null || value === undefined ? EMPTY : `${num(value, 2)} km`
}

export function hours(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY
  const whole = Math.floor(value)
  const minutes = Math.round((value - whole) * 60)
  return `${whole}h ${String(minutes).padStart(2, '0')}m`
}

export function exposure(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? EMPTY
    : value.toFixed(3)
}

export function count(value: number | null | undefined): string {
  return value === null || value === undefined ? 'not measured' : String(value)
}

export function timestamp(iso: string | null | undefined): string {
  if (!iso) return EMPTY
  return iso.replace('T', ' ').slice(0, 19) + 'Z'
}

/**
 * The estimated fuel proxy, in open-water-equivalent metres.
 *
 * OWE-m is a RELATIVE index: one unit is the fuel used steaming one metre in
 * ice-free water at the reference speed. It is never rendered as a volume or a
 * mass, because this project holds no engine, consumption, displacement or
 * efficiency data with which to resolve one.
 */
export function owe(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value))
    return EMPTY
  return `${num(value)} OWE-m`
}

/** Mean route difficulty: open-water-equivalent metres per kilometre travelled. */
export function owePerKm(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value))
    return EMPTY
  return `${num(value)} OWE-m/km`
}
