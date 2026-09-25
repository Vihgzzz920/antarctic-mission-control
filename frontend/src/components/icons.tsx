// Inline objective and map glyphs. Line art only: no gradients, no glow, no
// stock imagery. Each is a 16-unit square so they align on a text baseline.

import type { ProfileName } from '../api/types'

type Props = { size?: number; className?: string }

const wrap = (size: number, className: string | undefined, kids: JSX.Element) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.3"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    aria-hidden="true"
  >
    {kids}
  </svg>
)

/** Fastest: a clock face. */
export const ClockIcon = ({ size = 16, className }: Props) =>
  wrap(size, className, (
    <>
      <circle cx="8" cy="8" r="6" />
      <path d="M8 4.6V8l2.4 1.6" />
    </>
  ))

/** Risk-oriented: a contoured cost surface, not a shield. */
export const ContourIcon = ({ size = 16, className }: Props) =>
  wrap(size, className, (
    <>
      <path d="M1.8 10.6c1.9-2.6 4-3.9 6.2-3.9s4.3 1.3 6.2 3.9" />
      <path d="M3.6 13c1.3-1.7 2.8-2.6 4.4-2.6S11.1 11.3 12.4 13" />
      <circle cx="8" cy="3.6" r="1.5" />
    </>
  ))

/** Shortest distance: a measured straight leg. */
export const RulerIcon = ({ size = 16, className }: Props) =>
  wrap(size, className, (
    <>
      <path d="M2.6 12.4 13.4 3.6" />
      <circle cx="2.6" cy="12.4" r="1.6" />
      <circle cx="13.4" cy="3.6" r="1.6" />
    </>
  ))

export const PROFILE_ICON: Record<
  ProfileName,
  (props: Props) => JSX.Element
> = {
  fastest: ClockIcon,
  risk_oriented: ContourIcon,
  shortest_distance: RulerIcon,
}

/** Departure: a ringed fix. */
export const DepartureIcon = ({ size = 16, className }: Props) =>
  wrap(size, className, (
    <>
      <circle cx="8" cy="8" r="4.2" />
      <path d="M8 1.4v2.2M8 12.4v2.2M1.4 8h2.2M12.4 8h2.2" />
    </>
  ))

/** Destination: a target diamond. */
export const DestinationIcon = ({ size = 16, className }: Props) =>
  wrap(size, className, (
    <>
      <path d="M8 2.2 13.8 8 8 13.8 2.2 8z" />
      <circle cx="8" cy="8" r="1.4" fill="currentColor" stroke="none" />
    </>
  ))

/** Iceberg: a berg above the waterline with its mass below. */
export const IcebergIcon = ({ size = 16, className }: Props) =>
  wrap(size, className, (
    <>
      <path d="M8 2.2 12 8H4z" />
      <path d="M2.4 8h11.2" strokeDasharray="2 1.6" />
      <path d="M4 8l-1.4 3.4h10.8L12 8" />
    </>
  ))

export const AlertIcon = ({ size = 16, className }: Props) =>
  wrap(size, className, (
    <>
      <path d="M8 2.6 14.4 13H1.6z" />
      <path d="M8 6.4v3.1" />
      <circle cx="8" cy="11.3" r="0.55" fill="currentColor" stroke="none" />
    </>
  ))
