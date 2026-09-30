import '@testing-library/jest-dom/vitest'

// OpenLayers needs a few browser APIs jsdom does not implement. The map itself
// is stubbed in the component tests; these keep module import from throwing.
if (!globalThis.ResizeObserver) {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver
}
