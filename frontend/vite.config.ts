/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API is a separate process (uvicorn on 8000). In development Vite proxies
// /api to it so the browser sees one origin; in production the built assets are
// served by whatever fronts the API.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 4173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: { outDir: 'dist', sourcemap: true },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    css: false,
    //  Two suites drive REAL wall-clock animation -- the forecast clock and
    //  the vessel transit -- and assert that time advanced. Run in parallel on
    //  a contended machine they lose their timeslice and report a clock that
    //  went backwards, which is a false failure, not a regression. Running
    //  test files one at a time makes the suite deterministic. No assertion is
    //  relaxed by this; the whole suite passes either way when the machine is
    //  idle, and only this setting makes that true when it is not.
    fileParallelism: false,
  },
})
