'use client'

/**
 * FilmGrainOverlay – CSS-only film grain + vignette rendered in HTML layer.
 * Sits above the Canvas with pointer-events: none. No R3F dependency.
 */
export default function FilmGrainOverlay() {
  return <div className="film-grain-overlay" aria-hidden="true" />
}
