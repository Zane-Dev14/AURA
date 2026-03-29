'use client'
import { useSceneStore } from '@/store/useSceneStore'
import { useEffect } from 'react'

export default function LoadingGate() {
  const { assetsLoaded, setAssetsLoaded } = useSceneStore()

  // Mark loaded after a small delay — assets preload via R3F Suspense
  useEffect(() => {
    const t = setTimeout(() => setAssetsLoaded(true), 2500)
    return () => clearTimeout(t)
  }, [setAssetsLoaded])

  if (assetsLoaded) return null

  return (
    <div className="loading-gate">
      {/* Crosshair reticle behind content */}
      <div className="loading-reticle" />

      <div className="loading-content">
        {/* Corner brackets around loading area */}
        <div className="loading-bracket-wrap">
          <div className="loading-bracket-bottom" />
          <div className="loading-logo">AURA</div>
          <div className="loading-subtitle">Kubernetes Intelligence System</div>
          <div className="loading-bar-wrap">
            <div className="loading-bar-fill" />
          </div>
          <div className="loading-text">Initializing Cluster Simulation…</div>
        </div>
      </div>
    </div>
  )
}
