'use client'
import { useSceneStore } from '@/store/useSceneStore'
import { useEffect, useState, useRef } from 'react'

export default function LoadingGate() {
  const { assetsLoaded, setAssetsLoaded } = useSceneStore()
  const [progress, setProgress] = useState(0)
  const [visible, setVisible] = useState(true)
  const progressRef = useRef(0)
  const rafRef = useRef<number | undefined>(undefined)

  // Ultra-fast loading with requestAnimationFrame for smooth 60fps
  useEffect(() => {
    let lastTime = performance.now()
    
    const animate = (currentTime: number) => {
      const deltaTime = currentTime - lastTime
      lastTime = currentTime
      
      // Fast progress increment
      progressRef.current += (deltaTime / 1000) * 150 // Complete in ~0.67 seconds
      
      if (progressRef.current >= 100) {
        progressRef.current = 100
        setProgress(100)
        setAssetsLoaded(true)
        
        // Quick fade out
        setTimeout(() => setVisible(false), 200)
      } else {
        setProgress(Math.min(progressRef.current, 100))
        rafRef.current = requestAnimationFrame(animate)
      }
    }
    
    rafRef.current = requestAnimationFrame(animate)

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [setAssetsLoaded])

  if (!visible) return null

  return (
    <div
      className="loading-gate-awwwards"
      style={{
        opacity: progress >= 100 ? 0 : 1,
        transition: 'opacity 0.2s ease-out',
        pointerEvents: progress >= 100 ? 'none' : 'auto'
      }}
    >
      {/* Static background - no animation */}
      <div className="loading-radial-overlay" />

      {/* Main content - minimal animations */}
      <div className="loading-content-awwwards">
        {/* Logo - simple fade in */}
        <div className="loading-logo-awwwards">
          <div className="logo-letters">
            {['A', 'U', 'R', 'A'].map((letter, i) => (
              <span key={i} style={{ opacity: 1 }}>
                {letter}
              </span>
            ))}
          </div>
        </div>

        {/* Tagline */}
        <div className="loading-tagline">
          Autonomous Kubernetes Intelligence
        </div>

        {/* Progress bar - CSS only, GPU accelerated */}
        <div className="loading-progress-minimal">
          <div
            className="progress-bar-fill"
            style={{
              width: `${progress}%`,
              transform: 'translateZ(0)', // Force GPU acceleration
              willChange: 'width'
            }}
          />
        </div>
      </div>
    </div>
  )
}
