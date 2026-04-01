'use client'
import { useSceneStore } from '@/store/useSceneStore'
import { useEffect, useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export default function LoadingGate() {
  const { assetsLoaded, setAssetsLoaded } = useSceneStore()
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState<'loading' | 'complete' | 'exit'>('loading')
  const progressRef = useRef(0)

  // Instant loading - no fake delays
  useEffect(() => {
    const interval = setInterval(() => {
      progressRef.current += Math.random() * 30 + 20
      if (progressRef.current >= 100) {
        progressRef.current = 100
        clearInterval(interval)
        setTimeout(() => {
          setAssetsLoaded(true)
          setPhase('complete')
        }, 100)
      }
      setProgress(Math.min(progressRef.current, 100))
    }, 80)

    return () => clearInterval(interval)
  }, [setAssetsLoaded])

  // Exit immediately after complete
  useEffect(() => {
    if (phase === 'complete') {
      setTimeout(() => setPhase('exit'), 300)
    }
  }, [phase])

  if (phase === 'exit') return null

  return (
    <AnimatePresence>
      <motion.div
        className="loading-gate-awwwards"
        initial={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.6, ease: [0.43, 0.13, 0.23, 0.96] }}
      >
        {/* Animated mesh background */}
        <div className="loading-mesh-bg" />
        
        {/* Radial gradient overlay */}
        <div className="loading-radial-overlay" />

        {/* Main content */}
        <div className="loading-content-awwwards">
          {/* Logo with dramatic entrance */}
          <motion.div
            className="loading-logo-awwwards"
            initial={{ scale: 0.8, opacity: 0, rotateX: -20 }}
            animate={{ scale: 1, opacity: 1, rotateX: 0 }}
            transition={{
              duration: 1.2,
              ease: [0.16, 1, 0.3, 1]
            }}
          >
            <div className="logo-letters">
              {['A', 'U', 'R', 'A'].map((letter, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0, y: 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    delay: 0.1 + i * 0.08,
                    duration: 0.8,
                    ease: [0.16, 1, 0.3, 1]
                  }}
                >
                  {letter}
                </motion.span>
              ))}
            </div>
            <motion.div
              className="logo-glow"
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.3, 0.6, 0.3]
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: 'easeInOut'
              }}
            />
          </motion.div>

          {/* Tagline */}
          <motion.div
            className="loading-tagline"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.8 }}
          >
            Autonomous Kubernetes Intelligence
          </motion.div>

          {/* Minimal progress indicator */}
          <motion.div
            className="loading-progress-minimal"
            initial={{ opacity: 0, scaleX: 0 }}
            animate={{ opacity: 1, scaleX: 1 }}
            transition={{ delay: 0.8, duration: 0.6 }}
          >
            <motion.div
              className="progress-bar-fill"
              style={{ width: `${progress}%` }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
            />
          </motion.div>
        </div>

        {/* Floating orbs - premium effect */}
        <div className="loading-orbs">
          {Array.from({ length: 20 }).map((_, i) => (
            <motion.div
              key={i}
              className="loading-orb"
              style={{
                left: `${10 + Math.random() * 80}%`,
                top: `${10 + Math.random() * 80}%`,
              }}
              animate={{
                y: [0, -30 - Math.random() * 40, 0],
                x: [0, (Math.random() - 0.5) * 40, 0],
                opacity: [0, 0.6, 0],
                scale: [0, 1, 0]
              }}
              transition={{
                duration: 4 + Math.random() * 3,
                repeat: Infinity,
                delay: Math.random() * 2,
                ease: 'easeInOut'
              }}
            />
          ))}
        </div>
      </motion.div>
    </AnimatePresence>
  )
}
