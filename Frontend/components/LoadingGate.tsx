'use client'
import { useSceneStore } from '@/store/useSceneStore'
import { useEffect, useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import gsap from 'gsap'

export default function LoadingGate() {
  const { assetsLoaded, setAssetsLoaded } = useSceneStore()
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState<'loading' | 'complete' | 'exit'>('loading')
  const progressRef = useRef(0)

  // Simulate loading progress with realistic timing
  useEffect(() => {
    const interval = setInterval(() => {
      progressRef.current += Math.random() * 15
      if (progressRef.current >= 100) {
        progressRef.current = 100
        clearInterval(interval)
        setTimeout(() => {
          setAssetsLoaded(true)
          setPhase('complete')
        }, 300)
      }
      setProgress(Math.min(progressRef.current, 100))
    }, 150)

    return () => clearInterval(interval)
  }, [setAssetsLoaded])

  // Exit animation sequence
  useEffect(() => {
    if (phase === 'complete') {
      setTimeout(() => setPhase('exit'), 800)
    }
  }, [phase])

  if (phase === 'exit') return null

  return (
    <AnimatePresence>
      <motion.div
        className="loading-gate-premium"
        initial={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 1, ease: [0.43, 0.13, 0.23, 0.96] }}
      >
        {/* Animated gradient background */}
        <div className="loading-bg-gradient" />
        
        {/* Grid overlay */}
        <div className="loading-grid-overlay" />

        {/* Main content */}
        <div className="loading-content-premium">
          {/* Top label */}
          <motion.div
            className="loading-top-label"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.8 }}
          >
            KUBERNETES INTELLIGENCE PLATFORM
          </motion.div>

          {/* Main logo with stagger animation */}
          <div className="loading-logo-container">
            {['A', 'U', 'R', 'A'].map((letter, i) => (
              <motion.div
                key={i}
                className="loading-logo-letter"
                initial={{ opacity: 0, y: 40, rotateX: -90 }}
                animate={{ opacity: 1, y: 0, rotateX: 0 }}
                transition={{
                  delay: 0.4 + i * 0.1,
                  duration: 0.8,
                  ease: [0.43, 0.13, 0.23, 0.96]
                }}
              >
                {letter}
              </motion.div>
            ))}
          </div>

          {/* Subtitle */}
          <motion.div
            className="loading-subtitle-premium"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1, duration: 0.8 }}
          >
            Autonomous Resource Optimization
          </motion.div>

          {/* Progress section */}
          <motion.div
            className="loading-progress-section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2, duration: 0.8 }}
          >
            {/* Progress bar */}
            <div className="loading-progress-container">
              <div className="loading-progress-track">
                <motion.div
                  className="loading-progress-fill"
                  style={{ width: `${progress}%` }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                />
                <motion.div
                  className="loading-progress-glow"
                  style={{ left: `${progress}%` }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                />
              </div>
              
              {/* Progress percentage */}
              <motion.div
                className="loading-progress-text"
                key={Math.floor(progress)}
                initial={{ opacity: 0.5 }}
                animate={{ opacity: 1 }}
              >
                {Math.floor(progress)}%
              </motion.div>
            </div>

            {/* Status text */}
            <motion.div
              className="loading-status-text"
              animate={{
                opacity: [0.4, 1, 0.4]
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut'
              }}
            >
              {progress < 30 && 'Initializing Neural Network...'}
              {progress >= 30 && progress < 60 && 'Loading Cluster Topology...'}
              {progress >= 60 && progress < 90 && 'Calibrating AI Agents...'}
              {progress >= 90 && progress < 100 && 'Preparing Simulation...'}
              {progress === 100 && 'Ready'}
            </motion.div>
          </motion.div>

          {/* Bottom decorative elements */}
          <motion.div
            className="loading-bottom-decoration"
            initial={{ opacity: 0, scaleX: 0 }}
            animate={{ opacity: 1, scaleX: 1 }}
            transition={{ delay: 1.4, duration: 1, ease: 'easeOut' }}
          >
            <div className="loading-decoration-line" />
            <div className="loading-decoration-dot" />
            <div className="loading-decoration-line" />
          </motion.div>
        </div>

        {/* Floating particles */}
        <div className="loading-particles-premium">
          {Array.from({ length: 40 }).map((_, i) => (
            <motion.div
              key={i}
              className="loading-particle-premium"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                y: [0, -100 - Math.random() * 200],
                opacity: [0, 1, 0],
                scale: [0, 1, 0.5]
              }}
              transition={{
                duration: 3 + Math.random() * 4,
                repeat: Infinity,
                delay: Math.random() * 3,
                ease: 'easeOut'
              }}
            />
          ))}
        </div>

        {/* Corner brackets */}
        <div className="loading-corner-brackets">
          <div className="loading-bracket loading-bracket-tl" />
          <div className="loading-bracket loading-bracket-tr" />
          <div className="loading-bracket loading-bracket-bl" />
          <div className="loading-bracket loading-bracket-br" />
        </div>
      </motion.div>
    </AnimatePresence>
  )
}

// Made with Bob
