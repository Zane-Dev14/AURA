'use client'
import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useSceneStore } from '@/store/useSceneStore'

/**
 * InteractionHints - Subtle UI hints for user interaction
 * 
 * Shows hints that:
 * - Fade in after intro completes
 * - Fade out after user starts interacting
 * - Are non-intrusive and elegant
 * - Guide without being annoying
 */
export default function InteractionHints() {
  const { introComplete } = useSceneStore()
  const [showHints, setShowHints] = useState(false)
  const [hasInteracted, setHasInteracted] = useState(false)

  // Show hints 1 second after intro completes
  useEffect(() => {
    if (introComplete && !hasInteracted) {
      const timer = setTimeout(() => setShowHints(true), 1000)
      return () => clearTimeout(timer)
    }
  }, [introComplete, hasInteracted])

  // Hide hints after 10 seconds or on interaction
  useEffect(() => {
    if (showHints) {
      const timer = setTimeout(() => setShowHints(false), 10000)
      return () => clearTimeout(timer)
    }
  }, [showHints])

  // Detect user interaction
  useEffect(() => {
    const handleInteraction = () => {
      setHasInteracted(true)
      setShowHints(false)
    }

    const events = ['keydown', 'mousemove', 'wheel', 'click']
    events.forEach(event => {
      window.addEventListener(event, handleInteraction, { once: true })
    })

    return () => {
      events.forEach(event => {
        window.removeEventListener(event, handleInteraction)
      })
    }
  }, [])

  if (!introComplete) return null

  return (
    <>
      <AnimatePresence>
        {showHints && (
          <motion.div
            key="interaction-hints"
            className="interaction-hints"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 1 }}
          >
          {/* WASD hint */}
          <motion.div
            className="hint hint-wasd"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ delay: 0.2, duration: 0.8 }}
          >
            <div className="hint-keys">
              <span className="key">W</span>
              <span className="key">A</span>
              <span className="key">S</span>
              <span className="key">D</span>
            </div>
            <span className="hint-text">Fly Around</span>
          </motion.div>

          {/* Mouse hint */}
          <motion.div
            className="hint hint-mouse"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ delay: 0.4, duration: 0.8 }}
          >
            <div className="hint-icon">🖱️</div>
            <span className="hint-text">Look Around</span>
          </motion.div>

          {/* Scroll hint */}
          <motion.div
            className="hint hint-scroll"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ delay: 0.6, duration: 0.8 }}
          >
            <div className="hint-icon">⬇️</div>
            <span className="hint-text">Scroll to Zoom</span>
          </motion.div>

          </motion.div>
        )}
      </AnimatePresence>
      <style jsx>{`
        .interaction-hints {
          position: fixed;
          bottom: 40px;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          gap: 30px;
          z-index: 100;
          pointer-events: none;
        }

        .hint {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
          padding: 12px 20px;
          background: rgba(0, 10, 20, 0.7);
          border: 1px solid rgba(0, 229, 255, 0.3);
          border-radius: 8px;
          backdrop-filter: blur(10px);
        }

        .hint-keys {
          display: flex;
          gap: 6px;
        }

        .key {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 32px;
          height: 32px;
          background: rgba(0, 229, 255, 0.1);
          border: 1px solid rgba(0, 229, 255, 0.4);
          border-radius: 4px;
          color: #00e5ff;
          font-family: 'Courier New', monospace;
          font-size: 14px;
          font-weight: bold;
          box-shadow: 0 2px 8px rgba(0, 229, 255, 0.2);
        }

        .hint-icon {
          font-size: 24px;
          filter: drop-shadow(0 0 8px rgba(0, 229, 255, 0.5));
        }

        .hint-text {
          color: rgba(255, 255, 255, 0.8);
          font-size: 12px;
          font-weight: 500;
          text-transform: uppercase;
          letter-spacing: 1px;
          text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
        }

        .hint-skip {
          position: fixed;
          bottom: 20px;
          left: 50%;
          transform: translateX(-50%);
        }

        @media (max-width: 768px) {
          .interaction-hints {
            flex-direction: column;
            gap: 15px;
            bottom: 20px;
          }

          .hint {
            padding: 10px 16px;
          }

          .key {
            width: 28px;
            height: 28px;
            font-size: 12px;
          }

          .hint-text {
            font-size: 11px;
          }
        }
      `}</style>
    </>
  )
}

// Made with Bob