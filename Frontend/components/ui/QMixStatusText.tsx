'use client'
import { useSceneStore } from '@/store/useSceneStore'
import { useState } from 'react'

const QMIX_MESSAGES = [
  'Initializing intelligent scaling agent...',
  'Analyzing traffic patterns...',
  'Predicting load spike in T-5s...',
  'Deploying optimized scaling strategy...',
  'System stabilized. SLO met. ✓',
]

export default function QMixStatusText() {
  const { qmixStatusMsg, scene } = useSceneStore()
  if (scene !== 'transform' && scene !== 'recovery') return null
  if (!qmixStatusMsg) return null

  const isDone = qmixStatusMsg.includes('✓')

  return (
    <div className="qmix-status-bar">
      <div className={`qmix-status-dot ${isDone ? 'done' : 'active'}`} />
      <span className="qmix-status-text">{qmixStatusMsg}</span>
    </div>
  )
}
