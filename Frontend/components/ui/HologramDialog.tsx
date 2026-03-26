'use client'
import { useSceneStore } from '@/store/useSceneStore'
import { useState } from 'react'

export default function HologramDialog() {
  const { demoMode, setScene, setAirshipState, setQmixStatusMsg } = useSceneStore()
  const [status, setStatus] = useState<'idle' | 'connecting' | 'running'>('idle')

  const handleStart = async () => {
    setStatus('connecting')
    setAirshipState('locked')

    if (!demoMode) {
      try {
        const res = await fetch('/api/start-qmix', { method: 'POST' })
        const data = await res.json()
        console.log('QMix PID:', data.pid)
      } catch (e) {
        console.error('QMix start failed:', e)
      }
    }

    setStatus('running')
    setTimeout(() => {
      setScene('transform')
    }, 800)
  }

  return (
    <div className="hologram-dialog">
      <div className="holo-scanline" />
      <div className="holo-header">◈ AURA CONTROL SYSTEM ◈</div>
      <div className="holo-divider" />
      <div className="holo-body">
        <div className="holo-alert">! SYSTEM CRITICAL</div>
        <div className="holo-text">Intelligent scaling agent ready.</div>
        <div className="holo-text dim">QMix Controller — MARL v2.4</div>
        <div className="holo-text dim">Cluster: k3d-aura | Mode: {demoMode ? 'DEMO' : 'LIVE'}</div>
      </div>
      <div className="holo-divider" />
      <button
        className={`holo-btn ${status === 'running' ? 'running' : ''}`}
        onClick={handleStart}
        disabled={status !== 'idle'}
      >
        {status === 'idle' && '⚡ Start QMix Controller'}
        {status === 'connecting' && '● Connecting...'}
        {status === 'running' && '✓ Agent Active'}
      </button>
    </div>
  )
}
