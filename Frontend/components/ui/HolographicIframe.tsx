'use client'
import { useState, useEffect } from 'react'
import { useSceneStore } from '@/store/useSceneStore'

interface Props {
  src: string
  title: string
}

export default function HolographicIframe({ src, title }: Props) {
  const { demoMode } = useSceneStore()
  const [ready, setReady] = useState(demoMode)
  const [connecting, setConnecting] = useState(!demoMode)

  useEffect(() => {
    if (demoMode) { setReady(true); return }
    // Trigger port-forward then show iframe
    const connect = async () => {
      try {
        setConnecting(true)
        await fetch('/api/port-forward-locust', { method: 'POST' })
        setReady(true)
      } catch {
        setReady(true) // still show, might work
      } finally {
        setConnecting(false)
      }
    }
    connect()
  }, [demoMode])

  return (
    <div className="holo-iframe-shell" style={{ transform: 'perspective(800px) rotateY(-8deg)' }}>
      <div className="holo-iframe-header">
        <span className="holo-iframe-dot red" />
        <span className="holo-iframe-dot yellow" />
        <span className="holo-iframe-dot green" />
        <span className="holo-iframe-title">{title}</span>
      </div>
      <div className="holo-iframe-body">
        {connecting && (
          <div className="holo-iframe-loading">
            <div className="holo-spinner" />
            <span>Connecting to {title}…</span>
          </div>
        )}
        {ready && !connecting && !demoMode && (
          <iframe src={src} width="640" height="360" title={title} className="holo-frame" />
        )}
        {ready && demoMode && (
          <div className="holo-demo-chart">
            <div className="demo-chart-title">📊 {title} — Demo Mode</div>
            <DemoBarChart />
          </div>
        )}
      </div>
      <div className="holo-scanlines" />
    </div>
  )
}

function DemoBarChart() {
  const bars = [
    { label: 'req/s', value: 85 },
    { label: 'errors', value: 45 },
    { label: 'p99 ms', value: 60 },
  ]
  return (
    <div className="demo-bars">
      {bars.map((b, i) => (
        <div key={i} className="demo-bar-row">
          <span className="demo-bar-label">{b.label}</span>
          <div className="demo-bar-track">
            <div
              className="demo-bar-fill"
              style={{ width: `${b.value}%`, animationDelay: `${i * 0.2}s` }}
            />
          </div>
          <span className="demo-bar-val">{b.value}</span>
        </div>
      ))}
    </div>
  )
}
