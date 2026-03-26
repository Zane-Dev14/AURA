'use client'
import { useSceneStore } from '@/store/useSceneStore'
import { useEffect, useRef } from 'react'

function AnimatedValue({ value, suffix = '', decimals = 0 }: { value: number; suffix?: string; decimals?: number }) {
  const displayRef = useRef<HTMLSpanElement>(null)
  const current = useRef(value)

  useEffect(() => {
    const target = value
    const start = current.current
    const duration = 1200
    const startTime = performance.now()

    function tick(now: number) {
      const elapsed = Math.min(now - startTime, duration)
      const t = elapsed / duration
      const eased = 1 - Math.pow(1 - t, 3) // ease-out cubic
      const val = start + (target - start) * eased
      current.current = val
      if (displayRef.current) {
        displayRef.current.textContent = val.toFixed(decimals) + suffix
      }
      if (elapsed < duration) requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
  }, [value, suffix, decimals])

  return <span ref={displayRef}>{value.toFixed(decimals) + suffix}</span>
}

export default function MetricsDashboard() {
  const { metrics, scene } = useSceneStore()

  const rows = [
    { label: 'API P99', value: metrics.latencyMs, suffix: 'ms', decimals: 0, good: metrics.latencyMs < 50 },
    { label: 'Error Rate', value: metrics.failures, suffix: ' errs', decimals: 0, good: metrics.failures === 0 },
    { label: 'Active Pods', value: metrics.pods, suffix: '', decimals: 0, good: metrics.pods >= 6 },
    { label: 'CPU Util', value: metrics.cpuPercent, suffix: '%', decimals: 0, good: metrics.cpuPercent < 80 },
    { label: 'RPS', value: metrics.rps, suffix: '', decimals: 0, good: metrics.rps > 100 },
  ]

  return (
    <div className="metrics-panel">
      <div className="metrics-title">SYSTEM METRICS</div>
      {rows.map((r) => (
        <div key={r.label} className="metrics-row">
          <span className="metrics-label">{r.label}</span>
          <span className={`metrics-value ${r.good ? 'good' : 'bad'}`}>
            <AnimatedValue value={r.value} suffix={r.suffix} decimals={r.decimals} />
          </span>
        </div>
      ))}
      {scene === 'comparison' && (
        <div className="metrics-comparison-hint">
          QMix: 88.6% faster P99 · 50pp safer CPU
        </div>
      )}
    </div>
  )
}
