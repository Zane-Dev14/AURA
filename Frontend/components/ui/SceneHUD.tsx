'use client'
import { useSceneStore, Scene } from '@/store/useSceneStore'
import QMixStatusText from './QMixStatusText'
import MetricsDashboard from './MetricsDashboard'

const SCENE_BUTTONS: Array<{
  scene: Scene
  label: string
  action: Scene
  styles: string
}> = [
  { scene: 'calm',      label: '→ Enter System',        action: 'system',     styles: 'btn-primary' },
  { scene: 'system',    label: '▶ Start Traffic',        action: 'traffic',    styles: 'btn-danger' },
  { scene: 'traffic',   label: '→ Proceed',              action: 'failure',    styles: 'btn-secondary' },
  { scene: 'failure',   label: '→ Continue',             action: 'emotional',  styles: 'btn-secondary' },
  { scene: 'emotional', label: '→ Activate QMix',        action: 'qmix',       styles: 'btn-primary' },
  { scene: 'qmix',      label: '',                       action: 'qmix',       styles: '' },
  { scene: 'transform', label: '',                       action: 'transform',  styles: '' },
  { scene: 'recovery',  label: '→ Comparison',           action: 'comparison', styles: 'btn-primary' },
  { scene: 'comparison',label: '',                       action: 'comparison', styles: '' },
]

const SCENE_LABELS: Record<Scene, string> = {
  calm: '① Calm World',
  system: '② Kubernetes Cluster',
  traffic: '③ Traffic Surge',
  failure: '④ System Failure',
  emotional: '⑤ Critical State',
  qmix: '⑥ QMix Ready',
  transform: '⑦ Transforming...',
  recovery: '⑧ Recovery',
  comparison: '⑨ HPA vs QMix',
}

export default function SceneHUD() {
  const { scene, setScene, skipTo, resetDemo, assetsLoaded } = useSceneStore()
  if (!assetsLoaded) return null

  const btn = SCENE_BUTTONS.find((b) => b.scene === scene)

  return (
    <div className="hud-root">
      {/* Top bar */}
      <div className="hud-topbar">
        <div className="hud-logo">AURA</div>
        <div className="hud-scene-label">{SCENE_LABELS[scene]}</div>
        <div className="hud-controls">
          <button className="btn-hud-sm" onClick={() => resetDemo()}>↺ Restart</button>
          <button className="btn-hud-sm" onClick={() => skipTo('transform')}>⚡ Skip to QMix</button>
          <button className="btn-hud-sm" onClick={() => skipTo('comparison')}>📊 Comparison</button>
        </div>
      </div>

      {/* QMix status msgs */}
      <QMixStatusText />

      {/* Main CTA */}
      {btn?.label && (
        <div className="hud-cta">
          <button
            className={`btn-cta ${btn.styles}`}
            onClick={() => setScene(btn.action)}
          >
            {btn.label}
          </button>
        </div>
      )}

      {/* Metrics sidebar (scenes recovery+) */}
      {(scene === 'recovery' || scene === 'comparison') && <MetricsDashboard />}
    </div>
  )
}
