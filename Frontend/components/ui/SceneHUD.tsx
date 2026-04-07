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

// Tactical [XX] labels — darknode style
const SCENE_LABELS: Record<Scene, string> = {
  calm:       '[01] CALM WORLD',
  system:     '[02] KUBERNETES CLUSTER',
  traffic:    '[03] TRAFFIC SURGE',
  failure:    '[04] SYSTEM FAILURE',
  emotional:  '[05] CRITICAL STATE',
  qmix:      '[06] QMIX READY',
  transform:  '[07] TRANSFORMING',
  recovery:   '[08] RECOVERY',
  comparison: '[09] HPA vs QMIX',
}

const SCENE_ORDER: Scene[] = [
  'calm', 'system', 'traffic', 'failure', 'emotional',
  'qmix', 'transform', 'recovery', 'comparison',
]

export default function SceneHUD() {
  const { scene, setScene, skipTo, resetDemo, assetsLoaded } = useSceneStore()
  if (!assetsLoaded) return null

  const btn = SCENE_BUTTONS.find((b) => b.scene === scene)
  const currentIndex = SCENE_ORDER.indexOf(scene)

  return (
    <div className="hud-root">
      {/* Top bar */}
      <div className="hud-topbar">
        <div className="hud-logo">AURA</div>

        {/* Stage dots */}
        <div className="stage-dots">
          {SCENE_ORDER.map((s, i) => (
            <div
              key={s}
              className={`stage-dot ${i === currentIndex ? 'active' : i < currentIndex ? 'past' : ''}`}
            />
          ))}
        </div>

        <div className="hud-scene-label">{SCENE_LABELS[scene]}</div>
        <div className="hud-controls">
          <button className="btn-hud-sm" onClick={() => resetDemo()}>↺ RESTART</button>
          <button className="btn-hud-sm" onClick={() => skipTo('transform')}>⚡ SKIP</button>
          <button className="btn-hud-sm" onClick={() => skipTo('comparison')}>📊 COMPARE</button>
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
