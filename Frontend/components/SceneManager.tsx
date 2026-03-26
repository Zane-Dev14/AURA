'use client'
import { useSceneStore } from '@/store/useSceneStore'
import { Suspense } from 'react'
import dynamic from 'next/dynamic'

const CalmWorld = dynamic(() => import('./scenes/CalmWorld'), { ssr: false })
const SystemReveal = dynamic(() => import('./scenes/SystemReveal'), { ssr: false })
const TrafficScene = dynamic(() => import('./scenes/TrafficScene'), { ssr: false })
const FailureScene = dynamic(() => import('./scenes/FailureScene'), { ssr: false })
const EmotionalBeat = dynamic(() => import('./scenes/EmotionalBeat'), { ssr: false })
const QMixActivation = dynamic(() => import('./scenes/QMixActivation'), { ssr: false })
const Transformation = dynamic(() => import('./scenes/Transformation'), { ssr: false })
const RecoveryScene = dynamic(() => import('./scenes/RecoveryScene'), { ssr: false })
const ComparisonScene = dynamic(() => import('./scenes/ComparisonScene'), { ssr: false })

export default function SceneManager() {
  const { scene } = useSceneStore()

  return (
    <Suspense fallback={null}>
      {scene === 'calm' && <CalmWorld />}
      {scene === 'system' && <SystemReveal />}
      {scene === 'traffic' && <TrafficScene />}
      {scene === 'failure' && <FailureScene />}
      {scene === 'emotional' && <EmotionalBeat />}
      {scene === 'qmix' && <QMixActivation />}
      {scene === 'transform' && <Transformation />}
      {scene === 'recovery' && <RecoveryScene />}
      {scene === 'comparison' && <ComparisonScene />}
    </Suspense>
  )
}
