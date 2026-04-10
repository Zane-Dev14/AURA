'use client'
import { useSceneStore } from '@/store/useSceneStore'
import { Suspense } from 'react'
import dynamic from 'next/dynamic'
import CameraController from './r3f/CameraController'

function loadWithChunkRecovery<T>(importer: () => Promise<T>): Promise<T> {
  return importer().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error)
    const isRecoverableLoadError =
      message.includes('ChunkLoadError') ||
      message.includes('Loading chunk') ||
      message.includes('a[d] is not a function') ||
      message.includes('Failed to read source code') ||
      message.includes('Cannot read properties of undefined')

    if (typeof window !== 'undefined' && isRecoverableLoadError) {
      window.location.reload()
      return new Promise<T>(() => {
        // Intentionally unresolved: page is reloading.
      })
    }

    throw error
  })
}

const CalmWorld = dynamic(
  () => loadWithChunkRecovery(() => import('./scenes/CalmWorld')),
  { ssr: false }
)
const SystemReveal = dynamic(
  () => loadWithChunkRecovery(() => import('./scenes/SystemReveal')),
  { ssr: false }
)
const TrafficScene = dynamic(
  () => loadWithChunkRecovery(() => import('./scenes/TrafficScene')),
  { ssr: false }
)
const FailureScene = dynamic(
  () => loadWithChunkRecovery(() => import('./scenes/FailureScene')),
  { ssr: false }
)
const EmotionalBeat = dynamic(
  () => loadWithChunkRecovery(() => import('./scenes/EmotionalBeat')),
  { ssr: false }
)
const QMixActivation = dynamic(
  () => loadWithChunkRecovery(() => import('./scenes/QMixActivation')),
  { ssr: false }
)
const Transformation = dynamic(
  () => loadWithChunkRecovery(() => import('./scenes/Transformation')),
  { ssr: false }
)
const RecoveryScene = dynamic(
  () => loadWithChunkRecovery(() => import('./scenes/RecoveryScene')),
  { ssr: false }
)
const ComparisonScene = dynamic(
  () => loadWithChunkRecovery(() => import('./scenes/ComparisonScene')),
  { ssr: false }
)

export default function SceneManager() {
  const { scene } = useSceneStore()

  return (
    <>
      <CameraController />
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
    </>
  )
}
