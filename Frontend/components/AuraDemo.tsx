'use client'
import { Canvas } from '@react-three/fiber'
import { Environment, PerformanceMonitor, Preload, Stars } from '@react-three/drei'
import { Suspense, useState, useEffect } from 'react'
import { EffectComposer, Bloom, ChromaticAberration } from '@react-three/postprocessing'
import { BlendFunction } from 'postprocessing'
import { Vector2 } from 'three'
import { useSceneStore } from '@/store/useSceneStore'
import SceneManager from './SceneManager'
import SceneHUD from './ui/SceneHUD'
import LoadingGate from './LoadingGate'
import { unlockAudio } from '@/lib/sound'

export default function AuraDemo() {
  const [dpr, setDpr] = useState(1.5)
  const { glitchIntensity, scene, unlockAudio: storeUnlockAudio } = useSceneStore()

  const handleFirstInteraction = () => {
    unlockAudio()
    storeUnlockAudio()
  }

  const bloomIntensity = scene === 'transform' ? 8 : scene === 'recovery' ? 3 : 1.2
  const chromaticOffset = glitchIntensity * 0.005

  return (
    <div
      className="aura-root"
      onClick={handleFirstInteraction}
      onKeyDown={handleFirstInteraction}
    >
      <Canvas
        dpr={[1, dpr]}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        camera={{ position: [0, 4, 12], fov: 60 }}
        style={{ background: '#000' }}
      >
        <PerformanceMonitor
          onDecline={() => setDpr(1)}
          onIncline={() => setDpr(1.5)}
        />
        <Suspense fallback={null}>
          <Environment
            files={
              scene === 'recovery' || scene === 'comparison'
                ? '/models/spruit_sunrise_4k.exr'
                : '/models/evening_meadow_4k.exr'
            }
            background
            backgroundBlurriness={0.3}
          />
          <Stars radius={80} depth={50} count={3000} factor={3} fade speed={0.5} />
          <SceneManager />
          <Preload all />
        </Suspense>
        <EffectComposer>
          <Bloom
            intensity={bloomIntensity}
            luminanceThreshold={0.3}
            luminanceSmoothing={0.9}
            mipmapBlur
          />
          <ChromaticAberration
            blendFunction={BlendFunction.NORMAL}
            offset={new Vector2(chromaticOffset, chromaticOffset)}
          />
        </EffectComposer>
      </Canvas>
      <LoadingGate />
      <SceneHUD />
    </div>
  )
}
