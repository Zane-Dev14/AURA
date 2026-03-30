'use client'
import { Canvas } from '@react-three/fiber'
import { Environment, PerformanceMonitor, Preload, Stars } from '@react-three/drei'
import { Suspense, useState } from 'react'
import { EffectComposer, Bloom, ChromaticAberration, Vignette } from '@react-three/postprocessing'
import { BlendFunction } from 'postprocessing'
import * as THREE from 'three'
import { useSceneStore } from '@/store/useSceneStore'
import SceneManager from './SceneManager'
import SceneHUD from './ui/SceneHUD'
import LoadingGate from './LoadingGate'
import FilmGrainOverlay from './ui/FilmGrainOverlay'
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

  // Vignette darkness varies per scene mood
  const vignetteDarkness =
    scene === 'failure' || scene === 'emotional' ? 0.85
    : scene === 'transform' ? 0.5
    : 0.7

  return (
    <div
      className="aura-root"
      data-scene={scene}
      onClick={handleFirstInteraction}
      onKeyDown={handleFirstInteraction}
    >
      <Canvas
        dpr={[1, dpr]}
        gl={{
          antialias: true,
          powerPreference: 'high-performance',
          toneMappingExposure: 1.2,
          toneMapping: THREE.ACESFilmicToneMapping
        }}
        camera={{ position: [0, 8, 20], fov: 55, near: 0.1, far: 200 }}
        style={{ background: '#000' }}
      >
        <fog attach="fog" args={['#010308', 15, 100]} />
        <color attach="background" args={['#000205']} />
        <PerformanceMonitor
          onDecline={() => setDpr(1)}
          onIncline={() => setDpr(1.5)}
        />
        <Suspense fallback={null}>
          <Environment
            files={
              scene === 'recovery' || scene === 'comparison'
                ? '/models/spruit_sunrise_4k.exr'
                : '/models/1082ab60-0925-4509-9e69-90a7dfce573c.hdr'
            }
            background
            backgroundBlurriness={0}
            backgroundIntensity={0.4}
            environmentIntensity={1.5}
          />
          <Stars radius={100} depth={80} count={5000} factor={4} fade speed={0.3} />
          <SceneManager />
          <Preload all />
        </Suspense>
        <EffectComposer>
          <Bloom
            intensity={bloomIntensity * 0.3}
            luminanceThreshold={0.8}
            luminanceSmoothing={0.9}
            mipmapBlur
          />
          <ChromaticAberration
            blendFunction={BlendFunction.NORMAL}
            offset={new THREE.Vector2(chromaticOffset, chromaticOffset)}
          />
          <Vignette
            eskil={false}
            offset={0.3}
            darkness={vignetteDarkness}
          />
        </EffectComposer>
      </Canvas>
      <LoadingGate />
      <SceneHUD />
      <FilmGrainOverlay />
    </div>
  )
}
