'use client'
import { Canvas } from '@react-three/fiber'
import { Environment, PerformanceMonitor, Preload, Stars } from '@react-three/drei'
import { Suspense, useState, useEffect, useRef, lazy } from 'react'
import { EffectComposer, Bloom, ChromaticAberration, Vignette } from '@react-three/postprocessing'
import { BlendFunction } from 'postprocessing'
import * as THREE from 'three'
import { useSceneStore } from '@/store/useSceneStore'
import LoadingGate from './LoadingGate'
import FilmGrainOverlay from './ui/FilmGrainOverlay'
import InteractionHints from './ui/InteractionHints'
import { unlockAudio } from '@/lib/sound'

// Lazy load heavy components for better initial performance
const SceneManager = lazy(() => import('./SceneManager'))
const SceneHUD = lazy(() => import('./ui/SceneHUD'))
const LightingRig = lazy(() => import('./r3f/LightingRig'))
const SceneTransitionManager = lazy(() => import('./r3f/SceneTransitionManager'))
const IntroSequence = lazy(() => import('./r3f/IntroSequence'))

// Fog presets per scene for atmospheric depth
const fogPresets: Record<string, { color: string; density: number }> = {
  calm: { color: '#010308', density: 0.015 },
  system: { color: '#020510', density: 0.012 },
  traffic: { color: '#220800', density: 0.018 },
  failure: { color: '#110000', density: 0.025 },
  emotional: { color: '#0a0202', density: 0.022 },
  qmix: { color: '#020a15', density: 0.01 },
  transform: { color: '#0a0515', density: 0.008 },
  recovery: { color: '#020a08', density: 0.012 },
  comparison: { color: '#020810', density: 0.013 }
}

function FogController() {
  const { scene } = useSceneStore()
  const fogRef = useRef<THREE.FogExp2>(null)

  useEffect(() => {
    if (fogRef.current) {
      const preset = fogPresets[scene] || fogPresets.calm
      const targetColor = new THREE.Color(preset.color)
      
      // Smooth color transition
      const currentColor = fogRef.current.color
      currentColor.lerp(targetColor, 0.1)
      
      // Smooth density transition
      fogRef.current.density = THREE.MathUtils.lerp(
        fogRef.current.density,
        preset.density,
        0.05
      )
    }
  }, [scene])

  const initialPreset = fogPresets[useSceneStore.getState().scene] || fogPresets.calm

  return <fogExp2 ref={fogRef} attach="fog" args={[initialPreset.color, initialPreset.density]} />
}

export default function AuraDemo() {
  const [dpr, setDpr] = useState(1.5)
  const { glitchIntensity, scene, unlockAudio: storeUnlockAudio } = useSceneStore()

  const handleFirstInteraction = () => {
    unlockAudio()
    storeUnlockAudio()
  }

  // Enhanced bloom for dramatic glow effects
  const bloomIntensity =
    scene === 'transform' ? 8 :
    scene === 'qmix' ? 4 :
    scene === 'recovery' ? 3 :
    scene === 'failure' ? 2 :
    1.2
  
  const chromaticOffset = glitchIntensity * 0.005

  // Vignette darkness varies per scene mood
  const vignetteDarkness =
    scene === 'failure' || scene === 'emotional' ? 0.85
    : scene === 'transform' ? 0.5
    : scene === 'qmix' ? 0.6
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
          toneMapping: THREE.ACESFilmicToneMapping,
          // Performance optimizations
          alpha: false,
          stencil: false,
          depth: true,
        }}
        shadows="soft"
        camera={{ position: [0, 8, 20], fov: 55, near: 0.1, far: 200 }}
        style={{ background: '#000' }}
        frameloop="always" // Ensure smooth rendering
      >
        <FogController />
        <color attach="background" args={['#000205']} />
        
        {/* Performance monitoring with adaptive quality */}
        <PerformanceMonitor
          onDecline={() => setDpr(1)}
          onIncline={() => setDpr(1.5)}
          flipflops={3}
          factor={0.5}
        />
        
        {/* Suspense with proper fallback for smooth loading */}
        <Suspense fallback={null}>
          {/* Preload all assets FIRST for smooth intro transition */}
          <Preload all />
          
          <LightingRig />
          <Environment
            files={
              scene === 'recovery' || scene === 'comparison'
                ? '/models/spruit_sunrise_4k.exr'
                : '/models/1082ab60-0925-4509-9e69-90a7dfce573c.hdr'
            }
            background
            backgroundBlurriness={0}
            backgroundIntensity={0.3}
            environmentIntensity={2.0}
          />
          
          {/* Optimized star count for better performance */}
          <Stars radius={100} depth={80} count={3000} factor={4} fade speed={0.3} />
          
          <IntroSequence />
          <SceneManager />
          <SceneTransitionManager />
        </Suspense>
        
        {/* Post-processing with optimized settings */}
        <EffectComposer multisampling={4}>
          <Bloom
            intensity={bloomIntensity * 0.4}
            luminanceThreshold={0.7}
            luminanceSmoothing={0.9}
            mipmapBlur
            radius={0.8}
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
      
      {/* Lazy load UI components */}
      <Suspense fallback={null}>
        <SceneHUD />
      </Suspense>
      
      <FilmGrainOverlay />
      <InteractionHints />
    </div>
  )
}
