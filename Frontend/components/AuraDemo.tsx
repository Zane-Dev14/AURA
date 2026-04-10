'use client'
import { Canvas } from '@react-three/fiber'
import { Environment, PerformanceMonitor, Preload, Stars } from '@react-three/drei'
import { useState, useEffect, useRef } from 'react'
import { EffectComposer, Bloom, ChromaticAberration, Vignette } from '@react-three/postprocessing'
import { BlendFunction } from 'postprocessing'
import * as THREE from 'three'
import { useSceneStore } from '@/store/useSceneStore'
import LoadingGate from './LoadingGate'
import FilmGrainOverlay from './ui/FilmGrainOverlay'
import InteractionHints from './ui/InteractionHints'
import StoryText from './ui/StoryText'
import { unlockAudio } from '@/lib/sound'
import type { Scene } from '@/store/useSceneStore'
import SceneManager from './SceneManager'
import SceneHUD from './ui/SceneHUD'
import LightingRig from './r3f/LightingRig'
import SceneTransitionManager from './r3f/SceneTransitionManager'
import IntroSequence from './r3f/IntroSequence'

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

const sceneAudioMap: Record<Scene, string> = {
  calm: '/audio/calmMusic.mp3',
  system: '/audio/kubernetesScene.mp3',
  traffic: '/audio/kubernetesScene.mp3',
  failure: '/audio/criticalState.mp3',
  emotional: '/audio/criticalState.mp3',
  qmix: '/audio/qmixReady.mp3',
  transform: '/audio/qmixReady.mp3',
  recovery: '/audio/successFinal.mp3',
  comparison: '/audio/successFinal.mp3',
}

function SceneAudioManager() {
  const scene = useSceneStore((state) => state.scene)
  const audioUnlocked = useSceneStore((state) => state.audioUnlocked)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    if (!audioRef.current) {
      const audio = new Audio()
      audio.loop = true
      audio.preload = 'auto'
      audio.volume = 0.42
      audioRef.current = audio
    }

    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current.src = ''
        audioRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    if (!audioUnlocked) {
      audio.pause()
      return
    }

    const nextSrc = sceneAudioMap[scene]
    if (!nextSrc) return

    if (!audio.src.endsWith(nextSrc)) {
      audio.src = nextSrc
      audio.currentTime = 0
    }

    const playPromise = audio.play()
    if (playPromise && typeof playPromise.catch === 'function') {
      playPromise.catch(() => {
        // Browser autoplay policy can still block before a valid interaction.
      })
    }
  }, [audioUnlocked, scene])

  return null
}

export default function AuraDemo() {
  const [dpr, setDpr] = useState(2) // M4 Max can handle 2x pixel ratio
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
        dpr={[1.5, dpr]}
        gl={{
          antialias: true,
          powerPreference: 'high-performance',
          toneMappingExposure: 1.3,
          toneMapping: THREE.ACESFilmicToneMapping,
          // M4 Max optimizations
          alpha: false,
          stencil: false,
          depth: true,
          logarithmicDepthBuffer: true, // Better depth precision
          preserveDrawingBuffer: false,
        }}
        shadows="percentage" // Higher quality shadows for M4 Max
        camera={{ position: [0, 8, 20], fov: 55, near: 0.1, far: 200 }}
        style={{ background: '#000' }}
        frameloop="always"
      >
        <FogController />
        <color attach="background" args={['#000205']} />
        
        {/* Performance monitoring optimized for M4 Max */}
        <PerformanceMonitor
          onDecline={() => setDpr(1.5)}
          onIncline={() => setDpr(2)}
          flipflops={3}
          factor={0.8}
        />
        
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
          backgroundBlurriness={0.1}
          backgroundIntensity={0.4}
          environmentIntensity={2.5}
          resolution={1024}
        />
        
        {/* Enhanced star count for M4 Max */}
        <Stars radius={100} depth={80} count={5000} factor={5} fade speed={0.3} />
        
        <IntroSequence />
        <SceneManager />
        <SceneTransitionManager />
        
        {/* Post-processing optimized for M4 Max - higher quality */}
        <EffectComposer multisampling={8} enabled={true}>
          <Bloom
            intensity={bloomIntensity * 0.5}
            luminanceThreshold={0.6}
            luminanceSmoothing={0.95}
            mipmapBlur
            radius={0.9}
            levels={8}
          />
          <ChromaticAberration
            blendFunction={BlendFunction.NORMAL}
            offset={new THREE.Vector2(chromaticOffset, chromaticOffset)}
            radialModulation={true}
            modulationOffset={0.5}
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
      
      <StoryText />
      <FilmGrainOverlay />
      <InteractionHints />
      <SceneAudioManager />
    </div>
  )
}
