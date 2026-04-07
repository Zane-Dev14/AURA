'use client'
import { useRef, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'
import gsap from 'gsap'

// Type for lighting preset
type LightingPreset = {
  ambient: { color: string; intensity: number }
  key: { position: [number, number, number]; color: string; intensity: number }
  fill: { position: [number, number, number]; color: string; intensity: number }
  rim: { position: [number, number, number]; color: string; intensity: number }
  hemisphere: { skyColor: string; groundColor: string; intensity: number }
  fog: { color: string; density: number }
  flicker?: boolean
  flickerIntensity?: number
  pulse?: boolean
  pulseSpeed?: number
}

// Premium lighting presets for each scene
const lightingPresets: Record<string, LightingPreset> = {
  calm: {
    ambient: { color: '#1a2332', intensity: 0.3 },
    key: { position: [10, 15, 10], color: '#4a9eff', intensity: 2 },
    fill: { position: [-10, 8, -5], color: '#2a4a7f', intensity: 0.8 },
    rim: { position: [0, 5, -15], color: '#00e5ff', intensity: 1.2 },
    hemisphere: { skyColor: '#1a3a5f', groundColor: '#0a0f1a', intensity: 0.5 },
    fog: { color: '#010308', density: 0.015 }
  },
  system: {
    ambient: { color: '#1a2840', intensity: 0.4 },
    key: { position: [0, 20, 15], color: '#5599ff', intensity: 2.5 },
    fill: { position: [-15, 10, -10], color: '#2255aa', intensity: 1 },
    rim: { position: [15, 8, -8], color: '#00ccff', intensity: 1.5 },
    hemisphere: { skyColor: '#2a4a7f', groundColor: '#0f1a2a', intensity: 0.6 },
    fog: { color: '#020510', density: 0.012 }
  },
  traffic: {
    ambient: { color: '#331100', intensity: 0.2 },
    key: { position: [0, 10, 0], color: '#ff6600', intensity: 2.5 },
    fill: { position: [0, 5, -20], color: '#ff4400', intensity: 1.5 },
    rim: { position: [10, 10, 10], color: '#ff8800', intensity: 1 },
    hemisphere: { skyColor: '#442200', groundColor: '#110800', intensity: 0.4 },
    fog: { color: '#220800', density: 0.018 },
    flicker: true
  },
  failure: {
    ambient: { color: '#220000', intensity: 0.1 },
    key: { position: [0, 4, 0], color: '#ff2200', intensity: 3 },
    fill: { position: [0, 8, 4], color: '#440000', intensity: 1.2 },
    rim: { position: [-10, 5, -5], color: '#ff0000', intensity: 2 },
    hemisphere: { skyColor: '#330000', groundColor: '#110000', intensity: 0.3 },
    fog: { color: '#110000', density: 0.025 },
    flicker: true,
    flickerIntensity: 0.5
  },
  emotional: {
    ambient: { color: '#1a0a0a', intensity: 0.15 },
    key: { position: [-5, 6, 8], color: '#ff3344', intensity: 2 },
    fill: { position: [8, 4, -6], color: '#661122', intensity: 0.8 },
    rim: { position: [0, 10, -12], color: '#ff0044', intensity: 1.5 },
    hemisphere: { skyColor: '#2a1a1a', groundColor: '#0a0505', intensity: 0.4 },
    fog: { color: '#0a0202', density: 0.022 }
  },
  qmix: {
    ambient: { color: '#0a1a2a', intensity: 0.5 },
    key: { position: [0, 15, 0], color: '#00ffff', intensity: 3 },
    fill: { position: [-12, 8, -8], color: '#0088ff', intensity: 1.5 },
    rim: { position: [12, 8, -8], color: '#00ffaa', intensity: 2 },
    hemisphere: { skyColor: '#1a3a5a', groundColor: '#0a1520', intensity: 0.7 },
    fog: { color: '#020a15', density: 0.01 },
    pulse: true
  },
  transform: {
    ambient: { color: '#2a1a4a', intensity: 0.6 },
    key: { position: [0, 20, 0], color: '#aa44ff', intensity: 4 },
    fill: { position: [-15, 12, 10], color: '#4400ff', intensity: 2 },
    rim: { position: [15, 12, -10], color: '#ff00ff', intensity: 2.5 },
    hemisphere: { skyColor: '#3a2a5a', groundColor: '#1a0a2a', intensity: 0.8 },
    fog: { color: '#0a0515', density: 0.008 },
    pulse: true,
    pulseSpeed: 2
  },
  recovery: {
    ambient: { color: '#1a3a2a', intensity: 0.4 },
    key: { position: [12, 18, 12], color: '#44ff88', intensity: 2.5 },
    fill: { position: [-10, 10, -8], color: '#22aa66', intensity: 1.2 },
    rim: { position: [0, 8, -15], color: '#00ffaa', intensity: 1.8 },
    hemisphere: { skyColor: '#2a4a3a', groundColor: '#0f1a15', intensity: 0.6 },
    fog: { color: '#020a08', density: 0.012 }
  },
  comparison: {
    ambient: { color: '#1a2a3a', intensity: 0.45 },
    key: { position: [15, 15, 15], color: '#66aaff', intensity: 2.2 },
    fill: { position: [-12, 10, -10], color: '#3366aa', intensity: 1 },
    rim: { position: [0, 8, -18], color: '#00ddff', intensity: 1.4 },
    hemisphere: { skyColor: '#2a3a4a', groundColor: '#0f1520', intensity: 0.55 },
    fog: { color: '#020810', density: 0.013 }
  }
}

type SceneType = keyof typeof lightingPresets

export default function LightingRig() {
  const { scene, frozen } = useSceneStore()
  
  // Light refs
  const ambientRef = useRef<THREE.AmbientLight>(null)
  const keyRef = useRef<THREE.DirectionalLight>(null)
  const fillRef = useRef<THREE.DirectionalLight>(null)
  const rimRef = useRef<THREE.PointLight>(null)
  const hemisphereRef = useRef<THREE.HemisphereLight>(null)
  
  // Animation state
  const flickerPhase = useRef(0)
  const pulsePhase = useRef(0)
  const currentPreset = useRef<SceneType>('calm')

  // Smooth transition to new lighting preset
  useEffect(() => {
    const preset = lightingPresets[scene as SceneType] || lightingPresets.calm
    currentPreset.current = scene as SceneType

    // Ambient light transition
    if (ambientRef.current) {
      gsap.to(ambientRef.current.color, {
        r: new THREE.Color(preset.ambient.color).r,
        g: new THREE.Color(preset.ambient.color).g,
        b: new THREE.Color(preset.ambient.color).b,
        duration: 2,
        ease: 'power2.inOut'
      })
      gsap.to(ambientRef.current, {
        intensity: preset.ambient.intensity,
        duration: 2,
        ease: 'power2.inOut'
      })
    }

    // Key light transition
    if (keyRef.current) {
      gsap.to(keyRef.current.position, {
        x: preset.key.position[0],
        y: preset.key.position[1],
        z: preset.key.position[2],
        duration: 2,
        ease: 'power2.inOut'
      })
      gsap.to(keyRef.current.color, {
        r: new THREE.Color(preset.key.color).r,
        g: new THREE.Color(preset.key.color).g,
        b: new THREE.Color(preset.key.color).b,
        duration: 2,
        ease: 'power2.inOut'
      })
      gsap.to(keyRef.current, {
        intensity: preset.key.intensity,
        duration: 2,
        ease: 'power2.inOut'
      })
    }

    // Fill light transition
    if (fillRef.current) {
      gsap.to(fillRef.current.position, {
        x: preset.fill.position[0],
        y: preset.fill.position[1],
        z: preset.fill.position[2],
        duration: 2,
        ease: 'power2.inOut'
      })
      gsap.to(fillRef.current.color, {
        r: new THREE.Color(preset.fill.color).r,
        g: new THREE.Color(preset.fill.color).g,
        b: new THREE.Color(preset.fill.color).b,
        duration: 2,
        ease: 'power2.inOut'
      })
      gsap.to(fillRef.current, {
        intensity: preset.fill.intensity,
        duration: 2,
        ease: 'power2.inOut'
      })
    }

    // Rim light transition
    if (rimRef.current) {
      gsap.to(rimRef.current.position, {
        x: preset.rim.position[0],
        y: preset.rim.position[1],
        z: preset.rim.position[2],
        duration: 2,
        ease: 'power2.inOut'
      })
      gsap.to(rimRef.current.color, {
        r: new THREE.Color(preset.rim.color).r,
        g: new THREE.Color(preset.rim.color).g,
        b: new THREE.Color(preset.rim.color).b,
        duration: 2,
        ease: 'power2.inOut'
      })
      gsap.to(rimRef.current, {
        intensity: preset.rim.intensity,
        duration: 2,
        ease: 'power2.inOut'
      })
    }

    // Hemisphere light transition
    if (hemisphereRef.current) {
      gsap.to(hemisphereRef.current.color, {
        r: new THREE.Color(preset.hemisphere.skyColor).r,
        g: new THREE.Color(preset.hemisphere.skyColor).g,
        b: new THREE.Color(preset.hemisphere.skyColor).b,
        duration: 2,
        ease: 'power2.inOut'
      })
      gsap.to(hemisphereRef.current.groundColor, {
        r: new THREE.Color(preset.hemisphere.groundColor).r,
        g: new THREE.Color(preset.hemisphere.groundColor).g,
        b: new THREE.Color(preset.hemisphere.groundColor).b,
        duration: 2,
        ease: 'power2.inOut'
      })
      gsap.to(hemisphereRef.current, {
        intensity: preset.hemisphere.intensity,
        duration: 2,
        ease: 'power2.inOut'
      })
    }
  }, [scene])

  // Dynamic lighting effects per frame
  useFrame((state, delta) => {
    if (frozen) return
    
    const preset = lightingPresets[currentPreset.current]
    const t = state.clock.elapsedTime

    // Flicker effect for dramatic scenes
    if (preset.flicker && keyRef.current) {
      flickerPhase.current += delta * 10
      const flickerAmount = preset.flickerIntensity || 0.3
      const flicker = Math.sin(flickerPhase.current * 3) * 
                      Math.sin(flickerPhase.current * 7) * 
                      flickerAmount
      keyRef.current.intensity = preset.key.intensity * (1 + flicker)
      
      // Occasional complete flicker
      if (Math.random() > 0.98) {
        keyRef.current.intensity *= 0.3
      }
    }

    // Pulse effect for energy scenes
    if (preset.pulse && rimRef.current) {
      pulsePhase.current += delta * (preset.pulseSpeed || 1)
      const pulse = Math.sin(pulsePhase.current * 2) * 0.3 + 1
      rimRef.current.intensity = preset.rim.intensity * pulse
    }

    // Subtle key light movement for organic feel
    if (keyRef.current && !preset.flicker) {
      const drift = Math.sin(t * 0.3) * 0.5
      keyRef.current.position.x = preset.key.position[0] + drift
    }

    // Rim light subtle rotation
    if (rimRef.current && !preset.pulse) {
      const angle = t * 0.2
      const radius = 15
      rimRef.current.position.x = Math.cos(angle) * radius
      rimRef.current.position.z = Math.sin(angle) * radius
    }
  })

  const initialPreset = lightingPresets[scene as SceneType] || lightingPresets.calm

  return (
    <>
      {/* Ambient base illumination */}
      <ambientLight
        ref={ambientRef}
        color={initialPreset.ambient.color}
        intensity={initialPreset.ambient.intensity}
      />

      {/* Hemisphere light for ambient occlusion feel */}
      <hemisphereLight
        ref={hemisphereRef}
        color={initialPreset.hemisphere.skyColor}
        groundColor={initialPreset.hemisphere.groundColor}
        intensity={initialPreset.hemisphere.intensity}
        position={[0, 50, 0]}
      />

      {/* Key light - main directional light with shadows */}
      <directionalLight
        ref={keyRef}
        position={initialPreset.key.position}
        color={initialPreset.key.color}
        intensity={initialPreset.key.intensity}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-far={100}
        shadow-camera-left={-30}
        shadow-camera-right={30}
        shadow-camera-top={30}
        shadow-camera-bottom={-30}
        shadow-bias={-0.0001}
      />

      {/* Fill light - softer secondary light */}
      <directionalLight
        ref={fillRef}
        position={initialPreset.fill.position}
        color={initialPreset.fill.color}
        intensity={initialPreset.fill.intensity}
      />

      {/* Rim light - backlight for depth and separation */}
      <pointLight
        ref={rimRef}
        position={initialPreset.rim.position}
        color={initialPreset.rim.color}
        intensity={initialPreset.rim.intensity}
        distance={50}
        decay={2}
      />
    </>
  )
}

// Made with Bob
