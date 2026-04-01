'use client'
import { useEffect, useRef, useState } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import { Text } from '@react-three/drei'
import gsap from 'gsap'
import * as THREE from 'three'

/**
 * IntroSequence - ELEGANT REVEAL
 *
 * Professional, smooth intro inspired by Lusion, Aristide Benoist, and Active Theory
 * - 2.5 seconds total duration
 * - Smooth fade-in with gentle camera movement
 * - Elegant typography with subtle animations
 * - Ambient particles for atmosphere
 * - No aggressive effects (no flash, shake, or slam)
 */
export default function IntroSequence() {
  const { camera, scene } = useThree()
  const { introComplete, setIntroComplete, assetsLoaded } = useSceneStore()
  const timelineRef = useRef<gsap.core.Timeline | null>(null)
  
  // Animation states
  const [fadeIn, setFadeIn] = useState(0) // Overall fade from black
  const [textOpacity, setTextOpacity] = useState(0)
  const [textScale, setTextScale] = useState(0.95)
  const [glowIntensity, setGlowIntensity] = useState(0)
  
  // Ambient particles for atmosphere
  const [particles, setParticles] = useState<Array<{
    id: number
    x: number
    y: number
    z: number
    vx: number
    vy: number
    opacity: number
  }>>([])
  
  const baseCameraPos = useRef(new THREE.Vector3(0, 5, 16))
  const targetCameraPos = useRef(new THREE.Vector3(0, 5, 15))

  // Initialize ambient particles
  useEffect(() => {
    if (!assetsLoaded || introComplete) return

    // Create subtle ambient particles
    const ambientParticles = []
    for (let i = 0; i < 80; i++) {
      ambientParticles.push({
        id: i,
        x: (Math.random() - 0.5) * 30,
        y: Math.random() * 15 - 2,
        z: (Math.random() - 0.5) * 20 - 5,
        vx: (Math.random() - 0.5) * 0.3,
        vy: Math.random() * 0.2 + 0.1,
        opacity: Math.random() * 0.4 + 0.1
      })
    }
    setParticles(ambientParticles)
  }, [assetsLoaded, introComplete])

  useEffect(() => {
    if (!assetsLoaded || introComplete) return

    // Set camera to starting position
    camera.position.copy(baseCameraPos.current)
    camera.lookAt(0, 0, 0)

    // Create elegant timeline - 2.5 seconds total
    const timeline = gsap.timeline({
      onComplete: () => {
        setTimeout(() => setIntroComplete(true), 200)
      }
    })

    timelineRef.current = timeline

    // PHASE 1: Fade In (0.0 - 0.8s)
    // Gentle fade from black with camera dolly
    timeline.to({ value: 0 }, {
      value: 1,
      duration: 0.8,
      ease: 'power2.out',
      onUpdate: function() {
        setFadeIn(this.targets()[0].value)
      }
    })

    // Camera: Slow dolly forward
    timeline.to(camera.position, {
      z: targetCameraPos.current.z,
      duration: 2.5,
      ease: 'power2.inOut'
    }, 0)

    // PHASE 2: Logo Reveal (0.8 - 1.8s)
    // Text fades in with subtle scale animation
    timeline.to({ opacity: 0, scale: 0.95 }, {
      opacity: 1,
      scale: 1.0,
      duration: 1.0,
      ease: 'expo.out',
      onUpdate: function() {
        const target = this.targets()[0]
        setTextOpacity(target.opacity)
        setTextScale(target.scale)
      }
    }, 0.8)

    // Glow intensity builds up
    timeline.to({ glow: 0 }, {
      glow: 1,
      duration: 1.0,
      ease: 'power2.out',
      onUpdate: function() {
        setGlowIntensity(this.targets()[0].glow)
      }
    }, 0.8)

    // PHASE 3: Hold & Transition (1.8 - 2.5s)
    // Hold for a moment, then gentle fade
    timeline.to({}, { duration: 0.4 }, 1.8)
    
    // Gentle fade out
    timeline.to({ opacity: 1 }, {
      opacity: 0,
      duration: 0.3,
      ease: 'power2.in',
      onUpdate: function() {
        setTextOpacity(this.targets()[0].opacity)
      }
    }, 2.2)

    return () => {
      if (timelineRef.current) {
        timelineRef.current.kill()
      }
    }
  }, [assetsLoaded, introComplete, camera, setIntroComplete])

  // Animate ambient particles
  useFrame((state, delta) => {
    if (introComplete) return

    // Gentle particle float
    setParticles(prev =>
      prev.map(p => {
        const newY = p.y + p.vy * delta
        return {
          ...p,
          x: p.x + p.vx * delta,
          y: newY > 15 ? -2 : newY
        }
      })
    )

    // Ensure camera looks at center
    camera.lookAt(0, 0, 0)
  })

  // Allow skip with Enter or Space
  useEffect(() => {
    const handleSkip = (e: KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        if (!introComplete && timelineRef.current) {
          timelineRef.current.progress(1)
          setIntroComplete(true)
        }
      }
    }

    window.addEventListener('keydown', handleSkip)
    return () => window.removeEventListener('keydown', handleSkip)
  }, [introComplete, setIntroComplete])

  if (introComplete) return null

  return (
    <>
      {/* Fade from black overlay */}
      <mesh position={[0, 0, -5]}>
        <planeGeometry args={[200, 200]} />
        <meshBasicMaterial
          color="#000000"
          transparent
          opacity={1 - fadeIn}
          depthWrite={false}
        />
      </mesh>

      {/* Soft ambient lighting */}
      <ambientLight intensity={0.15 * fadeIn} color="#4a90e2" />
      
      {/* Subtle directional light */}
      <directionalLight
        position={[5, 10, 5]}
        intensity={0.3 * fadeIn}
        color="#ffffff"
      />

      {/* AURA Text - Elegant reveal */}
      <group position={[0, 0, 0]} scale={textScale}>
        <Text
          fontSize={3.5}
          color="#00e5ff"
          anchorX="center"
          anchorY="middle"
          letterSpacing={0.1}
        >
          AURA
        </Text>
        
        {/* Subtle glow around text */}
        <pointLight
          position={[0, 0, 2]}
          intensity={glowIntensity * 8}
          color="#00e5ff"
          distance={15}
          decay={2}
        />
        
        {/* Soft rim light */}
        <pointLight
          position={[0, 0, -3]}
          intensity={glowIntensity * 3}
          color="#0088cc"
          distance={10}
          decay={2}
        />
      </group>

      {/* Text opacity control */}
      <mesh position={[0, 0, 0.1]}>
        <planeGeometry args={[20, 10]} />
        <meshBasicMaterial
          color="#000000"
          transparent
          opacity={(1 - textOpacity) * fadeIn}
          depthWrite={false}
        />
      </mesh>

      {/* Ambient particles - subtle atmosphere */}
      {particles.map(p => (
        <mesh key={p.id} position={[p.x, p.y, p.z]}>
          <sphereGeometry args={[0.08, 6, 6]} />
          <meshBasicMaterial
            color="#00e5ff"
            transparent
            opacity={p.opacity * fadeIn * 0.6}
          />
        </mesh>
      ))}

      {/* Subtle atmospheric fog */}
      <mesh position={[0, -3, -15]} rotation={[-Math.PI / 6, 0, 0]}>
        <planeGeometry args={[60, 40]} />
        <meshBasicMaterial
          color="#001a33"
          transparent
          opacity={0.4 * fadeIn}
        />
      </mesh>

      {/* Skip hint - subtle */}
      {fadeIn > 0.5 && (
        <group position={[0, -6, 0]}>
          <Text
            fontSize={0.4}
            color="#ffffff"
            anchorX="center"
            anchorY="middle"
            fillOpacity={0.3 * fadeIn}
          >
            Press SPACE or ENTER to skip
          </Text>
        </group>
      )}
    </>
  )
}

// Made with Bob - Elegant Intro Redesign