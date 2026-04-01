'use client'
import { useEffect, useRef, useState } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import { Text } from '@react-three/drei'
import gsap from 'gsap'
import * as THREE from 'three'

/**
 * IntroSequence - CINEMATIC REVEAL
 * Award-winning style intro with dramatic camera work and particle effects
 */
export default function IntroSequence() {
  const { camera } = useThree()
  const { introComplete, setIntroComplete, assetsLoaded } = useSceneStore()
  const timelineRef = useRef<gsap.core.Timeline | null>(null)
  
  const [textOpacity, setTextOpacity] = useState(0)
  const [textScale, setTextScale] = useState(0.5)
  const [glowIntensity, setGlowIntensity] = useState(0)
  const [ringsOpacity, setRingsOpacity] = useState(0)
  const [particleIntensity, setParticleIntensity] = useState(0)
  
  // Cinematic particles
  const particlesRef = useRef<THREE.Points>(null)
  const particleCount = 2000
  
  useEffect(() => {
    if (!assetsLoaded || introComplete) return

    // Set dramatic camera start position
    camera.position.set(0, 10, 30)
    camera.lookAt(0, 0, 0)

    // Create cinematic timeline - 2.5 seconds
    const timeline = gsap.timeline({
      onComplete: () => {
        setTimeout(() => setIntroComplete(true), 100)
      }
    })

    timelineRef.current = timeline

    // Camera: Dramatic dolly in
    timeline.to(camera.position, {
      z: 15,
      y: 5,
      duration: 2.5,
      ease: 'power2.inOut'
    }, 0)

    // Text: Scale up with elastic bounce
    timeline.to({ scale: 0.5, opacity: 0 }, {
      scale: 1,
      opacity: 1,
      duration: 1.2,
      ease: 'elastic.out(1, 0.6)',
      onUpdate: function() {
        const target = this.targets()[0]
        setTextScale(target.scale)
        setTextOpacity(target.opacity)
      }
    }, 0.3)

    // Glow: Intense build-up
    timeline.to({ glow: 0 }, {
      glow: 1,
      duration: 1.5,
      ease: 'power3.out',
      onUpdate: function() {
        setGlowIntensity(this.targets()[0].glow)
      }
    }, 0.5)

    // Rings: Expand outward
    timeline.to({ rings: 0 }, {
      rings: 1,
      duration: 1.0,
      ease: 'power2.out',
      onUpdate: function() {
        setRingsOpacity(this.targets()[0].rings)
      }
    }, 0.8)

    // Particles: Burst effect
    timeline.to({ particles: 0 }, {
      particles: 1,
      duration: 1.2,
      ease: 'power2.out',
      onUpdate: function() {
        setParticleIntensity(this.targets()[0].particles)
      }
    }, 0.6)

    // Fade out
    timeline.to({ opacity: 1 }, {
      opacity: 0,
      duration: 0.4,
      ease: 'power2.in',
      onUpdate: function() {
        setTextOpacity(this.targets()[0].opacity)
        setRingsOpacity(this.targets()[0].opacity * 0.5)
      }
    }, 2.1)

    return () => {
      if (timelineRef.current) {
        timelineRef.current.kill()
      }
    }
  }, [assetsLoaded, introComplete, camera, setIntroComplete])

  // Animate particles
  useFrame((state) => {
    if (introComplete || !particlesRef.current) return
    
    const positions = particlesRef.current.geometry.attributes.position.array as Float32Array
    const time = state.clock.elapsedTime
    
    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3
      const x = positions[i3]
      const z = positions[i3 + 2]
      
      // Spiral outward motion
      positions[i3 + 1] += Math.sin(time + i * 0.1) * 0.02
    }
    
    particlesRef.current.geometry.attributes.position.needsUpdate = true
    particlesRef.current.rotation.y = time * 0.1
  })

  // Skip handler
  useEffect(() => {
    const handleSkip = (e: KeyboardEvent) => {
      if ((e.key === 'Enter' || e.key === ' ') && !introComplete && timelineRef.current) {
        timelineRef.current.progress(1)
        setIntroComplete(true)
      }
    }
    window.addEventListener('keydown', handleSkip)
    return () => window.removeEventListener('keydown', handleSkip)
  }, [introComplete, setIntroComplete])

  if (introComplete) return null

  // Generate particle positions
  const particlePositions = new Float32Array(particleCount * 3)
  for (let i = 0; i < particleCount; i++) {
    const i3 = i * 3
    const radius = 5 + Math.random() * 15
    const theta = Math.random() * Math.PI * 2
    const phi = Math.random() * Math.PI
    
    particlePositions[i3] = radius * Math.sin(phi) * Math.cos(theta)
    particlePositions[i3 + 1] = (Math.random() - 0.5) * 10
    particlePositions[i3 + 2] = radius * Math.sin(phi) * Math.sin(theta)
  }

  return (
    <>
      {/* Dramatic lighting */}
      <ambientLight intensity={0.1} color="#0a1428" />
      <pointLight position={[0, 0, 0]} intensity={glowIntensity * 15} color="#00e5ff" distance={50} />
      <spotLight
        position={[0, 20, 0]}
        angle={0.5}
        penumbra={0.5}
        intensity={glowIntensity * 8}
        color="#00d4ff"
        target-position={[0, 0, 0]}
      />

      {/* Main AURA text */}
      <group scale={textScale}>
        <Text
          fontSize={4}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
          letterSpacing={0.15}
          fillOpacity={textOpacity}
          outlineWidth={0.02}
          outlineColor="#00e5ff"
          outlineOpacity={textOpacity * 0.8}
        >
          AURA
        </Text>
        
        {/* Intense glow */}
        <pointLight
          position={[0, 0, 2]}
          intensity={glowIntensity * 20}
          color="#00e5ff"
          distance={30}
          decay={2}
        />
      </group>

      {/* Expanding energy rings */}
      {[1, 2, 3].map((ring, i) => (
        <mesh
          key={i}
          position={[0, 0, 0]}
          rotation={[Math.PI / 2, 0, 0]}
          scale={ringsOpacity * (1 + i * 0.5)}
        >
          <ringGeometry args={[3 + i * 2, 3.2 + i * 2, 64]} />
          <meshBasicMaterial
            color="#00e5ff"
            transparent
            opacity={ringsOpacity * (0.4 - i * 0.1)}
            side={THREE.DoubleSide}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      ))}

      {/* Particle burst */}
      <points ref={particlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={particleCount}
            array={particlePositions}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.08}
          color="#00e5ff"
          transparent
          opacity={particleIntensity * 0.6}
          blending={THREE.AdditiveBlending}
          sizeAttenuation
        />
      </points>

      {/* Atmospheric fog plane */}
      <mesh position={[0, -5, -10]} rotation={[-Math.PI / 4, 0, 0]}>
        <planeGeometry args={[80, 80]} />
        <meshBasicMaterial
          color="#001a3a"
          transparent
          opacity={0.2}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </>
  )
}