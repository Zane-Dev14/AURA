'use client'
import { useEffect, useRef, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import ServiceBeam from '@/components/r3f/ServiceBeam'
import { Text } from '@react-three/drei'
import * as THREE from 'three'
import { playGlitchBurst } from '@/lib/sound'
import gsap from 'gsap'
import { cameraShake } from '@/lib/animations'
import { createPodExplosion } from '@/lib/sceneAnimations'

// Exploding pod with GSAP animation
function ExplodingPod({ position, delay }: { position: [number, number, number]; delay: number }) {
  const groupRef = useRef<THREE.Group>(null)
  const meshRef = useRef<THREE.Mesh>(null)
  const [exploded, setExploded] = useState(false)
  const fragments = useRef<THREE.Vector3[]>([])
  const velocities = useRef<THREE.Vector3[]>([])
  const explosionTimeline = useRef<gsap.core.Timeline | null>(null)
  
  useEffect(() => {
    // Create fragment positions
    for (let i = 0; i < 8; i++) {
      fragments.current.push(new THREE.Vector3(0, 0, 0))
      velocities.current.push(
        new THREE.Vector3(
          (Math.random() - 0.5) * 3,
          Math.random() * 2 + 1,
          (Math.random() - 0.5) * 3
        )
      )
    }
    
    // Create explosion timeline with GSAP
    const tl = gsap.timeline({ delay })
    
    if (meshRef.current) {
      // Anticipation - squash down
      tl.to(meshRef.current.scale, {
        x: 0.7,
        y: 0.7,
        z: 0.7,
        duration: 0.15,
        ease: 'power2.in'
      })
      
      // Explosion trigger
      tl.call(() => setExploded(true))
    }
    
    explosionTimeline.current = tl
    
    return () => {
      tl.kill()
    }
  }, [delay])
  
  useFrame((state, delta) => {
    if (!groupRef.current || !exploded) return
    
    // Update fragment positions with physics
    for (let i = 0; i < fragments.current.length; i++) {
      fragments.current[i].add(velocities.current[i].clone().multiplyScalar(delta))
      velocities.current[i].y -= delta * 5 // gravity
    }
  })
  
  if (!exploded) {
    return (
      <mesh ref={meshRef} position={position}>
        <boxGeometry args={[0.8, 0.8, 0.8]} />
        <meshStandardMaterial
          color="#ff0000"
          emissive="#ff0000"
          emissiveIntensity={1 + Math.sin(Date.now() * 0.01) * 0.5}
        />
      </mesh>
    )
  }
  
  return (
    <group ref={groupRef} position={position}>
      {fragments.current.map((frag, i) => (
        <mesh key={i} position={frag}>
          <boxGeometry args={[0.3, 0.3, 0.3]} />
          <meshStandardMaterial
            color="#ff4400"
            emissive="#ff4400"
            emissiveIntensity={0.8}
            transparent
            opacity={Math.max(0, 1 - frag.length() * 0.1)}
          />
        </mesh>
      ))}
      {/* Explosion flash */}
      <mesh>
        <sphereGeometry args={[2, 16, 16]} />
        <meshBasicMaterial
          color="#ff6600"
          transparent
          opacity={0.3}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  )
}

// Glitch particles that spawn and disappear
function GlitchParticles() {
  const pointsRef = useRef<THREE.Points>(null)
  const positions = useRef(new Float32Array(500 * 3))
  const lifetimes = useRef(new Float32Array(500))
  
  useEffect(() => {
    // Initialize particles
    for (let i = 0; i < 500; i++) {
      positions.current[i * 3] = (Math.random() - 0.5) * 30
      positions.current[i * 3 + 1] = Math.random() * 15 - 5
      positions.current[i * 3 + 2] = (Math.random() - 0.5) * 30
      lifetimes.current[i] = Math.random()
    }
  }, [])
  
  useFrame((state, delta) => {
    if (!pointsRef.current) return
    const pos = pointsRef.current.geometry.attributes.position.array as Float32Array
    
    for (let i = 0; i < 500; i++) {
      const idx = i * 3
      
      // Update lifetime
      lifetimes.current[i] -= delta * 0.5
      
      // Respawn if dead
      if (lifetimes.current[i] <= 0) {
        pos[idx] = (Math.random() - 0.5) * 30
        pos[idx + 1] = Math.random() * 15 - 5
        pos[idx + 2] = (Math.random() - 0.5) * 30
        lifetimes.current[i] = Math.random() * 2
      } else {
        // Glitch movement
        pos[idx] += (Math.random() - 0.5) * 0.5
        pos[idx + 1] += (Math.random() - 0.5) * 0.5
        pos[idx + 2] += (Math.random() - 0.5) * 0.5
      }
    }
    
    pointsRef.current.geometry.attributes.position.needsUpdate = true
  })
  
  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions.current, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.2}
        color="#ff0000"
        transparent
        opacity={0.6}
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

// Falling debris
function FallingDebris() {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const dummy = useRef(new THREE.Object3D())
  const velocities = useRef<THREE.Vector3[]>([])
  
  useEffect(() => {
    for (let i = 0; i < 30; i++) {
      velocities.current.push(
        new THREE.Vector3(
          (Math.random() - 0.5) * 2,
          -Math.random() * 3 - 1,
          (Math.random() - 0.5) * 2
        )
      )
    }
  }, [])
  
  useFrame((state, delta) => {
    if (!meshRef.current) return
    
    for (let i = 0; i < 30; i++) {
      dummy.current.position.x = (i % 6 - 2.5) * 3 + velocities.current[i].x * state.clock.elapsedTime
      dummy.current.position.y = 10 + velocities.current[i].y * state.clock.elapsedTime
      dummy.current.position.z = (Math.floor(i / 6) - 2.5) * 3 + velocities.current[i].z * state.clock.elapsedTime
      
      // Reset if too low
      if (dummy.current.position.y < -5) {
        dummy.current.position.y = 10
      }
      
      dummy.current.rotation.x += delta * 2
      dummy.current.rotation.y += delta * 3
      dummy.current.updateMatrix()
      meshRef.current.setMatrixAt(i, dummy.current.matrix)
    }
    
    meshRef.current.instanceMatrix.needsUpdate = true
  })
  
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, 30]}>
      <boxGeometry args={[0.5, 0.5, 0.5]} />
      <meshStandardMaterial
        color="#440000"
        emissive="#ff0000"
        emissiveIntensity={0.5}
      />
    </instancedMesh>
  )
}

export default function FailureScene() {
  const { camera } = useThree()
  const {
    setGlitchIntensity, setPodHealth, setPodCount,
    setAirshipState, audioUnlocked, setMetrics, setScene,
  } = useSceneStore()
  const transitionTriggered = useRef(false)
  const masterTimeline = useRef<gsap.core.Timeline | null>(null)

  useEffect(() => {
    setPodCount(2)
    setPodHealth(0.1)
    setAirshipState('falling')
    if (audioUnlocked) playGlitchBurst()
    setMetrics({ failures: 47, latencyMs: 4200, cpuPercent: 116 })
    
    // Create dramatic failure sequence with GSAP
    const tl = gsap.timeline()
    
    // Brief pause for anticipation
    tl.to({}, { duration: 0.3 })
    
    // Camera shake sequence
    tl.add(cameraShake(camera, 0.8, 2), 0.3)
    
    // Continuous rumble
    const rumbleObj = { intensity: 0 }
    tl.to(rumbleObj, {
      intensity: 1,
      duration: 3,
      ease: 'power2.in',
      onUpdate: () => {
        camera.position.x += (Math.random() - 0.5) * 0.1 * rumbleObj.intensity
        camera.position.y += (Math.random() - 0.5) * 0.05 * rumbleObj.intensity
      }
    }, 0.5)
    
    // Glitch intensity ramps up with steps for glitchy feel
    const glitchObj = { value: 0 }
    tl.to(glitchObj, {
      value: 0.3,
      duration: 0.8,
      ease: 'steps(5)',
      onUpdate: () => setGlitchIntensity(glitchObj.value)
    }, 0.5)
    
    tl.to(glitchObj, {
      value: 0.7,
      duration: 1.2,
      ease: 'steps(8)',
      onUpdate: () => setGlitchIntensity(glitchObj.value)
    }, 1.3)
    
    tl.to(glitchObj, {
      value: 1,
      duration: 1.5,
      ease: 'steps(12)',
      onUpdate: () => setGlitchIntensity(glitchObj.value)
    }, 2.5)
    
    // Pod health crashes dramatically
    const healthObj = { value: 0.1 }
    tl.to(healthObj, {
      value: 0.05,
      duration: 2,
      ease: 'power4.in',
      onUpdate: () => setPodHealth(healthObj.value)
    }, 0.5)
    
    // Camera settles to final position
    tl.to(camera.position, {
      x: 0,
      y: 7,
      z: 16,
      duration: 2,
      ease: 'power2.out'
    }, 4)
    
    masterTimeline.current = tl
    
    // Transition to next scene
    const transitionTimer = setTimeout(() => {
      if (!transitionTriggered.current) {
        transitionTriggered.current = true
        setScene('emotional')
      }
    }, 12000)
    
    return () => {
      clearTimeout(transitionTimer)
      tl.kill()
    }
  }, [])

  useFrame(() => {
    // Keep camera looking at failure point
    camera.lookAt(0, -1, 0)
  })

  return (
    <>
      <ambientLight intensity={0.1} color="#220000" />
      <pointLight position={[0, 4, 0]} intensity={2} color="#ff2200" />
      <pointLight position={[0, 8, 4]} intensity={1.2} color="#440000" />
      
      {/* Flickering lights */}
      <pointLight 
        position={[-10, 5, -5]} 
        intensity={Math.random() * 2} 
        color="#ff0000" 
      />
      <pointLight 
        position={[10, 5, -5]} 
        intensity={Math.random() * 2} 
        color="#ff0000" 
      />
      
      {/* Warning text */}
      <Text
        position={[0, 8, -5]}
        fontSize={0.8}
        color="#ff0000"
        anchorX="center"
        anchorY="middle"
      >
        SYSTEM FAILURE
      </Text>
      
      {/* Exploding pods at different times */}
      <ExplodingPod position={[-3, 2, -5]} delay={1} />
      <ExplodingPod position={[3, 3, -8]} delay={2} />
      <ExplodingPod position={[-5, 1, -10]} delay={3} />
      <ExplodingPod position={[5, 4, -12]} delay={4} />
      <ExplodingPod position={[0, 2, -15]} delay={5} />
      
      {/* Glitch particles */}
      <GlitchParticles />
      
      {/* Falling debris */}
      <FallingDebris />
      
      {/* Cracked ground effect */}
      <mesh position={[0, -1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[40, 40, 20, 20]} />
        <meshStandardMaterial
          color="#220000"
          emissive="#ff0000"
          emissiveIntensity={0.2}
          wireframe
        />
      </mesh>
      
      <PodGrid />
      <ServiceBeam />
      <Airship />
    </>
  )
}

// Made with Bob
