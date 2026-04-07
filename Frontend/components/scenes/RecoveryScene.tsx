'use client'
import { useEffect, useRef, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import ServiceBeam from '@/components/r3f/ServiceBeam'
import { Text } from '@react-three/drei'
import * as THREE from 'three'
import { playRecoveryChime } from '@/lib/sound'

// Healing pod that grows and repairs
function HealingPod({ 
  position, 
  delay 
}: { 
  position: [number, number, number]
  delay: number
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [healing, setHealing] = useState(false)
  const [health, setHealth] = useState(0)
  
  useEffect(() => {
    setTimeout(() => setHealing(true), delay * 1000)
  }, [delay])
  
  useFrame((state, delta) => {
    if (!meshRef.current) return
    const t = state.clock.elapsedTime
    
    if (healing && health < 1) {
      setHealth(h => Math.min(1, h + delta * 0.3))
    }
    
    // Grow as health increases
    const scale = 0.3 + health * 0.5
    meshRef.current.scale.setScalar(scale)
    
    // Float
    meshRef.current.position.y = position[1] + Math.sin(t * 2 + delay) * 0.2
    
    // Rotate
    meshRef.current.rotation.y = t * 0.5
  })
  
  return (
    <mesh ref={meshRef} position={position}>
      <boxGeometry args={[0.8, 0.8, 0.8]} />
      <meshStandardMaterial
        color={health > 0.8 ? '#00ff88' : health > 0.4 ? '#88ff00' : '#ffff00'}
        emissive={health > 0.8 ? '#00ff88' : health > 0.4 ? '#88ff00' : '#ffff00'}
        emissiveIntensity={0.5 + health * 0.5}
        metalness={0.8}
        roughness={0.2}
      />
      
      {/* Healing particles */}
      {healing && health < 1 && (
        <>
          {Array.from({ length: 8 }).map((_, i) => {
            const angle = (i / 8) * Math.PI * 2
            const radius = 1.5
            return (
              <mesh
                key={i}
                position={[
                  Math.cos(angle) * radius,
                  Math.sin(angle * 2) * 0.3,
                  Math.sin(angle) * radius
                ]}
              >
                <sphereGeometry args={[0.08, 8, 8]} />
                <meshBasicMaterial
                  color="#00ff88"
                  transparent
                  opacity={0.8}
                />
              </mesh>
            )
          })}
        </>
      )}
      
      {/* Health bar above pod */}
      {healing && health < 1 && (
        <mesh position={[0, 1.2, 0]}>
          <boxGeometry args={[1 * health, 0.1, 0.1]} />
          <meshBasicMaterial color="#00ff88" />
        </mesh>
      )}
    </mesh>
  )
}

// Repair beam that user can guide
function RepairBeam({ 
  active, 
  targetPos 
}: { 
  active: boolean
  targetPos: [number, number, number]
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  useFrame((state) => {
    if (!meshRef.current || !active) return
    const t = state.clock.elapsedTime
    
    // Pulse
    const pulse = 1 + Math.sin(t * 5) * 0.2
    meshRef.current.scale.set(pulse, 1, pulse)
  })
  
  if (!active) return null
  
  return (
    <group>
      {/* Beam from top */}
      <mesh position={[targetPos[0], 8, targetPos[2]]}>
        <cylinderGeometry args={[0.3, 0.5, 16, 16]} />
        <meshBasicMaterial
          color="#00ff88"
          transparent
          opacity={0.6}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      
      {/* Impact point */}
      <mesh ref={meshRef} position={targetPos}>
        <sphereGeometry args={[0.5, 16, 16]} />
        <meshBasicMaterial
          color="#00ff88"
          transparent
          opacity={0.8}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  )
}

// Healing particles flowing upward
function HealingParticles() {
  const pointsRef = useRef<THREE.Points>(null)
  const positions = useRef(new Float32Array(400 * 3))
  
  useEffect(() => {
    for (let i = 0; i < 400; i++) {
      positions.current[i * 3] = (Math.random() - 0.5) * 30
      positions.current[i * 3 + 1] = Math.random() * 15 - 5
      positions.current[i * 3 + 2] = (Math.random() - 0.5) * 30
    }
  }, [])
  
  useFrame((state, delta) => {
    if (!pointsRef.current) return
    const pos = pointsRef.current.geometry.attributes.position.array as Float32Array
    
    for (let i = 0; i < 400; i++) {
      const idx = i * 3
      
      // Float upward
      pos[idx + 1] += delta * 2
      
      // Spiral
      const angle = state.clock.elapsedTime + i * 0.1
      pos[idx] += Math.cos(angle) * delta * 0.5
      pos[idx + 2] += Math.sin(angle) * delta * 0.5
      
      // Reset if too high
      if (pos[idx + 1] > 15) {
        pos[idx] = (Math.random() - 0.5) * 30
        pos[idx + 1] = -5
        pos[idx + 2] = (Math.random() - 0.5) * 30
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
        size={0.12}
        color="#00ff88"
        transparent
        opacity={0.6}
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

// Energy waves expanding from center
function EnergyWaves() {
  const meshRefs = useRef<THREE.Mesh[]>([])
  
  useFrame((state) => {
    const t = state.clock.elapsedTime
    
    meshRefs.current.forEach((mesh, i) => {
      if (!mesh) return
      const offset = i * 0.5
      const scale = 1 + ((t + offset) % 3) * 2
      mesh.scale.setScalar(scale)
      const mat = mesh.material as THREE.MeshBasicMaterial
      if (mat) mat.opacity = 1 - ((t + offset) % 3) / 3
    })
  })
  
  return (
    <>
      {[0, 1, 2].map((i) => (
        <mesh
          key={i}
          ref={(el) => { if (el) meshRefs.current[i] = el }}
          position={[0, 0, 0]}
          rotation={[-Math.PI / 2, 0, 0]}
        >
          <ringGeometry args={[2, 2.5, 32]} />
          <meshBasicMaterial
            color="#00ff88"
            transparent
            opacity={0.5}
            blending={THREE.AdditiveBlending}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </>
  )
}

export default function RecoveryScene() {
  const { camera } = useThree()
  const {
    setPodCount, setPodHealth, setAirshipState,
    setGlitchIntensity, setTrafficLevel, setMetrics, audioUnlocked, setScene,
  } = useSceneStore()
  const [phase, setPhase] = useState(0)
  const transitionTriggered = useRef(false)

  useEffect(() => {
    setPodCount(9)
    setPodHealth(1)
    setAirshipState('stable')
    setGlitchIntensity(0)
    setTrafficLevel(0.3)
    setMetrics({ failures: 0, pods: 9, latencyMs: 5, cpuPercent: 66, rps: 430 })
    if (audioUnlocked) playRecoveryChime()
    
    // Progress through phases
    setTimeout(() => setPhase(1), 3000)
    setTimeout(() => setPhase(2), 6000)
    setTimeout(() => setPhase(3), 9000)
  }, [])

  useFrame((_, delta) => {
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, 0, 0.02)
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, 8, 0.02)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, 18, 0.02)
    camera.lookAt(0, 0, 0)
    
    // Auto-transition after 15 seconds
    if (phase >= 3 && !transitionTriggered.current) {
      transitionTriggered.current = true
      setTimeout(() => {
        setScene('comparison')
      }, 5000)
    }
  })

  return (
    <>
      <ambientLight intensity={0.4} color="#002211" />
      <pointLight position={[0, 10, 0]} intensity={4} color="#00ff88" />
      <pointLight position={[0, -2, 8]} intensity={2} color="#00aaff" />
      <directionalLight position={[10, 10, 5]} intensity={1.5} color="#aaffdd" />
      
      {/* Healing spotlight */}
      <spotLight
        position={[0, 20, 0]}
        angle={0.6}
        penumbra={0.5}
        intensity={3}
        color="#00ff88"
        target-position={[0, 0, 0]}
      />
      
      {/* Status text */}
      <Text
        position={[0, 9, 0]}
        fontSize={0.6}
        color="#00ff88"
        anchorX="center"
        anchorY="middle"
      >
        {phase === 0 && 'System Recovery Initiated'}
        {phase === 1 && 'Healing Pods...'}
        {phase === 2 && 'Restoring Services...'}
        {phase === 3 && 'Recovery Complete!'}
      </Text>
      
      {/* Healing pods appearing in sequence */}
      <HealingPod position={[-4, 2, -5]} delay={0.5} />
      <HealingPod position={[4, 2, -5]} delay={1} />
      <HealingPod position={[-6, 3, -8]} delay={1.5} />
      <HealingPod position={[6, 3, -8]} delay={2} />
      <HealingPod position={[0, 4, -10]} delay={2.5} />
      <HealingPod position={[-8, 2, -12]} delay={3} />
      <HealingPod position={[8, 2, -12]} delay={3.5} />
      <HealingPod position={[-4, 5, -15]} delay={4} />
      <HealingPod position={[4, 5, -15]} delay={4.5} />
      
      {/* Repair beams */}
      <RepairBeam active={phase >= 1} targetPos={[-4, 2, -5]} />
      <RepairBeam active={phase >= 2} targetPos={[4, 2, -5]} />
      <RepairBeam active={phase >= 3} targetPos={[0, 4, -10]} />
      
      {/* Healing particles */}
      <HealingParticles />
      
      {/* Energy waves */}
      <EnergyWaves />
      
      {/* Restored grid floor */}
      <mesh position={[0, -1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[40, 40, 20, 20]} />
        <meshBasicMaterial
          color="#00ff88"
          transparent
          opacity={0.15}
          wireframe
        />
      </mesh>
      
      {/* Glowing platform */}
      <mesh position={[0, -0.5, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[12, 64]} />
        <meshStandardMaterial
          color="#002211"
          emissive="#00ff88"
          emissiveIntensity={0.3}
          transparent
          opacity={0.5}
        />
      </mesh>
      
      <PodGrid />
      <ServiceBeam />
      <Airship />
    </>
  )
}

// Made with Bob
