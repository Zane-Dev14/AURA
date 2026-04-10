'use client'
import { useEffect, useRef, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html, Text } from '@react-three/drei'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import HologramDialog from '@/components/ui/HologramDialog'
import * as THREE from 'three'

// Interactive activation button
function ActivationButton({ 
  position, 
  label, 
  activated, 
  onActivate 
}: { 
  position: [number, number, number]
  label: string
  activated: boolean
  onActivate: () => void
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)
  
  useFrame((state) => {
    if (!meshRef.current) return
    const t = state.clock.elapsedTime
    
    if (!activated) {
      // Pulse when not activated
      const pulse = 1 + Math.sin(t * 3) * 0.1
      meshRef.current.scale.setScalar(pulse)
      
      // Rotate
      meshRef.current.rotation.y = t * 0.5
    } else {
      // Spin fast when activated
      meshRef.current.rotation.y += 0.1
    }
  })
  
  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onClick={activated ? undefined : onActivate}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <cylinderGeometry args={[0.8, 0.8, 0.3, 32]} />
        <meshStandardMaterial
          color={activated ? '#00ff00' : (hovered ? '#0088ff' : '#0044ff')}
          emissive={activated ? '#00ff00' : (hovered ? '#0088ff' : '#0044ff')}
          emissiveIntensity={activated ? 1 : (hovered ? 0.8 : 0.5)}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>
      
      {/* Glow ring */}
      <mesh position={[0, 0, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.9, 1.1, 32]} />
        <meshBasicMaterial
          color={activated ? '#00ff00' : '#0044ff'}
          transparent
          opacity={0.5}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      
      {/* Label */}
      <Text
        position={[0, 1, 0]}
        fontSize={0.3}
        color={activated ? '#00ff00' : '#ffffff'}
        anchorX="center"
        anchorY="middle"
      >
        {activated ? '✓ ' + label : label}
      </Text>
      
      {/* Activation beam when activated */}
      {activated && (
        <mesh position={[0, 5, 0]}>
          <cylinderGeometry args={[0.1, 0.1, 10, 16]} />
          <meshBasicMaterial
            color="#00ff00"
            transparent
            opacity={0.6}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      )}
    </group>
  )
}

// Energy particles flowing to center
function ActivationParticles({ active }: { active: boolean }) {
  const pointsRef = useRef<THREE.Points>(null)
  const positions = useRef(new Float32Array(200 * 3))
  const velocities = useRef<THREE.Vector3[]>([])
  
  useEffect(() => {
    for (let i = 0; i < 200; i++) {
      const angle = (i / 200) * Math.PI * 2
      const radius = 15
      positions.current[i * 3] = Math.cos(angle) * radius
      positions.current[i * 3 + 1] = Math.random() * 10 - 5
      positions.current[i * 3 + 2] = Math.sin(angle) * radius
      
      velocities.current.push(new THREE.Vector3(0, 0, 0))
    }
  }, [])
  
  useFrame((state, delta) => {
    if (!pointsRef.current || !active) return
    const pos = pointsRef.current.geometry.attributes.position.array as Float32Array
    
    for (let i = 0; i < 200; i++) {
      const idx = i * 3
      const x = pos[idx]
      const y = pos[idx + 1]
      const z = pos[idx + 2]
      
      // Move toward center
      const dist = Math.sqrt(x * x + y * y + z * z)
      if (dist > 0.5) {
        pos[idx] -= (x / dist) * delta * 5
        pos[idx + 1] -= (y / dist) * delta * 5
        pos[idx + 2] -= (z / dist) * delta * 5
      } else {
        // Reset to edge
        const angle = Math.random() * Math.PI * 2
        const radius = 15
        pos[idx] = Math.cos(angle) * radius
        pos[idx + 1] = Math.random() * 10 - 5
        pos[idx + 2] = Math.sin(angle) * radius
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
        size={0.15}
        color="#00ff88"
        transparent
        opacity={0.8}
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

// Central core that powers up
function ActivationCore({ progress }: { progress: number }) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  useFrame((state) => {
    if (!meshRef.current) return
    const t = state.clock.elapsedTime
    
    meshRef.current.rotation.y = t * 0.5
    meshRef.current.rotation.x = Math.sin(t * 0.3) * 0.2
    
    const scale = 1 + progress * 2
    meshRef.current.scale.setScalar(scale)
  })
  
  return (
    <mesh ref={meshRef} position={[0, 3, 0]}>
      <octahedronGeometry args={[1, 0]} />
      <meshStandardMaterial
        color="#00ff88"
        emissive="#00ff88"
        emissiveIntensity={progress * 3}
        metalness={0.9}
        roughness={0.1}
        transparent
        opacity={0.8}
      />
      {/* Outer glow */}
      <mesh>
        <octahedronGeometry args={[1.5, 0]} />
        <meshBasicMaterial
          color="#00ff88"
          transparent
          opacity={progress * 0.3}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </mesh>
  )
}

export default function QMixActivation() {
  const { camera } = useThree()
  const { demoMode, setAirshipState, setGlitchIntensity, setQmixPid, setQmixStatusMsg, setScene } = useSceneStore()
  const [activatedButtons, setActivatedButtons] = useState<number[]>([])
  const [showInstructions, setShowInstructions] = useState(true)
  const transitionTriggered = useRef(false)
  
  const totalButtons = 4
  const progress = activatedButtons.length / totalButtons

  useEffect(() => {
    setAirshipState('locked')
    setGlitchIntensity(0.5)
    
    // Hide instructions after 5 seconds
    setTimeout(() => setShowInstructions(false), 5000)
  }, [])
  
  useEffect(() => {
    // When all buttons activated, transition
    if (activatedButtons.length === totalButtons && !transitionTriggered.current) {
      transitionTriggered.current = true

      const startController = async () => {
        setQmixStatusMsg('Bootstrapping AURA controller...')
        if (!demoMode) {
          try {
            const res = await fetch('/api/start-qmix', { method: 'POST' })
            const data = await res.json()
            if (typeof data?.pid === 'number') {
              setQmixPid(data.pid)
            }
            setQmixStatusMsg('AURA controller active')
          } catch {
            setQmixStatusMsg('Controller start failed')
          }
        } else {
          setQmixPid(99999)
          setQmixStatusMsg('Demo controller active')
        }

        setGlitchIntensity(0)
        setTimeout(() => {
          setScene('transform')
        }, 3000)
      }

      void startController()
    }
  }, [
    activatedButtons.length,
    demoMode,
    setGlitchIntensity,
    setQmixPid,
    setQmixStatusMsg,
    setScene,
  ])

  useFrame((_, delta) => {
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, 0, 0.02)
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, 6, 0.02)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, 13, 0.02)
    camera.lookAt(0, 3, 0)
  })
  
  const handleActivate = (index: number) => {
    if (!activatedButtons.includes(index)) {
      setActivatedButtons([...activatedButtons, index])
    }
  }

  return (
    <>
      <ambientLight intensity={0.15} color="#001133" />
      <pointLight position={[0, 8, 0]} intensity={2 + progress * 2} color="#00ff88" />
      <pointLight position={[0, -2, 0]} intensity={1} color="#003388" />
      
      {/* Spotlight on core */}
      <spotLight
        position={[0, 15, 0]}
        angle={0.5}
        penumbra={0.5}
        intensity={progress * 4}
        color="#00ff88"
        target-position={[0, 3, 0]}
      />
      
      {/* Instructions */}
      {showInstructions && (
        <Text
          position={[0, 8, 0]}
          fontSize={0.4}
          color="#00ffff"
          anchorX="center"
          anchorY="middle"
        >
          Activate all 4 agent nodes to start the real controller
        </Text>
      )}
      
      {/* Progress indicator */}
      <Text
        position={[0, 7, 0]}
        fontSize={0.5}
        color="#00ff88"
        anchorX="center"
        anchorY="middle"
      >
        {activatedButtons.length === totalButtons 
          ? 'QMix Activated!' 
          : `${activatedButtons.length}/${totalButtons} Nodes Active`}
      </Text>
      
      {/* Activation buttons in a circle */}
      <ActivationButton
        position={[-5, 2, -3]}
        label="Node 1"
        activated={activatedButtons.includes(0)}
        onActivate={() => handleActivate(0)}
      />
      <ActivationButton
        position={[5, 2, -3]}
        label="Node 2"
        activated={activatedButtons.includes(1)}
        onActivate={() => handleActivate(1)}
      />
      <ActivationButton
        position={[-5, 2, 3]}
        label="Node 3"
        activated={activatedButtons.includes(2)}
        onActivate={() => handleActivate(2)}
      />
      <ActivationButton
        position={[5, 2, 3]}
        label="Node 4"
        activated={activatedButtons.includes(3)}
        onActivate={() => handleActivate(3)}
      />
      
      {/* Central core */}
      <ActivationCore progress={progress} />
      
      {/* Energy particles */}
      <ActivationParticles active={progress > 0} />
      
      {/* Holographic grid floor */}
      <mesh position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[30, 30, 15, 15]} />
        <meshBasicMaterial
          color="#00ff88"
          transparent
          opacity={0.1 + progress * 0.2}
          wireframe
        />
      </mesh>
      
      {/* Energy rings expanding from center */}
      {progress > 0 && [1, 2, 3].map((i) => (
        <mesh key={i} position={[0, 3, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[i * 2 * progress, i * 2 * progress + 0.2, 32]} />
          <meshBasicMaterial
            color="#00ff88"
            transparent
            opacity={0.3 * (1 - progress)}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      ))}
      
      <PodGrid />
      <Airship />
      
      {/* 3D hologram dialog */}
      <Html position={[0, 5, 5]} transform distanceFactor={8} style={{ pointerEvents: 'auto' }}>
        <HologramDialog />
      </Html>
    </>
  )
}

// Made with Bob
