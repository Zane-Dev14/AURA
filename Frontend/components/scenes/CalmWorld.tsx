'use client'
import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import FloatingRing from '@/components/r3f/FloatingRing'
import Ground from '@/components/r3f/Ground'
import AtmosphericParticles from '@/components/r3f/AtmosphericParticles'
import BackgroundElements from '@/components/r3f/BackgroundElements'
import { Text } from '@react-three/drei'
import * as THREE from 'three'

export default function CalmWorld() {
  const { setScene } = useSceneStore()
  const lightRef = useRef<THREE.PointLight>(null)
  const spotlightRef = useRef<THREE.SpotLight>(null)
  const [ringsCollected, setRingsCollected] = useState(0)
  const transitionTriggered = useRef(false)

  // Camera now controlled by Airship component
  useFrame((state) => {
    const t = state.clock.elapsedTime

    // Animate lights
    if (lightRef.current) {
      lightRef.current.intensity = 3 + Math.sin(t * 0.5) * 0.5
      lightRef.current.position.x = 10 + Math.sin(t * 0.3) * 3
      lightRef.current.position.z = 12 + Math.cos(t * 0.4) * 3
    }

    if (spotlightRef.current) {
      spotlightRef.current.intensity = 4 + Math.sin(t * 0.7) * 0.8
    }
  })
  
  const handleRingCollect = () => {
    setRingsCollected(prev => {
      const newCount = prev + 1
      // Transition to next scene when all 8 rings collected
      if (newCount >= 8 && !transitionTriggered.current) {
        transitionTriggered.current = true
        setTimeout(() => {
          setScene('system')
        }, 2000) // 2 second delay for celebration
      }
      return newCount
    })
  }

  return (
    <>
      {/* HERO LIGHTING SETUP */}
      
      {/* Ambient base - dark and moody */}
      <ambientLight intensity={0.15} color="#0a1428" />
      
      {/* Hemisphere for subtle sky/ground gradient */}
      <hemisphereLight
        args={['#1a4a7a', '#050a15', 0.4]}
      />
      
      {/* KEY LIGHT - Main hero light from front-right */}
      <pointLight
        ref={lightRef}
        position={[8, 8, 10]}
        intensity={2.5}
        color="#00d4ff"
        distance={40}
        decay={2}
        castShadow
      />
      
      {/* SPOTLIGHT - Dramatic top-down hero spotlight */}
      <spotLight
        ref={spotlightRef}
        position={[0, 15, 5]}
        angle={0.4}
        penumbra={0.5}
        intensity={3}
        color="#00e5ff"
        distance={30}
        decay={2}
        castShadow
        target-position={[0, 2, 0]}
      />
      
      {/* RIM LIGHT - Edge definition from behind */}
      <directionalLight
        position={[-10, 8, -12]}
        intensity={1.2}
        color="#4a9aff"
      />
      
      {/* ACCENT LIGHT - Cyan accent from left */}
      <pointLight
        position={[-12, 5, 8]}
        intensity={1.5}
        color="#00ffaa"
        distance={35}
        decay={2}
      />
      
      {/* FILL LIGHT - Soft fill from below */}
      <pointLight
        position={[0, -2, 10]}
        intensity={0.8}
        color="#0066ff"
        distance={25}
        decay={2}
      />


      {/* NEW ENVIRONMENTAL COMPONENTS */}
      <Ground />
      <AtmosphericParticles />
      <BackgroundElements />
      {/* AIRSHIP */}
      <Airship />

      {/* Progress indicator */}
      {ringsCollected > 0 && (
        <Text
          position={[0, 8, -5]}
          fontSize={0.8}
          color="#00e5ff"
          anchorX="center"
          anchorY="middle"
        >
          {ringsCollected >= 8 ? '✓ All Rings Collected!' : `Rings: ${ringsCollected}/8`}
        </Text>
      )}

      {/* FLOATING RINGS TO COLLECT - Tutorial Course */}
      <FloatingRing position={[0, 3, -8]} onCollect={handleRingCollect} />
      <FloatingRing position={[-6, 4, -12]} onCollect={handleRingCollect} />
      <FloatingRing position={[6, 5, -15]} onCollect={handleRingCollect} />
      <FloatingRing position={[-8, 3, -20]} onCollect={handleRingCollect} />
      <FloatingRing position={[8, 6, -25]} onCollect={handleRingCollect} />
      <FloatingRing position={[0, 4, -30]} onCollect={handleRingCollect} />
      <FloatingRing position={[-10, 5, -35]} onCollect={handleRingCollect} />
      <FloatingRing position={[10, 3, -40]} onCollect={handleRingCollect} />

      {/* ATMOSPHERIC ELEMENTS */}
      
      {/* Volumetric light shafts effect */}
      <mesh position={[0, 8, -5]} rotation={[Math.PI / 6, 0, 0]}>
        <planeGeometry args={[30, 20]} />
        <meshBasicMaterial
          color="#00e5ff"
          transparent
          opacity={0.02}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Enhanced floating particles */}
      {Array.from({ length: 80 }).map((_, i) => {
        const angle = (i / 50) * Math.PI * 2
        const radius = 15 + Math.random() * 10
        const x = Math.cos(angle) * radius
        const z = Math.sin(angle) * radius
        const y = -5 + Math.random() * 20
        
        const size = 0.05 + Math.random() * 0.15
        
        return (
          <mesh key={i} position={[x, y, z]}>
            <sphereGeometry args={[size, 8, 8]} />
            <meshBasicMaterial
              color={i % 3 === 0 ? "#00ffaa" : "#00e5ff"}
              transparent
              opacity={0.4 + Math.random() * 0.5}
              blending={THREE.AdditiveBlending}
            />
          </mesh>
        )
      })}

      {/* Depth fog layers - creates atmosphere */}
      <mesh position={[0, -3, -15]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[80, 80]} />
        <meshBasicMaterial
          color="#001a3a"
          transparent
          opacity={0.15}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      <mesh position={[0, -5, -25]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[120, 120]} />
        <meshBasicMaterial
          color="#000a1a"
          transparent
          opacity={0.25}
          depthWrite={false}
        />
      </mesh>

      <mesh position={[0, -8, -40]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[180, 180]} />
        <meshBasicMaterial
          color="#000510"
          transparent
          opacity={0.4}
          depthWrite={false}
        />
      </mesh>

      {/* Circular platform/stage for hero */}
      <mesh position={[0, -0.5, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <circleGeometry args={[8, 64]} />
        <meshStandardMaterial
          color="#001a3a"
          transparent
          opacity={0.3}
          roughness={0.8}
          metalness={0.2}
          emissive="#00e5ff"
          emissiveIntensity={0.05}
        />
      </mesh>

      {/* Glowing ring around platform */}
      <mesh position={[0, -0.4, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[7.8, 8.2, 64]} />
        <meshBasicMaterial
          color="#00e5ff"
          transparent
          opacity={0.4}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Vertical light beams for drama */}
      {[0, 120, 240].map((angle, i) => {
        const rad = (angle * Math.PI) / 180
        const x = Math.cos(rad) * 12
        const z = Math.sin(rad) * 12
        
        return (
          <mesh key={i} position={[x, 5, z]} rotation={[0, 0, 0]}>
            <cylinderGeometry args={[0.1, 0.1, 20, 8]} />
            <meshBasicMaterial
              color="#00e5ff"
              transparent
              opacity={0.1}
              blending={THREE.AdditiveBlending}
            />
          </mesh>
        )
      })}

      {/* Holographic grid floor - extended */}
      <mesh position={[0, -1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[100, 100, 40, 40]} />
        <meshBasicMaterial
          color="#00e5ff"
          transparent
          opacity={0.08}
          wireframe
        />
      </mesh>

      {/* Energy field boundaries */}
      {[
        { pos: [0, 5, -50], rot: [0, 0, 0] },
        { pos: [-30, 5, -25], rot: [0, Math.PI / 2, 0] },
        { pos: [30, 5, -25], rot: [0, -Math.PI / 2, 0] },
      ].map((wall, i) => (
        <mesh
          key={i}
          position={wall.pos as [number, number, number]}
          rotation={wall.rot as [number, number, number]}
        >
          <planeGeometry args={[60, 20, 10, 10]} />
          <meshBasicMaterial
            color="#00e5ff"
            transparent
            opacity={0.03}
            wireframe
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </>
  )
}

// Made with Bob
