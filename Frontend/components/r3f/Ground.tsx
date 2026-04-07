'use client'
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

/**
 * Ground - Reflective ground plane with Tron-style grid
 * 
 * Creates a premium reflective surface that:
 * - Receives shadows from objects above
 * - Has a subtle grid pattern
 * - Features radial gradient from center
 * - Adds depth and scale to the scene
 */
export default function Ground() {
  const meshRef = useRef<THREE.Mesh>(null)
  const gridRef = useRef<THREE.Mesh>(null)

  // Subtle animation for living feel
  useFrame((state) => {
    const t = state.clock.elapsedTime
    
    if (meshRef.current) {
      // Very subtle pulsing opacity
      const material = meshRef.current.material as THREE.MeshStandardMaterial
      material.opacity = 0.3 + Math.sin(t * 0.5) * 0.05
    }

    if (gridRef.current) {
      // Subtle grid animation
      const material = gridRef.current.material as THREE.MeshBasicMaterial
      material.opacity = 0.15 + Math.sin(t * 0.3) * 0.03
    }
  })

  return (
    <group position={[0, -1, 0]}>
      {/* Main reflective ground plane */}
      <mesh
        ref={meshRef}
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
        position={[0, 0, 0]}
      >
        <planeGeometry args={[200, 200]} />
        <meshStandardMaterial
          color="#001a3a"
          transparent
          opacity={0.3}
          roughness={0.1}
          metalness={0.9}
          envMapIntensity={1.5}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Tron-style grid overlay */}
      <mesh
        ref={gridRef}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0.01, 0]}
      >
        <planeGeometry args={[200, 200, 50, 50]} />
        <meshBasicMaterial
          color="#00e5ff"
          transparent
          opacity={0.15}
          wireframe
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Radial gradient glow from center */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
        <circleGeometry args={[30, 64]} />
        <meshBasicMaterial
          color="#00e5ff"
          transparent
          opacity={0.1}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Inner glow ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.03, 0]}>
        <ringGeometry args={[8, 12, 64]} />
        <meshBasicMaterial
          color="#00ffaa"
          transparent
          opacity={0.2}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Outer boundary ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.04, 0]}>
        <ringGeometry args={[28, 30, 64]} />
        <meshBasicMaterial
          color="#00e5ff"
          transparent
          opacity={0.15}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  )
}

// Made with Bob