'use client'
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

/**
 * BackgroundElements - Distant geometric shapes for depth
 * 
 * Creates floating geometric shapes that:
 * - Slowly rotate in the background
 * - Have low opacity (0.1-0.2)
 * - Create sense of scale and depth
 * - Add visual interest without distraction
 */
export default function BackgroundElements() {
  const group1Ref = useRef<THREE.Group>(null)
  const group2Ref = useRef<THREE.Group>(null)
  const group3Ref = useRef<THREE.Group>(null)

  useFrame((state, delta) => {
    const t = state.clock.elapsedTime

    // Rotate groups at different speeds for parallax effect
    if (group1Ref.current) {
      group1Ref.current.rotation.y += delta * 0.05
      group1Ref.current.rotation.x = Math.sin(t * 0.1) * 0.1
    }

    if (group2Ref.current) {
      group2Ref.current.rotation.y -= delta * 0.03
      group2Ref.current.rotation.z = Math.cos(t * 0.15) * 0.1
    }

    if (group3Ref.current) {
      group3Ref.current.rotation.x += delta * 0.04
      group3Ref.current.rotation.y = Math.sin(t * 0.12) * 0.15
    }
  })

  return (
    <>
      {/* Group 1: Large distant cubes */}
      <group ref={group1Ref} position={[0, 0, -60]}>
        {[
          { pos: [-30, 15, 0], scale: 8 },
          { pos: [35, 20, -10], scale: 6 },
          { pos: [-25, 25, -15], scale: 7 },
          { pos: [30, 10, 5], scale: 5 },
        ].map((cube, i) => (
          <mesh
            key={i}
            position={cube.pos as [number, number, number]}
            rotation={[Math.PI / 4, Math.PI / 4, 0]}
          >
            <boxGeometry args={[cube.scale, cube.scale, cube.scale]} />
            <meshBasicMaterial
              color="#00e5ff"
              transparent
              opacity={0.08}
              wireframe
            />
          </mesh>
        ))}
      </group>

      {/* Group 2: Medium spheres */}
      <group ref={group2Ref} position={[0, 0, -45]}>
        {[
          { pos: [40, 18, 0], scale: 4 },
          { pos: [-35, 22, -8], scale: 5 },
          { pos: [25, 12, 10], scale: 3.5 },
          { pos: [-30, 15, 5], scale: 4.5 },
        ].map((sphere, i) => (
          <mesh
            key={i}
            position={sphere.pos as [number, number, number]}
          >
            <sphereGeometry args={[sphere.scale, 16, 16]} />
            <meshBasicMaterial
              color="#00ffaa"
              transparent
              opacity={0.12}
              wireframe
            />
          </mesh>
        ))}
      </group>

      {/* Group 3: Torus rings for variety */}
      <group ref={group3Ref} position={[0, 0, -50]}>
        {[
          { pos: [-40, 20, -5], scale: 5, rot: [0, 0, Math.PI / 3] },
          { pos: [38, 16, 8], scale: 4, rot: [Math.PI / 4, 0, 0] },
          { pos: [0, 25, -10], scale: 6, rot: [0, Math.PI / 4, Math.PI / 6] },
        ].map((torus, i) => (
          <mesh
            key={i}
            position={torus.pos as [number, number, number]}
            rotation={torus.rot as [number, number, number]}
          >
            <torusGeometry args={[torus.scale, torus.scale * 0.3, 16, 32]} />
            <meshBasicMaterial
              color="#4a9eff"
              transparent
              opacity={0.1}
              wireframe
            />
          </mesh>
        ))}
      </group>

      {/* Distant vertical pillars for scale */}
      {[
        { pos: [-50, 0, -70], height: 60 },
        { pos: [55, 0, -75], height: 50 },
        { pos: [-45, 0, -80], height: 55 },
        { pos: [50, 0, -85], height: 65 },
      ].map((pillar, i) => (
        <mesh
          key={`pillar-${i}`}
          position={pillar.pos as [number, number, number]}
        >
          <cylinderGeometry args={[0.5, 0.5, pillar.height, 8]} />
          <meshBasicMaterial
            color="#00e5ff"
            transparent
            opacity={0.06}
            wireframe
          />
        </mesh>
      ))}

      {/* Floating platforms for depth */}
      {[
        { pos: [-35, 5, -55], size: [15, 0.5, 15] },
        { pos: [40, 8, -60], size: [12, 0.5, 12] },
        { pos: [0, 12, -65], size: [18, 0.5, 18] },
      ].map((platform, i) => (
        <mesh
          key={`platform-${i}`}
          position={platform.pos as [number, number, number]}
          rotation={[-Math.PI / 2, 0, 0]}
        >
          <planeGeometry args={platform.size as [number, number]} />
          <meshBasicMaterial
            color="#00e5ff"
            transparent
            opacity={0.05}
            wireframe
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </>
  )
}

// Made with Bob