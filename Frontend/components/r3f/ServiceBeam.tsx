'use client'
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'

export default function ServiceBeam() {
  const meshRef = useRef<THREE.Mesh>(null)
  const matRef = useRef<THREE.MeshStandardMaterial>(null)
  const { podHealth, trafficLevel, frozen } = useSceneStore()

  useFrame((state, delta) => {
    if (!meshRef.current || !matRef.current || frozen) return
    const t = state.clock.elapsedTime
    // Pulse scale with traffic
    const pulse = 1 + Math.sin(t * (5 + trafficLevel * 10)) * 0.08 * trafficLevel
    meshRef.current.scale.x = pulse
    meshRef.current.scale.z = pulse
    // Flicker on failure
    if (podHealth < 0.3) {
      meshRef.current.visible = Math.random() > 0.15
    } else {
      meshRef.current.visible = true
    }
    // Color: green → red as health drops
    const col = new THREE.Color().lerpColors(
      new THREE.Color(0xff2200),
      new THREE.Color(0x00ffaa),
      podHealth
    )
    matRef.current.emissive = col
    matRef.current.emissiveIntensity = 1.5 + trafficLevel * 2 + Math.sin(t * 8) * 0.3
  })

  return (
    <group position={[0, -1, 0]}>
      <mesh ref={meshRef}>
        <cylinderGeometry args={[0.08, 0.08, 6, 8]} />
        <meshStandardMaterial
          ref={matRef}
          color="#001122"
          emissive="#00ffaa"
          emissiveIntensity={1.5}
          transparent
          opacity={0.85}
        />
      </mesh>
      {/* Base disc */}
      <mesh position={[0, -3, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[1.2, 32]} />
        <meshStandardMaterial color="#001a2e" emissive="#00aaff" emissiveIntensity={0.6} transparent opacity={0.5} />
      </mesh>
      {/* Top disc */}
      <mesh position={[0, 3, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.5, 32]} />
        <meshStandardMaterial color="#001a2e" emissive="#00ffaa" emissiveIntensity={1} transparent opacity={0.7} />
      </mesh>
    </group>
  )
}
