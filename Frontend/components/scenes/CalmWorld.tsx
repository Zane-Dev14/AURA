'use client'
import { useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Float } from '@react-three/drei'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import TrafficParticles from '@/components/r3f/TrafficParticles'
import * as THREE from 'three'

export default function CalmWorld() {
  const { camera } = useThree()
  const targetPos = new THREE.Vector3(0, 4, 12)
  const mouseRef = useRef({ x: 0, y: 0 })

  useFrame((state) => {
    // Mouse parallax
    mouseRef.current.x = state.mouse.x * 0.4
    mouseRef.current.y = state.mouse.y * 0.2
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, targetPos.x + mouseRef.current.x, 0.03)
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, targetPos.y + mouseRef.current.y, 0.03)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, targetPos.z, 0.03)
    camera.lookAt(0, 1, 0)
  })

  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={1.5} color="#ffd4aa" />
      <pointLight position={[0, 5, 0]} intensity={0.8} color="#88ddff" />
      <Airship />
      {/* Floating islands */}
      {[
        [-6, -1.5, -4],
        [7, -2, -6],
        [-4, -3, 4],
      ].map((pos, i) => (
        <Float key={i} speed={0.6 + i * 0.2} rotationIntensity={0.1} floatIntensity={0.3}>
          <mesh position={pos as [number, number, number]}>
            <cylinderGeometry args={[1.5 - i * 0.3, 1.2 - i * 0.2, 0.5, 8]} />
            <meshStandardMaterial color="#3a5a2a" roughness={0.9} />
          </mesh>
        </Float>
      ))}
      {/* Fog plane */}
      <mesh position={[0, -4, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[80, 80]} />
        <meshBasicMaterial color="#1a1a2e" transparent opacity={0.6} />
      </mesh>
      <TrafficParticles />
    </>
  )
}
