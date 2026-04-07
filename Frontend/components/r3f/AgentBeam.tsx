'use client'
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface Props {
  active: boolean
  from: THREE.Vector3   // airship position
  to: THREE.Vector3     // cluster core
}

export default function AgentBeam({ active, from, to }: Props) {
  const matRef = useRef<THREE.MeshStandardMaterial>(null)
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame((state, delta) => {
    if (!meshRef.current || !matRef.current || !active) return
    const t = state.clock.elapsedTime
    matRef.current.emissiveIntensity = 2 + Math.sin(t * 20) * 0.5
    // Pulse thickness
    const pulse = 1 + Math.sin(t * 30) * 0.15
    meshRef.current.scale.x = pulse
    meshRef.current.scale.z = pulse
  })

  if (!active) return null

  const mid = new THREE.Vector3().addVectors(from, to).multiplyScalar(0.5)
  const dir = new THREE.Vector3().subVectors(to, from)
  const length = dir.length()
  const quaternion = new THREE.Quaternion()
  quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize())

  return (
    <mesh ref={meshRef} position={mid} quaternion={quaternion}>
      <cylinderGeometry args={[0.04, 0.04, length, 8]} />
      <meshStandardMaterial
        ref={matRef}
        color="#001122"
        emissive="#00f0ff"
        emissiveIntensity={2}
        transparent
        opacity={0.9}
      />
    </mesh>
  )
}
