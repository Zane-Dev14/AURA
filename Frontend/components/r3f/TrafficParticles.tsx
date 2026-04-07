'use client'
import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

const COUNT = 1600
const TUNNEL_DEPTH = 120

export default function TrafficParticles() {
  const pointsRef = useRef<THREE.Points>(null)

  const positions = useMemo(() => {
    const arr = new Float32Array(COUNT * 3)
    for (let i = 0; i < COUNT; i++) {
      // Cylinder distribution to form an infinite space tunnel
      const radius = 5 + Math.random() * 30
      const angle = Math.random() * Math.PI * 2
      arr[i * 3 + 0] = Math.cos(angle) * radius 
      // Add slight offset so it feels scattered, not purely cylindrical
      arr[i * 3 + 1] = Math.sin(angle) * radius + (Math.random() - 0.5) * 5
      
      // Z-depth placement (-120 to +10)
      arr[i * 3 + 2] = 10 - Math.random() * TUNNEL_DEPTH
    }
    return arr
  }, [])

  useFrame((state, delta) => {
    if (!pointsRef.current) return
    const pos = pointsRef.current.geometry.attributes.position.array as Float32Array
    
    for (let i = 0; i < COUNT; i++) {
      const idx = i * 3
      // Rapid forward movement creates the illusion of ship traveling
      pos[idx + 2] += delta * 18.0
      
      // Organic swaying over time
      pos[idx + 0] += Math.sin(state.clock.elapsedTime + pos[idx+2] * 0.1) * 0.015
      
      // Wrap around when passing the camera (+Z)
      if (pos[idx + 2] > 10) {
        pos[idx + 2] = -TUNNEL_DEPTH
      }
    }
    
    pointsRef.current.geometry.attributes.position.needsUpdate = true
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={COUNT} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.15}
        color="#aaddff"
        transparent
        opacity={0.6}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}
