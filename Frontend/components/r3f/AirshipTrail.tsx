'use client'
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface AirshipTrailProps {
  position: THREE.Vector3
  speed: number
}

export default function AirshipTrail({ position, speed }: AirshipTrailProps) {
  const trailRef = useRef<THREE.Points>(null)
  const particlesRef = useRef<Float32Array>(new Float32Array(150))
  const velocitiesRef = useRef<Float32Array>(new Float32Array(150))
  const lifetimesRef = useRef<Float32Array>(new Float32Array(50))

  useFrame((_, delta) => {
    if (!trailRef.current) return

    const positions = particlesRef.current
    const velocities = velocitiesRef.current
    const lifetimes = lifetimesRef.current

    // Update existing particles
    for (let i = 0; i < 50; i++) {
      const i3 = i * 3
      
      // Age particles
      lifetimes[i] -= delta * 2
      
      // Reset dead particles at airship position
      if (lifetimes[i] <= 0 && speed > 0.5) {
        positions[i3] = position.x + (Math.random() - 0.5) * 0.5
        positions[i3 + 1] = position.y + (Math.random() - 0.5) * 0.5
        positions[i3 + 2] = position.z + (Math.random() - 0.5) * 0.5
        
        velocities[i3] = (Math.random() - 0.5) * 0.5
        velocities[i3 + 1] = (Math.random() - 0.5) * 0.5
        velocities[i3 + 2] = (Math.random() - 0.5) * 0.5
        
        lifetimes[i] = 1
      }
      
      // Move particles
      positions[i3] += velocities[i3] * delta * 5
      positions[i3 + 1] += velocities[i3 + 1] * delta * 5
      positions[i3 + 2] += velocities[i3 + 2] * delta * 5
    }

    trailRef.current.geometry.attributes.position.needsUpdate = true
  })

  return (
    <points ref={trailRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={50}
          array={particlesRef.current}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.15}
        color="#00e5ff"
        transparent
        opacity={0.6}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  )
}

// Made with Bob
