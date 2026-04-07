'use client'
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface AirshipTrailProps {
  position: THREE.Vector3
  velocity?: THREE.Vector3
  speed: number
}

export default function AirshipTrail({ position, velocity, speed }: AirshipTrailProps) {
  const trailRef = useRef<THREE.Points>(null)
  const particlesRef = useRef<Float32Array>(new Float32Array(150))
  const velocitiesRef = useRef<Float32Array>(new Float32Array(150))
  const lifetimesRef = useRef<Float32Array>(new Float32Array(50))
  const opacitiesRef = useRef<Float32Array>(new Float32Array(50))

  useFrame((_, delta) => {
    if (!trailRef.current) return

    const positions = particlesRef.current
    const velocities = velocitiesRef.current
    const lifetimes = lifetimesRef.current
    const opacities = opacitiesRef.current

    // Update existing particles
    for (let i = 0; i < 50; i++) {
      const i3 = i * 3
      
      // Age particles
      lifetimes[i] -= delta * 2
      
      // Fade out particles based on lifetime
      opacities[i] = Math.max(0, lifetimes[i])
      
      // Reset dead particles at airship position
      if (lifetimes[i] <= 0 && speed > 0.5) {
        // Spawn particles behind the airship based on velocity
        const spawnOffset = velocity ? velocity.clone().normalize().multiplyScalar(-0.5) : new THREE.Vector3()
        
        positions[i3] = position.x + spawnOffset.x + (Math.random() - 0.5) * 0.5
        positions[i3 + 1] = position.y + spawnOffset.y + (Math.random() - 0.5) * 0.5
        positions[i3 + 2] = position.z + spawnOffset.z + (Math.random() - 0.5) * 0.5
        
        // Inherit some of the airship's velocity (opposite direction)
        if (velocity) {
          velocities[i3] = -velocity.x * 0.3 + (Math.random() - 0.5) * 0.5
          velocities[i3 + 1] = -velocity.y * 0.3 + (Math.random() - 0.5) * 0.5
          velocities[i3 + 2] = -velocity.z * 0.3 + (Math.random() - 0.5) * 0.5
        } else {
          velocities[i3] = (Math.random() - 0.5) * 0.5
          velocities[i3 + 1] = (Math.random() - 0.5) * 0.5
          velocities[i3 + 2] = (Math.random() - 0.5) * 0.5
        }
        
        lifetimes[i] = 1
        opacities[i] = 1
      }
      
      // Move particles with damping
      const damping = 0.95
      velocities[i3] *= damping
      velocities[i3 + 1] *= damping
      velocities[i3 + 2] *= damping
      
      positions[i3] += velocities[i3] * delta * 5
      positions[i3 + 1] += velocities[i3 + 1] * delta * 5
      positions[i3 + 2] += velocities[i3 + 2] * delta * 5
    }

    trailRef.current.geometry.attributes.position.needsUpdate = true
    
    // Update opacity attribute if it exists
    const opacityAttr = trailRef.current.geometry.attributes.opacity
    if (opacityAttr) {
      opacityAttr.needsUpdate = true
    }
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
        <bufferAttribute
          attach="attributes-opacity"
          count={50}
          array={opacitiesRef.current}
          itemSize={1}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.15}
        color="#00e5ff"
        transparent
        opacity={0.6}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        vertexColors={false}
      />
    </points>
  )
}

// Made with Bob
