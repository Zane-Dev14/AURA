'use client'
import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

/**
 * AtmosphericParticles - Floating dust particles for depth
 *
 * Creates 500 particles (optimized for performance) that:
 * - Float organically through the scene
 * - Fade with distance (depth-based opacity)
 * - Add subtle glow
 * - Create sense of scale and atmosphere
 */
export default function AtmosphericParticles() {
  const particlesRef = useRef<THREE.Points>(null)
  
  // Generate particle positions and properties
  const particleData = useMemo(() => {
    const count = 500 // Reduced from 1000 for better performance
    const positions = new Float32Array(count * 3)
    const velocities = new Float32Array(count * 3)
    const sizes = new Float32Array(count)
    const opacities = new Float32Array(count)
    
    for (let i = 0; i < count; i++) {
      const i3 = i * 3
      
      // Spread particles in a large volume
      positions[i3] = (Math.random() - 0.5) * 100
      positions[i3 + 1] = Math.random() * 40 - 5
      positions[i3 + 2] = (Math.random() - 0.5) * 100
      
      // Random velocities for organic movement
      velocities[i3] = (Math.random() - 0.5) * 0.02
      velocities[i3 + 1] = Math.random() * 0.01 + 0.005
      velocities[i3 + 2] = (Math.random() - 0.5) * 0.02
      
      // Varying sizes
      sizes[i] = Math.random() * 0.15 + 0.05
      
      // Base opacity
      opacities[i] = Math.random() * 0.6 + 0.2
    }
    
    return { positions, velocities, sizes, opacities, count }
  }, [])

  // Animate particles
  useFrame((state, delta) => {
    if (!particlesRef.current) return
    
    const positions = particlesRef.current.geometry.attributes.position.array as Float32Array
    const { velocities, count } = particleData
    
    for (let i = 0; i < count; i++) {
      const i3 = i * 3
      
      // Update positions
      positions[i3] += velocities[i3]
      positions[i3 + 1] += velocities[i3 + 1]
      positions[i3 + 2] += velocities[i3 + 2]
      
      // Wrap around boundaries
      if (positions[i3 + 1] > 35) {
        positions[i3 + 1] = -5
        positions[i3] = (Math.random() - 0.5) * 100
        positions[i3 + 2] = (Math.random() - 0.5) * 100
      }
      
      if (Math.abs(positions[i3]) > 50) {
        positions[i3] = (Math.random() - 0.5) * 100
      }
      
      if (Math.abs(positions[i3 + 2]) > 50) {
        positions[i3 + 2] = (Math.random() - 0.5) * 100
      }
    }
    
    particlesRef.current.geometry.attributes.position.needsUpdate = true
    
    // Subtle rotation for variety
    particlesRef.current.rotation.y += delta * 0.01
  })

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleData.count}
          array={particleData.positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-size"
          count={particleData.count}
          array={particleData.sizes}
          itemSize={1}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.1}
        color="#00e5ff"
        transparent
        opacity={0.4}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        fog
      />
    </points>
  )
}

// Made with Bob