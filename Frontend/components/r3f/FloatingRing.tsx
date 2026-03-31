'use client'
import { useRef, useState, useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

interface FloatingRingProps {
  position: [number, number, number]
  onCollect?: () => void
}

export default function FloatingRing({ position, onCollect }: FloatingRingProps) {
  const ringRef = useRef<THREE.Mesh>(null)
  const [collected, setCollected] = useState(false)
  const [scale, setScale] = useState(1)
  const [shouldUnmount, setShouldUnmount] = useState(false)
  const rotationSpeed = useRef(Math.random() * 0.5 + 0.5)
  const { scene } = useThree()

  useFrame((state, delta) => {
    if (!ringRef.current) return
    
    const t = state.clock.elapsedTime
    
    if (!collected) {
      // Rotate
      ringRef.current.rotation.y += delta * rotationSpeed.current
      ringRef.current.rotation.x = Math.sin(t * 0.5) * 0.2
      
      // Float up and down
      ringRef.current.position.y = position[1] + Math.sin(t * 2) * 0.3
      
      // Pulse scale
      const pulseScale = 1 + Math.sin(t * 3) * 0.1
      ringRef.current.scale.setScalar(scale * pulseScale)
      
      // Check collision with airship
      const airship = scene.getObjectByName('airship')
      if (airship) {
        const distance = ringRef.current.position.distanceTo(airship.position)
        const collectionRadius = 2.5 // Collection distance
        
        if (distance < collectionRadius) {
          handleCollision()
        }
      }
    } else {
      // Shrink and fade when collected
      const newScale = Math.max(0, scale - delta * 4)
      setScale(newScale)
      ringRef.current.scale.setScalar(newScale)
      
      // Trigger unmount when animation completes
      if (newScale <= 0.01 && !shouldUnmount) {
        setShouldUnmount(true)
      }
    }
  })

  // Cleanup effect - dispose of geometries and materials
  useEffect(() => {
    return () => {
      if (ringRef.current) {
        ringRef.current.geometry?.dispose()
        if (ringRef.current.material) {
          if (Array.isArray(ringRef.current.material)) {
            ringRef.current.material.forEach(mat => mat.dispose())
          } else {
            ringRef.current.material.dispose()
          }
        }
      }
    }
  }, [])

  const handleCollision = () => {
    if (!collected) {
      setCollected(true)
      onCollect?.()
    }
  }

  // Don't render if unmount is triggered
  if (shouldUnmount) {
    return null
  }

  return (
    <mesh
      ref={ringRef}
      position={position}
      onClick={handleCollision}
    >
      <torusGeometry args={[1.5, 0.15, 16, 32]} />
      <meshStandardMaterial
        color="#00e5ff"
        emissive="#00e5ff"
        emissiveIntensity={collected ? 2 : 0.5}
        metalness={0.8}
        roughness={0.2}
        transparent
        opacity={collected ? 0.3 : 0.8}
      />
      
      {/* Inner glow */}
      <mesh>
        <torusGeometry args={[1.5, 0.3, 16, 32]} />
        <meshBasicMaterial
          color="#00e5ff"
          transparent
          opacity={0.2}
          blending={THREE.AdditiveBlending}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Particles around ring */}
      {!collected && Array.from({ length: 12 }).map((_, i) => {
        const angle = (i / 12) * Math.PI * 2
        const radius = 1.8
        return (
          <mesh
            key={i}
            position={[
              Math.cos(angle) * radius,
              Math.sin(angle * 2) * 0.2,
              Math.sin(angle) * radius
            ]}
          >
            <sphereGeometry args={[0.08, 8, 8]} />
            <meshBasicMaterial
              color="#00e5ff"
              transparent
              opacity={0.6}
            />
          </mesh>
        )
      })}
    </mesh>
  )
}

// Made with Bob
