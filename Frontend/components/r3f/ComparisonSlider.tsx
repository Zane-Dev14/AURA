'use client'
import { useRef, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Text } from '@react-three/drei'
import * as THREE from 'three'

interface ComparisonSliderProps {
  onPositionChange: (x: number) => void
}

export default function ComparisonSlider({ onPositionChange }: ComparisonSliderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [hovered, setHovered] = useState(false)
  const position = useRef(0)
  const groupRef = useRef<THREE.Group>(null)
  const handleRef = useRef<THREE.Mesh>(null)
  const { gl } = useThree()

  // Handle pointer events
  const handlePointerDown = (e: any) => {
    e.stopPropagation()
    setIsDragging(true)
    gl.domElement.style.cursor = 'grabbing'
  }

  const handlePointerUp = () => {
    setIsDragging(false)
    gl.domElement.style.cursor = hovered ? 'grab' : 'default'
  }

  const handlePointerOver = () => {
    setHovered(true)
    if (!isDragging) {
      gl.domElement.style.cursor = 'grab'
    }
  }

  const handlePointerOut = () => {
    setHovered(false)
    if (!isDragging) {
      gl.domElement.style.cursor = 'default'
    }
  }

  useFrame((state) => {
    if (isDragging) {
      // Map pointer.x (-1 to 1) to slider range (-10 to 10)
      const targetX = state.pointer.x * 10
      position.current = THREE.MathUtils.lerp(position.current, targetX, 0.15)
    }
    
    // Update group position
    if (groupRef.current) {
      groupRef.current.position.x = position.current
    }
    
    // Animate handle scale on hover
    if (handleRef.current) {
      const targetScale = hovered ? 1.2 : 1.0
      handleRef.current.scale.lerp(
        new THREE.Vector3(targetScale, targetScale, targetScale),
        0.1
      )
    }
    
    // Notify parent of position change
    onPositionChange(position.current)
  })

  return (
    <group ref={groupRef}>
      {/* Vertical bar */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[0.15, 12, 0.15]} />
        <meshStandardMaterial
          color="#00ffff"
          emissive="#00ffff"
          emissiveIntensity={hovered ? 2.5 : 1.5}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>

      {/* Draggable handle */}
      <mesh
        ref={handleRef}
        position={[0, 0, 0]}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
      >
        <cylinderGeometry args={[0.8, 0.8, 0.4, 32]} />
        <meshStandardMaterial
          color="#00ffff"
          emissive="#00ffff"
          emissiveIntensity={hovered ? 3 : 2}
          metalness={0.9}
          roughness={0.1}
        />
      </mesh>

      {/* Top cap */}
      <mesh position={[0, 6, 0]}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial
          color="#00ffff"
          emissive="#00ffff"
          emissiveIntensity={2}
        />
      </mesh>

      {/* Bottom cap */}
      <mesh position={[0, -6, 0]}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial
          color="#00ffff"
          emissive="#00ffff"
          emissiveIntensity={2}
        />
      </mesh>

      {/* Left arrow */}
      <Text
        position={[-1.5, 0, 0]}
        fontSize={0.6}
        color="#00ffff"
        anchorX="center"
        anchorY="middle"
      >
        ←
      </Text>

      {/* Right arrow */}
      <Text
        position={[1.5, 0, 0]}
        fontSize={0.6}
        color="#00ffff"
        anchorX="center"
        anchorY="middle"
      >
        →
      </Text>

      {/* Instruction text */}
      <Text
        position={[0, -7, 0]}
        fontSize={0.3}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        Drag to Compare
      </Text>
    </group>
  )
}

// Made with Bob
