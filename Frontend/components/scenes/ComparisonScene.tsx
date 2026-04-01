'use client'
import { useEffect, useRef, useState, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import { Text } from '@react-three/drei'
import * as THREE from 'three'
import ComparisonSlider from '@/components/r3f/ComparisonSlider'

// Mini pod for comparison side with clipping
function MiniPodGrid({
  side,
  podCount,
  health,
  sliderPosition
}: {
  side: 'left' | 'right'
  podCount: number
  health: number
  sliderPosition: number
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const color = health > 0.5 ? new THREE.Color(0x00ff66) : new THREE.Color(0xff2200)
  
  // Create clipping plane
  const clippingPlane = useMemo(() => {
    if (side === 'left') {
      // Left side: clip when slider moves right (positive x)
      return new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)
    } else {
      // Right side: clip when slider moves left (negative x)
      return new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0)
    }
  }, [side])

  useFrame((state) => {
    if (!meshRef.current) return
    
    // Update clipping plane based on slider position
    if (side === 'left') {
      clippingPlane.constant = sliderPosition
    } else {
      clippingPlane.constant = -sliderPosition
    }
    
    const xOffset = side === 'left' ? -9 : 9
    for (let i = 0; i < 10; i++) {
      const scale = i < podCount ? 1 : 0.01
      dummy.position.set(
        xOffset + (i % 5) * 1.8 - 3.6,
        -1 + Math.sin(state.clock.elapsedTime + i) * 0.05,
        (Math.floor(i / 5)) * 1.8 - 0.9
      )
      dummy.scale.setScalar(scale * 0.7)
      dummy.updateMatrix()
      meshRef.current.setMatrixAt(i, dummy.matrix)
    }
    meshRef.current.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, 10]}>
      <boxGeometry args={[0.6, 0.6, 0.6]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.5}
        clippingPlanes={[clippingPlane]}
        clipShadows
      />
    </instancedMesh>
  )
}

// Particles for each side with clipping
function ComparisonParticles({
  side,
  active,
  sliderPosition
}: {
  side: 'left' | 'right'
  active: boolean
  sliderPosition: number
}) {
  const pointsRef = useRef<THREE.Points>(null)
  const xOffset = side === 'left' ? -9 : 9
  const isHPA = side === 'left'

  const positions = useMemo(() => {
    const arr = new Float32Array(200 * 3)
    for (let i = 0; i < 200; i++) {
      arr[i * 3] = xOffset + (Math.random() - 0.5) * 8
      arr[i * 3 + 1] = Math.random() * 4 - 2
      arr[i * 3 + 2] = (Math.random() - 0.5) * 6
    }
    return arr
  }, [xOffset])
  
  // Create clipping plane
  const clippingPlane = useMemo(() => {
    if (side === 'left') {
      return new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)
    } else {
      return new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0)
    }
  }, [side])

  useFrame((state, delta) => {
    if (!pointsRef.current || !active) return
    
    // Update clipping plane
    if (side === 'left') {
      clippingPlane.constant = sliderPosition
    } else {
      clippingPlane.constant = -sliderPosition
    }
    
    const pos = pointsRef.current.geometry.attributes.position.array as Float32Array
    for (let i = 0; i < 200; i++) {
      const idx = i * 3
      if (isHPA) {
        // Pile up — converge but not flow through
        pos[idx] += (xOffset - pos[idx]) * 0.01
        pos[idx + 1] = Math.max(-1.5, pos[idx + 1] - delta * 0.3)
        if (pos[idx + 1] < -1.5) {
          pos[idx + 1] = 3
          pos[idx] = xOffset + (Math.random() - 0.5) * 8
        }
      } else {
        // Flow smoothly toward service
        pos[idx] += (xOffset * 0.5 - pos[idx]) * 0.02
        pos[idx + 1] -= delta * 0.8
        if (pos[idx + 1] < -2) {
          pos[idx + 1] = 3
          pos[idx] = xOffset + (Math.random() - 0.5) * 8
        }
      }
    }
    pointsRef.current.geometry.attributes.position.needsUpdate = true
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.08}
        color={isHPA ? '#ff4400' : '#00ccff'}
        transparent
        opacity={active ? 0.7 : 0.2}
        sizeAttenuation
        depthWrite={false}
        clippingPlanes={[clippingPlane]}
      />
    </points>
  )
}

// Metric display
function MetricDisplay({
  position,
  label,
  value,
  color,
  opacity = 1
}: {
  position: [number, number, number]
  label: string
  value: string
  color: string
  opacity?: number
}) {
  return (
    <group position={position}>
      <Text
        position={[0, 0.3, 0]}
        fontSize={0.3}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        fillOpacity={opacity}
      >
        {label}
      </Text>
      <Text
        position={[0, -0.3, 0]}
        fontSize={0.5}
        color={color}
        anchorX="center"
        anchorY="middle"
        fillOpacity={opacity}
      >
        {value}
      </Text>
    </group>
  )
}

export default function ComparisonScene() {
  const { camera, gl } = useThree()
  const { setAirshipState } = useSceneStore()
  const [sliderPosition, setSliderPosition] = useState(0)
  const [showInstructions, setShowInstructions] = useState(true)

  useEffect(() => {
    setAirshipState('stable')
    
    // Enable local clipping
    gl.localClippingEnabled = true
    
    // Hide instructions after 5 seconds
    setTimeout(() => setShowInstructions(false), 5000)
    
    return () => {
      gl.localClippingEnabled = false
    }
  }, [gl, setAirshipState])

  useFrame(() => {
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, 0, 0.02)
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, 8, 0.02)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, 25, 0.02)
    camera.lookAt(0, 0, 0)
  })
  
  // Calculate opacity based on slider position for fade effect
  const leftOpacity = THREE.MathUtils.clamp(1 - (sliderPosition + 10) / 20, 0.3, 1)
  const rightOpacity = THREE.MathUtils.clamp(1 - (10 - sliderPosition) / 20, 0.3, 1)

  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[0, 10, 5]} intensity={1.2} />
      <pointLight position={[-9, 4, 0]} intensity={2} color="#ff4400" />
      <pointLight position={[9, 4, 0]} intensity={2} color="#00ccff" />
      
      {/* Instructions */}
      {showInstructions && (
        <Text
          position={[0, 9, 0]}
          fontSize={0.4}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
        >
          Drag the slider to compare HPA vs QMix
        </Text>
      )}
      
      {/* Title */}
      <Text
        position={[0, 7, 0]}
        fontSize={0.6}
        color="#00ffff"
        anchorX="center"
        anchorY="middle"
      >
        Performance Comparison
      </Text>
      
      {/* Side labels */}
      <Text
        position={[-9, 5, 0]}
        fontSize={0.5}
        color="#ff4400"
        anchorX="center"
        anchorY="middle"
        fillOpacity={leftOpacity}
      >
        HPA (Slow)
      </Text>
      <Text
        position={[9, 5, 0]}
        fontSize={0.5}
        color="#00ccff"
        anchorX="center"
        anchorY="middle"
        fillOpacity={rightOpacity}
      >
        QMix (Fast)
      </Text>
      
      {/* Metrics */}
      <MetricDisplay
        position={[-9, 3.5, 0]}
        label="Scale Time"
        value="~4s"
        color="#ff4400"
        opacity={leftOpacity}
      />
      <MetricDisplay
        position={[9, 3.5, 0]}
        label="Scale Time"
        value="~0.5s"
        color="#00ccff"
        opacity={rightOpacity}
      />
      
      <MetricDisplay
        position={[-9, -3, 0]}
        label="Pods"
        value="2/10"
        color="#ff4400"
        opacity={leftOpacity}
      />
      <MetricDisplay
        position={[9, -3, 0]}
        label="Pods"
        value="9/10"
        color="#00ccff"
        opacity={rightOpacity}
      />

      {/* HPA side (left) - Always rendered */}
      <MiniPodGrid side="left" podCount={2} health={0.2} sliderPosition={sliderPosition} />
      <ComparisonParticles side="left" active={true} sliderPosition={sliderPosition} />

      {/* QMix side (right) - Always rendered */}
      <MiniPodGrid side="right" podCount={9} health={1} sliderPosition={sliderPosition} />
      <ComparisonParticles side="right" active={true} sliderPosition={sliderPosition} />
      
      {/* Comparison Slider */}
      <ComparisonSlider onPositionChange={setSliderPosition} />
      
      {/* Grid floor */}
      <mesh position={[0, -6, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[40, 30, 20, 15]} />
        <meshBasicMaterial
          color="#0088ff"
          transparent
          opacity={0.1}
          wireframe
        />
      </mesh>
      
      {/* Highlight zones with fade */}
      <mesh position={[-9, 0, 0]}>
        <boxGeometry args={[10, 10, 8]} />
        <meshBasicMaterial
          color="#ff4400"
          transparent
          opacity={0.05 * leftOpacity}
          wireframe
        />
      </mesh>
      
      <mesh position={[9, 0, 0]}>
        <boxGeometry args={[10, 10, 8]} />
        <meshBasicMaterial
          color="#00ccff"
          transparent
          opacity={0.05 * rightOpacity}
          wireframe
        />
      </mesh>
    </>
  )
}

// Made with Bob
