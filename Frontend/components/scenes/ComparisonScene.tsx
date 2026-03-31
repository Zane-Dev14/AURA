'use client'
import { useEffect, useRef, useState, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import { Text } from '@react-three/drei'
import * as THREE from 'three'

// Mini pod for comparison side
function MiniPodGrid({ 
  side, 
  podCount, 
  health 
}: { 
  side: 'left' | 'right'
  podCount: number
  health: number 
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const color = health > 0.5 ? new THREE.Color(0x00ff66) : new THREE.Color(0xff2200)

  useFrame((state) => {
    if (!meshRef.current) return
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
      />
    </instancedMesh>
  )
}

// Particles for each side
function ComparisonParticles({ 
  side, 
  active 
}: { 
  side: 'left' | 'right'
  active: boolean 
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

  useFrame((state, delta) => {
    if (!pointsRef.current || !active) return
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
      />
    </points>
  )
}

// Toggle button to switch between views
function ToggleButton({ 
  position, 
  activeView, 
  onToggle 
}: { 
  position: [number, number, number]
  activeView: 'hpa' | 'qmix' | 'both'
  onToggle: () => void
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)
  
  useFrame((state) => {
    if (!meshRef.current) return
    const t = state.clock.elapsedTime
    
    // Pulse
    const pulse = 1 + Math.sin(t * 3) * 0.05
    meshRef.current.scale.setScalar(pulse)
    
    // Rotate
    meshRef.current.rotation.y = t * 0.5
  })
  
  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onClick={onToggle}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <cylinderGeometry args={[1, 1, 0.4, 32]} />
        <meshStandardMaterial
          color={hovered ? '#00ffff' : '#0088ff'}
          emissive={hovered ? '#00ffff' : '#0088ff'}
          emissiveIntensity={hovered ? 1 : 0.5}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>
      
      {/* Label */}
      <Text
        position={[0, 1.5, 0]}
        fontSize={0.4}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        Toggle View
      </Text>
      
      {/* Current mode indicator */}
      <Text
        position={[0, -1.2, 0]}
        fontSize={0.3}
        color="#00ffff"
        anchorX="center"
        anchorY="middle"
      >
        {activeView === 'both' ? 'Both' : activeView === 'hpa' ? 'HPA Only' : 'QMix Only'}
      </Text>
    </group>
  )
}

// Metric display
function MetricDisplay({ 
  position, 
  label, 
  value, 
  color 
}: { 
  position: [number, number, number]
  label: string
  value: string
  color: string
}) {
  return (
    <group position={position}>
      <Text
        position={[0, 0.3, 0]}
        fontSize={0.3}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        {label}
      </Text>
      <Text
        position={[0, -0.3, 0]}
        fontSize={0.5}
        color={color}
        anchorX="center"
        anchorY="middle"
      >
        {value}
      </Text>
    </group>
  )
}

export default function ComparisonScene() {
  const { camera } = useThree()
  const { setAirshipState } = useSceneStore()
  const [activeView, setActiveView] = useState<'hpa' | 'qmix' | 'both'>('both')
  const [showInstructions, setShowInstructions] = useState(true)

  useEffect(() => {
    setAirshipState('stable')
    
    // Hide instructions after 5 seconds
    setTimeout(() => setShowInstructions(false), 5000)
  }, [])

  useFrame((_, delta) => {
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, 0, 0.02)
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, 6, 0.02)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, 22, 0.02)
    camera.lookAt(0, -1, 0)
  })
  
  const handleToggle = () => {
    setActiveView(prev => {
      if (prev === 'both') return 'hpa'
      if (prev === 'hpa') return 'qmix'
      return 'both'
    })
  }
  
  const showHPA = activeView === 'both' || activeView === 'hpa'
  const showQMix = activeView === 'both' || activeView === 'qmix'

  return (
    <>
      <ambientLight intensity={0.35} />
      <directionalLight position={[0, 10, 5]} intensity={1} />
      <pointLight position={[-9, 4, 0]} intensity={showHPA ? 2 : 0.5} color="#ff4400" />
      <pointLight position={[9, 4, 0]} intensity={showQMix ? 2 : 0.5} color="#00ccff" />
      
      {/* Instructions */}
      {showInstructions && (
        <Text
          position={[0, 8, 0]}
          fontSize={0.4}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
        >
          Click the button to compare HPA vs QMix
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
      >
        HPA (Slow)
      </Text>
      <Text
        position={[9, 5, 0]}
        fontSize={0.5}
        color="#00ccff"
        anchorX="center"
        anchorY="middle"
      >
        QMix (Fast)
      </Text>
      
      {/* Metrics */}
      <MetricDisplay 
        position={[-9, 3.5, 0]} 
        label="Scale Time" 
        value="~4s" 
        color="#ff4400" 
      />
      <MetricDisplay 
        position={[9, 3.5, 0]} 
        label="Scale Time" 
        value="~0.5s" 
        color="#00ccff" 
      />
      
      <MetricDisplay 
        position={[-9, -3, 0]} 
        label="Pods" 
        value="2/10" 
        color="#ff4400" 
      />
      <MetricDisplay 
        position={[9, -3, 0]} 
        label="Pods" 
        value="9/10" 
        color="#00ccff" 
      />

      {/* Divider */}
      <mesh position={[0, 1, 0]}>
        <boxGeometry args={[0.04, 8, 6]} />
        <meshStandardMaterial 
          color="#ffffff" 
          emissive="#ffffff" 
          emissiveIntensity={0.4} 
          transparent 
          opacity={0.3} 
        />
      </mesh>

      {/* HPA side (left) */}
      {showHPA && (
        <>
          <MiniPodGrid side="left" podCount={2} health={0.2} />
          <ComparisonParticles side="left" active={true} />
        </>
      )}

      {/* QMix side (right) */}
      {showQMix && (
        <>
          <MiniPodGrid side="right" podCount={9} health={1} />
          <ComparisonParticles side="right" active={true} />
        </>
      )}
      
      {/* Toggle button */}
      <ToggleButton 
        position={[0, -5, 0]} 
        activeView={activeView}
        onToggle={handleToggle}
      />
      
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
      
      {/* Highlight zones */}
      {showHPA && (
        <mesh position={[-9, 0, 0]}>
          <boxGeometry args={[10, 10, 8]} />
          <meshBasicMaterial
            color="#ff4400"
            transparent
            opacity={0.05}
            wireframe
          />
        </mesh>
      )}
      
      {showQMix && (
        <mesh position={[9, 0, 0]}>
          <boxGeometry args={[10, 10, 8]} />
          <meshBasicMaterial
            color="#00ccff"
            transparent
            opacity={0.05}
            wireframe
          />
        </mesh>
      )}
    </>
  )
}

// Made with Bob
