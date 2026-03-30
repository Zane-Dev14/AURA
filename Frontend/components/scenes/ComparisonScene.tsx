'use client'
import { useEffect, useRef, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'
import { COMPARISON_DATA } from '@/lib/mockMetrics'

// Simple pod for comparison side (not full InstancedMesh)
function MiniPodGrid({ side, podCount, health }: { side: 'left' | 'right'; podCount: number; health: number }) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const color = health > 0.5 ? new THREE.Color(0x00ff66) : new THREE.Color(0xff2200)

  useFrame((state) => {
    if (!meshRef.current) return
    const xOffset = side === 'left' ? -9 : 9
    for (let i = 0; i < 10; i++) {
      const scale = i < podCount ? 1 : 0.01
      dummy.position.set(xOffset + (i % 5) * 1.8 - 3.6, -1 + Math.sin(state.clock.elapsedTime + i) * 0.05, (Math.floor(i / 5)) * 1.8 - 0.9)
      dummy.scale.setScalar(scale * 0.7)
      dummy.updateMatrix()
      meshRef.current.setMatrixAt(i, dummy.matrix)
    }
    meshRef.current.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, 10]}>
      <boxGeometry args={[0.6, 0.6, 0.6]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
    </instancedMesh>
  )
}

// Particles for each side — HPA piles up (red), QMix flows (blue)
function ComparisonParticles({ side }: { side: 'left' | 'right' }) {
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
    if (!pointsRef.current) return
    const pos = pointsRef.current.geometry.attributes.position.array as Float32Array
    for (let i = 0; i < 200; i++) {
      const idx = i * 3
      if (isHPA) {
        // Pile up — converge but not flow through
        pos[idx] += (xOffset - pos[idx]) * 0.01
        pos[idx + 1] = Math.max(-1.5, pos[idx + 1] - delta * 0.3)
        if (pos[idx + 1] < -1.5) { pos[idx + 1] = 3; pos[idx] = xOffset + (Math.random() - 0.5) * 8 }
      } else {
        // Flow smoothly toward service
        pos[idx] += (xOffset * 0.5 - pos[idx]) * 0.02
        pos[idx + 1] -= delta * 0.8
        if (pos[idx + 1] < -2) { pos[idx + 1] = 3; pos[idx] = xOffset + (Math.random() - 0.5) * 8 }
      }
    }
    pointsRef.current.geometry.attributes.position.needsUpdate = true
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.08} color={isHPA ? '#ff4400' : '#00ccff'} transparent opacity={0.7} sizeAttenuation depthWrite={false} />
    </points>
  )
}

// HPA side: pods appear slowly (delayed)
function HPASide() {
  const podCount = useRef(1)
  const timerRef = useRef(0)
  const [count, setCount] = useDelayedCount(1, 4000, 1) // start at 1, add 1 every 4s

  return (
    <>
      <MiniPodGrid side="left" podCount={count} health={count / 10} />
      <ComparisonParticles side="left" />
    </>
  )
}

// Simple delayed count hook
function useDelayedCount(start: number, intervalMs: number, increment: number): [number, (n: number) => void] {
  const [count, setCount] = [useRef(start), (n: number) => { count.current = n }]
  const timerRef = useRef(0)
  useFrame((_, delta) => {
    timerRef.current += delta * 1000
    if (timerRef.current >= intervalMs && count.current < 10) {
      count.current += increment
      timerRef.current = 0
    }
  })
  return [count.current, (n) => { count.current = n }]
}

export default function ComparisonScene() {
  const { camera } = useThree()
  const { setAirshipState } = useSceneStore()

  useEffect(() => {
    setAirshipState('stable')
  }, [])

  useFrame((_, delta) => {
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, 0, 0.02)
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, 6, 0.02)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, 22, 0.02)
    camera.lookAt(0, -1, 0)
  })

  return (
    <>
      <ambientLight intensity={0.35} />
      <directionalLight position={[0, 10, 5]} intensity={1} />
      <pointLight position={[-9, 4, 0]} intensity={2} color="#ff4400" />
      <pointLight position={[9, 4, 0]} intensity={2} color="#00ccff" />
      
      

      {/* Divider */}
      <mesh position={[0, 1, 0]}>
        <boxGeometry args={[0.04, 8, 6]} />
        <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.4} transparent opacity={0.3} />
      </mesh>

      {/* HPA side (left) */}
      <MiniPodGrid side="left" podCount={2} health={0.2} />
      <ComparisonParticles side="left" />

      {/* QMix side (right) */}
      <MiniPodGrid side="right" podCount={9} health={1} />
      <ComparisonParticles side="right" />
    </>
  )
}
