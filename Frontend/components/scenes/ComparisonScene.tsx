'use client'
import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import { Text } from '@react-three/drei'
import * as THREE from 'three'

const FINAL_RESULTS = {
  hpa: {
    avgReplicasTotal: 6.93,
    cpuRequestedCores: 2.0,
    apiP99Ms: 99.87,
  },
  qmix: {
    avgReplicasTotal: 5.76,
    cpuRequestedCores: 0.9,
    apiP99Ms: 23.13,
  },
}

function MiniPodGrid({
  side,
  podCount,
  health,
}: {
  side: 'left' | 'right'
  podCount: number
  health: number
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const color = health > 0.5 ? new THREE.Color(0x00ff66) : new THREE.Color(0xff5500)
  const xOffset = side === 'left' ? -7.2 : 7.2

  useFrame((state) => {
    if (!meshRef.current) return

    for (let i = 0; i < 10; i++) {
      const scale = i < podCount ? 1 : 0.01
      dummy.position.set(
        xOffset + (i % 5) * 1.45 - 2.9,
        -1 + Math.sin(state.clock.elapsedTime + i) * 0.05,
        Math.floor(i / 5) * 1.45 - 0.72
      )
      dummy.scale.setScalar(scale * 0.68)
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

function ComparisonParticles({ side, active }: { side: 'left' | 'right'; active: boolean }) {
  const pointsRef = useRef<THREE.Points>(null)
  const xOffset = side === 'left' ? -7.2 : 7.2
  const isHpa = side === 'left'

  const positions = useMemo(() => {
    const arr = new Float32Array(200 * 3)
    for (let i = 0; i < 200; i++) {
      arr[i * 3] = xOffset + (Math.random() - 0.5) * 6.6
      arr[i * 3 + 1] = Math.random() * 4 - 2
      arr[i * 3 + 2] = (Math.random() - 0.5) * 5
    }
    return arr
  }, [xOffset])

  useFrame((_, delta) => {
    if (!pointsRef.current || !active) return

    const pos = pointsRef.current.geometry.attributes.position.array as Float32Array
    for (let i = 0; i < 200; i++) {
      const idx = i * 3
      if (isHpa) {
        pos[idx] += (xOffset - pos[idx]) * 0.01
        pos[idx + 1] = Math.max(-1.5, pos[idx + 1] - delta * 0.32)
        if (pos[idx + 1] < -1.5) {
          pos[idx + 1] = 3
          pos[idx] = xOffset + (Math.random() - 0.5) * 6.6
        }
      } else {
        pos[idx] += (xOffset * 0.5 - pos[idx]) * 0.02
        pos[idx + 1] -= delta * 0.82
        if (pos[idx + 1] < -2) {
          pos[idx + 1] = 3
          pos[idx] = xOffset + (Math.random() - 0.5) * 6.6
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
        color={isHpa ? '#ff4400' : '#00ccff'}
        transparent
        opacity={active ? 0.75 : 0.2}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  )
}

function MetricDisplay({
  position,
  label,
  value,
  color,
}: {
  position: [number, number, number]
  label: string
  value: string
  color: string
}) {
  return (
    <group position={position}>
      <Text position={[0, 0.3, 0]} fontSize={0.29} color="#ffffff" anchorX="center" anchorY="middle">
        {label}
      </Text>
      <Text position={[0, -0.3, 0]} fontSize={0.5} color={color} anchorX="center" anchorY="middle">
        {value}
      </Text>
    </group>
  )
}

export default function ComparisonScene() {
  const { camera } = useThree()
  const { setAirshipState, setMetrics } = useSceneStore()

  useEffect(() => {
    setAirshipState('stable')
    setMetrics({
      failures: 21,
      pods: Math.round(FINAL_RESULTS.qmix.avgReplicasTotal),
      latencyMs: Math.round(FINAL_RESULTS.qmix.apiP99Ms),
      cpuPercent: 11,
      rps: 663,
    })
  }, [setAirshipState, setMetrics])

  useFrame(() => {
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, 0, 0.03)
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, 8, 0.03)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, 23, 0.03)
    camera.lookAt(0, 0, 0)
  })

  return (
    <>
      <ambientLight intensity={0.42} />
      <directionalLight position={[0, 10, 5]} intensity={1.2} />
      <pointLight position={[-7.2, 4, 0]} intensity={2.1} color="#ff4400" />
      <pointLight position={[7.2, 4, 0]} intensity={2.1} color="#00ccff" />

      <Text position={[0, 7, 0]} fontSize={0.6} color="#00ffff" anchorX="center" anchorY="middle">
        Performance Comparison
      </Text>
      <Text position={[0, 6.35, 0]} fontSize={0.22} color="#cceeff" anchorX="center" anchorY="middle">
        Source: docs/Final Results (combined_hpa.json, combined_qmix.json)
      </Text>

      <mesh position={[-7.2, 1, -1.2]}>
        <planeGeometry args={[7.8, 7.2]} />
        <meshBasicMaterial color="#ff4400" transparent opacity={0.06} />
      </mesh>
      <mesh position={[7.2, 1, -1.2]}>
        <planeGeometry args={[7.8, 7.2]} />
        <meshBasicMaterial color="#00ccff" transparent opacity={0.07} />
      </mesh>

      <Text position={[-7.2, 5, 0]} fontSize={0.46} color="#ff4400" anchorX="center" anchorY="middle">
        HPA
      </Text>
      <Text position={[7.2, 5, 0]} fontSize={0.46} color="#00ccff" anchorX="center" anchorY="middle">
        QMix
      </Text>

      <MetricDisplay
        position={[-7.2, 3.5, 0]}
        label="Avg Replicas (All Services)"
        value={FINAL_RESULTS.hpa.avgReplicasTotal.toFixed(2)}
        color="#ff4400"
      />

      <MetricDisplay
        position={[7.2, 3.5, 0]}
        label="Avg Replicas (All Services)"
        value={FINAL_RESULTS.qmix.avgReplicasTotal.toFixed(2)}
        color="#00ccff"
      />

      <MetricDisplay
        position={[-7.2, -2.45, 0]}
        label="CPU Requested (cores)"
        value={FINAL_RESULTS.hpa.cpuRequestedCores.toFixed(2)}
        color="#ff4400"
      />
      <MetricDisplay
        position={[7.2, -2.45, 0]}
        label="CPU Requested (cores)"
        value={FINAL_RESULTS.qmix.cpuRequestedCores.toFixed(2)}
        color="#00ccff"
      />

      <MetricDisplay
        position={[-7.2, 1.1, 0]}
        label="API p99 (ms)"
        value={FINAL_RESULTS.hpa.apiP99Ms.toFixed(2)}
        color="#ff4400"
      />
      <MetricDisplay
        position={[7.2, 1.1, 0]}
        label="API p99 (ms)"
        value={FINAL_RESULTS.qmix.apiP99Ms.toFixed(2)}
        color="#00ccff"
      />

      <Text position={[0, -0.95, 0]} fontSize={0.3} color="#9fe8ff" anchorX="center" anchorY="middle">
        QMix strengths: API p99 23.13ms vs HPA 99.87ms, with lower requested CPU (0.90 vs 2.00 cores)
      </Text>

      <MiniPodGrid side="left" podCount={7} health={0.35} />
      <ComparisonParticles side="left" active={true} />

      <MiniPodGrid side="right" podCount={6} health={1} />
      <ComparisonParticles side="right" active={true} />

      <mesh position={[0, -6, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[40, 30, 20, 15]} />
        <meshBasicMaterial color="#0088ff" transparent opacity={0.1} wireframe />
      </mesh>

      <mesh position={[-7.2, 0, 0]}>
        <boxGeometry args={[8.5, 10, 8]} />
        <meshBasicMaterial color="#ff4400" transparent opacity={0.06} wireframe />
      </mesh>

      <mesh position={[7.2, 0, 0]}>
        <boxGeometry args={[8.5, 10, 8]} />
        <meshBasicMaterial color="#00ccff" transparent opacity={0.06} wireframe />
      </mesh>
    </>
  )
}
