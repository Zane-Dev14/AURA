'use client'
import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'

const COUNT = 800

export default function TrafficParticles() {
  const pointsRef = useRef<THREE.Points>(null)
  const { trafficLevel, podHealth, frozen } = useSceneStore()

  const positions = useMemo(() => {
    const arr = new Float32Array(COUNT * 3)
    for (let i = 0; i < COUNT; i++) {
      arr[i * 3 + 0] = (Math.random() - 0.5) * 20
      arr[i * 3 + 1] = Math.random() * 6 - 1
      arr[i * 3 + 2] = (Math.random() - 0.5) * 20
    }
    return arr
  }, [])

  const velocities = useMemo(() => {
    const arr = new Float32Array(COUNT * 3)
    for (let i = 0; i < COUNT; i++) {
      // Bias toward center (service beam at 0,0,0)
      arr[i * 3 + 0] = (Math.random() - 0.5) * 0.02
      arr[i * 3 + 1] = (Math.random() - 0.5) * 0.01
      arr[i * 3 + 2] = (Math.random() - 0.5) * 0.02
    }
    return arr
  }, [])

  useFrame((_, delta) => {
    if (!pointsRef.current || frozen) return
    const pos = pointsRef.current.geometry.attributes.position.array as Float32Array
    const speed = trafficLevel * 4 + 0.5

    for (let i = 0; i < COUNT; i++) {
      const idx = i * 3
      // Move toward center
      const cx = pos[idx] * -0.02 * speed
      const cz = pos[idx + 2] * -0.02 * speed
      pos[idx] += (velocities[idx] + cx) * delta * speed
      pos[idx + 1] += velocities[idx + 1] * delta
      pos[idx + 2] += (velocities[idx + 2] + cz) * delta * speed

      // Wrap around
      const dist = Math.sqrt(pos[idx] ** 2 + pos[idx + 2] ** 2)
      if (dist < 0.5 || dist > 14) {
        pos[idx] = (Math.random() - 0.5) * 20
        pos[idx + 1] = Math.random() * 6 - 1
        pos[idx + 2] = (Math.random() - 0.5) * 20
      }
    }
    pointsRef.current.geometry.attributes.position.needsUpdate = true

    // Point size driven by traffic
    const mat = pointsRef.current.material as THREE.PointsMaterial
    mat.size = 0.06 + trafficLevel * 0.08
    mat.opacity = 0.4 + trafficLevel * 0.5

    // Color: healthy=cyan, failing=red
    mat.color = new THREE.Color().lerpColors(
      new THREE.Color(0xff3300),
      new THREE.Color(0x00eeff),
      podHealth
    )
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.06}
        color="#00eeff"
        transparent
        opacity={0.5}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  )
}
