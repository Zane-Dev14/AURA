'use client'
import { useEffect, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import TrafficParticles from '@/components/r3f/TrafficParticles'
import { Text } from '@react-three/drei'
import * as THREE from 'three'

export default function EmotionalBeat() {
  const { camera } = useThree()
  const { setPodCount, setPodHealth, setAirshipState, setGlitchIntensity, setMetrics } = useSceneStore()

  useEffect(() => {
    setPodCount(1)
    setPodHealth(0)
    setAirshipState('falling')
    setGlitchIntensity(1)
    setMetrics({ failures: 120, pods: 1, latencyMs: 9000, cpuPercent: 116 })
  }, [])

  useFrame((_, delta) => {
    // Slow pull back, slightly tilted
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, 1.5, 0.02)
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, 5, 0.02)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, 14, 0.02)
    camera.lookAt(0, -1, 0)
  })

  return (
    <>
      <ambientLight intensity={0.1} color="#0000aa" />
      <pointLight position={[0, 4, 0]} intensity={0.8} color="#4400ff" />
      <pointLight position={[-5, 2, -3]} intensity={0.5} color="#220066" />

      {/* Critical failure board modeled after error.html with nginx 503 focus */}
      <group position={[0, 4.2, -8]}>
        <mesh>
          <planeGeometry args={[13.5, 8]} />
          <meshBasicMaterial color="#ffffff" side={THREE.DoubleSide} />
        </mesh>

        <mesh position={[0, 3.45, 0.01]}>
          <planeGeometry args={[13.5, 0.9]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>

        <Text position={[-6.2, 3.45, 0.02]} fontSize={0.16} color="#555555" anchorX="left" anchorY="middle" maxWidth={7.8}>
          e-Gov Platform for APJ Abdul Kalam Technological University
        </Text>
        <Text position={[2.8, 3.45, 0.02]} fontSize={0.15} color="#333333" anchorX="left" anchorY="middle">
          Home  Research  FAQ  Contact Us
        </Text>

        <mesh position={[0, -0.15, 0.01]}>
          <planeGeometry args={[12, 5.8]} />
          <meshBasicMaterial color="#f7f9fc" />
        </mesh>

        <Text position={[0, 0.8, 0.02]} fontSize={1.45} color="#1f3c88" anchorX="center" anchorY="middle">
          503
        </Text>
        <Text position={[0, -0.45, 0.02]} fontSize={0.42} color="#333333" anchorX="center" anchorY="middle">
          nginx 503 Service Temporarily Unavailable
        </Text>
        <Text position={[0, -1.05, 0.02]} fontSize={0.22} color="#666666" anchorX="center" anchorY="middle">
          Critical Failure: backend unavailable
        </Text>

        <mesh position={[0, -3.45, 0.01]}>
          <planeGeometry args={[13.5, 0.9]} />
          <meshBasicMaterial color="#f8f8f8" />
        </mesh>
        <Text position={[0, -3.45, 0.02]} fontSize={0.14} color="#777777" anchorX="center" anchorY="middle" maxWidth={12}>
          Copyright APJ Abdul Kalam Technological University 2014.
        </Text>
      </group>

      <PodGrid />
      <TrafficParticles />
      <Airship />
    </>
  )
}
