'use client'
import { useEffect, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import TrafficParticles from '@/components/r3f/TrafficParticles'
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
      {/* Darkened ground */}
      
      
      <PodGrid />
      <TrafficParticles />
      <Airship />
    </>
  )
}
