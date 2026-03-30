'use client'
import { useEffect, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import ServiceBeam from '@/components/r3f/ServiceBeam'
import TrafficParticles from '@/components/r3f/TrafficParticles'
import * as THREE from 'three'
import { playGlitchBurst } from '@/lib/sound'

export default function FailureScene() {
  const { camera } = useThree()
  const {
    setGlitchIntensity, setPodHealth, setPodCount,
    setAirshipState, audioUnlocked, setMetrics,
  } = useSceneStore()
  const shakeTimer = useRef(0)

  useEffect(() => {
    setPodCount(2)
    setPodHealth(0.1)
    setAirshipState('falling')
    if (audioUnlocked) playGlitchBurst()
    setMetrics({ failures: 47, latencyMs: 4200, cpuPercent: 116 })
  }, [])

  useFrame((_, delta) => {
    shakeTimer.current += delta
    // Camera shake
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, 0, 0.01) + (Math.random() - 0.5) * 0.06
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, 7, 0.01) + (Math.random() - 0.5) * 0.03
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, 16, 0.01)
    camera.lookAt(0, -1, 0)
    // Glitch fades in over 3s
    const g = Math.min(1, shakeTimer.current / 3)
    setGlitchIntensity(g)
  })

  return (
    <>
      <ambientLight intensity={0.15} color="#220000" />
      <pointLight position={[0, 4, 0]} intensity={1.5} color="#ff2200" />
      <pointLight position={[0, 8, 4]} intensity={0.8} color="#440000" />
      
      
      <PodGrid />
      <ServiceBeam />
      <TrafficParticles />
      <Airship />
    </>
  )
}
