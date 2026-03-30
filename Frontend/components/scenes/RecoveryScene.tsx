'use client'
import { useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import ServiceBeam from '@/components/r3f/ServiceBeam'
import TrafficParticles from '@/components/r3f/TrafficParticles'
import * as THREE from 'three'
import { playRecoveryChime } from '@/lib/sound'

export default function RecoveryScene() {
  const { camera } = useThree()
  const {
    setPodCount, setPodHealth, setAirshipState,
    setGlitchIntensity, setTrafficLevel, setMetrics, audioUnlocked,
  } = useSceneStore()

  useEffect(() => {
    setPodCount(9)
    setPodHealth(1)
    setAirshipState('stable')
    setGlitchIntensity(0)
    setTrafficLevel(0.3)
    setMetrics({ failures: 0, pods: 9, latencyMs: 5, cpuPercent: 66, rps: 430 })
    if (audioUnlocked) playRecoveryChime()
  }, [])

  useFrame((_, delta) => {
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, 0, 0.02)
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, 8, 0.02)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, 18, 0.02)
    camera.lookAt(0, -1, 0)
  })

  return (
    <>
      <ambientLight intensity={0.5} color="#002211" />
      <pointLight position={[0, 6, 0]} intensity={3} color="#00ff88" />
      <pointLight position={[0, -2, 8]} intensity={1.5} color="#00aaff" />
      <directionalLight position={[10, 10, 5]} intensity={1} color="#aaffdd" />
      
      
      <PodGrid />
      <ServiceBeam />
      <TrafficParticles />
      <Airship />
    </>
  )
}
