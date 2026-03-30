'use client'
import { useEffect, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import ServiceBeam from '@/components/r3f/ServiceBeam'
import TrafficParticles from '@/components/r3f/TrafficParticles'
import * as THREE from 'three'
import { playRisingHum } from '@/lib/sound'

export default function TrafficScene() {
  const { camera } = useThree()
  const { setTrafficLevel, setPodCount, setAirshipState, audioUnlocked, setMetrics } = useSceneStore()
  const rampRef = useRef(0)

  useEffect(() => {
    setPodCount(6)
    setAirshipState('stressed')
    if (audioUnlocked) playRisingHum()
  }, [])

  useFrame((_, delta) => {
    // Camera pulls back to see full cluster
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, 0, 0.02)
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, 7, 0.02)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, 16, 0.02)
    camera.lookAt(0, -1, 0)

    // Gradually ramp traffic over ~8s
    rampRef.current = Math.min(1, rampRef.current + delta * 0.1)
    setTrafficLevel(rampRef.current)
    setMetrics({
      rps: Math.round(rampRef.current * 120),
      cpuPercent: Math.round(20 + rampRef.current * 62),
    })
  })

  return (
    <>
      <ambientLight intensity={0.25} />
      <pointLight position={[0, 4, 0]} intensity={2} color="#00aaff" />
      
      
      <PodGrid />
      <ServiceBeam />
      <TrafficParticles />
      <Airship />
    </>
  )
}
