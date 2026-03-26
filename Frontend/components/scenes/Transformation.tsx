'use client'
import { useEffect, useRef, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import ServiceBeam from '@/components/r3f/ServiceBeam'
import TrafficParticles from '@/components/r3f/TrafficParticles'
import ShockwaveRing from '@/components/r3f/ShockwaveRing'
import AgentBeam from '@/components/r3f/AgentBeam'
import WhiteFlash from '@/components/r3f/WhiteFlash'
import * as THREE from 'three'
import { playBassDrop } from '@/lib/sound'

export default function Transformation() {
  const { camera } = useThree()
  const {
    setPodCount, setPodHealth, setAirshipState,
    setGlitchIntensity, audioUnlocked, setMetrics,
    setQmixStatusMsg,
  } = useSceneStore()
  const phase = useRef(0) // 0=beam 1=flash 2=shockwave 3=recovering
  const timer = useRef(0)
  const [showBeam, setShowBeam] = useState(false)
  const [showFlash, setShowFlash] = useState(false)
  const [showShockwave, setShowShockwave] = useState(false)
  const targetPods = 12

  const STATUS_MSGS = [
    'Initializing intelligent scaling agent...',
    'Analyzing traffic patterns...',
    'Predicting load spike in T-5s...',
    'Deploying optimized scaling strategy...',
    'System stabilized. SLO met. ✓',
  ]

  useEffect(() => {
    setAirshipState('powered')
    setGlitchIntensity(0.2)
    // Phase 0: show beam
    setTimeout(() => setShowBeam(true), 0)
    // Phase 1: white flash
    setTimeout(() => setShowFlash(true), 300)
    // Phase 2: shockwave after freeze ends
    setTimeout(() => {
      setShowShockwave(true)
      if (audioUnlocked) playBassDrop()
    }, 600)
    // Pod cascade
    let count = 2
    const podInterval = setInterval(() => {
      count += 3
      if (count >= targetPods) { clearInterval(podInterval); count = targetPods }
      setPodCount(count)
      setPodHealth(count / targetPods)
    }, 200)
    // Status messages
    STATUS_MSGS.forEach((msg, i) => setTimeout(() => setQmixStatusMsg(msg), i * 1200))
    setMetrics({ failures: 30, pods: 5, latencyMs: 800, cpuPercent: 66 })
    return () => clearInterval(podInterval)
  }, [])

  useFrame((_, delta) => {
    timer.current += delta
    // Rapid zoom in then pull out
    const camZ = timer.current < 0.6
      ? THREE.MathUtils.lerp(camera.position.z, 6, 0.15)
      : THREE.MathUtils.lerp(camera.position.z, 18, 0.03)
    camera.position.z = camZ
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, 8, 0.03)
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, 0, 0.05)
    camera.lookAt(0, 0, 0)
  })

  return (
    <>
      <ambientLight intensity={0.3} color="#001133" />
      <pointLight position={[0, 6, 0]} intensity={4} color="#00f0ff" />
      <pointLight position={[0, -2, 0]} intensity={2} color="#0099ff" />
      <mesh position={[0, -1.6, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[40, 40]} />
        <meshStandardMaterial color="#000510" transparent opacity={0.97} />
      </mesh>
      <gridHelper args={[30, 30, '#001a3e', '#000d20']} position={[0, -1.5, 0]} />
      <PodGrid />
      <ServiceBeam />
      <TrafficParticles />
      <Airship />
      <AgentBeam
        active={showBeam}
        from={new THREE.Vector3(0, 3, 0)}
        to={new THREE.Vector3(0, -1, 0)}
      />
      <WhiteFlash active={showFlash} onDone={() => {}} />
      <ShockwaveRing active={showShockwave} />
    </>
  )
}
