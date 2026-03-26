'use client'
import { useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import TrafficParticles from '@/components/r3f/TrafficParticles'
import HologramDialog from '@/components/ui/HologramDialog'
import * as THREE from 'three'

export default function QMixActivation() {
  const { camera } = useThree()
  const { setAirshipState, setGlitchIntensity } = useSceneStore()

  useEffect(() => {
    setAirshipState('locked')
    setGlitchIntensity(0.5)
  }, [])

  useFrame((_, delta) => {
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, 0, 0.02)
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, 6, 0.02)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, 13, 0.02)
    camera.lookAt(0, 0, 0)
  })

  return (
    <>
      <ambientLight intensity={0.15} color="#001133" />
      <pointLight position={[0, 4, 0]} intensity={1.5} color="#0044ff" />
      <pointLight position={[0, -2, 0]} intensity={0.8} color="#003388" />
      <mesh position={[0, -1.6, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[40, 40]} />
        <meshStandardMaterial color="#000008" transparent opacity={0.97} />
      </mesh>
      <gridHelper args={[30, 30, '#001133', '#000a1f']} position={[0, -1.5, 0]} />
      <PodGrid />
      <TrafficParticles />
      <Airship />
      {/* 3D hologram dialog */}
      <Html position={[0, 4, 2]} transform distanceFactor={8} style={{ pointerEvents: 'auto' }}>
        <HologramDialog />
      </Html>
    </>
  )
}
