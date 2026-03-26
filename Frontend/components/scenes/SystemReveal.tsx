'use client'
import { useRef, useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import ServiceBeam from '@/components/r3f/ServiceBeam'
import * as THREE from 'three'

useGLTF.preload('/models/hyrule_castle_interior.glb')

export default function SystemReveal() {
  const { scene: castle } = useGLTF('/models/hyrule_castle_interior.glb')
  const { camera } = useThree()
  const t = useRef(0)
  const { setPodCount, setAirshipState } = useSceneStore()

  useEffect(() => {
    setPodCount(6)
    setAirshipState('patrol')
  }, [])

  useFrame((state, delta) => {
    t.current += delta * 0.4
    const progress = Math.min(1, t.current)
    // Camera dives through castle tunnel then arrives at cluster
    if (progress < 0.5) {
      const p = progress / 0.5
      camera.position.set(
        0,
        THREE.MathUtils.lerp(4, -2, p),
        THREE.MathUtils.lerp(12, 2, p)
      )
      camera.lookAt(0, 0, 0)
    } else {
      const p = (progress - 0.5) / 0.5
      camera.position.set(
        0,
        THREE.MathUtils.lerp(-2, 6, p),
        THREE.MathUtils.lerp(2, 14, p)
      )
      camera.lookAt(0, -1, 0)
    }
  })

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[0, 4, 0]} intensity={2} color="#00aaff" />
      <pointLight position={[0, -4, 0]} intensity={1} color="#0044ff" />
      {/* Castle tunnel used for transition feel */}
      <primitive
        object={castle}
        scale={0.015}
        position={[0, -3, -2]}
        visible={t.current < 0.55}
      />
      {/* Grid floor for cluster world */}
      <gridHelper args={[30, 30, '#001a3e', '#001a3e']} position={[0, -1.5, 0]} />
      <mesh position={[0, -1.6, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[40, 40]} />
        <meshStandardMaterial color="#000814" transparent opacity={0.95} />
      </mesh>
      <PodGrid />
      <ServiceBeam />
      <Airship />
    </>
  )
}
