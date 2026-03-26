'use client'
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'

interface Props {
  active: boolean
  onDone?: () => void
}

export default function WhiteFlash({ active, onDone }: Props) {
  const meshRef = useRef<THREE.Mesh>(null)
  const matRef = useRef<THREE.MeshBasicMaterial>(null)
  const phase = useRef<'in' | 'hold' | 'out' | 'done'>('in')
  const timer = useRef(0)
  const { setFrozen } = useSceneStore()

  useFrame((_, delta) => {
    if (!matRef.current || !active) return

    timer.current += delta

    if (phase.current === 'in') {
      matRef.current.opacity = Math.min(1, timer.current / 0.12)
      if (matRef.current.opacity >= 1) {
        phase.current = 'hold'
        timer.current = 0
        setFrozen(true)
      }
    } else if (phase.current === 'hold') {
      if (timer.current > 0.22) {
        phase.current = 'out'
        timer.current = 0
        setFrozen(false)
      }
    } else if (phase.current === 'out') {
      matRef.current.opacity = Math.max(0, 1 - timer.current / 0.25)
      if (matRef.current.opacity <= 0) {
        phase.current = 'done'
        onDone?.()
      }
    }
  })

  if (!active) return null

  return (
    <mesh ref={meshRef} renderOrder={999}>
      <planeGeometry args={[100, 100]} />
      <meshBasicMaterial ref={matRef} color="white" transparent opacity={0} depthTest={false} />
    </mesh>
  )
}
