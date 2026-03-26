'use client'
import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'

useGLTF.preload('/models/small_spaceship.glb')

export default function Airship() {
  const { scene: gltfScene } = useGLTF('/models/small_spaceship.glb')
  const groupRef = useRef<THREE.Group>(null)
  const { airshipState, frozen } = useSceneStore()

  const baseY = useRef(2)
  const shakeOffset = useRef(new THREE.Vector3())
  const emissiveRef = useRef(0)

  // Apply emissive to all materials when powered
  useMemo(() => {
    gltfScene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh
        const mat = mesh.material as THREE.MeshStandardMaterial
        if (mat && mat.type === 'MeshStandardMaterial') {
          mat.emissive = new THREE.Color(0x00f0ff)
          mat.emissiveIntensity = 0
        }
      }
    })
  }, [gltfScene])

  useFrame((state, delta) => {
    if (!groupRef.current || frozen) return
    const t = state.clock.elapsedTime
    const g = groupRef.current

    switch (airshipState) {
      case 'patrol': {
        // Gentle bob and slow patrol orbit
        g.position.y = baseY.current + Math.sin(t * 0.8) * 0.15
        g.position.x = Math.sin(t * 0.2) * 3
        g.rotation.y = Math.sin(t * 0.2) * 0.3
        g.rotation.z = Math.sin(t * 0.4) * 0.05
        break
      }
      case 'stressed': {
        // Faster, more erratic
        g.position.y = baseY.current + Math.sin(t * 2) * 0.2 + Math.sin(t * 3.7) * 0.05
        g.position.x = Math.sin(t * 0.8) * 2 + Math.sin(t * 2.1) * 0.3
        g.rotation.z = Math.sin(t * 3) * 0.1
        break
      }
      case 'falling': {
        // Drift and tilt downward
        baseY.current = Math.max(-1, baseY.current - delta * 0.3)
        g.position.y = baseY.current + Math.sin(t * 1.5) * 0.1
        g.rotation.z = Math.sin(t * 0.5) * 0.3 + 0.2
        // Random shake
        shakeOffset.current.set(
          (Math.random() - 0.5) * 0.1,
          (Math.random() - 0.5) * 0.05,
          0
        )
        g.position.x += shakeOffset.current.x
        g.position.y += shakeOffset.current.y
        break
      }
      case 'locked': {
        // Lock onto center, aiming forward
        g.position.x = THREE.MathUtils.lerp(g.position.x, 0, 0.05)
        g.position.y = THREE.MathUtils.lerp(g.position.y, 2.5, 0.05)
        g.rotation.y = THREE.MathUtils.lerp(g.rotation.y, -Math.PI * 0.1, 0.05)
        g.rotation.z = THREE.MathUtils.lerp(g.rotation.z, 0, 0.05)
        break
      }
      case 'powered': {
        // Emissive ramp + lock
        emissiveRef.current = Math.min(2, emissiveRef.current + delta * 3)
        g.position.x = THREE.MathUtils.lerp(g.position.x, 0, 0.08)
        g.position.y = THREE.MathUtils.lerp(g.position.y, 3, 0.08)
        g.rotation.y = 0
        gltfScene.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial
            if (mat) mat.emissiveIntensity = emissiveRef.current
          }
        })
        break
      }
      case 'stable': {
        // Hover centered, green glow
        emissiveRef.current = Math.max(0, emissiveRef.current - delta * 0.5)
        g.position.x = THREE.MathUtils.lerp(g.position.x, 0, 0.03)
        g.position.y = baseY.current + Math.sin(t * 0.6) * 0.1
        g.rotation.z = THREE.MathUtils.lerp(g.rotation.z, 0, 0.03)
        gltfScene.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial
            if (mat) {
              mat.emissive = new THREE.Color(0x00ff88)
              mat.emissiveIntensity = 0.4
            }
          }
        })
        break
      }
    }
  })

  // Reset baseY when state changes to patrol/stable
  useMemo(() => {
    if (airshipState === 'patrol' || airshipState === 'stable') {
      baseY.current = 2
    }
  }, [airshipState])

  return (
    <group ref={groupRef} position={[0, 2, 0]} scale={0.8}>
      <primitive object={gltfScene} />
    </group>
  )
}
