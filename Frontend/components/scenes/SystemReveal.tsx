'use client'
import { useRef, useEffect, useMemo } from 'react'
import { useGLTF } from '@react-three/drei'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import ServiceBeam from '@/components/r3f/ServiceBeam'
import * as THREE from 'three'

useGLTF.preload('/models/heart_of_the_forest.glb')

export default function SystemReveal() {
  const { scene: swordGltf } = useGLTF('/models/heart_of_the_forest.glb')
  const swordRef = useRef<THREE.Group>(null)
  const { setPodCount, setAirshipState } = useSceneStore()

  // Clone, fit-to-size, and keep materials readable (avoid turning into a bloom blob)
  const swordFit = useMemo(() => {
    const cloned = swordGltf.clone(true)

    // Keep only upper meshes to avoid clipping artifacts while rendering top section only.
    const modelBox = new THREE.Box3().setFromObject(cloned)
    const modelSize = new THREE.Vector3()
    const modelCenter = new THREE.Vector3()
    modelBox.getSize(modelSize)
    modelBox.getCenter(modelCenter)
    const topThreshold = modelCenter.y + modelSize.y * 0.1

    cloned.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh
        const original = mesh.material as THREE.MeshStandardMaterial

        const meshBox = new THREE.Box3().setFromObject(mesh)
        if (meshBox.max.y < topThreshold) {
          mesh.visible = false
          return
        }

        // Some GLTFs can have arrays of materials; leave those alone for safety.
        if (Array.isArray(original)) return

        const mat = original?.clone?.() ?? new THREE.MeshStandardMaterial()
        mat.roughness = THREE.MathUtils.clamp(mat.roughness ?? 0.4, 0.15, 0.75)
        mat.metalness = THREE.MathUtils.clamp(mat.metalness ?? 0.2, 0, 1)
        mat.envMapIntensity = 1.8

        // Keep emissive subtle so bloom doesn’t wipe the model.
        if (mat.emissive) {
          mat.emissiveIntensity = Math.min(mat.emissiveIntensity ?? 0, 0.12)
        }

        mesh.castShadow = true
        mesh.receiveShadow = true
        mesh.material = mat
      }
    })

    // Fit the model to a stable world-space size and center it.
    const box = new THREE.Box3().setFromObject(cloned)
    const size = new THREE.Vector3()
    const center = new THREE.Vector3()
    box.getSize(size)
    box.getCenter(center)

    const maxDim = Math.max(size.x, size.y, size.z) || 1
    // Scene 2 should feel like a world the airship can explore.
    const targetSize = 24 // world units
    const scale = targetSize / maxDim
    const offset = center.multiplyScalar(-scale)

    return { object: cloned, scale, offset }
  }, [swordGltf])

  useEffect(() => {
    setPodCount(6)
    setAirshipState('patrol')
  }, [setPodCount, setAirshipState])

  // Intentionally no per-frame animation here: this model is the locked "world" for Scene 2.

  return (
    <>
      {/* Enhanced lighting for better visibility */}
      <ambientLight intensity={0.35} color="#102438" />
      <hemisphereLight args={['#2a5a8a', '#050b14', 0.45]} />
      
      {/* Key lights for the sword */}
      <pointLight position={[0, 8, 0]} intensity={2.2} color="#00d4ff" distance={25} castShadow />
      <pointLight position={[5, 5, 5]} intensity={1.4} color="#00aaff" distance={20} />
      <pointLight position={[-5, 5, 5]} intensity={1.4} color="#0088ff" distance={20} />
      
      {/* Directional light for depth */}
      <directionalLight
        position={[10, 10, 10]}
        intensity={1.4}
        color="#00e5ff"
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      
      {/* HEART OF THE FOREST SWORD - Central Landmark */}
      <group
        ref={swordRef}
        position={[0, 10, 0]}
      >
        <primitive
          object={swordFit.object}
          scale={swordFit.scale}
          position={swordFit.offset as any}
          rotation={[0, 0, 0]}
        />
        
        {/* Multiple point lights for dramatic effect */}
        <pointLight
          position={[0, 2, 0]}
          intensity={3.2}
          color="#00e5ff"
          distance={20}
          decay={2}
        />
        <pointLight
          position={[0, -1, 0]}
          intensity={1.6}
          color="#00aaff"
          distance={15}
          decay={2}
        />
        
        {/* Glowing ring around sword */}
        <mesh position={[0, 0, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[2, 2.5, 64]} />
          <meshStandardMaterial
            color="#062033"
            emissive="#00e5ff"
            emissiveIntensity={0.55}
            transparent
            opacity={0.28}
            side={THREE.DoubleSide}
            metalness={0.6}
            roughness={0.3}
          />
        </mesh>
      </group>
      
      {/* Atmospheric particles around the sword */}
      {Array.from({ length: 50 }).map((_, i) => {
        const angle = (i / 50) * Math.PI * 2
        const radius = 3 + Math.random() * 2
        const x = Math.cos(angle) * radius
        const z = Math.sin(angle) * radius
        const y = Math.random() * 6
        
        return (
          <mesh key={i} position={[x, y, z]}>
            <sphereGeometry args={[0.05, 8, 8]} />
            <meshBasicMaterial
              color="#00e5ff"
              transparent
              opacity={0.35}
            />
          </mesh>
        )
      })}
      
      <PodGrid />
      <ServiceBeam />
      <Airship />
    </>
  )
}
