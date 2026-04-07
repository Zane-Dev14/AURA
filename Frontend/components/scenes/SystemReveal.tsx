'use client'
import { useRef, useEffect, useMemo } from 'react'
import { Text, useGLTF } from '@react-three/drei'
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
    // Keep only the upper crown of the asset; bottom geometry is intentionally removed.
    const topThreshold = modelCenter.y + modelSize.y * 0.35

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
    // Make Scene 2 landmark bigger so the top section fills the playable area.
    const targetSize = 36 // world units
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

      {/* KTU dashboard panel anchored to the system camera snap target (north view) */}
      <group position={[8, 31.5, 0]} rotation={[0, -Math.PI / 2, 0]} scale={1.35}>
        <mesh>
          <planeGeometry args={[18, 10.5]} />
          <meshBasicMaterial color="#ffffff" side={THREE.DoubleSide} />
        </mesh>

        {/* Navbar */}
        <mesh position={[0, 4.65, 0.01]}>
          <planeGeometry args={[18, 1.1]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
        <Text
          position={[-7.9, 4.67, 0.02]}
          fontSize={0.2}
          color="#555555"
          anchorX="left"
          anchorY="middle"
          maxWidth={10}
        >
          e-Gov Platform for APJ Abdul Kalam Technological University
        </Text>
        <Text position={[4.7, 4.67, 0.02]} fontSize={0.19} color="#333333" anchorX="left" anchorY="middle">
          Home  Research  FAQ  Contact Us
        </Text>

        {/* Portal title strip */}
        <mesh position={[-2.1, 3.55, 0.01]}>
          <planeGeometry args={[11.6, 0.9]} />
          <meshBasicMaterial color="#f5f5f5" />
        </mesh>
        <Text
          position={[-7.55, 3.55, 0.02]}
          fontSize={0.18}
          color="#333333"
          anchorX="left"
          anchorY="middle"
          maxWidth={9.8}
        >
          APJ Abdul Kalam Technological University e-Governance Portal
        </Text>
        <Text position={[2.95, 3.55, 0.02]} fontSize={0.15} color="#428bca" anchorX="left" anchorY="middle">
          ktuapp4
        </Text>

        {/* Left column tiles */}
        {[2.05, 0.7, -0.65].map((y, i) => (
          <mesh key={`tile-${i}`} position={[-2.1, y, 0.01]}>
            <planeGeometry args={[11.6, 1.15]} />
            <meshBasicMaterial color="#ffffff" />
          </mesh>
        ))}

        {/* Icon circles */}
        <mesh position={[-7.2, 2.05, 0.02]}>
          <circleGeometry args={[0.26, 24]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
        <mesh position={[-7.2, 2.05, 0.021]}>
          <ringGeometry args={[0.22, 0.26, 24]} />
          <meshBasicMaterial color="#3ab0e2" />
        </mesh>

        <mesh position={[-7.2, 0.7, 0.02]}>
          <circleGeometry args={[0.26, 24]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
        <mesh position={[-7.2, 0.7, 0.021]}>
          <ringGeometry args={[0.22, 0.26, 24]} />
          <meshBasicMaterial color="#5cb85c" />
        </mesh>

        <mesh position={[-7.2, -0.65, 0.02]}>
          <circleGeometry args={[0.26, 24]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
        <mesh position={[-7.2, -0.65, 0.021]}>
          <ringGeometry args={[0.22, 0.26, 24]} />
          <meshBasicMaterial color="#ffcc00" />
        </mesh>

        {/* Section titles */}
        <Text position={[-6.7, 2.2, 0.02]} fontSize={0.26} color="#3ab0e2" anchorX="left" anchorY="middle">
          Institutions
        </Text>
        <Text position={[-6.7, 0.85, 0.02]} fontSize={0.26} color="#5cb85c" anchorX="left" anchorY="middle">
          Students
        </Text>
        <Text position={[-6.7, -0.5, 0.02]} fontSize={0.26} color="#ffcc00" anchorX="left" anchorY="middle">
          University
        </Text>

        {/* Section body previews */}
        <Text
          position={[-6.7, 1.84, 0.02]}
          fontSize={0.13}
          color="#666666"
          anchorX="left"
          anchorY="top"
          maxWidth={9.2}
          lineHeight={1.25}
        >
          Institutions affiliated/applying for affiliation can login here for affiliation,
          registration, academics and fee-related workflows.
        </Text>
        <Text
          position={[-6.7, 0.49, 0.02]}
          fontSize={0.13}
          color="#666666"
          anchorX="left"
          anchorY="top"
          maxWidth={9.2}
          lineHeight={1.25}
        >
          Registered students can access attendance, marks, grade sheets,
          and academic history from the student portal.
        </Text>
        <Text
          position={[-6.7, -0.86, 0.02]}
          fontSize={0.13}
          color="#666666"
          anchorX="left"
          anchorY="top"
          maxWidth={9.2}
          lineHeight={1.25}
        >
          University staff can manage programs, curriculum, calendars,
          clusters and communication with colleges.
        </Text>

        {/* Right sign-in panel */}
        <mesh position={[5.75, 1.2, 0.01]}>
          <planeGeometry args={[4.0, 5.2]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
        <mesh position={[5.75, 3.35, 0.02]}>
          <planeGeometry args={[4.0, 0.8]} />
          <meshBasicMaterial color="#f5f5f5" />
        </mesh>
        <Text position={[4.05, 3.35, 0.03]} fontSize={0.24} color="#333333" anchorX="left" anchorY="middle">
          Sign In
        </Text>

        <Text position={[4.05, 2.7, 0.03]} fontSize={0.14} color="#333333" anchorX="left" anchorY="middle">
          Username
        </Text>
        <mesh position={[5.75, 2.38, 0.02]}>
          <planeGeometry args={[3.2, 0.45]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
        <mesh position={[5.75, 2.38, 0.021]}>
          <lineSegments>
            <edgesGeometry args={[new THREE.PlaneGeometry(3.2, 0.45)]} />
            <lineBasicMaterial color="#dddddd" />
          </lineSegments>
        </mesh>

        <Text position={[4.05, 1.78, 0.03]} fontSize={0.14} color="#333333" anchorX="left" anchorY="middle">
          Password
        </Text>
        <mesh position={[5.75, 1.46, 0.02]}>
          <planeGeometry args={[3.2, 0.45]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
        <mesh position={[5.75, 1.46, 0.021]}>
          <lineSegments>
            <edgesGeometry args={[new THREE.PlaneGeometry(3.2, 0.45)]} />
            <lineBasicMaterial color="#dddddd" />
          </lineSegments>
        </mesh>

        <mesh position={[5.75, 0.55, 0.02]}>
          <planeGeometry args={[3.2, 0.62]} />
          <meshBasicMaterial color="#5cb85c" />
        </mesh>
        <Text position={[5.75, 0.55, 0.03]} fontSize={0.18} color="#ffffff" anchorX="center" anchorY="middle">
          Login
        </Text>
        <Text position={[5.75, -0.05, 0.03]} fontSize={0.12} color="#428bca" anchorX="center" anchorY="middle">
          Forgot password?
        </Text>

        {/* Footer */}
        <mesh position={[0, -4.45, 0.01]}>
          <planeGeometry args={[18, 0.95]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
        <Text
          position={[0, -4.45, 0.02]}
          fontSize={0.16}
          color="#777777"
          anchorX="center"
          anchorY="middle"
          maxWidth={16}
        >
          Copyright APJ Abdul Kalam Technological University 2014.
        </Text>

        <Text position={[0, -3.6, 0.03]} fontSize={0.62} color="#111111" anchorX="center" anchorY="middle">
          KTU SITE UP
        </Text>

        <pointLight position={[0, 0, 1.2]} intensity={1.25} color="#ffffff" distance={24} />
      </group>
      
      <PodGrid />
      <ServiceBeam />
      <Airship />
    </>
  )
}
