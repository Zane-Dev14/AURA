'use client'
import { useRef, useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import * as THREE from 'three'
import gsap from 'gsap'

export default function CalmWorld() {
  const { camera } = useThree()
  const { assetsLoaded } = useSceneStore()
  const lightRef = useRef<THREE.PointLight>(null)
  const spotlightRef = useRef<THREE.SpotLight>(null)
  const cameraRigRef = useRef({ phase: 0, radius: 15, height: 5 })

  // Cinematic camera intro - dramatic reveal
  useEffect(() => {
    if (assetsLoaded) {
      // Start with dramatic high angle
      camera.position.set(-20, 12, 20)
      camera.lookAt(0, 2, 0)

      // Create cinematic camera movement sequence
      const timeline = gsap.timeline()
      
      // Phase 1: Dramatic sweep in
      timeline.to(camera.position, {
        x: -8,
        y: 6,
        z: 15,
        duration: 4,
        ease: 'power2.inOut',
      })
      
      // Phase 2: Circle around to hero angle
      timeline.to(camera.position, {
        x: 5,
        y: 4,
        z: 12,
        duration: 3.5,
        ease: 'power1.inOut',
      }, '-=1')
      
      // Phase 3: Settle into showcase position
      timeline.to(camera.position, {
        x: 0,
        y: 5,
        z: 14,
        duration: 2.5,
        ease: 'power2.out',
      })
    }
  }, [assetsLoaded, camera])

  // Dynamic camera orbit for showcase feel
  useFrame((state, delta) => {
    const t = state.clock.elapsedTime

    // After intro, gentle orbital movement
    if (t > 10) {
      cameraRigRef.current.phase += delta * 0.08
      
      // Smooth orbital path
      const orbitX = Math.sin(cameraRigRef.current.phase) * cameraRigRef.current.radius
      const orbitZ = Math.cos(cameraRigRef.current.phase) * cameraRigRef.current.radius
      const orbitY = cameraRigRef.current.height + Math.sin(cameraRigRef.current.phase * 0.5) * 1.5
      
      // Smooth interpolation to orbital position
      camera.position.x += (orbitX - camera.position.x) * 0.01
      camera.position.y += (orbitY - camera.position.y) * 0.01
      camera.position.z += (orbitZ - camera.position.z) * 0.01
    }

    // Always look at airship with slight offset for drama
    const lookTarget = new THREE.Vector3(
      Math.sin(t * 0.2) * 0.5,
      2.5 + Math.cos(t * 0.3) * 0.3,
      0
    )
    camera.lookAt(lookTarget)

    // Animate key light for drama
    if (lightRef.current) {
      lightRef.current.intensity = 2.5 + Math.sin(t * 0.5) * 0.3
      lightRef.current.position.x = 8 + Math.sin(t * 0.3) * 2
      lightRef.current.position.z = 10 + Math.cos(t * 0.4) * 2
    }

    // Animate spotlight for hero lighting
    if (spotlightRef.current) {
      spotlightRef.current.intensity = 3 + Math.sin(t * 0.7) * 0.5
    }
  })

  return (
    <>
      {/* HERO LIGHTING SETUP */}
      
      {/* Ambient base - dark and moody */}
      <ambientLight intensity={0.15} color="#0a1428" />
      
      {/* Hemisphere for subtle sky/ground gradient */}
      <hemisphereLight
        args={['#1a4a7a', '#050a15', 0.4]}
      />
      
      {/* KEY LIGHT - Main hero light from front-right */}
      <pointLight
        ref={lightRef}
        position={[8, 8, 10]}
        intensity={2.5}
        color="#00d4ff"
        distance={40}
        decay={2}
        castShadow
      />
      
      {/* SPOTLIGHT - Dramatic top-down hero spotlight */}
      <spotLight
        ref={spotlightRef}
        position={[0, 15, 5]}
        angle={0.4}
        penumbra={0.5}
        intensity={3}
        color="#00e5ff"
        distance={30}
        decay={2}
        castShadow
        target-position={[0, 2, 0]}
      />
      
      {/* RIM LIGHT - Edge definition from behind */}
      <directionalLight
        position={[-10, 8, -12]}
        intensity={1.2}
        color="#4a9aff"
      />
      
      {/* ACCENT LIGHT - Cyan accent from left */}
      <pointLight
        position={[-12, 5, 8]}
        intensity={1.5}
        color="#00ffaa"
        distance={35}
        decay={2}
      />
      
      {/* FILL LIGHT - Soft fill from below */}
      <pointLight
        position={[0, -2, 10]}
        intensity={0.8}
        color="#0066ff"
        distance={25}
        decay={2}
      />

      {/* THE HERO - AIRSHIP */}
      <Airship />

      {/* ATMOSPHERIC ELEMENTS */}
      
      {/* Volumetric light shafts effect */}
      <mesh position={[0, 8, -5]} rotation={[Math.PI / 6, 0, 0]}>
        <planeGeometry args={[30, 20]} />
        <meshBasicMaterial
          color="#00e5ff"
          transparent
          opacity={0.02}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Floating light particles for depth */}
      {Array.from({ length: 50 }).map((_, i) => {
        const angle = (i / 50) * Math.PI * 2
        const radius = 15 + Math.random() * 10
        const x = Math.cos(angle) * radius
        const z = Math.sin(angle) * radius
        const y = -5 + Math.random() * 20
        
        return (
          <mesh key={i} position={[x, y, z]}>
            <sphereGeometry args={[0.05 + Math.random() * 0.1, 8, 8]} />
            <meshBasicMaterial
              color="#00e5ff"
              transparent
              opacity={0.3 + Math.random() * 0.4}
            />
          </mesh>
        )
      })}

      {/* Depth fog layers - creates atmosphere */}
      <mesh position={[0, -3, -15]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[80, 80]} />
        <meshBasicMaterial
          color="#001a3a"
          transparent
          opacity={0.15}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      <mesh position={[0, -5, -25]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[120, 120]} />
        <meshBasicMaterial
          color="#000a1a"
          transparent
          opacity={0.25}
          depthWrite={false}
        />
      </mesh>

      <mesh position={[0, -8, -40]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[180, 180]} />
        <meshBasicMaterial
          color="#000510"
          transparent
          opacity={0.4}
          depthWrite={false}
        />
      </mesh>

      {/* Circular platform/stage for hero */}
      <mesh position={[0, -0.5, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <circleGeometry args={[8, 64]} />
        <meshStandardMaterial
          color="#001a3a"
          transparent
          opacity={0.3}
          roughness={0.8}
          metalness={0.2}
          emissive="#00e5ff"
          emissiveIntensity={0.05}
        />
      </mesh>

      {/* Glowing ring around platform */}
      <mesh position={[0, -0.4, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[7.8, 8.2, 64]} />
        <meshBasicMaterial
          color="#00e5ff"
          transparent
          opacity={0.4}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Vertical light beams for drama */}
      {[0, 120, 240].map((angle, i) => {
        const rad = (angle * Math.PI) / 180
        const x = Math.cos(rad) * 12
        const z = Math.sin(rad) * 12
        
        return (
          <mesh key={i} position={[x, 5, z]} rotation={[0, 0, 0]}>
            <cylinderGeometry args={[0.1, 0.1, 20, 8]} />
            <meshBasicMaterial
              color="#00e5ff"
              transparent
              opacity={0.1}
              blending={THREE.AdditiveBlending}
            />
          </mesh>
        )
      })}

      {/* Holographic grid floor */}
      <mesh position={[0, -1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[50, 50, 20, 20]} />
        <meshBasicMaterial
          color="#00e5ff"
          transparent
          opacity={0.05}
          wireframe
        />
      </mesh>
    </>
  )
}

// Made with Bob
