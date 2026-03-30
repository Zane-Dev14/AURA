'use client'
import { useRef, useMemo, useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'
import gsap from 'gsap'

useGLTF.preload('/models/small_spaceship.glb')

export default function Airship() {
  const { scene: gltfScene } = useGLTF('/models/small_spaceship.glb')
  const groupRef = useRef<THREE.Group>(null)
  const { airshipState, frozen, assetsLoaded } = useSceneStore()
  const { pointer, gl } = useThree()

  const innerGroupRef = useRef<THREE.Group>(null)
  const velocityRef = useRef(new THREE.Vector3())
  const baseY = useRef(2)
  const shakeOffset = useRef(new THREE.Vector3())
  const emissiveRef = useRef(0)
  const hoverPhase = useRef(0)
  const rotationVelocity = useRef(0)

  // Apply premium showcase materials with enhanced emissive
  useMemo(() => {
    gltfScene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh
        const mat = mesh.material as THREE.MeshStandardMaterial
        if (mat) {
          // Premium metallic finish for hero showcase
          mat.roughness = 0.15
          mat.metalness = 0.95
          mat.envMapIntensity = 3.0
          
          // Cyan emissive glow for dramatic hero effect
          mat.emissive = new THREE.Color(0x00e5ff)
          mat.emissiveIntensity = 0.2
        }
      }
    })
  }, [gltfScene])

  // Cinematic intro animation
  useEffect(() => {
    if (assetsLoaded && groupRef.current) {
      // Start ship slightly behind and lower
      groupRef.current.position.set(0, 0, -5)
      groupRef.current.rotation.y = Math.PI * 0.2

      // Smooth GSAP reveal
      gsap.to(groupRef.current.position, {
        y: 2,
        z: 0,
        duration: 3,
        ease: 'power2.out',
      })
      gsap.to(groupRef.current.rotation, {
        y: 0,
        duration: 3.5,
        ease: 'power2.inOut',
      })
    }
  }, [assetsLoaded])

  // Click to impulse forward
  useEffect(() => {
    const handleClick = () => {
      if (airshipState === 'patrol' && velocityRef.current) {
        // Add forward impulse
        velocityRef.current.z -= 0.5
        velocityRef.current.y += 0.2
      }
    }

    gl.domElement.addEventListener('click', handleClick)
    return () => gl.domElement.removeEventListener('click', handleClick)
  }, [airshipState, gl.domElement])

  useFrame((state, delta) => {
    if (!groupRef.current || frozen) return
    const t = state.clock.elapsedTime
    const g = groupRef.current

    // Apply velocity with damping (physics-like)
    if (velocityRef.current) {
      g.position.x += velocityRef.current.x * delta * 10
      g.position.y += velocityRef.current.y * delta * 10
      g.position.z += velocityRef.current.z * delta * 10

      // Damping
      velocityRef.current.multiplyScalar(0.95)

      // Return to base position
      const returnForce = 0.02
      velocityRef.current.x += (0 - g.position.x) * returnForce
      velocityRef.current.y += (baseY.current - g.position.y) * returnForce
      velocityRef.current.z += (0 - g.position.z) * returnForce
    }

    switch (airshipState) {
      case 'patrol': {
        // Gentle hover with multiple sine waves for organic feel
        hoverPhase.current += delta * 0.8
        const hover1 = Math.sin(hoverPhase.current) * 0.12
        const hover2 = Math.sin(hoverPhase.current * 1.7) * 0.05
        const hover3 = Math.cos(hoverPhase.current * 0.6) * 0.08
        g.position.y = baseY.current + hover1 + hover2 + hover3 + velocityRef.current.y

        // Slow natural rotation
        rotationVelocity.current += (Math.sin(t * 0.3) * 0.002 - rotationVelocity.current) * 0.05
        g.rotation.y += rotationVelocity.current

        // Subtle roll based on velocity
        g.rotation.z = velocityRef.current.x * 0.3
        break
      }
      case 'stressed': {
        g.position.y = baseY.current + Math.sin(t * 2) * 0.2 + Math.sin(t * 3.7) * 0.05
        g.position.x = Math.sin(t * 0.8) * 2 + Math.sin(t * 2.1) * 0.3
        g.rotation.z = Math.sin(t * 3) * 0.1
        break
      }
      case 'falling': {
        baseY.current = Math.max(-1, baseY.current - delta * 0.3)
        g.position.y = baseY.current + Math.sin(t * 1.5) * 0.1
        g.rotation.z = Math.sin(t * 0.5) * 0.3 + 0.2
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
        g.position.x = THREE.MathUtils.lerp(g.position.x, 0, 0.05)
        g.position.y = THREE.MathUtils.lerp(g.position.y, 2.5, 0.05)
        g.rotation.y = THREE.MathUtils.lerp(g.rotation.y, -Math.PI * 0.1, 0.05)
        g.rotation.z = THREE.MathUtils.lerp(g.rotation.z, 0, 0.05)
        break
      }
      case 'powered': {
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

    // Mouse tilt - responsive and physical
    if (innerGroupRef.current) {
      const ig = innerGroupRef.current
      
      // Mouse influence with smooth interpolation
      const mouseInfluence = airshipState === 'patrol' ? 1 : 0.3
      const targetTiltZ = -pointer.x * 0.25 * mouseInfluence
      const targetTiltX = pointer.y * 0.15 * mouseInfluence
      
      // Natural drift
      const naturalDriftY = Math.sin(t * 0.5) * 0.03
      const naturalDriftX = Math.cos(t * 0.7) * 0.02

      // Smooth interpolation with inertia
      ig.rotation.z = THREE.MathUtils.lerp(ig.rotation.z, targetTiltZ + naturalDriftX, 0.08)
      ig.rotation.x = THREE.MathUtils.lerp(ig.rotation.x, targetTiltX, 0.08)
      ig.rotation.y = THREE.MathUtils.lerp(ig.rotation.y, naturalDriftY, 0.05)

      // Subtle position offset based on mouse (parallax within ship)
      const targetOffsetX = pointer.x * 0.3 * mouseInfluence
      const targetOffsetY = pointer.y * 0.15 * mouseInfluence
      ig.position.x = THREE.MathUtils.lerp(ig.position.x, targetOffsetX, 0.06)
      ig.position.y = THREE.MathUtils.lerp(ig.position.y, targetOffsetY, 0.06)
    }
  })

  useMemo(() => {
    if (airshipState === 'patrol' || airshipState === 'stable') {
      baseY.current = 2
    }
  }, [airshipState])

  return (
    <group ref={groupRef} position={[0, 2, 0]} scale={0.8}>
      <group ref={innerGroupRef}>
        <primitive object={gltfScene} />
      </group>
    </group>
  )
}

// Made with Bob
