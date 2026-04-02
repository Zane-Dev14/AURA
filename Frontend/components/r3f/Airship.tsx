'use client'
import { useRef, useMemo, useEffect } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'
import gsap from 'gsap'
import { getTimelineController } from '@/lib/timelineController'
import { floatAnimation, pulseGlow, EASINGS } from '@/lib/animationPresets'
import { createNoise3D } from 'simplex-noise'

useGLTF.preload('/models/small_spaceship.glb')

// Movement profiles for different states
const movementProfiles = {
  calm: { acceleration: 20, maxSpeed: 10, damping: 0.93, angularDamping: 0.90 },
  patrol: { acceleration: 18, maxSpeed: 9, damping: 0.93, angularDamping: 0.90 },
  stressed: { acceleration: 25, maxSpeed: 12, damping: 0.90, angularDamping: 0.88 },
  falling: { acceleration: 5, maxSpeed: 15, damping: 0.98, angularDamping: 0.95 },
  locked: { acceleration: 0, maxSpeed: 0, damping: 0.85, angularDamping: 0.85 },
  powered: { acceleration: 22, maxSpeed: 11, damping: 0.92, angularDamping: 0.89 },
  stable: { acceleration: 20, maxSpeed: 10, damping: 0.93, angularDamping: 0.90 },
}

export default function Airship() {
  const { scene: gltfScene } = useGLTF('/models/small_spaceship.glb')
  const groupRef = useRef<THREE.Group>(null)
  const { airshipState, frozen, assetsLoaded } = useSceneStore()
  const { camera } = useThree()
  
  const innerGroupRef = useRef<THREE.Group>(null)
  const arrowRef = useRef<THREE.Group>(null)
  
  // Physics state
  const velocity = useRef(new THREE.Vector3())
  const angularVelocity = useRef(new THREE.Euler())
  const currentInput = useRef(new THREE.Vector3())
  const keys = useRef({ w: false, s: false, a: false, d: false, q: false, e: false, ctrl: false })
  
  // Perlin noise for environmental effects
  const noise3D = useMemo(() => createNoise3D(), [])
  const noiseTime = useRef(0)
  
  // GSAP-based animation refs
  const stateTimeline = useRef<gsap.core.Timeline | null>(null)
  const floatTimeline = useRef<gsap.core.Timeline | null>(null)
  const glowTimeline = useRef<gsap.core.Timeline | null>(null)
  const physicsTimeline = useRef<gsap.core.Timeline | null>(null)

  // Apply premium showcase materials with enhanced emissive and clearcoat
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
          
          // Premium clearcoat for glossy finish (using any to bypass TS limitation)
          ;(mat as any).clearcoat = 0.6
          ;(mat as any).clearcoatRoughness = 0.1
          
          // Cyan emissive glow for dramatic hero effect
          mat.emissive = new THREE.Color(0x00e5ff)
          mat.emissiveIntensity = 0.2
          
          // Enable shadow casting
          mesh.castShadow = true
          mesh.receiveShadow = true
          
          mat.needsUpdate = true
        }
      }
    })
  }, [gltfScene])

  // Keyboard input handling
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase()
      if (key === 'w') keys.current.w = true
      if (key === 's') keys.current.s = true
      if (key === 'a') keys.current.a = true
      if (key === 'd') keys.current.d = true
      if (key === 'q') keys.current.q = true
      if (key === 'e') keys.current.e = true
      if (key === 'control') keys.current.ctrl = true
    }

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase()
      if (key === 'w') keys.current.w = false
      if (key === 's') keys.current.s = false
      if (key === 'a') keys.current.a = false
      if (key === 'd') keys.current.d = false
      if (key === 'q') keys.current.q = false
      if (key === 'e') keys.current.e = false
      if (key === 'control') keys.current.ctrl = false
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [])

  // Cinematic intro animation using GSAP
  useEffect(() => {
    if (assetsLoaded && groupRef.current) {
      const controller = getTimelineController()
      
      // Start ship slightly behind and lower
      groupRef.current.position.set(0, 0, -5)
      groupRef.current.rotation.y = Math.PI * 0.2

      // Create intro timeline
      const introTl = gsap.timeline()
      
      introTl.to(groupRef.current.position, {
        y: 2,
        z: 0,
        duration: 3,
        ease: EASINGS.SMOOTH_OUT,
      })
      
      introTl.to(groupRef.current.rotation, {
        y: 0,
        duration: 3.5,
        ease: EASINGS.SMOOTH_IN_OUT,
      }, 0)

      // Register with timeline controller
      controller.registerScene('airship-intro', introTl, {
        autoPlay: true,
        onComplete: () => {
          // Start floating animation after intro
          if (groupRef.current) {
            floatTimeline.current = floatAnimation(groupRef.current, {
              amplitude: 0.15,
              speed: 2,
              axis: 'y',
            })
          }
        },
      })

      return () => {
        controller.unregisterScene('airship-intro')
      }
    }
  }, [assetsLoaded])

  // Arrow animation using GSAP
  useEffect(() => {
    if (arrowRef.current) {
      const arrowGroup = arrowRef.current
      
      // Pulsing scale animation
      const pulseTl = gsap.timeline({ repeat: -1, yoyo: true })
      pulseTl.to(arrowGroup.scale, {
        x: 1.1,
        y: 1.1,
        z: 1.1,
        duration: 0.6,
        ease: EASINGS.SINE_IN_OUT,
      })

      // Bobbing animation
      const bobTl = gsap.timeline({ repeat: -1, yoyo: true })
      bobTl.to(arrowGroup.position, {
        y: 1.25,
        duration: 1,
        ease: EASINGS.SINE_IN_OUT,
      })

      // Glow pulse
      arrowGroup.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial
          if (mat && mat.emissive) {
            gsap.to(mat, {
              emissiveIntensity: 1.2,
              duration: 0.6,
              ease: EASINGS.SINE_IN_OUT,
              repeat: -1,
              yoyo: true,
            })
          }
        }
      })

      return () => {
        pulseTl.kill()
        bobTl.kill()
      }
    }
  }, [])

  // State-based animations using GSAP (for non-interactive states)
  useEffect(() => {
    if (!groupRef.current || frozen) return

    const group = groupRef.current
    
    // Kill existing state timeline
    if (stateTimeline.current) {
      stateTimeline.current.kill()
    }

    // Only use GSAP animations for locked states (non-interactive)
    if (airshipState === 'locked' || airshipState === 'powered' || airshipState === 'stable') {
      const tl = gsap.timeline()

      switch (airshipState) {
        case 'locked': {
          // Smooth lock into position
          tl.to(group.position, {
            x: 0,
            y: 2.5,
            z: 0,
            duration: 1.5,
            ease: EASINGS.SMOOTH_OUT,
          })

          tl.to(group.rotation, {
            y: -Math.PI * 0.1,
            z: 0,
            x: 0,
            duration: 1.5,
            ease: EASINGS.SMOOTH_OUT,
          }, 0)
          break
        }

        case 'powered': {
          // Rise up with power
          tl.to(group.position, {
            x: 0,
            y: 3,
            z: 0,
            duration: 2,
            ease: EASINGS.SMOOTH_OUT,
          })

          tl.to(group.rotation, {
            y: 0,
            z: 0,
            x: 0,
            duration: 2,
            ease: EASINGS.SMOOTH_OUT,
          }, 0)

          // Emissive intensity ramp
          gltfScene.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
              const mat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial
              if (mat) {
                tl.to(mat, {
                  emissiveIntensity: 2,
                  duration: 2,
                  ease: EASINGS.SMOOTH_OUT,
                }, 0)
              }
            }
          })
          break
        }

        case 'stable': {
          // Gentle stable hover
          tl.to(group.position, {
            x: 0,
            y: 2.1,
            z: 0,
            duration: 2,
            ease: EASINGS.SMOOTH_OUT,
          })

          tl.to(group.rotation, {
            z: 0,
            x: 0,
            duration: 2,
            ease: EASINGS.SMOOTH_OUT,
          }, 0)

          // Gentle bob
          tl.to(group.position, {
            y: '+=0.1',
            duration: 1.67,
            ease: EASINGS.SINE_IN_OUT,
            repeat: -1,
            yoyo: true,
          }, 2)

          // Change emissive to green
          gltfScene.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
              const mat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial
              if (mat) {
                const targetColor = new THREE.Color(0x00ff88)
                tl.to(mat.emissive, {
                  r: targetColor.r,
                  g: targetColor.g,
                  b: targetColor.b,
                  duration: 1,
                  ease: EASINGS.SMOOTH_OUT,
                }, 0)

                tl.to(mat, {
                  emissiveIntensity: 0.4,
                  duration: 1,
                  ease: EASINGS.SMOOTH_OUT,
                }, 0)
              }
            }
          })
          break
        }
      }

      stateTimeline.current = tl
    }

    return () => {
      if (stateTimeline.current) {
        stateTimeline.current.kill()
      }
    }
  }, [airshipState, frozen, gltfScene])

  // Natural drift animation for inner group (GSAP-based)
  useEffect(() => {
    if (innerGroupRef.current) {
      const ig = innerGroupRef.current

      // Natural drift animations
      const driftTl = gsap.timeline({ repeat: -1 })

      driftTl.to(ig.rotation, {
        z: 0.03,
        duration: 2,
        ease: EASINGS.SINE_IN_OUT,
        yoyo: true,
        repeat: -1,
      }, 0)

      driftTl.to(ig.rotation, {
        y: 0.02,
        duration: 2.8,
        ease: EASINGS.SINE_IN_OUT,
        yoyo: true,
        repeat: -1,
      }, 0)

      return () => {
        driftTl.kill()
      }
    }
  }, [])

  // Physics-based movement system (useFrame)
  useFrame((state, delta) => {
    if (!groupRef.current || frozen) return

    const group = groupRef.current
    const profile = movementProfiles[airshipState] || movementProfiles.calm

    // Skip physics for locked states (handled by GSAP)
    if (airshipState === 'locked' || airshipState === 'powered' || airshipState === 'stable') {
      return
    }

    // === INPUT HANDLING ===
    const targetInput = new THREE.Vector3()
    const inputSmoothing = 0.15

    // Get raw input
    if (keys.current.w) targetInput.z += 1
    if (keys.current.s) targetInput.z -= 1
    if (keys.current.a) targetInput.x -= 1
    if (keys.current.d) targetInput.x += 1
    if (keys.current.q) targetInput.y += 1
    if (keys.current.e) targetInput.y -= 1

    // Smooth input
    currentInput.current.lerp(targetInput, inputSmoothing)

    // === BOOST SYSTEM ===
    const boostMultiplier = keys.current.ctrl ? 2.0 : 1.0
    const boostDamping = keys.current.ctrl ? 0.96 : profile.damping

    // Visual feedback for boost
    if (keys.current.ctrl && 'fov' in camera) {
      const perspectiveCamera = camera as THREE.PerspectiveCamera
      perspectiveCamera.fov = THREE.MathUtils.lerp(perspectiveCamera.fov, 80, 0.1)
      perspectiveCamera.updateProjectionMatrix()
    } else if ('fov' in camera) {
      const perspectiveCamera = camera as THREE.PerspectiveCamera
      perspectiveCamera.fov = THREE.MathUtils.lerp(perspectiveCamera.fov, 75, 0.1)
      perspectiveCamera.updateProjectionMatrix()
    }

    // === PHYSICS SYSTEM ===
    // Apply input forces
    if (currentInput.current.length() > 0.01) {
      const force = currentInput.current.clone()
        .normalize()
        .multiplyScalar(profile.acceleration * boostMultiplier * delta)
      
      velocity.current.add(force)
    }

    // Clamp to max speed
    const currentSpeed = velocity.current.length()
    if (currentSpeed > profile.maxSpeed * boostMultiplier) {
      velocity.current.normalize().multiplyScalar(profile.maxSpeed * boostMultiplier)
    }

    // Apply damping (air resistance)
    velocity.current.multiplyScalar(boostDamping)

    // === ENVIRONMENTAL EFFECTS ===
    noiseTime.current += delta

    // Perlin noise for turbulence
    const noiseScale = airshipState === 'stressed' ? 0.08 : 0.02
    const noiseSpeed = airshipState === 'stressed' ? 2.0 : 0.5
    
    const noiseX = noise3D(noiseTime.current * noiseSpeed, 0, 0) * noiseScale
    const noiseY = noise3D(0, noiseTime.current * noiseSpeed, 0) * noiseScale * 0.75
    const noiseZ = noise3D(0, 0, noiseTime.current * noiseSpeed) * noiseScale

    // Add turbulence to velocity
    velocity.current.x += noiseX * delta * 10
    velocity.current.y += noiseY * delta * 10
    velocity.current.z += noiseZ * delta * 10

    // Random gusts for stressed state
    if (airshipState === 'stressed' && Math.random() < 0.01) {
      velocity.current.add(new THREE.Vector3(
        (Math.random() - 0.5) * 5,
        (Math.random() - 0.5) * 3,
        (Math.random() - 0.5) * 5
      ))
    }

    // Gentle floating for patrol state
    if (airshipState === 'patrol') {
      const floatAmount = Math.sin(state.clock.elapsedTime * 0.5) * 0.3
      group.position.y += floatAmount * delta
    }

    // Falling physics
    if (airshipState === 'falling') {
      velocity.current.y -= 9.8 * delta // Gravity
      
      // Shake effect
      group.position.x += (Math.random() - 0.5) * 0.1 * delta * 10
      group.position.z += (Math.random() - 0.5) * 0.1 * delta * 10
    }

    // === UPDATE POSITION ===
    group.position.add(velocity.current.clone().multiplyScalar(delta))

    // === SOFT COLLISION RESPONSE ===
    const bounds = { x: 50, y: 30, z: 50 }

    if (Math.abs(group.position.x) > bounds.x) {
      velocity.current.x *= -0.5
      group.position.x = Math.sign(group.position.x) * bounds.x
    }

    if (group.position.y > bounds.y) {
      velocity.current.y *= -0.5
      group.position.y = bounds.y
    } else if (group.position.y < 0.5) {
      velocity.current.y *= -0.5
      group.position.y = 0.5
    }

    if (Math.abs(group.position.z) > bounds.z) {
      velocity.current.z *= -0.5
      group.position.z = Math.sign(group.position.z) * bounds.z
    }

    // === ROTATION & TILTING ===
    // Only apply rotation if moving significantly
    if (currentSpeed > 0.5) {
      // Banking on turns (tilt based on horizontal velocity)
      const tiltAmount = velocity.current.x * 0.15
      group.rotation.z = THREE.MathUtils.lerp(group.rotation.z, tiltAmount, 0.1)

      // Pitch on vertical movement
      const pitchAmount = -velocity.current.y * 0.1
      group.rotation.x = THREE.MathUtils.lerp(group.rotation.x, pitchAmount, 0.1)

      // Yaw to face movement direction (only for horizontal movement)
      const horizontalVelocity = new THREE.Vector3(velocity.current.x, 0, velocity.current.z)
      if (horizontalVelocity.length() > 0.5) {
        const targetYaw = Math.atan2(velocity.current.x, velocity.current.z)
        group.rotation.y = THREE.MathUtils.lerp(group.rotation.y, targetYaw, 0.05)
      }
    } else {
      // Return to neutral rotation when stopped
      group.rotation.z = THREE.MathUtils.lerp(group.rotation.z, 0, 0.05)
      group.rotation.x = THREE.MathUtils.lerp(group.rotation.x, 0, 0.05)
    }

    // === STRESSED STATE ERRATIC MOVEMENT ===
    if (airshipState === 'stressed') {
      group.rotation.z += Math.sin(state.clock.elapsedTime * 10) * 0.02
    }

    // === FALLING STATE ROTATION ===
    if (airshipState === 'falling') {
      group.rotation.z += delta * 0.3
      group.rotation.x += delta * 0.2
    }
  })

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stateTimeline.current) stateTimeline.current.kill()
      if (floatTimeline.current) floatTimeline.current.kill()
      if (glowTimeline.current) glowTimeline.current.kill()
      if (physicsTimeline.current) physicsTimeline.current.kill()
      
      const controller = getTimelineController()
      controller.unregisterScene('airship-intro')
    }
  }, [])

  return (
    <group ref={groupRef} name="airship" position={[0, 2, 0]} scale={0.8}>
      <group ref={innerGroupRef}>
        <primitive object={gltfScene} />
        
        {/* Direction Indicator Arrow - Shows forward direction */}
        <group ref={arrowRef} position={[0, 1.2, 0]}>
          {/* Main arrow shaft */}
          <mesh position={[0, 0, -0.8]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.08, 0.08, 0.6, 8]} />
            <meshStandardMaterial
              color="#00e5ff"
              emissive="#00e5ff"
              emissiveIntensity={0.8}
              metalness={0.9}
              roughness={0.2}
            />
          </mesh>
          
          {/* Arrow head (cone) */}
          <mesh position={[0, 0, -1.2]} rotation={[Math.PI, 0, 0]}>
            <coneGeometry args={[0.2, 0.4, 8]} />
            <meshStandardMaterial
              color="#00e5ff"
              emissive="#00e5ff"
              emissiveIntensity={1.2}
              metalness={0.9}
              roughness={0.1}
            />
          </mesh>
          
          {/* Glow effect around arrow */}
          <pointLight
            position={[0, 0, -1]}
            intensity={2}
            color="#00e5ff"
            distance={3}
            decay={2}
          />
          
          {/* Pulsing ring at base */}
          <mesh position={[0, 0, -0.5]} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.15, 0.03, 8, 16]} />
            <meshStandardMaterial
              color="#00e5ff"
              emissive="#00e5ff"
              emissiveIntensity={0.6}
              transparent
              opacity={0.7}
            />
          </mesh>
        </group>
      </group>
    </group>
  )
}

// Made with Bob
