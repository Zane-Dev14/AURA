'use client'
import { useRef, useMemo, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'
import gsap from 'gsap'
import { EASINGS } from '@/lib/animationPresets'
import { createNoise3D } from 'simplex-noise'

useGLTF.preload('/models/small_spaceship.glb')

// Movement profiles for different states
const movementProfiles = {
  calm: { acceleration: 28, maxSpeed: 16, damping: 0.95, angularDamping: 0.90 },
  patrol: { acceleration: 26, maxSpeed: 15, damping: 0.95, angularDamping: 0.90 },
  stressed: { acceleration: 25, maxSpeed: 12, damping: 0.90, angularDamping: 0.88 },
  falling: { acceleration: 5, maxSpeed: 15, damping: 0.98, angularDamping: 0.95 },
  locked: { acceleration: 0, maxSpeed: 0, damping: 0.85, angularDamping: 0.85 },
  powered: { acceleration: 22, maxSpeed: 11, damping: 0.92, angularDamping: 0.89 },
  stable: { acceleration: 20, maxSpeed: 10, damping: 0.93, angularDamping: 0.90 },
}

export default function Airship() {
  const { scene: gltfScene } = useGLTF('/models/small_spaceship.glb')
  const groupRef = useRef<THREE.Group>(null)
  const { airshipState, frozen, assetsLoaded, scene, introComplete } = useSceneStore()
  
  const innerGroupRef = useRef<THREE.Group>(null)
  const arrowRef = useRef<THREE.Group>(null)
  
  // Physics state
  const velocity = useRef(new THREE.Vector3())
  const angularVelocity = useRef(new THREE.Euler())
  const currentInput = useRef(new THREE.Vector3())
  const rotationInput = useRef(0)
  const keys = useRef({
    w: false, s: false, a: false, d: false,
    space: false, shift: false, ctrl: false
  })
  
  // Perlin noise for environmental effects
  const noise3D = useMemo(() => createNoise3D(), [])
  const noiseTime = useRef(0)
  
  // GSAP-based animation refs
  const stateTimeline = useRef<gsap.core.Timeline | null>(null)
  const floatTimeline = useRef<gsap.core.Timeline | null>(null)
  const glowTimeline = useRef<gsap.core.Timeline | null>(null)
  const physicsTimeline = useRef<gsap.core.Timeline | null>(null)

  const SYSTEM_SPAWN = useMemo(() => new THREE.Vector3(0, 30.5, 0), [])
  const SYSTEM_FOCUS = useMemo(() => new THREE.Vector3(0, 30.5, -18), [])

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

      if (e.code === 'Space') {
        e.preventDefault()
        keys.current.space = true
      }

      if (key === ' ' || key === 'spacebar') {
        e.preventDefault()
        keys.current.space = true
      }

      if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') {
        e.preventDefault()
        keys.current.shift = true
      }

      if (key === 'shift') {
        e.preventDefault()
        keys.current.shift = true
      }

      if (e.code === 'ControlLeft' || e.code === 'ControlRight') {
        keys.current.ctrl = true
      }
    }

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase()

      if (key === 'w') keys.current.w = false
      if (key === 's') keys.current.s = false
      if (key === 'a') keys.current.a = false
      if (key === 'd') keys.current.d = false

      if (e.code === 'Space') keys.current.space = false
      if (key === ' ' || key === 'spacebar') keys.current.space = false

      if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') {
        keys.current.shift = false
      }

      if (key === 'shift') {
        keys.current.shift = false
      }

      if (e.code === 'ControlLeft' || e.code === 'ControlRight') {
        keys.current.ctrl = false
      }
    }

    const resetKeys = () => {
      keys.current = {
        w: false,
        s: false,
        a: false,
        d: false,
        space: false,
        shift: false,
        ctrl: false,
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    window.addEventListener('blur', resetKeys)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      window.removeEventListener('blur', resetKeys)
    }
  }, [])

  // Scene 2 spawn point: place airship above the world so exploration starts at the top.
  useEffect(() => {
    if (!groupRef.current || scene !== 'system') return

    groupRef.current.position.copy(SYSTEM_SPAWN)
    groupRef.current.lookAt(SYSTEM_FOCUS)
    groupRef.current.rotation.x = 0
    groupRef.current.rotation.z = 0
    velocity.current.set(0, 0, 0)
    currentInput.current.set(0, 0, 0)
    rotationInput.current = 0
  }, [scene, SYSTEM_SPAWN, SYSTEM_FOCUS])

  // Avoid timeline-vs-physics conflicts in controllable exploration scenes.
  useEffect(() => {
    if (!floatTimeline.current) return
    if (scene === 'system') {
      floatTimeline.current.kill()
      floatTimeline.current = null
    }
  }, [scene])

  // Keep controls responsive immediately in calm scene; no auto-flight timeline.
  useEffect(() => {
    if (!assetsLoaded || !groupRef.current) return
    if (scene === 'calm') {
      groupRef.current.position.set(0, 3.5, 4)
      groupRef.current.rotation.set(0, 0, 0)
      velocity.current.set(0, 0, 0)
      currentInput.current.set(0, 0, 0)
      rotationInput.current = 0
    }
  }, [assetsLoaded, scene])

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

    // Ensure manual controls are not fighting GSAP float timeline.
    if (floatTimeline.current && scene === 'system') {
      floatTimeline.current.kill()
      floatTimeline.current = null
    }

    // === INPUT HANDLING ===
    const targetInput = new THREE.Vector3()
    const inputSmoothing = 0.15

    // Get raw input - FIXED CONTROLS
    // W/S: Forward/Backward (negative Z is forward in Three.js)
    if (keys.current.w) targetInput.z -= 1  // Forward
    if (keys.current.s) targetInput.z += 1  // Backward
    
    // Space/Shift: Up/Down
    if (keys.current.space) targetInput.y += 1  // Up
    if (keys.current.shift) targetInput.y -= 1  // Down
    
    // A/D: Rotation (handled separately below)
    let targetRotation = 0
    if (keys.current.a) targetRotation = 1   // Rotate left
    if (keys.current.d) targetRotation = -1  // Rotate right
    rotationInput.current = THREE.MathUtils.lerp(rotationInput.current, targetRotation, 0.1)

    // Smooth horizontal movement, but keep vertical input immediate for responsive Space/Shift.
    currentInput.current.x = THREE.MathUtils.lerp(currentInput.current.x, targetInput.x, inputSmoothing)
    currentInput.current.z = THREE.MathUtils.lerp(currentInput.current.z, targetInput.z, inputSmoothing)
    currentInput.current.y = targetInput.y

    // === BOOST SYSTEM ===
    const boostMultiplier = keys.current.ctrl ? 2.0 : 1.0
    const boostDamping = keys.current.ctrl ? 0.96 : profile.damping

    // === ROTATION SYSTEM ===
    // Apply rotation from A/D keys
    if (Math.abs(rotationInput.current) > 0.01) {
      group.rotation.y += rotationInput.current * delta * 3.4
    }

    // === PHYSICS SYSTEM ===
    // Apply input forces - horizontal in local space, vertical in world space
    if (currentInput.current.length() > 0.01) {
      // Separate horizontal (XZ) and vertical (Y) movement
      const horizontalInput = new THREE.Vector3(0, 0, currentInput.current.z)
      const verticalInput = currentInput.current.y
      
      // Transform horizontal input to world space based on ship's rotation
      const worldHorizontal = horizontalInput.applyEuler(new THREE.Euler(0, group.rotation.y, 0))
      
      // Combine: horizontal in local space + vertical in world space
      const finalForce = new THREE.Vector3(
        worldHorizontal.x,
        verticalInput, // Y always in world space
        worldHorizontal.z
      )
      
      const force = finalForce
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
    const noiseScale = airshipState === 'stressed' ? 0.08 : (scene === 'system' ? 0 : 0.003)
    const noiseSpeed = airshipState === 'stressed' ? 2.0 : 0.2
    
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
    if (airshipState === 'patrol' && scene !== 'system') {
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
    const bounds = scene === 'system'
      ? { x: 16, yMin: 27.5, yMax: 46, z: 16 }
      : { x: 50, yMin: 0.5, yMax: 30, z: 50 }

    if (Math.abs(group.position.x) > bounds.x) {
      velocity.current.x *= -0.5
      group.position.x = Math.sign(group.position.x) * bounds.x
    }

    if (group.position.y > bounds.yMax) {
      velocity.current.y *= -0.5
      group.position.y = bounds.yMax
    } else if (group.position.y < bounds.yMin) {
      velocity.current.y *= -0.5
      group.position.y = bounds.yMin
    }

    if (Math.abs(group.position.z) > bounds.z) {
      velocity.current.z *= -0.5
      group.position.z = Math.sign(group.position.z) * bounds.z
    }

    // === ROTATION & TILTING ===
    // Banking on turns (tilt based on rotation input)
    const tiltAmount = -rotationInput.current * 0.3
    group.rotation.z = THREE.MathUtils.lerp(group.rotation.z, tiltAmount, 0.1)

    // Pitch on vertical movement
    const pitchAmount = -velocity.current.y * 0.08
    group.rotation.x = THREE.MathUtils.lerp(group.rotation.x, pitchAmount, 0.1)

    // Return to neutral tilt when not turning
    if (Math.abs(rotationInput.current) < 0.01) {
      group.rotation.z = THREE.MathUtils.lerp(group.rotation.z, 0, 0.05)
    }
    
    // Return to neutral pitch when not moving vertically
    if (Math.abs(velocity.current.y) < 0.1) {
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
    }
  }, [])

  return (
    <group ref={groupRef} name="airship" position={[0, 2, 0]} scale={0.8}>
      <group ref={innerGroupRef}>
        <primitive object={gltfScene} />
        
        {/* Direction Indicator Arrow - Shows forward direction */}
        {scene !== 'calm' && introComplete && <group ref={arrowRef} position={[0, 1.2, 0]}>
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
          
          {/* Arrow head */}
          <mesh position={[0, 0, -1.2]}>
            <sphereGeometry args={[0.16, 10, 10]} />
            <meshStandardMaterial
              color="#00e5ff"
              emissive="#00e5ff"
              emissiveIntensity={1.0}
              metalness={0.9}
              roughness={0.15}
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
        </group>}
      </group>
    </group>
  )
}

// Made with Bob
