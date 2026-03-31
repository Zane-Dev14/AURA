'use client'
import { useEffect, useRef, useState } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import { Text } from '@react-three/drei'
import gsap from 'gsap'
import * as THREE from 'three'

/**
 * IntroSequence - DRAMATIC cinematic intro
 * 
 * Lightning flashes, letters SLAM down with impact,
 * screen shake, particle explosions - pure cinema!
 */
export default function IntroSequence() {
  const { camera, scene } = useThree()
  const { introComplete, setIntroComplete, assetsLoaded } = useSceneStore()
  const timelineRef = useRef<gsap.core.Timeline | null>(null)
  
  // Letter states
  const [letterStates, setLetterStates] = useState([
    { letter: 'A', visible: false, y: 50, landed: false },
    { letter: 'U', visible: false, y: 50, landed: false },
    { letter: 'R', visible: false, y: 50, landed: false },
    { letter: 'A', visible: false, y: 50, landed: false },
  ])
  
  // Effects states
  const [lightningFlash, setLightningFlash] = useState(0)
  const [particles, setParticles] = useState<Array<{id: number, x: number, y: number, z: number, vx: number, vy: number, vz: number}>>([])
  const cameraShake = useRef(new THREE.Vector3())
  const baseCameraPos = useRef(new THREE.Vector3(0, 5, 15))
  const particleIdCounter = useRef(0)

  useEffect(() => {
    if (!assetsLoaded || introComplete) return

    // Set camera to dramatic starting position
    camera.position.copy(baseCameraPos.current)
    camera.lookAt(0, 0, 0)

    // Create dramatic timeline
    const timeline = gsap.timeline({
      onComplete: () => {
        setTimeout(() => setIntroComplete(true), 500)
      }
    })

    timelineRef.current = timeline

    // PHASE 1: Initial lightning flash (0-0.3s)
    timeline.call(() => {
      setLightningFlash(1)
      setTimeout(() => setLightningFlash(0), 100)
    })

    // PHASE 2: Letter A SLAMS down (0.5s)
    timeline.call(() => {
      setLetterStates(prev => {
        const next = [...prev]
        next[0].visible = true
        return next
      })
      
      // Lightning flash
      setLightningFlash(1)
      setTimeout(() => setLightningFlash(0), 80)
      
      // Camera shake
      gsap.to(cameraShake.current, {
        x: Math.random() * 0.5 - 0.25,
        y: Math.random() * 0.5 - 0.25,
        z: Math.random() * 0.3 - 0.15,
        duration: 0.1,
        onComplete: () => {
          gsap.to(cameraShake.current, { x: 0, y: 0, z: 0, duration: 0.3 })
        }
      })
      
      // Letter slam animation
      gsap.to(letterStates[0], {
        y: 0,
        duration: 0.4,
        ease: 'power4.in',
        onUpdate: () => {
          setLetterStates(prev => {
            const next = [...prev]
            next[0].y = letterStates[0].y
            return next
          })
        },
        onComplete: () => {
          setLetterStates(prev => {
            const next = [...prev]
            next[0].landed = true
            return next
          })
          // Particle explosion
          createParticleExplosion(-6, 0, 0)
        }
      })
    }, [], 0.5)

    // PHASE 3: Letter U SLAMS down (1.0s)
    timeline.call(() => {
      setLetterStates(prev => {
        const next = [...prev]
        next[1].visible = true
        return next
      })
      
      setLightningFlash(1)
      setTimeout(() => setLightningFlash(0), 80)
      
      gsap.to(cameraShake.current, {
        x: Math.random() * 0.4 - 0.2,
        y: Math.random() * 0.4 - 0.2,
        z: Math.random() * 0.2 - 0.1,
        duration: 0.1,
        onComplete: () => {
          gsap.to(cameraShake.current, { x: 0, y: 0, z: 0, duration: 0.3 })
        }
      })
      
      gsap.to(letterStates[1], {
        y: 0,
        duration: 0.4,
        ease: 'power4.in',
        onUpdate: () => {
          setLetterStates(prev => {
            const next = [...prev]
            next[1].y = letterStates[1].y
            return next
          })
        },
        onComplete: () => {
          setLetterStates(prev => {
            const next = [...prev]
            next[1].landed = true
            return next
          })
          createParticleExplosion(-2, 0, 0)
        }
      })
    }, [], 1.0)

    // PHASE 4: Letter R SLAMS down (1.5s)
    timeline.call(() => {
      setLetterStates(prev => {
        const next = [...prev]
        next[2].visible = true
        return next
      })
      
      setLightningFlash(1)
      setTimeout(() => setLightningFlash(0), 80)
      
      gsap.to(cameraShake.current, {
        x: Math.random() * 0.4 - 0.2,
        y: Math.random() * 0.4 - 0.2,
        z: Math.random() * 0.2 - 0.1,
        duration: 0.1,
        onComplete: () => {
          gsap.to(cameraShake.current, { x: 0, y: 0, z: 0, duration: 0.3 })
        }
      })
      
      gsap.to(letterStates[2], {
        y: 0,
        duration: 0.4,
        ease: 'power4.in',
        onUpdate: () => {
          setLetterStates(prev => {
            const next = [...prev]
            next[2].y = letterStates[2].y
            return next
          })
        },
        onComplete: () => {
          setLetterStates(prev => {
            const next = [...prev]
            next[2].landed = true
            return next
          })
          createParticleExplosion(2, 0, 0)
        }
      })
    }, [], 1.5)

    // PHASE 5: Letter A SLAMS down (2.0s)
    timeline.call(() => {
      setLetterStates(prev => {
        const next = [...prev]
        next[3].visible = true
        return next
      })
      
      setLightningFlash(1.5) // Bigger flash for final letter
      setTimeout(() => setLightningFlash(0), 120)
      
      gsap.to(cameraShake.current, {
        x: Math.random() * 0.6 - 0.3,
        y: Math.random() * 0.6 - 0.3,
        z: Math.random() * 0.3 - 0.15,
        duration: 0.15,
        onComplete: () => {
          gsap.to(cameraShake.current, { x: 0, y: 0, z: 0, duration: 0.4 })
        }
      })
      
      gsap.to(letterStates[3], {
        y: 0,
        duration: 0.4,
        ease: 'power4.in',
        onUpdate: () => {
          setLetterStates(prev => {
            const next = [...prev]
            next[3].y = letterStates[3].y
            return next
          })
        },
        onComplete: () => {
          setLetterStates(prev => {
            const next = [...prev]
            next[3].landed = true
            return next
          })
          createParticleExplosion(6, 0, 0)
        }
      })
    }, [], 2.0)

    // PHASE 6: Final dramatic pause and fade (2.5-3.5s)
    timeline.to({}, { duration: 1.5 })

    return () => {
      if (timelineRef.current) {
        timelineRef.current.kill()
      }
    }
  }, [assetsLoaded, introComplete, camera, setIntroComplete])

  // Create particle explosion effect
  const createParticleExplosion = (x: number, y: number, z: number) => {
    const newParticles = []
    for (let i = 0; i < 30; i++) {
      const angle = (Math.PI * 2 * i) / 30
      const speed = 2 + Math.random() * 3
      newParticles.push({
        id: particleIdCounter.current++,
        x,
        y,
        z,
        vx: Math.cos(angle) * speed,
        vy: Math.random() * 4 + 2,
        vz: Math.sin(angle) * speed,
      })
    }
    setParticles(prev => [...prev, ...newParticles])
  }

  // Animate particles
  useFrame((state, delta) => {
    if (!introComplete) {
      // Apply camera shake
      camera.position.copy(baseCameraPos.current).add(cameraShake.current)
      camera.lookAt(0, 0, 0)
    }

    // Update particles
    setParticles(prev => 
      prev
        .map(p => ({
          ...p,
          x: p.x + p.vx * delta,
          y: p.y + p.vy * delta,
          z: p.z + p.vz * delta,
          vy: p.vy - 9.8 * delta, // Gravity
        }))
        .filter(p => p.y > -5) // Remove particles that fall too far
    )
  })

  // Skip intro on user interaction
  useEffect(() => {
    const handleSkip = (e: KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        if (!introComplete && timelineRef.current) {
          timelineRef.current.progress(1)
          setIntroComplete(true)
        }
      }
    }

    window.addEventListener('keydown', handleSkip)
    return () => window.removeEventListener('keydown', handleSkip)
  }, [introComplete, setIntroComplete])

  if (introComplete) return null

  return (
    <>
      {/* Lightning flash effect */}
      {lightningFlash > 0 && (
        <pointLight
          position={[0, 10, 5]}
          intensity={lightningFlash * 50}
          color="#ffffff"
          distance={100}
        />
      )}
      
      {/* Dark moody ambient */}
      <ambientLight intensity={0.05} />
      
      {/* Background lightning flashes */}
      {lightningFlash > 0 && (
        <>
          <mesh position={[0, 0, -20]}>
            <planeGeometry args={[100, 100]} />
            <meshBasicMaterial
              color="#ffffff"
              transparent
              opacity={lightningFlash * 0.3}
            />
          </mesh>
        </>
      )}

      {/* AURA Letters */}
      {letterStates.map((state, i) => {
        if (!state.visible) return null
        
        const xPos = -6 + i * 4
        
        return (
          <group key={i} position={[xPos, state.y, 0]}>
            <Text
              fontSize={3}
              color="#00e5ff"
              anchorX="center"
              anchorY="middle"
            >
              {state.letter}
            </Text>
            
            {/* Glow effect */}
            <pointLight
              position={[0, 0, 2]}
              intensity={state.landed ? 5 : 10}
              color="#00e5ff"
              distance={10}
            />
            
            {/* Impact ring when landed */}
            {state.landed && (
              <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.5, 0]}>
                <ringGeometry args={[0.5, 3, 32]} />
                <meshBasicMaterial
                  color="#00e5ff"
                  transparent
                  opacity={0.5}
                  side={THREE.DoubleSide}
                />
              </mesh>
            )}
          </group>
        )
      })}

      {/* Particle explosions */}
      {particles.map(p => (
        <mesh key={p.id} position={[p.x, p.y, p.z]}>
          <sphereGeometry args={[0.1, 8, 8]} />
          <meshBasicMaterial
            color="#00e5ff"
            transparent
            opacity={0.8}
          />
        </mesh>
      ))}

      {/* Atmospheric fog */}
      <mesh position={[0, -2, -10]}>
        <planeGeometry args={[50, 50]} />
        <meshBasicMaterial
          color="#000510"
          transparent
          opacity={0.8}
        />
      </mesh>
    </>
  )
}

// Made with Bob