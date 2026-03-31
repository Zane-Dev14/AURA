'use client'
import { useEffect, useRef, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from '@/components/r3f/Airship'
import PodGrid from '@/components/r3f/PodGrid'
import ServiceBeam from '@/components/r3f/ServiceBeam'
import { Text } from '@react-three/drei'
import * as THREE from 'three'
import { playRisingHum } from '@/lib/sound'
import gsap from 'gsap'
import { createObstacleAnimation } from '@/lib/sceneAnimations'

// Moving obstacle that the player must dodge - now with GSAP
function MovingObstacle({
  startPos,
  speed,
  axis,
  range,
  color = '#ff4400'
}: {
  startPos: [number, number, number]
  speed: number
  axis: 'x' | 'y' | 'z'
  range: number
  color?: string
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const timelineRef = useRef<gsap.core.Timeline | null>(null)
  
  useEffect(() => {
    if (!meshRef.current) return
    
    const mesh = meshRef.current
    const startVec = new THREE.Vector3(...startPos)
    
    // Kill existing animation
    if (timelineRef.current) {
      timelineRef.current.kill()
    }
    
    // Create GSAP timeline for movement
    timelineRef.current = createObstacleAnimation(mesh, startVec, axis, range, speed)
    
    return () => {
      timelineRef.current?.kill()
    }
  }, [startPos, speed, axis, range])
  
  return (
    <mesh ref={meshRef} position={startPos}>
      <boxGeometry args={[1.5, 1.5, 1.5]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.8}
        transparent
        opacity={0.7}
      />
      {/* Warning glow */}
      <mesh>
        <boxGeometry args={[2, 2, 2]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.2}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </mesh>
  )
}

// Traffic particles flowing through the scene
function TrafficFlow() {
  const pointsRef = useRef<THREE.Points>(null)
  
  const positions = useRef(
    new Float32Array(
      Array.from({ length: 300 }, () => [
        (Math.random() - 0.5) * 30,
        Math.random() * 15 - 5,
        Math.random() * 60 - 30
      ]).flat()
    )
  )
  
  useFrame((state, delta) => {
    if (!pointsRef.current) return
    const pos = pointsRef.current.geometry.attributes.position.array as Float32Array
    
    for (let i = 0; i < 300; i++) {
      const idx = i * 3
      // Flow forward
      pos[idx + 2] += delta * 8
      
      // Reset if too far
      if (pos[idx + 2] > 30) {
        pos[idx + 2] = -30
        pos[idx] = (Math.random() - 0.5) * 30
        pos[idx + 1] = Math.random() * 15 - 5
      }
    }
    
    pointsRef.current.geometry.attributes.position.needsUpdate = true
  })
  
  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions.current, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.15}
        color="#ff6600"
        transparent
        opacity={0.6}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

export default function TrafficScene() {
  const { camera } = useThree()
  const { setTrafficLevel, setPodCount, setAirshipState, audioUnlocked, setMetrics, setScene } = useSceneStore()
  const [showInstructions, setShowInstructions] = useState(true)
  const transitionTriggered = useRef(false)
  const sceneTimeline = useRef<gsap.core.Timeline | null>(null)
  const cameraTimeline = useRef<gsap.core.Timeline | null>(null)

  useEffect(() => {
    setPodCount(6)
    setAirshipState('stressed')
    if (audioUnlocked) playRisingHum()
    
    // Hide instructions with fade
    const instructionTimer = setTimeout(() => setShowInstructions(false), 5000)
    
    // Create master scene timeline with GSAP
    const masterTL = gsap.timeline()
    
    // Traffic level ramp with dramatic easing
    const trafficObj = { value: 0 }
    masterTL.to(trafficObj, {
      value: 1,
      duration: 8,
      ease: 'power2.in',
      onUpdate: () => {
        setTrafficLevel(trafficObj.value)
        setMetrics({
          rps: Math.round(trafficObj.value * 120),
          cpuPercent: Math.round(20 + trafficObj.value * 62),
        })
      }
    }, 0)
    
    // Camera smooth entry
    const camTL = gsap.timeline()
    camTL.to(camera.position, {
      x: 0,
      y: 8,
      z: 20,
      duration: 2,
      ease: 'power3.out',
      onUpdate: () => {
        camera.lookAt(0, 2, -10)
      }
    })
    
    cameraTimeline.current = camTL
    sceneTimeline.current = masterTL
    
    // Auto-progress to failure scene
    const transitionTimer = setTimeout(() => {
      if (!transitionTriggered.current) {
        transitionTriggered.current = true
        setScene('failure')
      }
    }, 23000) // 23 seconds total (8s ramp + 15s at peak)
    
    return () => {
      clearTimeout(instructionTimer)
      clearTimeout(transitionTimer)
      masterTL.kill()
      camTL.kill()
    }
  }, [])

  useFrame(() => {
    // Keep camera looking at target (GSAP handles position)
    camera.lookAt(0, 2, -10)
  })

  return (
    <>
      <ambientLight intensity={0.2} />
      <pointLight position={[0, 10, 0]} intensity={2} color="#ff6600" />
      <pointLight position={[0, 5, -20]} intensity={1.5} color="#ff4400" />
      <directionalLight position={[10, 10, 10]} intensity={0.5} color="#ff8800" />
      
      {/* Instructions */}
      {showInstructions && (
        <Text
          position={[0, 6, -5]}
          fontSize={0.5}
          color="#ffaa00"
          anchorX="center"
          anchorY="middle"
        >
          Navigate through the traffic!{'\n'}Avoid the obstacles
        </Text>
      )}
      
      {/* Moving obstacles - create a challenging course */}
      <MovingObstacle startPos={[-8, 3, -15]} speed={4} axis="x" range={6} color="#ff4400" />
      <MovingObstacle startPos={[8, 5, -20]} speed={3} axis="x" range={7} color="#ff6600" />
      <MovingObstacle startPos={[0, 2, -25]} speed={5} axis="y" range={3} color="#ff8800" />
      <MovingObstacle startPos={[-6, 6, -30]} speed={4} axis="x" range={5} color="#ff4400" />
      <MovingObstacle startPos={[6, 3, -35]} speed={3.5} axis="y" range={4} color="#ff6600" />
      <MovingObstacle startPos={[0, 7, -40]} speed={4.5} axis="x" range={8} color="#ff8800" />
      <MovingObstacle startPos={[-10, 4, -45]} speed={5} axis="y" range={3.5} color="#ff4400" />
      <MovingObstacle startPos={[10, 5, -50]} speed={3} axis="x" range={6} color="#ff6600" />
      
      {/* Traffic flow particles */}
      <TrafficFlow />
      
      {/* Warning barriers on sides */}
      {[-15, 15].map((x, i) => (
        <mesh key={i} position={[x, 5, -30]}>
          <boxGeometry args={[0.5, 15, 80]} />
          <meshStandardMaterial
            color="#ff0000"
            emissive="#ff0000"
            emissiveIntensity={0.5}
            transparent
            opacity={0.3}
            wireframe
          />
        </mesh>
      ))}
      
      {/* Ground grid for speed sense */}
      <mesh position={[0, -2, -30]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[40, 100, 20, 50]} />
        <meshBasicMaterial
          color="#ff4400"
          transparent
          opacity={0.1}
          wireframe
        />
      </mesh>
      
      <PodGrid />
      <ServiceBeam />
      <Airship />
    </>
  )
}

// Made with Bob
