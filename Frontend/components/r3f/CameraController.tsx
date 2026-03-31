'use client'
import { useEffect, useRef } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import { useSceneStore, Scene } from '@/store/useSceneStore'
import * as THREE from 'three'
import gsap from 'gsap'

type CameraMode = 'cinematic' | 'follow' | 'fixed' | 'orbit'

interface CameraPreset {
  position: [number, number, number]
  lookAt: [number, number, number]
  fov: number
  mode: CameraMode
  shake?: boolean // Enable camera shake for dramatic scenes
  orbitSpeed?: number // For orbit mode
  orbitRadius?: number // For orbit mode
}

const cameraPresets: Record<Scene, CameraPreset> = {
  calm: { 
    position: [0, 8, 20], 
    lookAt: [0, 2, 0], 
    fov: 55, 
    mode: 'follow' 
  },
  system: { 
    position: [5, 10, 18], 
    lookAt: [0, 3, 0], 
    fov: 55, 
    mode: 'cinematic' 
  },
  traffic: { 
    position: [0, 8, 20], 
    lookAt: [0, 2, -10], 
    fov: 55, 
    mode: 'follow' 
  },
  failure: { 
    position: [0, 7, 16], 
    lookAt: [0, -1, 0], 
    fov: 60, 
    mode: 'cinematic',
    shake: true 
  },
  emotional: { 
    position: [2, 5, 12], 
    lookAt: [0, 2, 0], 
    fov: 65, 
    mode: 'cinematic',
    shake: true 
  },
  qmix: { 
    position: [0, 6, 15], 
    lookAt: [0, 2.5, 0], 
    fov: 55, 
    mode: 'fixed' 
  },
  transform: { 
    position: [0, 10, 25], 
    lookAt: [0, 3, 0], 
    fov: 50, 
    mode: 'orbit',
    orbitSpeed: 0.15,
    orbitRadius: 25 
  },
  recovery: { 
    position: [-8, 12, 22], 
    lookAt: [0, 4, 0], 
    fov: 55, 
    mode: 'cinematic' 
  },
  comparison: { 
    position: [0, 15, 30], 
    lookAt: [0, 5, 0], 
    fov: 50, 
    mode: 'cinematic' 
  }
}

export default function CameraController() {
  const { camera, scene: threeScene } = useThree()
  const scene = useSceneStore((state) => state.scene)
  
  // Refs for smooth camera control
  const currentMode = useRef<CameraMode>('cinematic')
  const targetPosition = useRef(new THREE.Vector3())
  const targetLookAt = useRef(new THREE.Vector3())
  const currentLookAt = useRef(new THREE.Vector3())
  const orbitAngle = useRef(0)
  const shakeOffset = useRef(new THREE.Vector3())
  const tweenRef = useRef<gsap.core.Tween | null>(null)
  const fovTweenRef = useRef<gsap.core.Tween | null>(null)
  
  // Airship reference for follow mode
  const airshipPosition = useRef(new THREE.Vector3(0, 2, 0))
  
  // Initialize camera look target
  useEffect(() => {
    const preset = cameraPresets[scene]
    currentLookAt.current.set(...preset.lookAt)
  }, [])
  
  // Handle scene changes with smooth GSAP transitions
  useEffect(() => {
    const preset = cameraPresets[scene]
    currentMode.current = preset.mode
    
    // Kill existing tweens
    if (tweenRef.current) tweenRef.current.kill()
    if (fovTweenRef.current) fovTweenRef.current.kill()
    
    // Set target position and lookAt
    targetPosition.current.set(...preset.position)
    targetLookAt.current.set(...preset.lookAt)
    
    // Choose easing based on scene mood
    const ease = 
      scene === 'failure' || scene === 'emotional' ? 'power3.inOut' :
      scene === 'transform' ? 'expo.inOut' :
      scene === 'recovery' ? 'power2.out' :
      'power2.inOut'
    
    // Smooth camera position transition
    tweenRef.current = gsap.to(camera.position, {
      x: preset.position[0],
      y: preset.position[1],
      z: preset.position[2],
      duration: 2.5,
      ease,
      onUpdate: () => {
        // Update target for orbit mode
        if (preset.mode === 'orbit') {
          targetPosition.current.copy(camera.position)
        }
      }
    })
    
    // Smooth lookAt transition
    gsap.to(currentLookAt.current, {
      x: preset.lookAt[0],
      y: preset.lookAt[1],
      z: preset.lookAt[2],
      duration: 2.5,
      ease
    })
    
    // Smooth FOV transition
    fovTweenRef.current = gsap.to(camera, {
      fov: preset.fov,
      duration: 2,
      ease: 'power2.inOut',
      onUpdate: () => {
        camera.updateProjectionMatrix()
      }
    })
    
    // Reset orbit angle for orbit mode
    if (preset.mode === 'orbit') {
      orbitAngle.current = Math.atan2(
        preset.position[2] - preset.lookAt[2],
        preset.position[0] - preset.lookAt[0]
      )
    }
    
    return () => {
      if (tweenRef.current) tweenRef.current.kill()
      if (fovTweenRef.current) fovTweenRef.current.kill()
    }
  }, [scene, camera])
  
  // Frame-by-frame camera updates
  useFrame((state, delta) => {
    const preset = cameraPresets[scene]
    
    // Camera shake for dramatic scenes
    if (preset.shake) {
      const shakeIntensity = scene === 'failure' ? 0.08 : 0.05
      const shakeSpeed = scene === 'failure' ? 15 : 10
      
      shakeOffset.current.set(
        Math.sin(state.clock.elapsedTime * shakeSpeed) * shakeIntensity,
        Math.cos(state.clock.elapsedTime * shakeSpeed * 1.3) * shakeIntensity,
        Math.sin(state.clock.elapsedTime * shakeSpeed * 0.8) * shakeIntensity * 0.5
      )
    } else {
      shakeOffset.current.lerp(new THREE.Vector3(0, 0, 0), 0.1)
    }
    
    // Mode-specific camera behavior
    switch (currentMode.current) {
      case 'follow': {
        // Smooth follow with damping
        const cameraOffset = new THREE.Vector3(0, 3, 8)
        const targetPos = airshipPosition.current.clone().add(cameraOffset)
        
        camera.position.lerp(targetPos, 0.05)
        
        const lookTarget = airshipPosition.current.clone().add(new THREE.Vector3(0, 1, 0))
        currentLookAt.current.lerp(lookTarget, 0.08)
        break
      }
      
      case 'orbit': {
        // Gentle orbit around focal point
        const speed = preset.orbitSpeed || 0.15
        const radius = preset.orbitRadius || 25
        
        orbitAngle.current += delta * speed
        
        const orbitX = Math.cos(orbitAngle.current) * radius
        const orbitZ = Math.sin(orbitAngle.current) * radius
        
        targetPosition.current.set(
          orbitX,
          preset.position[1],
          orbitZ
        )
        
        camera.position.lerp(targetPosition.current, 0.02)
        break
      }
      
      case 'cinematic': {
        // Smooth damping towards target (GSAP handles the main transition)
        camera.position.lerp(targetPosition.current, 0.02)
        break
      }
      
      case 'fixed': {
        // Minimal movement, just subtle breathing
        const breathe = Math.sin(state.clock.elapsedTime * 0.5) * 0.1
        camera.position.y = targetPosition.current.y + breathe
        break
      }
    }
    
    // Apply camera shake
    const finalPosition = camera.position.clone().add(shakeOffset.current)
    camera.position.copy(finalPosition)
    
    // Smooth lookAt with damping
    const lookAtTarget = currentLookAt.current.clone().add(shakeOffset.current)
    camera.lookAt(lookAtTarget)
    
    // Update camera matrix
    camera.updateMatrixWorld()
  })
  
  // Listen for airship position updates (for follow mode)
  useFrame(() => {
    // This will be updated by Airship component or we can query the scene
    // For now, we'll use a simple approach - the Airship will update this
    const airship = threeScene.getObjectByName('airship')
    if (airship) {
      airshipPosition.current.copy(airship.position)
    }
  })
  
  return null
}

// Made with Bob
