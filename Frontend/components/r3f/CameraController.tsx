'use client'
import { useEffect, useRef, useState } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import { useSceneStore, Scene } from '@/store/useSceneStore'
import * as THREE from 'three'
import gsap from 'gsap'

type CameraMode = 'cinematic' | 'follow' | 'fixed' | 'orbit' | 'mouse-orbit'

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
  const { camera, scene: threeScene, pointer } = useThree()
  const scene = useSceneStore((state) => state.scene)
  const introComplete = useSceneStore((state) => state.introComplete)
  
  // Refs for smooth camera control
  const currentMode = useRef<CameraMode>('cinematic')
  const targetPosition = useRef(new THREE.Vector3())
  const targetLookAt = useRef(new THREE.Vector3())
  const currentLookAt = useRef(new THREE.Vector3())
  const orbitAngle = useRef(0)
  const shakeOffset = useRef(new THREE.Vector3())
  const tweenRef = useRef<gsap.core.Tween | null>(null)
  const fovTweenRef = useRef<gsap.core.Tween | null>(null)
  
  // Mouse drag rotation control
  const [isDragging, setIsDragging] = useState(false)
  const dragAngles = useRef({ horizontal: 0, vertical: 0 })
  const defaultAngles = useRef({ horizontal: Math.PI, vertical: 0 }) // Behind airship
  const lastMousePos = useRef({ x: 0, y: 0 })
  const orbitDistance = useRef(20)
  
  // Airship reference for follow mode
  const airshipPosition = useRef(new THREE.Vector3(0, 2, 0))
  
  // Initialize camera look target
  useEffect(() => {
    const preset = cameraPresets[scene]
    currentLookAt.current.set(...preset.lookAt)
  }, [])

  // Mouse drag handlers for camera rotation
  useEffect(() => {
    if (!introComplete) return

    const handlePointerDown = (e: PointerEvent) => {
      // Right-click or left-click to drag
      if (e.button === 0 || e.button === 2) {
        setIsDragging(true)
        lastMousePos.current = { x: e.clientX, y: e.clientY }
        e.preventDefault()
      }
    }

    const handlePointerMove = (e: PointerEvent) => {
      if (!isDragging) return

      const deltaX = e.clientX - lastMousePos.current.x
      const deltaY = e.clientY - lastMousePos.current.y
      
      lastMousePos.current = { x: e.clientX, y: e.clientY }

      // Update drag angles based on mouse movement
      // Horizontal: full 360° rotation (sensitivity: 0.005)
      dragAngles.current.horizontal -= deltaX * 0.005
      
      // Vertical: limited to -45° to 45° (prevent flipping)
      const verticalRange = Math.PI / 4 // 45 degrees
      dragAngles.current.vertical = THREE.MathUtils.clamp(
        dragAngles.current.vertical + deltaY * 0.005,
        -verticalRange,
        verticalRange
      )
    }

    const handlePointerUp = () => {
      setIsDragging(false)
    }

    // Prevent context menu on right-click
    const handleContextMenu = (e: MouseEvent) => {
      e.preventDefault()
    }

    window.addEventListener('pointerdown', handlePointerDown)
    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp)
    window.addEventListener('contextmenu', handleContextMenu)

    return () => {
      window.removeEventListener('pointerdown', handlePointerDown)
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
      window.removeEventListener('contextmenu', handleContextMenu)
    }
  }, [introComplete, isDragging])
  
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
    
    // Third-person camera with drag rotation (for follow mode after intro)
    if (currentMode.current === 'follow' && introComplete) {
      // Calculate camera position using spherical coordinates around airship
      const horizontalAngle = dragAngles.current.horizontal
      const verticalAngle = dragAngles.current.vertical
      
      // Distance from airship
      const distance = 12
      
      // Convert spherical to cartesian coordinates
      const offsetX = distance * Math.cos(verticalAngle) * Math.sin(horizontalAngle)
      const offsetY = 3 + distance * Math.sin(verticalAngle)
      const offsetZ = distance * Math.cos(verticalAngle) * Math.cos(horizontalAngle)
      
      // Position camera relative to airship (follows airship movement)
      const targetPos = new THREE.Vector3(
        airshipPosition.current.x + offsetX,
        airshipPosition.current.y + offsetY,
        airshipPosition.current.z + offsetZ
      )
      
      // Smooth camera movement - follows airship
      camera.position.lerp(targetPos, 0.1)
      
      // Look at airship (slightly above center)
      const lookTarget = airshipPosition.current.clone().add(new THREE.Vector3(0, 1, 0))
      camera.lookAt(lookTarget)
      camera.updateMatrixWorld()
      return
    }
    
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
    
    // Mode-specific camera behavior (skip follow mode as it's handled above)
    switch (currentMode.current) {
      case 'follow': {
        // Handled above with drag rotation support
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
