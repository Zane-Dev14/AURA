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
    position: [0, 8, 18], 
    lookAt: [0, 2.5, 0], 
    fov: 60, 
    mode: 'follow' 
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
  const airshipVelocity = useRef(new THREE.Vector3())
  const smoothedAirshipVelocity = useRef(new THREE.Vector3())
  const lastAirshipPosition = useRef(new THREE.Vector3(0, 2, 0))
  
  // Mouse drag rotation control
  const [isDragging, setIsDragging] = useState(false)
  const dragAngles = useRef({ horizontal: 0, vertical: 0 })
  const defaultAngles = useRef({ horizontal: Math.PI, vertical: -0.15 }) // Slightly above/behind
  const lastMousePos = useRef({ x: 0, y: 0 })
  
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

    if (preset.mode === 'follow') {
      dragAngles.current.horizontal = defaultAngles.current.horizontal
      dragAngles.current.vertical = defaultAngles.current.vertical
    }
    
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

    const dt = Math.max(delta, 1 / 240)

    // Pull airship transform directly in this frame to avoid update-order jitter.
    const airship = threeScene.getObjectByName('airship')
    if (airship) {
      airshipPosition.current.copy(airship.position)
    }
    
    // Calculate airship velocity for prediction
    airshipVelocity.current.copy(airshipPosition.current).sub(lastAirshipPosition.current).divideScalar(dt)
    const velocitySmoothing = 1 - Math.exp(-10 * dt)
    smoothedAirshipVelocity.current.lerp(airshipVelocity.current, velocitySmoothing)
    lastAirshipPosition.current.copy(airshipPosition.current)
    
    // Third-person camera with drag rotation (for follow mode after intro)
    if (currentMode.current === 'follow' && introComplete) {
      // Calculate camera position using spherical coordinates around airship
      const horizontalAngle = dragAngles.current.horizontal
      const verticalAngle = dragAngles.current.vertical
      
      // Scene 2 uses a larger world-scale model, so keep the camera farther back.
      const distance = scene === 'system' ? 20 : 12
      const baseHeight = scene === 'system' ? 8 : 3
      const lookHeight = scene === 'system' ? 3 : 1
      
      // Convert spherical to cartesian coordinates
      const offsetX = distance * Math.cos(verticalAngle) * Math.sin(horizontalAngle)
      const offsetY = baseHeight + distance * Math.sin(verticalAngle)
      const offsetZ = distance * Math.cos(verticalAngle) * Math.cos(horizontalAngle)
      
      // Add velocity-based offset for cinematic lag
      const velocityOffset = smoothedAirshipVelocity.current.clone().multiplyScalar(-0.2)
      
      // Position camera relative to airship (follows airship movement)
      const targetPos = new THREE.Vector3(
        airshipPosition.current.x + offsetX + velocityOffset.x,
        airshipPosition.current.y + offsetY + velocityOffset.y,
        airshipPosition.current.z + offsetZ + velocityOffset.z
      )

      // Lock Scene 2 camera traversal to the model region.
      if (scene === 'system') {
        const bounds = { x: 16, yMin: 20, yMax: 36, z: 16 }
        targetPos.x = THREE.MathUtils.clamp(targetPos.x, -bounds.x, bounds.x)
        targetPos.y = THREE.MathUtils.clamp(targetPos.y, bounds.yMin, bounds.yMax)
        targetPos.z = THREE.MathUtils.clamp(targetPos.z, -bounds.z, bounds.z)
      }
      
      // Frame-rate independent smoothing for stable follow motion.
      const positionAlpha = 1 - Math.exp(-6 * dt)
      camera.position.lerp(targetPos, positionAlpha)
      
      // Look ahead based on velocity for predictive camera
      const predictedPosition = airshipPosition.current.clone()
        .add(smoothedAirshipVelocity.current.clone().multiplyScalar(0.18))
      
      // Look at predicted position (slightly above center)
      const lookTarget = predictedPosition.clone().add(new THREE.Vector3(0, lookHeight, 0))

      if (scene === 'system') {
        lookTarget.x = THREE.MathUtils.clamp(lookTarget.x, -14, 14)
        lookTarget.y = THREE.MathUtils.clamp(lookTarget.y, 18, 34)
        lookTarget.z = THREE.MathUtils.clamp(lookTarget.z, -14, 14)
      }

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
  
  return null
}

// Made with Bob
