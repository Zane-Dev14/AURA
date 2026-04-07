'use client'
import { useRef, useMemo, useEffect, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'
import { premiumMaterials } from '@/lib/materials'
import gsap from 'gsap'
import { float } from '@/lib/microAnimations'

const MAX_PODS = 50
const GRID_SIZE = 5

// Energy field fragment shader
const podFragShader = `
uniform float uTime;
uniform float uHealth;
uniform float uPulse;
varying vec3 vNormal;
varying vec3 vPosition;
void main() {
  vec3 col = mix(vec3(1.0, 0.15, 0.1), vec3(0.0, 1.0, 0.4), uHealth);
  float f = pow(1.0 - abs(dot(normalize(vNormal), vec3(0.0, 0.0, 1.0))), 3.0);
  float p = sin(uTime * 3.0 + vPosition.y * 4.0) * 0.5 + 0.5;
  vec3 finalCol = col + col * f * 0.8 + col * p * 0.15 * uPulse;
  gl_FragColor = vec4(finalCol, 0.88 + f * 0.12);
}
`

const podVertShader = `
varying vec3 vNormal;
varying vec3 vPosition;
void main() {
  vNormal = normalMatrix * normal;
  vPosition = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

export default function PodGrid() {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const { podCount, podHealth, frozen } = useSceneStore()

  const [hoveredInstanceId, setHoveredInstanceId] = useState<number | null>(null)
  const [selectedInstanceId, setSelectedInstanceId] = useState<number | null>(null)

  const dummy = useMemo(() => new THREE.Object3D(), [])
  const scales = useRef<Float32Array>(new Float32Array(MAX_PODS).fill(0))
  const targetScales = useRef<Float32Array>(new Float32Array(MAX_PODS).fill(0))
  const scaleTimelines = useRef<(gsap.core.Tween | null)[]>(new Array(MAX_PODS).fill(null))

  // Update target scales when podCount changes with GSAP
  useEffect(() => {
    if (!meshRef.current) return

    for (let i = 0; i < MAX_PODS; i++) {
      const targetScale = i < podCount ? 1 : 0
      targetScales.current[i] = targetScale

      // Kill existing animation
      if (scaleTimelines.current[i]) {
        scaleTimelines.current[i]?.kill()
      }

      // Animate scale with elastic easing for pods appearing, power2 for disappearing
      const currentScale = scales.current[i]
      const scaleObj = { value: currentScale }
      
      scaleTimelines.current[i] = gsap.to(scaleObj, {
        value: targetScale,
        duration: targetScale > currentScale ? 0.8 : 0.4,
        ease: targetScale > currentScale ? 'elastic.out(1, 0.5)' : 'power2.in',
        delay: i * 0.05, // Stagger effect
        onUpdate: () => {
          scales.current[i] = scaleObj.value
        }
      })
    }
  }, [podCount])

  // Calculate grid positions
  const positions = useMemo(() => {
    const pts: [number, number, number][] = []
    for (let i = 0; i < MAX_PODS; i++) {
      const col = i % GRID_SIZE
      const row = Math.floor(i / GRID_SIZE)
      pts.push([
        (col - GRID_SIZE / 2) * 2.2,
        0,
        (row - Math.floor(MAX_PODS / GRID_SIZE) / 2) * 2.2
      ])
    }
    return pts
  }, [])

  useFrame((state, delta) => {
    if (!meshRef.current) return

    for (let i = 0; i < MAX_PODS; i++) {
      // Use GSAP-animated scales (no more lerp!)
      const s = scales.current[i]
      const [x, y, z] = positions[i]
      
      // Gentle floating animation with sine wave
      const floatOffset = Math.sin(state.clock.elapsedTime * 1.5 + i * 0.4) * 0.05
      dummy.position.set(x, y + floatOffset, z)
      dummy.scale.setScalar(s * 0.8)
      dummy.updateMatrix()
      meshRef.current.setMatrixAt(i, dummy.matrix)
    }
    meshRef.current.instanceMatrix.needsUpdate = true
  })

  // Premium material that changes with health
  const podMaterial = useMemo(() => {
    const mat = new THREE.MeshStandardMaterial({
      ...premiumMaterials.podHealthy,
      transparent: true,
      opacity: 0.9
    })
    mat.vertexColors = true
    return mat
  }, [])

  // Initialize and maintain per-instance colors (for hover/select interaction)
  useEffect(() => {
    if (!meshRef.current) return

    // Instance colors multiply the material color. Keep default neutral (white)
    // so the GSAP-driven health color on the material remains accurate.
    const baseColor = new THREE.Color(1, 1, 1)
    const hoverTint = new THREE.Color(0.6, 1.0, 1.0)
    const selectedTint = new THREE.Color(1.0, 1.0, 1.0)

    for (let i = 0; i < MAX_PODS; i++) {
      // Hide inactive pods by keeping them dark; scale animation already handles visibility
      const isActive = i < podCount
      const isHovered = hoveredInstanceId === i
      const isSelected = selectedInstanceId === i

      const c = isSelected ? selectedTint : isHovered ? hoverTint : baseColor

      // Keep inactive pods visually subdued
      const finalColor = isActive ? c : baseColor.clone().multiplyScalar(0.05)
      meshRef.current.setColorAt(i, finalColor)
    }

    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true
    }
  }, [podCount, hoveredInstanceId, selectedInstanceId, podMaterial])

  // Update material based on health with GSAP color transitions
  const colorTransitionRef = useRef<gsap.core.Timeline | null>(null)
  
  useEffect(() => {
    if (!podMaterial) return
    
    // Kill existing color transition
    if (colorTransitionRef.current) {
      colorTransitionRef.current.kill()
    }
    
    // Determine target colors based on health
    let targetColor: THREE.Color
    let targetEmissive: THREE.Color
    let targetIntensity: number
    
    if (podHealth > 0.6) {
      targetColor = new THREE.Color(premiumMaterials.podHealthy.color)
      targetEmissive = new THREE.Color(premiumMaterials.podHealthy.emissive!)
      targetIntensity = premiumMaterials.podHealthy.emissiveIntensity!
    } else if (podHealth > 0.3) {
      targetColor = new THREE.Color(premiumMaterials.podStressed.color)
      targetEmissive = new THREE.Color(premiumMaterials.podStressed.emissive!)
      targetIntensity = premiumMaterials.podStressed.emissiveIntensity!
    } else {
      targetColor = new THREE.Color(premiumMaterials.podCritical.color)
      targetEmissive = new THREE.Color(premiumMaterials.podCritical.emissive!)
      targetIntensity = premiumMaterials.podCritical.emissiveIntensity!
    }
    
    // Animate color transition with GSAP
    const tl = gsap.timeline()
    
    tl.to(podMaterial.color, {
      r: targetColor.r,
      g: targetColor.g,
      b: targetColor.b,
      duration: 0.8,
      ease: 'power2.inOut'
    }, 0)
    
    tl.to(podMaterial.emissive, {
      r: targetEmissive.r,
      g: targetEmissive.g,
      b: targetEmissive.b,
      duration: 0.8,
      ease: 'power2.inOut'
    }, 0)
    
    tl.to(podMaterial, {
      emissiveIntensity: targetIntensity,
      duration: 0.8,
      ease: 'power2.inOut'
    }, 0)
    
    colorTransitionRef.current = tl
  }, [podHealth, podMaterial])

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, MAX_PODS]}
      position={[0, -1, 0]}
      name="podGrid"
      castShadow
      receiveShadow
      onPointerMove={(e) => {
        // Only treat active pods as interactive
        const id = typeof e.instanceId === 'number' ? e.instanceId : null
        setHoveredInstanceId(id !== null && id < podCount ? id : null)
      }}
      onPointerOut={() => setHoveredInstanceId(null)}
      onClick={(e) => {
        const id = typeof e.instanceId === 'number' ? e.instanceId : null
        if (id === null || id >= podCount) return
        setSelectedInstanceId((prev) => (prev === id ? null : id))
      }}
    >
      <boxGeometry args={[0.7, 0.7, 0.7]} />
      <primitive object={podMaterial} attach="material" />
    </instancedMesh>
  )
}
