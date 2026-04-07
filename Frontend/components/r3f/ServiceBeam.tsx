'use client'
import { useRef, useMemo, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'
import { premiumMaterials } from '@/lib/materials'
import gsap from 'gsap'

export default function ServiceBeam() {
  const meshRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.Mesh>(null)
  const matRef = useRef<THREE.MeshStandardMaterial>(null)
  const glowMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const baseMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const topMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const { podHealth, trafficLevel, frozen } = useSceneStore()
  
  const pulseTimeline = useRef<gsap.core.Timeline | null>(null)
  const colorTimeline = useRef<gsap.core.Timeline | null>(null)

  // Premium beam materials
  const beamCoreMaterial = useMemo(() => {
    return new THREE.MeshStandardMaterial({
      ...premiumMaterials.beamCore
    })
  }, [])

  const beamGlowMaterial = useMemo(() => {
    return new THREE.MeshStandardMaterial({
      ...premiumMaterials.beamGlow
    })
  }, [])

  // Setup GSAP pulse animation based on traffic level
  useEffect(() => {
    if (!meshRef.current || !glowRef.current || frozen) return
    
    // Kill existing pulse
    if (pulseTimeline.current) {
      pulseTimeline.current.kill()
    }
    
    // Create pulsing animation with traffic intensity
    const tl = gsap.timeline({ repeat: -1, yoyo: true })
    const pulseIntensity = 0.1 + trafficLevel * 0.2
    const speed = 1 / (1 + trafficLevel * 2)
    
    tl.to(meshRef.current.scale, {
      x: 1 + pulseIntensity,
      z: 1 + pulseIntensity,
      duration: speed,
      ease: 'sine.inOut'
    }, 0)
    
    tl.to(glowRef.current.scale, {
      x: (1 + pulseIntensity) * 1.3,
      z: (1 + pulseIntensity) * 1.3,
      duration: speed,
      ease: 'sine.inOut'
    }, 0)
    
    pulseTimeline.current = tl
    
    return () => {
      tl.kill()
    }
  }, [trafficLevel, frozen])
  
  // Setup GSAP color transitions based on health
  useEffect(() => {
    if (!matRef.current) return
    
    // Kill existing color transition
    if (colorTimeline.current) {
      colorTimeline.current.kill()
    }
    
    // Determine target color based on health
    const healthyColor = new THREE.Color('#00ffaa')
    const stressedColor = new THREE.Color('#ff8800')
    const criticalColor = new THREE.Color('#ff2200')
    
    let targetColor: THREE.Color
    if (podHealth > 0.6) {
      targetColor = healthyColor
    } else if (podHealth > 0.3) {
      targetColor = stressedColor
    } else {
      targetColor = criticalColor
    }
    
    // Animate color transition
    const tl = gsap.timeline()
    
    // Core beam color
    tl.to(matRef.current.emissive, {
      r: targetColor.r,
      g: targetColor.g,
      b: targetColor.b,
      duration: 0.8,
      ease: 'power2.inOut'
    }, 0)
    
    // Glow color
    if (glowMatRef.current) {
      tl.to(glowMatRef.current.emissive, {
        r: targetColor.r,
        g: targetColor.g,
        b: targetColor.b,
        duration: 0.8,
        ease: 'power2.inOut'
      }, 0)
    }
    
    // Base disc color
    if (baseMatRef.current) {
      tl.to(baseMatRef.current.emissive, {
        r: targetColor.r,
        g: targetColor.g,
        b: targetColor.b,
        duration: 0.8,
        ease: 'power2.inOut'
      }, 0)
    }
    
    // Top disc color
    if (topMatRef.current) {
      tl.to(topMatRef.current.emissive, {
        r: targetColor.r,
        g: targetColor.g,
        b: targetColor.b,
        duration: 0.8,
        ease: 'power2.inOut'
      }, 0)
    }
    
    colorTimeline.current = tl
    
    return () => {
      tl.kill()
    }
  }, [podHealth])
  
  useFrame((state) => {
    if (!meshRef.current || !matRef.current || frozen) return
    const t = state.clock.elapsedTime
    
    // Dramatic flicker on failure (keep this for immediate visual feedback)
    if (podHealth < 0.3) {
      meshRef.current.visible = Math.random() > 0.15
      if (glowRef.current) glowRef.current.visible = Math.random() > 0.2
    } else {
      meshRef.current.visible = true
      if (glowRef.current) glowRef.current.visible = true
    }
    
    // Dynamic intensity modulation (subtle sine wave on top of base)
    matRef.current.emissiveIntensity = 2.5 + trafficLevel * 2 + Math.sin(t * 8) * 0.4
    
    if (glowMatRef.current) {
      glowMatRef.current.emissiveIntensity = 1.8 + trafficLevel * 1.5 + Math.sin(t * 6) * 0.3
    }
    
    if (baseMatRef.current) {
      baseMatRef.current.emissiveIntensity = 0.8 + Math.sin(t * 4) * 0.2
    }
    
    if (topMatRef.current) {
      topMatRef.current.emissiveIntensity = 1.2 + Math.sin(t * 5) * 0.3
    }
  })

  return (
    <group position={[0, -1, 0]}>
      {/* Core beam with premium material */}
      <mesh ref={meshRef} name="serviceBeam" castShadow>
        <cylinderGeometry args={[0.08, 0.08, 6, 16]} />
        <primitive object={beamCoreMaterial} attach="material" ref={matRef} />
      </mesh>
      
      {/* Outer glow layer */}
      <mesh ref={glowRef} name="serviceBeamGlow">
        <cylinderGeometry args={[0.12, 0.12, 6, 16]} />
        <primitive object={beamGlowMaterial} attach="material" ref={glowMatRef} />
      </mesh>
      
      {/* Enhanced base disc with glow */}
      <mesh position={[0, -3, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <circleGeometry args={[1.2, 32]} />
        <meshStandardMaterial
          ref={baseMatRef}
          color="#001a2e"
          emissive="#00aaff"
          emissiveIntensity={0.8}
          transparent
          opacity={0.6}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>
      
      {/* Base glow ring */}
      <mesh position={[0, -2.98, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.2, 1.5, 32]} />
        <meshStandardMaterial
          color="#001a2e"
          emissive="#00aaff"
          emissiveIntensity={0.5}
          transparent
          opacity={0.4}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Enhanced top disc */}
      <mesh position={[0, 3, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.5, 32]} />
        <meshStandardMaterial
          ref={topMatRef}
          color="#001a2e"
          emissive="#00ffaa"
          emissiveIntensity={1.2}
          transparent
          opacity={0.8}
          metalness={0.9}
          roughness={0.1}
        />
      </mesh>
      
      {/* Top glow ring */}
      <mesh position={[0, 3.02, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.5, 0.7, 32]} />
        <meshStandardMaterial
          color="#001a2e"
          emissive="#00ffaa"
          emissiveIntensity={0.8}
          transparent
          opacity={0.5}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  )
}
