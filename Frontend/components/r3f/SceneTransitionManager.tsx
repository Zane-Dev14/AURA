'use client'
import { useEffect, useRef } from 'react'
import { useSceneStore } from '@/store/useSceneStore'
import { useThree } from '@react-three/fiber'
import { createSceneTimeline } from '@/lib/sceneAnimations'
import gsap from 'gsap'
import * as THREE from 'three'

export default function SceneTransitionManager() {
  const { scene, podHealth, trafficLevel, glitchIntensity } = useSceneStore()
  const { scene: threeScene, camera } = useThree()
  const currentTimeline = useRef<gsap.core.Timeline | null>(null)
  const prevScene = useRef(scene)

  useEffect(() => {
    if (prevScene.current !== scene) {
      // Kill previous timeline
      if (currentTimeline.current) {
        currentTimeline.current.kill()
      }

      // Gather scene objects
      const objects = {
        pods: threeScene.getObjectByName('podGrid') as THREE.InstancedMesh | undefined,
        beam: threeScene.getObjectByName('serviceBeam') as THREE.Mesh | undefined,
        beamGlow: threeScene.getObjectByName('serviceBeamGlow') as THREE.Mesh | undefined,
        airship: threeScene.getObjectByName('airship') as THREE.Group | undefined,
        camera,
        trafficLevel: { value: trafficLevel },
        glitchIntensity: { value: glitchIntensity },
        podHealth: { value: podHealth }
      }

      // Create and play new timeline for scene
      currentTimeline.current = createSceneTimeline(scene, objects)
      if (currentTimeline.current) {
        currentTimeline.current.play()
      }

      prevScene.current = scene
    }

    return () => {
      if (currentTimeline.current) {
        currentTimeline.current.kill()
      }
    }
  }, [scene, threeScene, camera, trafficLevel, glitchIntensity, podHealth])

  return null
}

// Made with Bob
