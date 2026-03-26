'use client'
import { useRef, useMemo, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'

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
  const matRef = useRef<THREE.ShaderMaterial>(null)
  const { podCount, podHealth, frozen } = useSceneStore()

  const dummy = useMemo(() => new THREE.Object3D(), [])
  const scales = useRef<Float32Array>(new Float32Array(MAX_PODS).fill(0))
  const targetScales = useRef<Float32Array>(new Float32Array(MAX_PODS).fill(0))

  // Update target scales when podCount changes
  useEffect(() => {
    for (let i = 0; i < MAX_PODS; i++) {
      targetScales.current[i] = i < podCount ? 1 : 0
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
    if (!meshRef.current || !matRef.current) return
    if (!frozen) matRef.current.uniforms.uTime.value = state.clock.elapsedTime
    matRef.current.uniforms.uHealth.value = THREE.MathUtils.lerp(
      matRef.current.uniforms.uHealth.value,
      podHealth,
      delta * 2
    )
    matRef.current.uniforms.uPulse.value = 1

    for (let i = 0; i < MAX_PODS; i++) {
      // Animate scale toward target (pods pop in/out)
      scales.current[i] = THREE.MathUtils.lerp(
        scales.current[i],
        targetScales.current[i],
        delta * 4
      )
      const s = scales.current[i]
      const [x, y, z] = positions[i]
      dummy.position.set(x, y + Math.sin(state.clock.elapsedTime * 1.5 + i * 0.4) * 0.05, z)
      dummy.scale.setScalar(s * 0.8)
      dummy.updateMatrix()
      meshRef.current.setMatrixAt(i, dummy.matrix)
    }
    meshRef.current.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, MAX_PODS]} position={[0, -1, 0]}>
      <boxGeometry args={[0.7, 0.7, 0.7]} />
      <shaderMaterial
        ref={matRef}
        vertexShader={podVertShader}
        fragmentShader={podFragShader}
        uniforms={{
          uTime: { value: 0 },
          uHealth: { value: 1 },
          uPulse: { value: 1 },
        }}
        transparent
      />
    </instancedMesh>
  )
}
