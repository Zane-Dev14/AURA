'use client'
import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import * as THREE from 'three'

const shockwaveVert = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const shockwaveFrag = `
uniform float uTime;
uniform float uProgress;
uniform vec3 uColor;
varying vec2 vUv;

void main() {
  vec2 uv = vUv - 0.5;
  float dist = length(uv);
  float ring = smoothstep(uProgress - 0.05, uProgress, dist)
             - smoothstep(uProgress, uProgress + 0.05, dist);
  float glow = pow(ring, 1.2) * 2.5;
  float angle = atan(uv.y, uv.x);
  float tendrils = sin(angle * 12.0 + uTime * 6.0) * 0.5 + 0.5;
  vec3 col = uColor * glow * (0.6 + 0.4 * tendrils);
  // Inner glow
  float inner = smoothstep(0.0, uProgress - 0.05, dist);
  col += uColor * (1.0 - inner) * 0.15 * uProgress;
  gl_FragColor = vec4(col, glow * 0.95);
}
`

interface Props {
  active: boolean
  onComplete?: () => void
}

export default function ShockwaveRing({ active, onComplete }: Props) {
  const meshRef = useRef<THREE.Mesh>(null)
  const matRef = useRef<THREE.ShaderMaterial>(null)
  const progressRef = useRef(0)
  const done = useRef(false)

  useFrame((state, delta) => {
    if (!matRef.current) return
    matRef.current.uniforms.uTime.value = state.clock.elapsedTime
    if (!active || done.current) return

    progressRef.current += delta * 0.55
    matRef.current.uniforms.uProgress.value = progressRef.current

    if (progressRef.current >= 1.2) {
      done.current = true
      onComplete?.()
    }
  })

  if (!active) return null

  return (
    <mesh ref={meshRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 0]}>
      <planeGeometry args={[30, 30, 1, 1]} />
      <shaderMaterial
        ref={matRef}
        vertexShader={shockwaveVert}
        fragmentShader={shockwaveFrag}
        uniforms={{
          uTime: { value: 0 },
          uProgress: { value: 0 },
          uColor: { value: new THREE.Color(0x00eeff) },
        }}
        transparent
        depthWrite={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}
