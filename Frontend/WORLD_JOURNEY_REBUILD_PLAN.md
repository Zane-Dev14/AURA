={3.0}  // Strong lighting from HDRI
/>

// Procedural sky gradient
<mesh>
  <sphereGeometry args={[500, 32, 32]} />
  <shaderMaterial
    vertexShader={skyVertexShader}
    fragmentShader={skyFragmentShader}
    side={THREE.BackSide}
    uniforms={{
      topColor: { value: new THREE.Color('#000510') },
      bottomColor: { value: new THREE.Color('#1a0a2e') },
      offset: { value: 33 },
      exponent: { value: 0.6 }
    }}
  />
</mesh>
```

#### PROBLEM 7: Placeholder Geometry
**Current:** Boxes and spheres for pods, generic shapes
**Fix:**
- Design custom pod geometry (icosahedron with displacement)
- Add detail to all interactive elements
- Use instanced meshes for performance
- Apply proper materials (metallic, emissive accents)

```typescript
// Custom pod geometry
function PodMesh() {
  const geometry = useMemo(() => {
    const geo = new THREE.IcosahedronGeometry(0.5, 2)
    const positions = geo.attributes.position
    
    // Add displacement for detail
    for (let i = 0; i < positions.count; i++) {
      const vertex = new THREE.Vector3()
      vertex.fromBufferAttribute(positions, i)
      vertex.normalize().multiplyScalar(
        0.5 + Math.random() * 0.1
      )
      positions.setXYZ(i, vertex.x, vertex.y, vertex.z)
    }
    
    geo.computeVertexNormals()
    return geo
  }, [])
  
  return (
    <mesh geometry={geometry} castShadow>
      <meshStandardMaterial
        color="#00ffaa"
        emissive="#00ffaa"
        emissiveIntensity={0.5}
        metalness={0.8}
        roughness={0.2}
      />
    </mesh>
  )
}
```

---

### Visual Language to Establish

**Color Palette:**
- **Primary:** Cyan (#00ffaa) for healthy/active states
- **Secondary:** Deep blue (#001a3a) for calm/stable
- **Accent:** Purple (#8844ff) for energy/transformation
- **Warning:** Orange (#ff6600) for stress/traffic
- **Danger:** Red (#ff0033) for failure/crisis
- **Background:** Very dark blue-black (#000510)

**Material Guidelines:**
- **Metals:** High metalness (0.9), low roughness (0.2)
- **Glass:** Transmission 1.0, thickness 0.5, roughness 0.1
- **Emissives:** Intensity 0.5-2.0 (never above 2.0)
- **Organics:** Medium roughness (0.6), low metalness (0.1)

**Lighting Ratios:**
- **Key:Fill:Rim = 3:1:2** (standard three-point)
- **Ambient:** Very low (0.2 intensity) for drama
- **Shadows:** Soft, PCF filtering, 2048x2048 maps

**Fog Layers:**
- **Near fog:** 0.008-0.015 density (atmospheric haze)
- **Mid fog:** 0.015-0.025 density (depth separation)
- **Far fog:** 0.025-0.035 density (distance fade)

---

## 8. IMPLEMENTATION MAPPING

### File-by-File Changes

---

#### [`components/AuraDemo.tsx`](components/AuraDemo.tsx)

**CHANGES:**
- Remove scene-based fog presets (lines 23-33)
- Remove scene-based bloom intensity (lines 72-77)
- Replace with zone-based system
- Update Canvas camera to start at intro position

**DELETE:**
```typescript
const fogPresets: Record<string, { color: string; density: number }> = { ... }
const bloomIntensity = scene === 'transform' ? 8 : ...
```

**REPLACE WITH:**
```typescript
const zonePresets: Record<string, { fog: FogConfig; bloom: number }> = {
  awakening: { fog: { color: '#010308', density: 0.008 }, bloom: 1.2 },
  discovery: { fog: { color: '#020510', density: 0.012 }, bloom: 1.2 },
  passage: { fog: { color: '#220800', density: 0.025 }, bloom: 1.5 },
  crisis: { fog: { color: '#110000', density: 0.008 }, bloom: 1.8 },
  transformation: { fog: { color: '#0a0515', density: 0.010 }, bloom: 2.0 },
  horizon: { fog: { color: '#020a08', density: 0.012 }, bloom: 1.2 }
}
```

**UPDATE:**
```typescript
camera={{ position: [0, 10, 30], fov: 55, near: 0.1, far: 1000 }}
```

---

#### [`components/SceneManager.tsx`](components/SceneManager.tsx)

**STATUS:** DELETE ENTIRELY

**REASON:** Scene-based switching is the core problem. Replace with zone-based world controller.

**CREATE NEW:** [`components/WorldController.tsx`](components/WorldController.tsx)

```typescript
'use client'
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { useSceneStore } from '@/store/useSceneStore'
import Airship from './r3f/Airship'
import ZoneAwakening from './zones/ZoneAwakening'
import ZoneDiscovery from './zones/ZoneDiscovery'
import ZonePassage from './zones/ZonePassage'
import ZoneCrisis from './zones/ZoneCrisis'
import ZoneTransformation from './zones/ZoneTransformation'
import ZoneHorizon from './zones/ZoneHorizon'
import CameraController from './r3f/CameraController'

export default function WorldController() {
  const airshipRef = useRef<THREE.Group>(null)
  const currentZone = useRef('awakening')
  
  useFrame(() => {
    if (!airshipRef.current) return
    
    const z = airshipRef.current.position.z
    
    // Determine zone based on Z position
    const newZone = 
      z > -100 ? 'awakening' :
      z > -250 ? 'discovery' :
      z > -400 ? 'passage' :
      z > -550 ? 'crisis' :
      z > -700 ? 'transformation' :
      'horizon'
    
    if (newZone !== currentZone.current) {
      handleZoneTransition(currentZone.current, newZone)
      currentZone.current = newZone
    }
  })
  
  return (
    <>
      <CameraController airshipRef={airshipRef} />
      <Airship ref={airshipRef} />
      
      {/* All zones render simultaneously, visibility controlled by Z position */}
      <ZoneAwakening />
      <ZoneDiscovery />
      <ZonePassage />
      <ZoneCrisis />
      <ZoneTransformation />
      <ZoneHorizon />
    </>
  )
}
```

---

#### [`components/LoadingGate.tsx`](components/LoadingGate.tsx)

**CHANGES:**
- Replace static loading with interactive version
- Add mouse-reactive AURA text
- Add jiggly loading bar
- Add particle attraction system

**KEEP:** Lines 1-11 (imports and setup)

**REPLACE:** Lines 39-145 with new interactive loading:

```typescript
return (
  <AnimatePresence>
    <motion.div className="loading-gate-interactive">
      {/* Interactive AURA text */}
      <InteractiveAURAText mousePos={mousePos} />
      
      {/* Jiggly loading bar */}
      <JigglyLoadingBar progress={progress} mousePos={mousePos} />
      
      {/* Particle attraction field */}
      <ParticleField mousePos={mousePos} />
    </motion.div>
  </AnimatePresence>
)
```

**ADD NEW COMPONENTS:**
- `InteractiveAURAText`: Letters react to mouse proximity
- `JigglyLoadingBar`: Segments bounce with mouse
- `ParticleField`: Particles attracted to cursor

---

#### [`components/r3f/CameraController.tsx`](components/r3f/CameraController.tsx)

**CHANGES:**
- Remove scene-based presets (lines 20-79)
- Replace with zone-based camera modes
- Fix broken follow camera (lines 247-283)
- Add smooth mode transitions

**DELETE:**
```typescript
const cameraPresets: Record<Scene, CameraPreset> = { ... }
```

**REPLACE WITH:**
```typescript
const cameraModes: Record<Zone, CameraMode> = {
  awakening: {
    type: 'cinematic-intro',
    offset: [0, 10, 30],
    lookAt: [0, 2, 0],
    fov: 55,
    lerpFactor: 0.05
  },
  discovery: {
    type: 'follow-drift',
    offset: [0, 6, 12],
    lookAt: 'predicted',
    fov: 55,
    lerpFactor: 0.05,
    mouseInfluence: 2
  },
  // ... etc
}
```

**FIX FOLLOW CAMERA (lines 247-283):**
```typescript
// CURRENT (BROKEN):
camera.position.lerp(targetPos, lerpFactor * 0.15)

// NEW (FIXED):
const targetOffset = new THREE.Vector3(0, 6, 12)
const targetPosition = airship.position.clone().add(targetOffset)

// Add velocity-based lag
const velocityLag = airshipVelocity.clone().multiplyScalar(-0.5)
targetPosition.add(velocityLag)

// Frame-rate independent lerp
const lerpFactor = 1 - Math.pow(0.001, delta)
camera.position.lerp(targetPosition, lerpFactor * 0.15)

// Look ahead based on velocity
const predictedPosition = airship.position.clone()
  .add(airshipVelocity.clone().multiplyScalar(0.3))
camera.lookAt(predictedPosition)
```

---

#### [`components/r3f/Ground.tsx`](components/r3f/Ground.tsx)

**STATUS:** DELETE ENTIRELY

**REASON:** Grid floor is immersion-breaking. Replace with fog layers and zone-specific platforms.

**NO REPLACEMENT FILE** — ground is handled per-zone in zone components.

---

#### [`components/scenes/`](components/scenes/) (All Scene Files)

**STATUS:** DELETE ALL SCENE FILES

**FILES TO DELETE:**
- `CalmWorld.tsx`
- `SystemReveal.tsx`
- `TrafficScene.tsx`
- `FailureScene.tsx`
- `EmotionalBeat.tsx`
- `QMixActivation.tsx`
- `Transformation.tsx`
- `RecoveryScene.tsx`
- `ComparisonScene.tsx`

**REASON:** Scene-based architecture is the problem. Replace with zone-based continuous world.

**CREATE NEW:** [`components/zones/`](components/zones/) directory

**NEW FILES:**
- `ZoneAwakening.tsx` (replaces CalmWorld)
- `ZoneDiscovery.tsx` (replaces SystemReveal)
- `ZonePassage.tsx` (replaces TrafficScene + FailureScene)
- `ZoneCrisis.tsx` (replaces EmotionalBeat + QMixActivation)
- `ZoneTransformation.tsx` (replaces Transformation)
- `ZoneHorizon.tsx` (replaces RecoveryScene + ComparisonScene)

---

#### [`store/useSceneStore.ts`](store/useSceneStore.ts)

**CHANGES:**
- Replace `Scene` type with `Zone` type
- Update state management for continuous world
- Add airship position tracking
- Remove scene-based presets

**REPLACE:**
```typescript
export type Scene = 'calm' | 'system' | ... // DELETE

export type Zone = 
  | 'awakening'
  | 'discovery'
  | 'passage'
  | 'crisis'
  | 'transformation'
  | 'horizon'
```

**ADD:**
```typescript
interface SceneStore {
  zone: Zone
  airshipPosition: THREE.Vector3
  forwardDrift: number
  // ... existing fields
}
```

---

#### NEW FILES TO CREATE

**1. [`components/WorldController.tsx`](components/WorldController.tsx)**
- Replaces SceneManager
- Handles zone detection based on airship Z position
- Manages zone transitions
- Coordinates camera and environment

**2. [`components/zones/ZoneAwakening.tsx`](components/zones/ZoneAwakening.tsx)**
- Renders geometry for Z: 0 to -100
- Floating rocks, fog layers, discovery nodes
- No grid floor

**3. [`components/zones/ZoneDiscovery.tsx`](components/zones/ZoneDiscovery.tsx)**
- Renders geometry for Z: -100 to -250
- Places heart_of_the_forest.glb at [0, -30, -400]
- Pod orbit visualization
- Energy beams

**4. [`components/zones/ZonePassage.tsx`](components/zones/ZonePassage.tsx)**
- Renders geometry for Z: -250 to -400
- Places hyrule_castle_interior.glb at [0, 0, -325]
- Traffic particles, failing pods
- Tunnel walls with emissive cracks

**5. [`components/zones/ZoneCrisis.tsx`](components/zones/ZoneCrisis.tsx)**
- Renders geometry for Z: -400 to -550
- Empty space with single platform
- QMix activation cube
- Debris field

**6. [`components/zones/ZoneTransformation.tsx`](components/zones/ZoneTransformation.tsx)**
- Renders geometry for Z: -550 to -700
- Healed pod grid
- Energy connections
- Particle explosion

**7. [`components/zones/ZoneHorizon.tsx`](components/zones/ZoneHorizon.tsx)**
- Renders geometry for Z: -700 to -800
- Comparison platforms
- Landing pad
- Portal loop mechanism

**8. [`components/r3f/FogLayers.tsx`](components/r3f/FogLayers.tsx)**
- Replaces Ground.tsx
- Renders invisible fog planes for depth
- No visible geometry

**9. [`lib/zoneTransitions.ts`](lib/zoneTransitions.ts)**
- Handles smooth transitions between zones
- Fog crossfades
- Lighting transitions
- Audio crossfades

**10. [`lib/proximitySystem.ts`](lib/proximitySystem.ts)**
- Manages node activation based on distance
- Handles approach feedback
- Triggers events on proximity

---

## 9. BUILD ORDER

### Phase 1: Foundation (Critical Path)
**Priority:** Immersion-breaking elements must go first

1. **DELETE immersion breakers**
   - Delete [`components/r3f/Ground.tsx`](components/r3f/Ground.tsx)
   - Delete all files in [`components/scenes/`](components/scenes/)
   - Delete scene-based logic from [`components/SceneManager.tsx`](components/SceneManager.tsx)

2. **Create zone architecture**
   - Create [`components/zones/`](components/zones/) directory
   - Create [`components/WorldController.tsx`](components/WorldController.tsx)
   - Update [`store/useSceneStore.ts`](store/useSceneStore.ts) with Zone types

3. **Fix camera system**
   - Rewrite follow camera in [`components/r3f/CameraController.tsx`](components/r3f/CameraController.tsx)
   - Add velocity-based lag
   - Add predictive look-ahead
   - Remove broken drag rotation

---

### Phase 2: World Building
**Priority:** Establish continuous space

4. **Create Zone 1: Awakening**
   - Build [`components/zones/ZoneAwakening.tsx`](components/zones/ZoneAwakening.tsx)
   - Add fog layers (no ground)
   - Add floating rocks
   - Add discovery nodes
   - Test Z: 0 to -100 progression

5. **Create Zone 2: Discovery**
   - Build [`components/zones/ZoneDiscovery.tsx`](components/zones/ZoneDiscovery.tsx)
   - Place [`heart_of_the_forest.glb`](public/models/heart_of_the_forest.glb)
   - Add pod orbit
   - Test Z: -100 to -250 progression

6. **Create Zone 3: Passage**
   - Build [`components/zones/ZonePassage.tsx`](components/zones/ZonePassage.tsx)
   - Place [`hyrule_castle_interior.glb`](public/models/hyrule_castle_interior.glb)
   - Add tunnel walls
   - Test Z: -250 to -400 progression

---

### Phase 3: Interaction Systems
**Priority:** Make world responsive

7. **Build proximity system**
   - Create [`lib/proximitySystem.ts`](lib/proximitySystem.ts)
   - Implement distance-based activation
   - Add approach feedback
   - Test with discovery nodes

8. **Build zone transitions**
   - Create [`lib/zoneTransitions.ts`](lib/zoneTransitions.ts)
   - Implement fog crossfades
   - Implement lighting transitions
   - Test smooth zone changes

9. **Add forward drift**
   - Update [`components/r3f/Airship.tsx`](components/r3f/Airship.tsx)
   - Add constant forward velocity
   - Add mouse influence
   - Test progression through zones

---

### Phase 4: Remaining Zones
**Priority:** Complete the journey

10. **Create Zone 4: Crisis**
    - Build [`components/zones/ZoneCrisis.tsx`](components/zones/ZoneCrisis.tsx)
    - Add QMix activation cube
    - Add debris field
    - Test Z: -400 to -550 progression

11. **Create Zone 5: Transformation**
    - Build [`components/zones/ZoneTransformation.tsx`](components/zones/ZoneTransformation.tsx)
    - Add pod grid healing
    - Add particle explosion
    - Test Z: -550 to -700 progression

12. **Create Zone 6: Horizon**
    - Build [`components/zones/ZoneHorizon.tsx`](components/zones/ZoneHorizon.tsx)
    - Add comparison platforms
    - Add portal loop
    - Test Z: -700 to -800 progression

---

### Phase 5: Visual Polish
**Priority:** Premium presentation

13. **Fix lighting**
    - Update [`components/r3f/LightingRig.tsx`](components/r3f/LightingRig.tsx)
    - Implement three-point lighting
    - Enable soft shadows
    - Reduce ambient intensity

14. **Fix post-processing**
    - Update bloom in [`components/AuraDemo.tsx`](components/AuraDemo.tsx)
    - Reduce max intensity to 2.0
    - Increase luminance threshold to 0.9
    - Add depth of field

15. **Fix HDRI usage**
    - Set background={false} in Environment
    - Reduce backgroundIntensity to 0.1
    - Increase environmentIntensity to 3.0
    - Add procedural sky gradient

---

### Phase 6: Loading Experience
**Priority:** First impression

16. **Rebuild loading screen**
    - Update [`components/LoadingGate.tsx`](components/LoadingGate.tsx)
    - Add interactive AURA text
    - Add jiggly loading bar
    - Add particle attraction

17. **Create intro sequence**
    - Update [`components/r3f/IntroSequence.tsx`](components/r3f/IntroSequence.tsx)
    - Implement fade from black
    - Add camera dolly forward
    - Add control handoff

---

### Phase 7: Testing & Refinement
**Priority:** Quality assurance

18. **Performance optimization**
    - Profile frame rate across all zones
    - Optimize particle counts
    - Implement LOD for distant geometry
    - Test on target hardware

19. **Interaction tuning**
    - Test all proximity triggers
    - Adjust activation radii
    - Fine-tune forward drift speeds
    - Test mouse influence

20. **Visual refinement**
    - Adjust fog densities
    - Balance lighting ratios
    - Fine-tune bloom
    - Test color palette consistency

---

## 10. SUCCESS CHECKLIST

### Immersion Criteria
**The experience should feel like a world, not a demo:**

- [ ] **No visible grid or floor** — only fog and atmospheric depth
- [ ] **No UI scene buttons** — progression is purely spatial
- [ ] **No placeholder geometry** — all objects have detail and purpose
- [ ] **No flat planes** — all surfaces have depth or are invisible
- [ ] **Camera feels natural** — smooth follow without jerkiness
- [ ] **Movement has consequence** — approaching objects triggers responses
- [ ] **Environment is continuous** — no loading or scene cuts
- [ ] **Fog creates depth** — multiple layers separate foreground/background
- [ ] **Lighting has direction** — clear key/fill/rim separation
- [ ] **Shadows are present** — objects cast and receive shadows
- [ ] **Bloom is subtle** — only brightest elements glow
- [ ] **HDRI lights the scene** — not visible as blurred background
- [ ] **Models are integrated** — heart_of_the_forest and hyrule_castle feel natural
- [ ] **Forward drift is constant** — user influences but doesn't control
- [ ] **Mouse creates subtle offset** — not direct control
- [ ] **Zones blend smoothly** — transitions are gradual, not abrupt
- [ ] **Intro feels cinematic** — fade from black, camera reveal
- [ ] **Loading is interactive** — AURA text and bar react to mouse
- [ ] **Portal loops gracefully** — experience can restart seamlessly

---

### Technical Criteria
**The implementation should be clean and performant:**

- [ ] **60 FPS on target hardware** — smooth throughout journey
- [ ] **No scene-based switching** — single continuous world
- [ ] **Zone detection is spatial** — based on airship Z position
- [ ] **Camera modes transition smoothly** — GSAP tweens between modes
- [ ] **Proximity system is efficient** — distance checks optimized
- [ ] **Fog transitions are smooth** — lerp between zone presets
- [ ] **Lighting transitions are smooth** — GSAP tweens for colors/intensities
- [ ] **Shadows are optimized** — 2048x2048 maps, soft filtering
- [ ] **Particles are instanced** — GPU instancing for performance
- [ ] **Models are loaded efficiently** — preload during loading screen
- [ ] **Audio crossfades work** — smooth transitions between zone tracks
- [ ] **Store is updated correctly** — zone state reflects airship position
- [ ] **No memory leaks** — proper cleanup on unmount
- [ ] **No console errors** — clean execution
- [ ] **Code is maintainable** — clear structure, good comments

---

### Emotional Criteria
**The experience should evoke specific feelings:**

- [ ] **Wonder** — initial reveal creates awe
- [ ] **Agency** — user feels they control the journey
- [ ] **Discovery** — finding nodes feels rewarding
- [ ] **Tension** — tunnel passage creates urgency
- [ ] **Vulnerability** — crisis zone feels isolating
- [ ] **Hope** — QMix activation is uplifting
- [ ] **Triumph** — transformation is celebratory
- [ ] **Satisfaction** — comparison view validates journey
- [ ] **Curiosity** — portal invites replay

---

### Comparison Test
**Before vs After:**

**BEFORE (Demo Feel):**
- Click button → scene loads → static view
- Grid floor visible
- Flat lighting
- Overbloom washes out details
- Camera feels floaty
- Movement is arbitrary
- No sense of place

**AFTER (World Feel):**
- Fade from black → camera reveals → smooth handoff
- Fog layers create depth
- Directional lighting with shadows
- Subtle bloom on emissives only
- Camera follows naturally
- Movement triggers events
- Strong sense of continuous space

**Test:** Show to someone unfamiliar. Ask: "Does this feel like a tech demo or a world?"
**Success:** They say "world" without hesitation.

---

## CONCLUSION

This plan transforms AURA from a scene-based technical demo into a continuous cinematic world journey. The key shifts are:

1. **Spatial progression** replaces UI-driven scene switching
2. **Continuous world** replaces discrete scenes
3. **Camera as director** replaces broken follow camera
4. **Purpose-driven interaction** replaces empty movement
5. **Atmospheric depth** replaces grid floor
6. **Integrated models** replace placeholder geometry
7. **Subtle bloom** replaces overbloom
8. **Directional lighting** replaces flat lighting
9. **Interactive loading** replaces static screen
10. **Zone-based architecture** replaces scene-based architecture

**Implementation is ready to begin.** Follow the build order strictly, starting with deletion of immersion-breaking elements, then building the zone architecture, then adding interaction systems, then polishing visuals.

**Estimated implementation time:** 40-60 hours for full rebuild.

**Critical success factor:** Maintain continuous spatial experience throughout. Every change must serve the goal of creating a believable world, not a demo.

---

*Document complete. Ready for code mode implementation.*