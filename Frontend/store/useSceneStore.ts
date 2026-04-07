import { create } from 'zustand'

export type Scene =
  | 'calm'
  | 'system'
  | 'traffic'
  | 'failure'
  | 'emotional'
  | 'qmix'
  | 'transform'
  | 'recovery'
  | 'comparison'

export type AirshipState =
  | 'patrol'
  | 'stressed'
  | 'falling'
  | 'locked'
  | 'powered'
  | 'stable'

export interface Metrics {
  failures: number
  pods: number
  latencyMs: number
  cpuPercent: number
  rps: number
}

interface SceneStore {
  scene: Scene
  demoMode: boolean
  podCount: number
  podHealth: number       // 0–1: 0=red, 1=green
  trafficLevel: number    // 0–1
  glitchIntensity: number // 0–1
  frozen: boolean         // for white flash freeze frame
  airshipState: AirshipState
  audioUnlocked: boolean
  metrics: Metrics
  qmixPid: number | null
  qmixStatusMsg: string
  assetsLoaded: boolean
  introComplete: boolean

  setScene: (s: Scene) => void
  setPodCount: (n: number) => void
  setPodHealth: (h: number) => void
  setTrafficLevel: (t: number) => void
  setGlitchIntensity: (g: number) => void
  setFrozen: (f: boolean) => void
  setAirshipState: (s: AirshipState) => void
  unlockAudio: () => void
  setMetrics: (m: Partial<Metrics>) => void
  setQmixPid: (pid: number | null) => void
  setQmixStatusMsg: (msg: string) => void
  setAssetsLoaded: (v: boolean) => void
  setIntroComplete: (v: boolean) => void
  resetDemo: () => void
  skipTo: (s: Scene) => void
}

const defaultMetrics: Metrics = {
  failures: 0,
  pods: 3,
  latencyMs: 12,
  cpuPercent: 45,
  rps: 0,
}

export const useSceneStore = create<SceneStore>((set) => ({
  scene: 'calm',
  demoMode: process.env.NEXT_PUBLIC_DEMO_MODE === 'true',
  podCount: 3,
  podHealth: 1,
  trafficLevel: 0,
  glitchIntensity: 0,
  frozen: false,
  airshipState: 'patrol',
  audioUnlocked: false,
  metrics: defaultMetrics,
  qmixPid: null,
  qmixStatusMsg: '',
  assetsLoaded: false,
  introComplete: false,

  setScene: (scene) => set({ scene }),
  setPodCount: (podCount) => set({ podCount }),
  setPodHealth: (podHealth) => set({ podHealth }),
  setTrafficLevel: (trafficLevel) => set({ trafficLevel }),
  setGlitchIntensity: (glitchIntensity) => set({ glitchIntensity }),
  setFrozen: (frozen) => set({ frozen }),
  setAirshipState: (airshipState) => set({ airshipState }),
  unlockAudio: () => set({ audioUnlocked: true }),
  setMetrics: (m) => set((s) => ({ metrics: { ...s.metrics, ...m } })),
  setQmixPid: (qmixPid) => set({ qmixPid }),
  setQmixStatusMsg: (qmixStatusMsg) => set({ qmixStatusMsg }),
  setAssetsLoaded: (assetsLoaded) => set({ assetsLoaded }),
  setIntroComplete: (introComplete) => set({ introComplete }),

  resetDemo: () =>
    set({
      scene: 'calm',
      podCount: 3,
      podHealth: 1,
      trafficLevel: 0,
      glitchIntensity: 0,
      frozen: false,
      airshipState: 'patrol',
      metrics: defaultMetrics,
      qmixPid: null,
      qmixStatusMsg: '',
    }),

  skipTo: (scene) => {
    const presets: Record<Scene, Partial<SceneStore>> = {
      calm: { podCount: 3, podHealth: 1, trafficLevel: 0, glitchIntensity: 0, airshipState: 'patrol' },
      system: { podCount: 6, podHealth: 1, trafficLevel: 0.1, airshipState: 'patrol' },
      traffic: { podCount: 6, podHealth: 0.9, trafficLevel: 0.7, airshipState: 'stressed' },
      failure: { podCount: 2, podHealth: 0.1, trafficLevel: 1, glitchIntensity: 0.8, airshipState: 'falling' },
      emotional: { podCount: 1, podHealth: 0, trafficLevel: 1, glitchIntensity: 1, airshipState: 'falling' },
      qmix: { podCount: 1, podHealth: 0, trafficLevel: 0.8, glitchIntensity: 0.6, airshipState: 'locked' },
      transform: { podCount: 1, podHealth: 0.5, trafficLevel: 0.5, glitchIntensity: 0.2, airshipState: 'powered' },
      recovery: { podCount: 9, podHealth: 1, trafficLevel: 0.3, glitchIntensity: 0, airshipState: 'stable' },
      comparison: { podCount: 9, podHealth: 1, trafficLevel: 0, glitchIntensity: 0, airshipState: 'stable' },
    }
    set({ scene, ...(presets[scene] || {}) })
  },
}))
