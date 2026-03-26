import { NextResponse } from 'next/server'
import { execSync } from 'child_process'

async function checkAppHealth(): Promise<{ status: number; ok: boolean }> {
  try {
    const res = await fetch('http://localhost:8080', {
      signal: AbortSignal.timeout(3000),
    })
    return { status: res.status, ok: res.ok }
  } catch {
    return { status: 0, ok: false }
  }
}

function getPodCount(svc: string): number {
  try {
    const out = execSync(
      `kubectl get deployment ${svc} -o jsonpath={.status.readyReplicas}`,
      { timeout: 3000 }
    ).toString().trim()
    return parseInt(out) || 0
  } catch {
    return -1
  }
}

export async function GET() {
  const demoMode = process.env.NEXT_PUBLIC_DEMO_MODE === 'true'

  if (demoMode) {
    return NextResponse.json({ appStatus: 200, pods: { api: 3, app: 3, db: 1 } })
  }

  const [health, apiPods, appPods, dbPods] = await Promise.all([
    checkAppHealth(),
    Promise.resolve(getPodCount('api')),
    Promise.resolve(getPodCount('app')),
    Promise.resolve(getPodCount('db')),
  ])

  return NextResponse.json({
    appStatus: health.status,
    appOk: health.ok,
    pods: { api: apiPods, app: appPods, db: dbPods },
  })
}
