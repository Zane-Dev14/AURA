import { NextResponse } from 'next/server'
import { execSync, spawn } from 'child_process'

async function getLocustPodName(): Promise<string | null> {
  try {
    const out = execSync('kubectl get pod -l app=locust -o jsonpath={.items[0].metadata.name}', {
      timeout: 5000,
    }).toString().trim()
    return out || null
  } catch {
    return null
  }
}

export async function POST() {
  const demoMode = process.env.NEXT_PUBLIC_DEMO_MODE === 'true'

  if (demoMode) {
    return NextResponse.json({ status: 'demo', message: 'Demo mode — traffic simulated' })
  }

  const podName = await getLocustPodName()
  if (!podName) {
    return NextResponse.json({ status: 'error', message: 'Locust pod not found' }, { status: 404 })
  }

  // Use CLI exec — more reliable than HTTP /swarm
  const proc = spawn('kubectl', [
    'exec', podName, '--',
    'locust', '-f', 'locustfile.py',
    '--headless', '-u', '100', '-r', '10',
    '--host=http://app:8080',
    '--run-time=5m',
  ], { detached: true, stdio: 'ignore' })
  proc.unref()

  return NextResponse.json({ status: 'started', podName })
}
