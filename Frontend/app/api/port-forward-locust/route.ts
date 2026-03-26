import { NextResponse } from 'next/server'
import { spawn } from 'child_process'

declare global {
  var __locustPortForward: ReturnType<typeof spawn> | undefined
  var __locustReady: boolean | undefined
}

export async function POST() {
  // Singleton: only start one port-forward process
  if (!globalThis.__locustPortForward) {
    const proc = spawn('kubectl', ['port-forward', 'svc/locust', '8089:8089'], {
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    proc.stdout.on('data', () => { globalThis.__locustReady = true })
    proc.stderr.on('data', (d) => {
      const msg = d.toString()
      if (msg.includes('Forwarding from')) globalThis.__locustReady = true
    })
    proc.on('close', () => {
      globalThis.__locustPortForward = undefined
      globalThis.__locustReady = false
    })
    globalThis.__locustPortForward = proc

    // Give it 2s to bind
    await new Promise((r) => setTimeout(r, 2000))
  }

  return NextResponse.json({
    status: globalThis.__locustReady ? 'ready' : 'starting',
    url: 'http://localhost:8089',
  })
}
