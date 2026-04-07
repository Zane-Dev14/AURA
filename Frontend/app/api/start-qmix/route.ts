import { NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

declare global {
  var __qmixProc: { pid: number | undefined; lastLog: string } | undefined
}

const REPO_ROOT = path.resolve(process.cwd(), '..')

export async function POST() {
  const demoMode = process.env.NEXT_PUBLIC_DEMO_MODE === 'true'

  if (demoMode) {
    globalThis.__qmixProc = { pid: 99999, lastLog: '✅ AURA Agent Controller Started (demo)' }
    return NextResponse.json({ status: 'started', pid: 99999, mode: 'demo' })
  }

  if (globalThis.__qmixProc) {
    return NextResponse.json({ status: 'already-running', pid: globalThis.__qmixProc.pid })
  }

  const proc = spawn('python', ['deployment/agent_controller.py'], {
    cwd: REPO_ROOT,
    env: { ...process.env, AURA_SHADOW_MODE: 'false' },
    stdio: ['ignore', 'pipe', 'pipe'],
  })

  let lastLog = 'Starting...'
  proc.stdout.on('data', (d: Buffer) => { lastLog = d.toString().trim() })
  proc.stderr.on('data', (d: Buffer) => { lastLog = `ERR: ${d.toString().trim()}` })
  proc.on('close', () => { globalThis.__qmixProc = undefined })

  globalThis.__qmixProc = { pid: proc.pid, lastLog }

  return NextResponse.json({ status: 'started', pid: proc.pid })
}
