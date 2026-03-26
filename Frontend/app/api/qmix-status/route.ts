import { NextResponse } from 'next/server'

export async function GET() {
  const p = globalThis.__qmixProc
  if (!p) {
    return NextResponse.json({ running: false, pid: null, lastLog: '' })
  }
  return NextResponse.json({
    running: true,
    pid: p.pid,
    lastLog: p.lastLog,
  })
}
