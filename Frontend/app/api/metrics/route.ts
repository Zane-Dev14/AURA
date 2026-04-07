import { NextResponse } from 'next/server'
import { getMockMetrics } from '@/lib/mockMetrics'

const PROM_URL = process.env.PROMETHEUS_URL ?? 'http://localhost:9090'

async function queryProm(q: string): Promise<number> {
  const url = `${PROM_URL}/api/v1/query?query=${encodeURIComponent(q)}`
  const res = await fetch(url, { signal: AbortSignal.timeout(3000) })
  const json = await res.json()
  const val = json?.data?.result?.[0]?.value?.[1]
  return val !== undefined ? parseFloat(val) : NaN
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url)
  const scene = (searchParams.get('scene') ?? 'calm') as Parameters<typeof getMockMetrics>[0]
  const demoMode = process.env.NEXT_PUBLIC_DEMO_MODE === 'true'

  if (demoMode) {
    return NextResponse.json(getMockMetrics(scene))
  }

  try {
    const [p99, rps] = await Promise.all([
      queryProm(`histogram_quantile(0.99, sum by (le) (rate(envoy_http_downstream_rq_time_bucket{job="api"}[1m])))`),
      queryProm(`sum(rate(envoy_http_downstream_rq_total{job="api"}[1m]))`),
    ])

    return NextResponse.json({
      failures: 0,
      pods: 3,
      latencyMs: isNaN(p99) ? 0 : Math.round(p99),
      cpuPercent: 0,
      rps: isNaN(rps) ? 0 : Math.round(rps),
    })
  } catch {
    return NextResponse.json(getMockMetrics(scene))
  }
}
