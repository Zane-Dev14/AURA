import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AURA — Kubernetes Intelligence System',
  description: 'Cinematic Kubernetes autoscaling demo powered by QMIX Multi-Agent Reinforcement Learning',
  keywords: ['kubernetes', 'autoscaling', 'MARL', 'QMIX', 'demo'],
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
