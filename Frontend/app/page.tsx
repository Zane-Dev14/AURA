import dynamic from 'next/dynamic'

const AuraDemo = dynamic(() => import('@/components/AuraDemo'), { ssr: false })

export default function Home() {
  return <AuraDemo />
}
