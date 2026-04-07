import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // Allow large model files to be served from public/
  experimental: {},
  // Transpile three.js ecosystem for Next
  transpilePackages: ['three'],
}

export default nextConfig
