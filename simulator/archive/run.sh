#!/bin/bash
set -e  # stop if any command fails

# Base URL for Alibaba v2021 microservice trace
url='http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces'

# Create directories
mkdir -p data/MSResource data/MSRTQps

echo "⬇️ Downloading a tiny subset of Alibaba trace (~1 GB compressed)..."

# --- 1. Download 1/12 of the MSResource dataset (CPU + MEM) ---
cd data/MSResource
for((i=0;i<=0;i++)); do
    wget -c --retry-connrefused --tries=0 --timeout=50 ${url}/MSResource/MSResource_${i}.tar.gz
done

# --- 2. Download 1/25 of the MSRTQps dataset (RPS) ---
cd ../MSRTQps
for((i=0;i<=0;i++)); do
    wget -c --retry-connrefused --tries=0 --timeout=50 ${url}/MSRTQps/MSRTQps_${i}.tar.gz
done

cd ../..

echo "✅ Downloads complete. Now extract..."
