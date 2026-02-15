#!/bin/bash
# Test different GKE machine type configurations

echo "====== GKE EXACT PRICING - SCENARIOS ======"
echo ""
echo "Scenario 1: Your Current k3d (2 nodes, n2-standard-2 equivalent)"
echo "  Cost: (2 × $0.097118) + $0.10 = $0.294236/hour"
echo ""

echo "Scenario 2: Small Production (1 node, e2-standard-2)"
echo "  Cost: (1 × $0.067011) + $0.10 = $0.167011/hour"
echo "  Monthly: ~$120.80"
echo ""

echo "Scenario 3: Standard Production (2 nodes, n2-standard-4)"
echo "  Cost: (2 × $0.194236) + $0.10 = $0.488472/hour"
echo "  Monthly: ~$354.50"
echo ""

echo "Scenario 4: High Availability (3 nodes, n2-standard-4)"
echo "  Cost: (3 × $0.194236) + $0.10 = $0.682708/hour"
echo "  Monthly: ~$496.47"
echo ""

echo "Scenario 5: Memory-Optimized (2 nodes, n2-highmem-4)"
echo "  Cost: (2 × $0.262658) + $0.10 = $0.625316/hour"
echo "  Monthly: ~$454.35"
echo ""

echo "====== HOW TO CHANGE SETTINGS ======"
echo ""
echo "Edit: tools/gke_cost_report.py"
echo ""
echo "Then update these 2 lines:"
echo "  NODE_PRICE_PER_HOUR = <machine-type-price>"
echo "  NUM_NODES = <number-of-nodes>"
echo ""
echo "Examples:"
echo "  NODE_PRICE_PER_HOUR = 0.067011  # e2-standard-2"
echo "  NODE_PRICE_PER_HOUR = 0.194236  # n2-standard-4"
echo "  NODE_PRICE_PER_HOUR = 0.262658  # n2-highmem-4"
echo ""
echo "See: https://cloud.google.com/compute/all-pricing"
echo "     (Filter to us-central1 for exact prices)"

