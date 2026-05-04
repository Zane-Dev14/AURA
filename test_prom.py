import requests
import json

PROM_URL = "http://127.0.0.1:9090"

def query_prom(query):
    try:
        r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": query}, timeout=10)
        result = r.json()
        if result["data"]["result"]:
            return float(result["data"]["result"][0]["value"][1])
    except Exception as e:
        print("Err:", e)
    return 0.0

print("api replicas:", query_prom('kube_deployment_spec_replicas{deployment="api"}'))
