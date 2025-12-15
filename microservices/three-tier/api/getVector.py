import numpy as np
import requests



# Per service (9 dimensions)
METRIC_ORDER = [
    "replicas",
    "cpu_util",
    "memory_util",
    "request_rate",
    "avg_latency",
    "error_rate",
    "queue_length",
    "pending_pods",
    "ready_ratio",
]

PROM_QUERIES = {
    "replicas": """
        kube_deployment_status_replicas_available{{deployment="{service}"}}
    """,

    "cpu_util": """
        sum(rate(container_cpu_usage_seconds_total{{
            namespace="{ns}",
            pod=~"{service}-.*",
            container!="POD"
        }}[30s]))
        /
        sum(kube_pod_container_resource_requests_cpu_cores{{
            namespace="{ns}",
            pod=~"{service}-.*"
        }})
    """,

    "memory_util": """
        sum(container_memory_working_set_bytes{{
            namespace="{ns}",
            pod=~"{service}-.*",
            container!="POD"
        }})
        /
        sum(kube_pod_container_resource_requests_memory_bytes{{
            namespace="{ns}",
            pod=~"{service}-.*"
        }})
    """,

    "request_rate": """
        sum(rate(http_requests_total{{service="{service}"}}[30s]))
    """,

    "avg_latency": """
        sum(rate(http_request_duration_seconds_sum{{service="{service}"}}[30s]))
        /
        sum(rate(http_request_duration_seconds_count{{service="{service}"}}[30s]))
    """,

    "error_rate": """
        sum(rate(http_requests_total{{service="{service}",status=~"5.."}}[30s]))
        /
        sum(rate(http_requests_total{{service="{service}"}}[30s]))
    """,

    "queue_length": """
        avg(request_queue_length{{service="{service}"}})
    """,

    "pending_pods": """
        count(kube_pod_status_phase{{
            namespace="{ns}",
            pod=~"{service}-.*",
            phase="Pending"
       }})
    """,

    "ready_ratio": """
        sum(kube_pod_status_ready{{
            namespace="{ns}",
            pod=~"{service}-.*",
            condition="true"
        }})
        /
        count(kube_pod_info{{
            namespace="{ns}",
            pod=~"{service}-.*"
        }})
    """
}


class PrometheusVectorClient:
    def __init__(self, prom_url="http://localhost:9090", namespace=""):
        self.prom_url = prom_url.rstrip("/")
        self.ns = namespace

    def _query(self, promql: str) -> float:
        resp = requests.get(
            f"{self.prom_url}/api/v1/query",
            params={"query": promql}
        )
        resp.raise_for_status()
        data = resp.json()["data"]["result"]

        if not data:
            return 0.0

        return float(data[0]["value"][1])

    def service_vector(self, service: str) -> np.ndarray:
        values = []

        for metric in METRIC_ORDER:
            q = PROM_QUERIES[metric].format(service=service, ns=self.ns )
            values.append(self._query(q))

        return np.array(values, dtype=np.float32)

    def system_vector(self, services: list[str]) -> np.ndarray:
        vectors = [self.service_vector(s) for s in services]
        return np.concatenate(vectors)


#---------------------------------------------------------------------------------------------
def getVector(prometheus_url:str= "http://localhost:7070",servicesList:list[str]=["api"]) -> np.ndarray:
    """servicesList is the list of service names whose metrics you need.\n
    Example: ["api","auth", "payments", "orders"]\n
    Default: ["api"]
     
    This function returns an ndarray of format:\n
    ["replicas",\n
    "cpu_util",\n
    "memory_util",\n
    "request_rate",\n
    "avg_latency",\n
    "error_rate",\n
    "queue_length",\n
    "pending_pods",\n
    "ready_ratio",\n
    ]"""

    client = PrometheusVectorClient(
		prom_url=prometheus_url,
		namespace=""
	)
    state_vector = client.system_vector(servicesList)

    print(state_vector.shape)   # (27,)
    return state_vector
if __name__=="__main__":
	print(getVector())