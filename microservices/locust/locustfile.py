from locust import HttpUser, task, between

class User(HttpUser):
    wait_time = between(1, 3)

    @task(4)
    def index(self):
        self.client.get("/health")

    @task(1)
    def api_sample(self):
        self.client.get("/api/quotes")
