from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)

    @task(4)
    def index(self):
        self.client.get("/")

    @task(1)
    def api_sample(self):
        # adjust path to a real API route
        self.client.get("/api/products")
