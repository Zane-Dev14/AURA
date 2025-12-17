from locust import HttpUser, task, between

class User(HttpUser):
    wait_time = between(0.5, 2)

    @task(3)
    def view_ui(self):
        self.client.get("/")

    @task(2)
    def read_quotes(self):
        self.client.get("/api/quotes")

    @task(1)
    def health(self):
        self.client.get("/health")
