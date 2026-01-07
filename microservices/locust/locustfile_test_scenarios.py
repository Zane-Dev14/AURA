from locust import HttpUser, task, between, LoadTestShape

class SimpleUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(3)
    def view_home(self):
        self.client.get("/", name="Homepage")

    @task(2)
    def view_api(self):
        self.client.get("/api/quotes", name="API Quotes")

class LightLoadShape(LoadTestShape):
    def tick(self):
        if self.get_run_time() < 600:
            return 100, 10
        return None

class ModerateLoadShape(LoadTestShape):
    def tick(self):
        t = self.get_run_time()
        if t < 200:
            return int(t * 500 / 200), 10
        elif t < 800:
            return 500, 10
        return None

class HighLoadShape(LoadTestShape):
    def tick(self):
        t = self.get_run_time()
        if t < 500:
            return int(t * 1000 / 500), 20
        elif t < 1100:
            return 1000, 20
        return None

class SpikeLoadShape(LoadTestShape):
    def tick(self):
        t = self.get_run_time()
        if t < 1000:
            return int(t * 2000 / 1000), 50
        elif t < 1600:
            return 2000, 50
        return None
