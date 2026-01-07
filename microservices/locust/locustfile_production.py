from locust import HttpUser, task, between, LoadTestShape
import random
import time

from locust import HttpUser, task, between
import random
import time

class UniversityUser(HttpUser):
    """
    Simulates 3 types of users:
    - Light (70%)
    - Regular (25%)
    - Power (5%)
    """

    # Default, will be overridden safely
    wait_time = between(1, 3)

    def on_start(self):
        r = random.random()

        if r < 0.70:
            self.user_type = "light"
            self.wait_time = between(3, 8)

        elif r < 0.95:
            self.user_type = "regular"
            self.wait_time = between(1, 3)

        else:
            self.user_type = "power"
            self.wait_time = between(0.1, 0.5)

    @task(10)
    def view_homepage(self):
        self.client.get("/", name="Homepage")

    @task(3)
    def read_items(self):
        if self.user_type in ["regular", "power"]:
            self.client.get("/api/items", name="List Items")

    @task(2)
    def search(self):
        if self.user_type in ["regular", "power"]:
            self.client.get("/api/search?q=test", name="Search")

    @task(1)
    def post_action(self):
        if self.user_type == "power":
            self.client.post(
                "/api/action",
                json={"data": "test"},
                name="Post Action"
            )

    @task(2)
    def refresh_spam(self):
        if self.user_type == "power":
            for _ in range(random.randint(3, 6)):
                self.client.get("/api/items", name="Refresh Spam")
                time.sleep(0.1)


class ProductionDayShape(LoadTestShape):
    """
    Load shape: short-cycled but realistic.
    Returns (user_count, spawn_rate) or None to stop.
    """

    def __init__(self):
        super().__init__()
        self.phases = [
            (300, 500, 50),    # 0-5min ramp to 500
            (600, 500, 20),    # 5-10min hold
            (900, 800, 50),    # 10-15min ramp to 800
            (1200, 800, 20),   # 15-20min hold
            (1500, 2000, 200), # 20-25min ramp to 2000 (spike)
            (1800, 2000, 50),  # 25-30min peak
            (2100, 100, 200),  # 30-35min drop to 100
            (3600, 100, 10),   # 35-60min night steady
        ]

    def tick(self):
        run_time = self.get_run_time()
        for end_t, users, spawn in self.phases:
            if run_time <= end_t:
                return users, spawn
        return None

