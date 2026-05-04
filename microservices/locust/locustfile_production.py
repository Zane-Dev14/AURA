"""
Production load test for AURA 3-tier quotes application.

Simulates realistic user behavior with 3 user types:
- Light users (70%): Occasional homepage views
- Regular users (25%): Browse quotes, view homepage
- Power users (5%): Frequent reads + writes

FIXED ENDPOINTS (matching actual API):
- GET /           → Homepage (via app service)
- GET /api/quotes → List all quotes (via app→api)
- POST /api/quotes → Create new quote (power users only)

Load shape: ProductionDayShape (30-min phased)
- 0-3min:   ramp to 500 users
- 3-8min:   hold 500
- 8-11min:  ramp to 800
- 11-16min: hold 800
- 16-19min: spike to 2000
- 19-24min: sustain 2000
- 24-27min: drop to 100
- 27-30min: steady 100 → stop
"""

from locust import HttpUser, task, between, LoadTestShape
from types import MethodType
import random
import time


class UniversityUser(HttpUser):
    """
    Simulates 3 types of users:
    - Light (70%): wait 3-8s between requests
    - Regular (25%): wait 1-3s between requests
    - Power (5%): wait 0.1-0.5s between requests
    """

    wait_time = between(1, 2)  # Default, overridden in on_start

    
    def on_start(self):
        r = random.random()
        if r < 0.70:
            self.user_type = "light"
            self.wait_time = MethodType(between(3, 8), self)   # casual browser
        elif r < 0.95:
            self.user_type = "regular"
            self.wait_time = MethodType(between(1, 3), self)   # normal user
        else:
            self.user_type = "power"
            self.wait_time = MethodType(between(0.2, 0.8), self)  # heavy but not insane


    @task(10)
    def view_homepage(self):
        """All users view the homepage."""
        self.client.get("/", name="Homepage")

    @task(5)
    def read_quotes(self):
        """Regular and power users fetch the quotes list."""
        if self.user_type in ["regular", "power"]:
            self.client.get("/api/quotes", name="List Quotes")

    @task(2)
    def browse_quotes(self):
        """Regular and power users browse quotes (same endpoint, different name for metrics)."""
        if self.user_type in ["regular", "power"]:
            self.client.get("/api/quotes", name="Browse Quotes")

    @task(1)
    def create_quote(self):
        """Power users create new quotes."""
        if self.user_type == "power":
            self.client.post(
                "/api/quotes",
                json={
                    "text": f"Test quote {random.randint(1, 10000)}",
                    "author": f"User_{random.randint(1, 100)}"
                },
                name="Create Quote"
            )

    @task(2)
    def refresh_spam(self):
        """Power users rapidly refresh the quotes list."""
        if self.user_type == "power":
            for _ in range(random.randint(3, 6)):
                self.client.get("/api/quotes", name="Refresh Spam")
                time.sleep(0.1)


class ProductionDayShape(LoadTestShape):
    """
    30-minute realistic production load shape.
    """

    def __init__(self):
        super().__init__()

        self.phases = [
            # (end_time_sec, users, spawn_rate)

            (180, 1000, 100),     # 0–3 min: ramp to 1000
            (480, 1000, 50),     # 3–8 min: hold 1000

            (660, 2000, 100),     # 8–11 min: ramp to 2000
            (960, 2000, 50),     # 11–16 min: hold 2000

            (1140, 4000, 300),  # 16–19 min: spike to 4000
            (1440, 4000, 100),   # 19–24 min: sustain peak

            (1620, 200, 200),   # 24–27 min: drop to 200
            (1800, 200, 20),    # 27–30 min: steady low
        ]

    def tick(self):
        run_time = self.get_run_time()

        for end_t, users, spawn in self.phases:
            if run_time <= end_t:
                return users, spawn

        return None

# Made with Bob
