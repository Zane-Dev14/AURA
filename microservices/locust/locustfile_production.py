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

Load shape: ProductionDayShape (60-min phased)
- 0-5min:   ramp to 500 users
- 5-10min:  hold 500
- 10-15min: ramp to 800
- 15-20min: hold 800
- 20-25min: spike to 2000
- 25-30min: peak 2000
- 30-35min: drop to 100
- 35-60min: steady 100
"""

from locust import HttpUser, task, between, LoadTestShape
import random
import time


class UniversityUser(HttpUser):
    """
    Simulates 3 types of users:
    - Light (70%): wait 3-8s between requests
    - Regular (25%): wait 1-3s between requests
    - Power (5%): wait 0.1-0.5s between requests
    """

    wait_time = between(1, 3)  # Default, overridden in on_start

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

