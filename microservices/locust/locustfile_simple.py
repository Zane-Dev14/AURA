"""
Simple Locust file for AURA benchmark experiments.
Controlled via HTTP API (POST /swarm with user_count and spawn_rate).
"""

from locust import HttpUser, task, between
import random


class QuoteUser(HttpUser):
    """
    Simulates users accessing the quotes application.
    Mix of reads (GET) and writes (POST).
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    @task(7)  # 70% of requests
    def get_quotes(self):
        """List all quotes"""
        self.client.get("/api/quotes", name="GET /api/quotes")
    
    @task(2)  # 20% of requests
    def get_homepage(self):
        """Get homepage"""
        self.client.get("/", name="GET /")
    
    @task(1)  # 10% of requests
    def create_quote(self):
        """Create a new quote"""
        quote_data = {
            "text": f"Test quote {random.randint(1, 10000)}",
            "author": f"Author {random.randint(1, 100)}"
        }
        self.client.post("/api/quotes", json=quote_data, name="POST /api/quotes")

# Made with Bob
