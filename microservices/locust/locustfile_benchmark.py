"""
Locust load test file for AURA benchmark experiments.
NO LoadTestShape - runs continuously until stopped by benchmark script.
"""

from locust import HttpUser, task, between
import random


class LightUser(HttpUser):
    """Light user - mostly browsing"""
    weight = 7
    wait_time = between(2, 5)
    
    @task(10)
    def homepage(self):
        self.client.get("/health")
    
    @task(3)
    def browse_quotes(self):
        self.client.get("/api/quotes")
    
    @task(1)
    def view_quote(self):
        # API doesn't have individual quote endpoint, use list instead
        self.client.get("/api/quotes")


class RegularUser(HttpUser):
    """Regular user - browsing and creating"""
    weight = 2
    wait_time = between(1, 3)
    
    @task(5)
    def homepage(self):
        self.client.get("/health")
    
    @task(5)
    def list_quotes(self):
        self.client.get("/api/quotes")
    
    @task(3)
    def view_quote(self):
        # API doesn't have individual quote endpoint, use list instead
        self.client.get("/api/quotes")
    
    @task(2)
    def create_quote(self):
        self.client.post("/api/quotes", json={
            "text": f"Test quote {random.randint(1, 10000)}",
            "author": f"User {random.randint(1, 100)}"
        })


class PowerUser(HttpUser):
    """Power user - heavy API usage"""
    weight = 1
    wait_time = between(0.5, 1.5)
    
    @task(10)
    def spam_requests(self):
        self.client.get("/health")
    
    @task(5)
    def create_many_quotes(self):
        for _ in range(3):
            self.client.post("/api/quotes", json={
                "text": f"Bulk quote {random.randint(1, 10000)}",
                "author": "PowerUser"
            })
    
    @task(3)
    def list_quotes(self):
        self.client.get("/api/quotes")

# Made with Bob
