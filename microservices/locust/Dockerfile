FROM python:3.11-slim

# Install locust
RUN pip install locust

# Copy locust files
# Use production locustfile (FIXED: /health and /api/quotes, NO LoadTestShape)
COPY locustfile_production.py /locustfile.py
# Keep other variants available
COPY locustfile_benchmark.py /locustfile_benchmark.py
COPY locustfile_test_scenarios.py /locustfile_test_scenarios.py
COPY locustfile_simple.py /locustfile_simple.py

WORKDIR /

EXPOSE 8089
CMD ["locust", "-f", "/locustfile.py"]
