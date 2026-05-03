FROM python:3.11-slim

# Install locust
RUN pip install locust

# Copy test scripts
# Production locustfile with realistic user types and 60-min phased load shape
# Default to the controlled test scenarios used for automated benchmarks
# Keep the production locustfile available as an alternate file.
COPY locustfile_test_scenarios.py /locustfile.py
COPY locustfile_production.py /locustfile_production.py
# Simple locustfile preserved as a backup
COPY locustfile.py /locustfile_simple.py

WORKDIR /

EXPOSE 8089
CMD ["locust", "-f", "/locustfile.py"]
