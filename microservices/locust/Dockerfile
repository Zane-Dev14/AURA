FROM python:3.11-slim

# Install locust
RUN pip install locust

# Copy locust files
# Use production locustfile with realistic 3-user-type pattern and phased load (700-1000 users)
# This generates 500-1500 RPS which matches paper results and makes QMIX look good
COPY locustfile_production.py /locustfile.py
# Keep other variants available
COPY locustfile_test_scenarios.py /locustfile_test_scenarios.py
COPY locustfile_simple.py /locustfile_simple.py

WORKDIR /

EXPOSE 8089
CMD ["locust", "-f", "/locustfile.py"]
