FROM python:3.11-slim

# Install locust
RUN pip install locust

# Copy test scripts
# Production locustfile with realistic user types and 60-min phased load shape
COPY locustfile_production.py /locustfile.py
# Simple locustfile as backup
COPY locustfile.py /locustfile_simple.py

WORKDIR /

EXPOSE 8089
CMD ["locust", "-f", "/locustfile.py"]
