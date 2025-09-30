FROM python:3.11-slim

# Install locust
RUN pip install locust

# Copy test script
COPY locustfile.py /locustfile.py

WORKDIR /

CMD ["locust", "-f", "/locustfile.py"]
