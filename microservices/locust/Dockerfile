FROM python:3.11-slim

# Install locust
RUN pip install locust
# Copy test script
COPY locustfile.py /locustfile.py

WORKDIR /

EXPOSE 8089
CMD ["locust", "-f", "/locustfile.py"]
