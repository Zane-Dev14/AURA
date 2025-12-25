FROM python:3.11-slim

WORKDIR /api

# install required packages for system --> MUST RUN THIS FOR THIS TO WORK!
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y gcc default-libmysqlclient-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python dependencies file first to leverage Docker cache layers
COPY requirements.txt ./

COPY . .

# RUN pip install mysqlclient - already in the requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

<<<<<<< Updated upstream
CMD ["python", "api.py"]
=======
CMD ["gunicorn", "api:api",
     "-k", "gevent",
     "-w", "4",
     "--worker-connections", "400",
     "--timeout", "60",
     "--bind", "0.0.0.0:5001"]
>>>>>>> Stashed changes
