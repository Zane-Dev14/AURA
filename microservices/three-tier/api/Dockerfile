FROM python:3.11-slim

WORKDIR /api

RUN apt-get update \
 && apt-get install -y gcc default-libmysqlclient-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["gunicorn", "api:api",     "-k", "gevent",     "-w", "4",     "--worker-connections", "200",    "--timeout", "60",     "--bind", "0.0.0.0:5001"]
