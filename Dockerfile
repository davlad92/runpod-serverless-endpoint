# This Dockerfile is intentionally minimal — RunPod handles the rest
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

# Do NOT add CMD — RunPod handles serverless handler automatically
