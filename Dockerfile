FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    DIFFUSERS_CACHE=/cache/huggingface

# Base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy app files
WORKDIR /app
COPY requirements.txt handler.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create HF cache dir
RUN mkdir -p /cache/huggingface

CMD ["python3", "handler.py"]
