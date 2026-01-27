FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    DIFFUSERS_CACHE=/cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-pip libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy code
COPY handler.py .

# Make sure cache folder exists
RUN mkdir -p /cache/huggingface

# Start the handler
CMD ["python3", "handler.py"]
