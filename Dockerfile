FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    DIFFUSERS_CACHE=/cache/huggingface

WORKDIR /app

# System dependencies (minimal)
RUN apt-get update && apt-get install -y \
    git python3-pip libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY handler.py .

RUN mkdir -p /cache/huggingface

CMD ["python3", "handler.py"]
