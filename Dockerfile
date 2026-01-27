FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    DIFFUSERS_CACHE=/cache/huggingface

# Paquetes base
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# PyTorch (CUDA 12.1)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Dependencias RunPod + Diffusers
RUN pip install \
    runpod \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    pillow

# (Opcional pero útil) xformers puede mejorar memoria/velocidad en algunos casos
# Si te da problemas, comenta esta línea.
RUN pip install xformers --no-deps || true

RUN pip install --no-cache-dir "protobuf>=3.20.3,<5" sentencepiece tokenizers

WORKDIR /app
COPY handler.py /app/handler.py

# Directorio de cache para HF
RUN mkdir -p /cache/huggingface

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

CMD ["python3", "-m", "runpod"]


