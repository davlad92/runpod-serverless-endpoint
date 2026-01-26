import os
import io
import base64
import threading
from typing import Any, Dict, Optional, Tuple

import runpod
import torch
from PIL import Image
from diffusers import FluxPipeline

MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")

# Límites recomendados (ajusta según tu GPU)
MAX_WIDTH = int(os.getenv("MAX_WIDTH", "1024"))
MAX_HEIGHT = int(os.getenv("MAX_HEIGHT", "1024"))
MIN_WIDTH = int(os.getenv("MIN_WIDTH", "256"))
MIN_HEIGHT = int(os.getenv("MIN_HEIGHT", "256"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))
MIN_STEPS = int(os.getenv("MIN_STEPS", "1"))
ALIGN = int(os.getenv("ALIGN", "8"))

PIPE_LOCK = threading.Lock()

PIPE = None
PIPE_STATE = {"loaded": False, "mode": "unknown", "device": "cpu"}

def ensure_hf_cache() -> None:
    """
    Evita 'No space left on device' moviendo el cache de HF a un path con más disco.
    Recomendado en RunPod: /workspace (si existe) o /cache.
    """
    # Si ya lo definiste en env vars, respétalo.
    base = os.getenv("HF_HOME")

    # Fallbacks razonables si no está definido:
    if not base:
        if os.path.isdir("/workspace"):
            base = "/workspace/hf"
        elif os.path.isdir("/cache"):
            base = "/cache/hf"
        else:
            base = "/tmp/hf"  # último recurso (puede seguir siendo pequeño)

    os.environ.setdefault("HF_HOME", base)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(base, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(base, "transformers"))
    os.environ.setdefault("DIFFUSERS_CACHE", os.path.join(base, "diffusers"))

    # Algunas librerías respetan XDG_CACHE_HOME
    if os.path.isdir("/workspace"):
        os.environ.setdefault("XDG_CACHE_HOME", "/workspace/.cache")
    else:
        os.environ.setdefault("XDG_CACHE_HOME", os.path.join(base, ".cache"))

    # Crear directorios para evitar fallos al escribir
    for p in [
        os.environ["HF_HOME"],
        os.environ["HF_HUB_CACHE"],
        os.environ["TRANSFORMERS_CACHE"],
        os.environ["DIFFUSERS_CACHE"],
        os.environ["XDG_CACHE_HOME"],
    ]:
        os.makedirs(p, exist_ok=True)

def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))

def align_to(x: int, base: int) -> int:
    if base <= 1:
        return x
    return (x // base) * base

def sanitize_prompt(p: Any, default: str) -> str:
    if not isinstance(p, str):
        return default
    p = p.strip()
    if not p:
        return default
    return p[:2000]

def load_pipe() -> Tuple[FluxPipeline, Dict[str, str]]:
    # >>> CAMBIO SOLO PARA ESPACIO: asegurar cache antes de descargar <<<
    ensure_hf_cache()

    token = os.getenv("HF_TOKEN")
    dtype = torch.float16

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=token
    )

    pipe.enable_attention_slicing()

    state = {"mode": "cpu", "device": "cpu"}

    if torch.cuda.is_available():
        try:
            pipe.to("cuda")
            state = {"mode": "cuda", "device": "cuda"}
        except torch.cuda.OutOfMemoryError:
            pipe.enable_sequential_cpu_offload()
            state = {"mode": "cpu_offload", "device": "cuda"}
    else:
        state = {"mode": "cpu", "device": "cpu"}

    return pipe, state

def make_generator(seed: Optional[Any], device: str) -> Optional[torch.Generator]:
    if seed is None:
        return None
    try:
        s = int(seed)
    except Exception:
        return None
    gen_device = device if device in ("cuda", "cpu") else "cpu"
    return torch.Generator(device=gen_device).manual_seed(s)

def handle_inference(pipe: FluxPipeline, prompt: str, width: int, height: int, steps: int, generator):
    with torch.inference_mode():
        return pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator
        )

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    global PIPE, PIPE_STATE

    if PIPE is None:
        print("Loading pipeline...")
        PIPE, st = load_pipe()
        PIPE_STATE.update({"loaded": True, **st})
        print(
            f"Pipeline loaded. mode={PIPE_STATE['mode']} device={PIPE_STATE['device']} "
            f"HF_HOME={os.getenv('HF_HOME')} HF_HUB_CACHE={os.getenv('HF_HUB_CACHE')}"
        )

    job_input = (event.get("input") or {}) if isinstance(event, dict) else {}
    default_prompt = "A cute robot barista, cinematic lighting"
    prompt = sanitize_prompt(job_input.get("prompt"), default_prompt)

    width  = clamp_int(job_input.get("width"),  MIN_WIDTH,  MAX_WIDTH,  512)
    height = clamp_int(job_input.get("height"), MIN_HEIGHT, MAX_HEIGHT, 512)
    steps  = clamp_int(job_input.get("steps"),  MIN_STEPS,  MAX_STEPS,  20)

    width = align_to(width, ALIGN)
    height = align_to(height, ALIGN)

    seed = job_input.get("seed")
    generator = make_generator(seed, PIPE_STATE.get("device", "cpu"))

    with PIPE_LOCK:
        result = handle_inference(PIPE, prompt, width, height, steps, generator)

    image = result.images[0]
    return {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "mode": PIPE_STATE.get("mode"),
        "image_base64": pil_to_base64_png(image),
        "content_type": "image/png"
    }

runpod.serverless.start({"handler": handler})
