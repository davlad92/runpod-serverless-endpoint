import torch
from diffusers import StableDiffusionPipeline
import base64
from io import BytesIO

# Load model (at container startup)
pipe = StableDiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
    use_safetensors=True,
    revision="fp16"
).to("cuda" if torch.cuda.is_available() else "cpu")

def handler(event):
    try:
        prompt = event["input"].get("prompt", "")
        if not prompt:
            return {"error": "Missing 'prompt'"}

        image = pipe(prompt).images[0]

        # Convert image to base64 string
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": img_str}

    except Exception as e:
        return {"error": str(e)}
