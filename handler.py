import runpod
from diffusers import FluxPipeline
import torch
from io import BytesIO
import base64

# ----------------------------
# Load the model at container startup
# ----------------------------
print("Loading FLUX.1-dev model...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
).to("cuda")

print("Model loaded successfully.")


# ----------------------------
# Helper: Convert PIL image to base64
# ----------------------------
def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str


# ----------------------------
# Main handler function
# ----------------------------
def handler(event):
    """
    event = {
      "input": {
         "prompt": "a futuristic city in the clouds"
      }
    }
    """

    try:
        prompt = event["input"].get("prompt", None)

        if not prompt:
            return {"error": "No prompt provided."}

        # Generate the image
        result = pipe(prompt)
        image = result.images[0]

        # Convert to base64
        img_base64 = pil_to_base64(image)

        return {
            "image_base64": img_base64,
            "message": "Image generated successfully."
        }

    except Exception as e:
        return {"error": str(e)}


# RunPod entrypoint
runpod.serverless.start({"handler": handler})
