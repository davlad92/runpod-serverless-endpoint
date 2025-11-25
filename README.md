RunPod Serverless Endpoint – FLUX.1-dev Image Generator

This project implements a custom RunPod Serverless Endpoint that loads the FLUX.1-dev diffusion model from HuggingFace and generates images from text prompts. The endpoint runs inside a GPU-enabled Docker container and returns base64-encoded PNG images.

Features
	•	Custom Python handler using runpod serverless SDK
	•	Loads the black-forest-labs/FLUX.1-dev model via diffusers
	•	GPU-accelerated inference (CUDA)
	•	Accepts simple JSON payloads
	•	Returns a base64 image suitable for browsers or API clients
	•	Clean Dockerfile + requirements for reproducible builds

Project Structure

.
├── handler.py          # Main RunPod serverless handler
├── Dockerfile          # Custom container definition
├── requirements.txt    # Python dependencies
└── README.md           # Documentation

How It Works

The serverless endpoint calls the handler(event) function from handler.py.

Example event:

{
  "input": {
    "prompt": "a futuristic neon city floating above the clouds"
  }
}

Example response:

{
  "image_base64": "<base64-string>",
  "message": "Image generated successfully."
}

Model

This endpoint uses the FLUX.1-dev model:

🔗 https://huggingface.co/black-forest-labs/FLUX.1-dev

The model is loaded at container startup using:

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
).to("cuda")

A HuggingFace token with read access is passed via the env variable:

HUGGINGFACE_HUB_TOKEN

Deployment on RunPod

This endpoint is deployed using:
	•	GitHub → RunPod build (Dockerfile-based)
	•	GPU queue worker
	•	Custom handler via runpod.serverless.start({"handler": handler})

Build + deploy workflow:
	1.	Commit changes to GitHub
	2.	RunPod automatically builds the Docker image
	3.	Endpoint becomes available through RunPod’s Serverless UI

Local Development

You can install dependencies locally using:

pip install -r requirements.txt

Note: GPU inference will not run locally without CUDA support.

Author

David Ladino
Technical Support Engineer | API Integrations | L2 Support
