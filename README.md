# RunPod Serverless Endpoint – Image Generation with FLUX.1

This project deploys a serverless GPU endpoint on [RunPod.io](https://runpod.io) to generate images from text prompts using the Hugging Face model [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev).

## 🔧 Technologies Used
- RunPod Serverless
- Hugging Face Transformers + Diffusers
- PyTorch
- Docker

---

## 📁 File Structure
├── Dockerfile              # Container setup using RunPod’s serverless base
├── handler.py              # Python handler for serverless inference
├── requirements.txt        # Python dependencies
├── test_request.json       # Sample input payload for testing
└── README.md               # Project documentation

---

## 🧠 Model

We use the `black-forest-labs/FLUX.1-dev` Stable Diffusion model for generating high-quality images from text prompts.

---

## 🚀 Deploying to RunPod

### Requirements
- A RunPod account
- Free credits (contact hailong.yang@runpod.io if needed)
- GitHub repo linked to RunPod

### Steps
1. Create a serverless endpoint on RunPod
2. Connect your GitHub repo
3. Use GPU (24GB+) and `runpod/serverless:3.10-cuda11.8` base image
4. No ports or env variables needed
5. Deploy!

---

## 🧪 Testing the Endpoint

Use the following test input:

```json
{
  "input": {
    "prompt": "A futuristic city skyline at sunset"
  }
}

<!-- trigger rebuild -->
