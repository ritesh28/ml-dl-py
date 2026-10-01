# Hugging Face Text-to-Image (Stable Diffusion v1.5)

Local text-to-image with **[stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)** via the [`diffusers`](https://github.com/huggingface/diffusers) library.

## What it does

1. **Load env** — Reads repo-root `.env` for `HF_HOME` / `HF_TOKEN` (see `.env.example`).
2. **Load pipeline** — Builds a `StableDiffusionPipeline` (weights download from Hugging Face on first run).
3. **Generate** — Creates an image from a text prompt and saves PNG files under `output/`.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for the project environment.

**Env (`.env`):** Copy the repo-root `.env.example` to `.env` and set `HF_HOME` (default `~/ml-models`) and optionally `HF_TOKEN`. The notebook loads these before downloading weights.

Then:

```bash
cd hf-text-to-image
uv sync
```

Register the kernel (optional, for Jupyter / VS Code / Cursor):

```bash
uv run python -m ipykernel install --user --name=hf-text-to-image --display-name="Python (hf-text-to-image)"
```

## Run the notebook

Open `hf-text-to-image.ipynb`, click **Select Kernel** → **Jupyter Kernel…**, and choose **Python (hf-text-to-image)**. Reload the IDE if you do not see it.

On Apple Silicon the notebook uses **MPS**; otherwise it falls back to CUDA or CPU.

## Demo prompt

The notebook generates a **Book Illustration** style image of a human thinking:

> A thoughtful human deep in thought, hand on chin, Book Illustration style, detailed ink and watercolor, storybook art

## Dependencies

| Package       | Role                                     |
| ------------- | ---------------------------------------- |
| diffusers     | Stable Diffusion pipeline                |
| torch         | Model runtime (MPS / CUDA / CPU)         |
| transformers  | Text encoder / tokenizer                 |
| accelerate    | Device placement helpers                 |
| pillow        | Save and display PNG output              |
| safetensors   | Safe weight loading                      |
| python-dotenv | Load repo-root `.env` (`HF_HOME` / token)|
| ipykernel     | Jupyter kernel for the local environment |
