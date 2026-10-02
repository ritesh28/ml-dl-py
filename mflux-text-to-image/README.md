# mflux Text-to-Image (FLUX.1-schnell)

Local text-to-image with **[black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)** using **[mflux](https://github.com/filipstrand/mflux)** on Apple **MLX** / Metal.

Weights are downloaded from the Hugging Face Hub into `HF_HOME`, and inference runs through mflux + MLX.

## Why mflux (not Hugging Face `diffusers`)?

| Approach                         | What it is                       | On this 16 GB Mac                                   |
| -------------------------------- | -------------------------------- | --------------------------------------------------- |
| **HF `diffusers` + PyTorch MPS** | Official FLUX pipeline           | Full weights ~24–34 GB → OOM / Jupyter kernel crash |
| **mflux (this project)**         | Apple Silicon–native FLUX on MLX | **4-bit ~10 GB** → runs on Metal                    |

We tried `diffusers` first. Loading the full model into unified memory failed (`MPS backend out of memory`), and CPU offload then killed the kernel under memory pressure. **mflux** is the practical stack for FLUX on 16 GB Apple Silicon: quantized MLX inference instead of PyTorch/`diffusers`.

Hugging Face Hub is still used only as the **weight source** (`HF_HOME` / optional `HF_TOKEN`). The runtime is mflux, not the HF Diffusers library.

## What it does

1. **Load env** — Reads repo-root `.env` for `HF_HOME` / `HF_TOKEN` (see `.env.example`).
2. **Load model** — `Flux1.from_name("schnell", quantize=4)` via MLX/Metal.
3. **Generate** — 4 steps at 768×768; saves PNGs under `output/`.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for the project environment.

**Env (`.env`):** Copy the repo-root `.env.example` to `.env` and set `HF_HOME` (default `~/ml-models`) and optionally `HF_TOKEN`.

```bash
cd mflux-text-to-image
uv sync
```

Register the kernel (optional):

```bash
uv run python -m ipykernel install --user --name=mflux-text-to-image --display-name="Python (mflux-text-to-image)"
```

## Run the notebook

1. Close other memory-heavy apps (browsers with many tabs, etc.).
2. Open `mflux-text-to-image.ipynb` → kernel **Python (mflux-text-to-image)**.
3. Restart the kernel if a previous run crashed, then run cells top to bottom.

## CLI batch generate

`generate.py` loads the model once, then runs a list of jobs with `id`, `prompt`, and `output_path`. It prints progress (id + file + seconds) and returns per-item `ok` / `error`.

```bash
cd mflux-text-to-image
uv run python generate.py jobs.example.json
uv run python generate.py --jobs '[{"id":"bike","prompt":"a red bicycle","output_path":"output/bike.png"}]' --results-json output/results.json
```

Jobs JSON (objects preferred; triples also work):

```json
[
  {"id": "one", "prompt": "prompt one", "output_path": "output/one.png"},
  ["two", "prompt two", "output/two.png"]
]
```

Each result includes `"id"`. Exit code `0` if all succeeded, `1` if any failed, `2` if jobs JSON is invalid. The final stdout line is the full results JSON list.

## Tips for 16 GB

- Keep **`quantize=4`** (do not use 8-bit or full precision).
- Prefer **768×768** (or smaller) over 1024×1024.
- Run **one** generation at a time.
- If the kernel dies again, free RAM and restart the kernel before reloading.

## Demo prompt

> A thoughtful human deep in thought, hand on chin, Book Illustration style, detailed ink and watercolor, storybook art

## Dependencies

| Package       | Role                                     |
| ------------- | ---------------------------------------- |
| mflux         | FLUX on Apple MLX (Metal)                |
| python-dotenv | Load repo-root `.env`                    |
| ipykernel     | Jupyter kernel for the local environment |
| pillow        | Image display / save helpers             |
