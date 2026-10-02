# rembg Background Removal

Local image background removal with **[danielgatis/rembg](https://github.com/danielgatis/rembg)** (ONNX Runtime). No notebook — CLI batch jobs only.

## What it does

1. **Load env** — Reads repo-root `.env` for optional `REMBG_HOME` (model cache; default `~/.rembg`).
2. **Load session** — `new_session(model)` once (default `bria-rmbg`); weights download on first use.
3. **Remove backgrounds** — Writes cutouts (or masks) under `output/` for each job.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for the project environment.

**Env (optional):** Set `REMBG_HOME` in the repo-root `.env` to put ONNX models somewhere other than `~/.rembg`.

```bash
cd rembg-background-removal
uv sync
```

Put source images under `input/` (or any path you reference in jobs JSON).

## CLI batch generate

`generate.py` loads the rembg session once, then runs a list of jobs with `id`, `input_path`, and `output_path`. It prints progress (id + file + seconds) and returns per-item `ok` / `error`.

```bash
cd rembg-background-removal
uv run python generate.py jobs.example.json
uv run python generate.py --jobs '[{"id":"cat","input_path":"input/cat.jpg","output_path":"output/cat.png"}]' --results-json output/results.json
uv run python generate.py jobs.example.json --decontaminate --model u2net
```

Jobs JSON (objects preferred; triples also work):

```json
[
  {"id": "one", "input_path": "input/one.png", "output_path": "output/one.png"},
  ["two", "input/two.png", "output/two.png"]
]
```

Object aliases: `input` / `image_path` / `image` for the source; `path` for the destination.

| Flag              | Role                                                                 |
| ----------------- | -------------------------------------------------------------------- |
| `--model`         | Model name (default `bria-rmbg`; try `u2net` for a smaller/faster run) |
| `--decontaminate` | Fix colored halos on soft edges (cheap; good for batches)            |
| `--alpha-matting` | Refine soft mask edges (slower)                                      |
| `--only-mask`     | Write the alpha mask only                                            |

Each result includes `"id"` and `"elapsed_s"`. Exit code `0` if all succeeded, `1` if any failed, `2` if jobs JSON is invalid. The final stdout line is the full results JSON list.

## Models

See the [rembg model list](https://github.com/danielgatis/rembg#models). Notes:

- **`bria-rmbg`** (default) is strong but large (~1 GB) and slower; check BRIA’s license for commercial use.
- **`u2net` / `u2netp` / `silueta`** are smaller and faster with firmer edges.
- This project installs **`rembg[cpu]`** (onnxruntime). NVIDIA GPU users can switch the extra to `rembg[gpu]` in `pyproject.toml`.

## Dependencies

| Package       | Role                                      |
| ------------- | ----------------------------------------- |
| rembg[cpu]    | Background removal + ONNX Runtime (CPU)   |
| python-dotenv | Load repo-root `.env` (`REMBG_HOME`, etc.) |
