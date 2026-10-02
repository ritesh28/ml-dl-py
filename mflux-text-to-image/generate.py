#!/usr/bin/env python3
"""Generate FLUX.1-schnell images via mflux from a list of jobs.

CLI examples:

  uv run python generate.py jobs.json
  uv run python generate.py --jobs '[{"id":"cat","prompt":"a cat","output_path":"output/cat.png"}]'

jobs.json format (either works):

  [
    {"id": "one", "prompt": "prompt one", "output_path": "output/one.png"},
    ["two", "prompt two", "output/two.png"]
  ]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_QUANTIZE = 4
DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 768
DEFAULT_STEPS = 4
DEFAULT_SEED = 42


def _load_env() -> None:
    """Load repo-root `.env` (or local `.env`) and expand HF_HOME."""
    here = Path(__file__).resolve().parent
    env_path = here.parent / ".env"
    if not env_path.is_file():
        env_path = here / ".env"
    load_dotenv(env_path)

    if "HF_HOME" in os.environ:
        os.environ["HF_HOME"] = os.path.expanduser(os.environ["HF_HOME"])


def _normalize_jobs(
    prompt_and_output_path_list: list[Any],
) -> list[tuple[str, str, Path]]:
    """Return list of (id, prompt, output_path)."""
    jobs: list[tuple[str, str, Path]] = []
    seen: set[str] = set()

    for i, item in enumerate(prompt_and_output_path_list):
        if isinstance(item, dict):
            job_id = item.get("id", item.get("identifier"))
            prompt = item.get("prompt")
            output_path = item.get("output_path") or item.get("path")
            if job_id is None or prompt is None or output_path is None:
                raise ValueError(
                    f"job[{i}] dict needs 'id' (or 'identifier'), 'prompt', and "
                    f"'output_path' (or 'path'), got {item!r}"
                )
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            job_id, prompt, output_path = item
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            # Back-compat: [prompt, path] → id from path stem
            prompt, output_path = item
            job_id = Path(str(output_path)).stem
        else:
            raise ValueError(
                f"job[{i}] must be {{'id', 'prompt', 'output_path'}}, "
                f"[id, prompt, output_path], or [prompt, output_path], got {item!r}"
            )

        job_id = str(job_id)
        if not job_id:
            raise ValueError(f"job[{i}] has empty id")
        if job_id in seen:
            raise ValueError(f"duplicate job id: {job_id!r}")
        seen.add(job_id)
        jobs.append((job_id, str(prompt), Path(output_path).expanduser()))
    return jobs


def generate_from_list(
    prompt_and_output_path_list: list[Any],
    *,
    quantize: int = DEFAULT_QUANTIZE,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    model_name: str = "schnell",
) -> list[dict[str, Any]]:
    """Generate one image per job.

    Returns a list aligned with the input, each item:

      {
        "id": str,
        "ok": bool,
        "prompt": str,
        "output_path": str,
        "elapsed_s": float | None,
        "error": str | None,
      }
    """
    _load_env()
    jobs = _normalize_jobs(prompt_and_output_path_list)
    total = len(jobs)

    if total == 0:
        print("no jobs", flush=True)
        return []

    print(
        f"loading mflux model={model_name!r} quantize={quantize} "
        f"size={width}x{height} steps={steps} seed={seed} jobs={total}",
        flush=True,
    )
    load_t0 = time.perf_counter()
    try:
        from mflux.models.flux.variants.txt2img.flux import Flux1

        flux = Flux1.from_name(model_name=model_name, quantize=quantize)
    except Exception as exc:  # noqa: BLE001 — surface any load failure per job
        load_err = f"model load failed: {type(exc).__name__}: {exc}"
        print(f"ERROR {load_err}", flush=True)
        return [
            {
                "id": job_id,
                "ok": False,
                "prompt": prompt,
                "output_path": str(path),
                "elapsed_s": None,
                "error": load_err,
            }
            for job_id, prompt, path in jobs
        ]

    print(f"model ready in {time.perf_counter() - load_t0:.1f}s", flush=True)

    results: list[dict[str, Any]] = []
    for i, (job_id, prompt, out_path) in enumerate(jobs, start=1):
        print(f"[{i}/{total}] id={job_id} start → {out_path}", flush=True)
        print(f"         prompt: {prompt[:120]}{'…' if len(prompt) > 120 else ''}", flush=True)
        t0 = time.perf_counter()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result = flux.generate_image(
                seed=seed + i - 1,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
            )
            result.save(out_path, overwrite=True)
            elapsed = time.perf_counter() - t0
            entry = {
                "id": job_id,
                "ok": True,
                "prompt": prompt,
                "output_path": str(out_path),
                "elapsed_s": round(elapsed, 2),
                "error": None,
            }
            print(f"[{i}/{total}] id={job_id} done  → {out_path} ({elapsed:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001 — continue remaining jobs
            elapsed = time.perf_counter() - t0
            err = f"{type(exc).__name__}: {exc}"
            entry = {
                "id": job_id,
                "ok": False,
                "prompt": prompt,
                "output_path": str(out_path),
                "elapsed_s": round(elapsed, 2),
                "error": err,
            }
            print(
                f"[{i}/{total}] id={job_id} FAIL  → {out_path} ({elapsed:.1f}s) {err}",
                flush=True,
            )
        results.append(entry)

    ok_n = sum(1 for r in results if r["ok"])
    print(f"finished {ok_n}/{total} ok", flush=True)
    return results


def _parse_jobs_arg(raw: str) -> list[Any]:
    """Parse jobs from a JSON file path or a JSON string."""
    stripped = raw.lstrip()
    # Inline JSON: don't call Path.is_file() — long strings raise ENAMETOOLONG on macOS.
    if stripped.startswith(("[", "{")):
        text = raw
    else:
        path = Path(raw)
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if is_file:
            text = path.read_text(encoding="utf-8")
        else:
            text = raw
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("jobs JSON must be a list")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-generate FLUX.1-schnell images with mflux.",
    )
    parser.add_argument(
        "jobs",
        nargs="?",
        help="Path to jobs JSON file, or inline JSON list",
    )
    parser.add_argument(
        "--jobs",
        dest="jobs_opt",
        help="Same as positional jobs (path or JSON string)",
    )
    parser.add_argument("--quantize", type=int, default=DEFAULT_QUANTIZE)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default="schnell")
    parser.add_argument(
        "--results-json",
        type=Path,
        help="Optional path to write the results list as JSON",
    )
    args = parser.parse_args(argv)

    jobs_raw = args.jobs_opt or args.jobs
    if not jobs_raw:
        parser.error("provide jobs JSON file/string as positional arg or --jobs")

    try:
        jobs = _parse_jobs_arg(jobs_raw)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR invalid jobs: {exc}", file=sys.stderr, flush=True)
        return 2

    try:
        results = generate_from_list(
            jobs,
            quantize=args.quantize,
            height=args.height,
            width=args.width,
            steps=args.steps,
            seed=args.seed,
            model_name=args.model,
        )
    except ValueError as exc:
        print(f"ERROR invalid jobs: {exc}", file=sys.stderr, flush=True)
        return 2

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote results: {args.results_json}", flush=True)

    # Always print machine-readable summary last for callers.
    print(json.dumps(results), flush=True)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
