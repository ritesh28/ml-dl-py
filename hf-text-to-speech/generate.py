#!/usr/bin/env python3
"""Synthesize Kokoro-82M speech from a list of jobs.

CLI examples:

  uv run python generate.py jobs.json
  uv run python generate.py --jobs '[{"id":"hello","text":"Hello world.","output_path":"output/hello.wav"}]'

jobs.json format (either works):

  [
    {"id": "one", "text": "Hello.", "output_path": "output/one.wav"},
    ["two", "Second line.", "output/two.wav"]
  ]

Optional per-job key on objects: speed.
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

DEFAULT_VOICE = "af_heart"
DEFAULT_LANG_CODE = "a"
DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"
DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_SPEED = 1.0


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
    text_and_output_path_list: list[Any],
    *,
    default_speed: float,
) -> list[tuple[str, str, Path, float]]:
    """Return list of (id, text, output_path, speed)."""
    jobs: list[tuple[str, str, Path, float]] = []
    seen: set[str] = set()

    for i, item in enumerate(text_and_output_path_list):
        speed = default_speed

        if isinstance(item, dict):
            job_id = item.get("id", item.get("identifier"))
            text = item.get("text", item.get("prompt"))
            output_path = item.get("output_path") or item.get("path")
            if "speed" in item and item["speed"] is not None:
                speed = float(item["speed"])
            if job_id is None or text is None or output_path is None:
                raise ValueError(
                    f"job[{i}] dict needs 'id' (or 'identifier'), 'text' (or 'prompt'), and "
                    f"'output_path' (or 'path'), got {item!r}"
                )
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            job_id, text, output_path = item
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            text, output_path = item
            job_id = Path(str(output_path)).stem
        else:
            raise ValueError(
                f"job[{i}] must be {{'id', 'text', 'output_path'}}, "
                f"[id, text, output_path], or [text, output_path], got {item!r}"
            )

        job_id = str(job_id)
        if not job_id:
            raise ValueError(f"job[{i}] has empty id")
        if job_id in seen:
            raise ValueError(f"duplicate job id: {job_id!r}")
        seen.add(job_id)
        jobs.append((job_id, str(text), Path(output_path).expanduser(), float(speed)))
    return jobs


def generate_from_list(
    text_and_output_path_list: list[Any],
    *,
    speed: float = DEFAULT_SPEED,
    lang_code: str = DEFAULT_LANG_CODE,
    repo_id: str = DEFAULT_REPO_ID,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> list[dict[str, Any]]:
    """Synthesize one WAV per job.

    Returns a list aligned with the input, each item:

      {
        "id": str,
        "ok": bool,
        "text": str,
        "output_path": str,
        "duration_s": float | None,
        "error": str | None,
      }
    """
    _load_env()
    jobs = _normalize_jobs(text_and_output_path_list, default_speed=speed)
    total = len(jobs)

    if total == 0:
        print("no jobs", flush=True)
        return []

    print(
        f"loading Kokoro pipeline repo={repo_id!r} lang_code={lang_code!r} jobs={total}",
        flush=True,
    )
    load_t0 = time.perf_counter()
    try:
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code=lang_code, repo_id=repo_id)
    except Exception as exc:  # noqa: BLE001 — surface any load failure per job
        load_err = f"pipeline load failed: {type(exc).__name__}: {exc}"
        print(f"ERROR {load_err}", flush=True)
        return [
            {
                "id": job_id,
                "ok": False,
                "text": text,
                "output_path": str(path),
                "duration_s": None,
                "error": load_err,
            }
            for job_id, text, path, _ in jobs
        ]

    print(f"pipeline ready in {time.perf_counter() - load_t0:.1f}s", flush=True)

    import numpy as np
    import soundfile as sf

    results: list[dict[str, Any]] = []
    for i, (job_id, text, out_path, job_speed) in enumerate(jobs, start=1):
        print(f"[{i}/{total}] id={job_id} start → {out_path}", flush=True)
        preview = text.replace("\n", " ").strip()
        print(
            f"         text: {preview[:120]}{'…' if len(preview) > 120 else ''}",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            chunks: list[Any] = []
            for _gs, _ps, audio in pipeline(
                text,
                voice=DEFAULT_VOICE,
                speed=job_speed,
                split_pattern=r"\n+",
            ):
                chunks.append(np.asarray(audio))
            if not chunks:
                raise RuntimeError("no audio chunks generated")
            audio_out = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            sf.write(out_path, audio_out, sample_rate)
            duration_s = float(len(audio_out)) / float(sample_rate)
            wall_s = time.perf_counter() - t0
            entry = {
                "id": job_id,
                "ok": True,
                "text": text,
                "output_path": str(out_path),
                "duration_s": round(duration_s, 3),
                "error": None,
            }
            print(
                f"[{i}/{total}] id={job_id} done  → {out_path} "
                f"(audio {duration_s:.2f}s, wall {wall_s:.1f}s)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — continue remaining jobs
            wall_s = time.perf_counter() - t0
            err = f"{type(exc).__name__}: {exc}"
            entry = {
                "id": job_id,
                "ok": False,
                "text": text,
                "output_path": str(out_path),
                "duration_s": None,
                "error": err,
            }
            print(
                f"[{i}/{total}] id={job_id} FAIL  → {out_path} ({wall_s:.1f}s) {err}",
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
        description="Batch-synthesize Kokoro-82M speech.",
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
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--lang-code", default=DEFAULT_LANG_CODE)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
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
            speed=args.speed,
            lang_code=args.lang_code,
            repo_id=args.repo_id,
            sample_rate=args.sample_rate,
        )
    except ValueError as exc:
        print(f"ERROR invalid jobs: {exc}", file=sys.stderr, flush=True)
        return 2

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote results: {args.results_json}", flush=True)

    print(json.dumps(results), flush=True)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
