"""Stable-whisper DTW transcription: 3 models, RAW sha256 cache, ProcessPool (Linux/CUDA).

Canon inherited from UGE:
  - ALWAYS stable_whisper DTW, never vanilla whisper (feedback_always_stable_ts_dtw).
  - word_timestamps=True + the anti-self-censor initial_prompt (whisper_flatten).
  - Cache the RAW result dict keyed by sha256(flac); NEVER cache derived windows
    (so a wordlist change re-derives correctly — UGE cache invariant).
  - ProcessPool with spawn context (fork poisons CUDA) + one model loaded per worker.

Models still transcribe ONE AT A TIME (operator directive 2026-07-27: no concurrent models
sharing the GPU) -- each model gets its own ProcessPoolExecutor, fully finishes, then the
next model starts. What changed is WORKERS_BY_MODEL itself: the counts are now sized for
this box's actual GPU (RTX 3080 Ti / 12 GB, confirmed via nvidia-smi) instead of the
previous RTX 5080 / ~16-17 GB assumption -- see config.py's docstring for the per-model
numbers and where they came from.

Heavy imports (torch/stable_whisper) happen only inside worker init, so this module is
importable on Windows for the non-GPU code paths (CLI wiring, tests).
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import CACHE_DIR, WHISPER_INITIAL_PROMPT, WORKERS_BY_MODEL
from .vendored.whisper_flatten import stable_result_to_dict


def require_linux() -> None:
    """Triton (needed for DTW word timestamps) is Linux-only. Fail loud elsewhere."""
    if platform.system() != "Linux":
        sys.stderr.write(
            "ERROR: audio transcription requires Linux/WSL2 with a CUDA stable-whisper venv.\n"
            "       Run the audio pass inside your Linux movie venv (see README).\n"
        )
        raise SystemExit(2)


def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def cache_path(cache_dir: Path, model: str, sha: str) -> Path:
    return cache_dir / model / f"{sha}.json"


# --- ProcessPool worker (module-level for picklability) ----------------------
_worker_model = None
_worker_model_name = None


def _worker_init(model_name: str) -> None:
    global _worker_model, _worker_model_name
    import stable_whisper  # Linux/CUDA only
    _worker_model = stable_whisper.load_model(model_name, device="cuda")
    _worker_model_name = model_name


def _worker_transcribe(args: tuple[str, str, str]) -> tuple[str, bool, str]:
    """Transcribe one flac, write RAW cache JSON. Returns (sha, ok, err)."""
    flac_str, sha, cache_file_str = args
    cache_file = Path(cache_file_str)
    try:
        result = _worker_model.transcribe(
            flac_str, language="en", initial_prompt=WHISPER_INITIAL_PROMPT,
            word_timestamps=True, verbose=False,
        )
        raw = result.to_dict() if hasattr(result, "to_dict") else stable_result_to_dict(result)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache_file)
        return (sha, True, "")
    except Exception as exc:  # noqa: BLE001 — report, don't crash the pool
        return (sha, False, f"{type(exc).__name__}: {exc}")


def transcribe_corpus(
    flacs: list[Path],
    models: list[str],
    cache_dir: Path = CACHE_DIR,
    workers_by_model: dict[str, int] | None = None,
    force: bool = False,
) -> dict[Path, dict[str, dict]]:
    """Transcribe every flac with every model, returning {flac: {model: raw_dict}}.

    Cache hits are reused (no GPU work). Only cache-miss (flac, model) pairs are
    transcribed. Raises SystemExit(2) on non-Linux if any transcription is actually needed.
    """
    workers_by_model = workers_by_model or WORKERS_BY_MODEL
    shas = {f: sha256_file(f) for f in flacs}
    out: dict[Path, dict[str, dict]] = {f: {} for f in flacs}

    for model in models:
        misses: list[tuple[str, str, str]] = []
        for f in flacs:
            sha = shas[f]
            cf = cache_path(cache_dir, model, sha)
            if cf.exists() and not force:
                try:
                    out[f][model] = json.loads(cf.read_text(encoding="utf-8"))
                    continue
                except (OSError, json.JSONDecodeError):
                    pass  # corrupt cache -> re-transcribe
            misses.append((str(f), sha, str(cf)))

        if not misses:
            continue

        require_linux()
        n_workers = max(1, min(workers_by_model.get(model, 3), len(misses)))
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp_ctx,
            initializer=_worker_init, initargs=(model,),
        ) as pool:
            results = list(pool.map(_worker_transcribe, misses))

        ok_by_sha = {sha: (ok, err) for sha, ok, err in results}
        for f in flacs:
            if model in out[f]:
                continue
            sha = shas[f]
            ok, err = ok_by_sha.get(sha, (False, "not transcribed"))
            cf = cache_path(cache_dir, model, sha)
            if ok and cf.exists():
                out[f][model] = json.loads(cf.read_text(encoding="utf-8"))
            else:
                sys.stderr.write(f"WARN: {model} transcription failed for {f.name}: {err}\n")
                out[f][model] = {"segments": [], "text": "", "language": "en"}
    return out
