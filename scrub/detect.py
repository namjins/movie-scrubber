"""Stage 1 (detect): discover .flac/.srt inputs and produce RAW audio transcripts.

Input discovery is pure/cheap and runs anywhere. Transcription delegates to transcribe.py
(Linux/CUDA) and is only invoked when there are audio inputs and audio is enabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import CACHE_DIR, DEFAULT_MODELS
from .transcribe import transcribe_corpus


@dataclass
class Inputs:
    flacs: list[Path] = field(default_factory=list)
    srts: list[Path] = field(default_factory=list)


def find_inputs(root: Path, do_audio: bool = True, do_subs: bool = True,
                limit: int | None = None) -> Inputs:
    root = Path(root)
    flacs: list[Path] = []
    srts: list[Path] = []
    if root.is_file():
        if root.suffix.lower() == ".flac":
            flacs.append(root)
        elif root.suffix.lower() == ".srt":
            srts.append(root)
    else:
        flacs = sorted(p for p in root.rglob("*.flac") if p.is_file())
        srts = sorted(p for p in root.rglob("*.srt") if p.is_file())
    if not do_audio:
        flacs = []
    if not do_subs:
        srts = []
    if limit is not None:
        flacs = flacs[:limit]
        srts = srts[:limit]
    return Inputs(flacs=flacs, srts=srts)


def transcribe_inputs(flacs: list[Path], models: list[str] | None = None,
                      cache_dir: Path = CACHE_DIR, force: bool = False,
                      workers_by_model: dict[str, int] | None = None
                      ) -> dict[Path, dict[str, dict]]:
    """Return {flac: {model: raw_dict}} for the given flacs (cache-aware)."""
    if not flacs:
        return {}
    return transcribe_corpus(flacs, models or DEFAULT_MODELS, cache_dir=cache_dir,
                             force=force, workers_by_model=workers_by_model)
