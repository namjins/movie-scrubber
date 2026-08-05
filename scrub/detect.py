"""Stage 1 (detect): discover .flac/.srt inputs and produce RAW audio transcripts.

Input discovery is pure/cheap and runs anywhere. Transcription delegates to transcribe.py
(Linux/CUDA) and is only invoked when there are audio inputs and audio is enabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import CACHE_DIR, DEFAULT_MODELS
from .transcribe import transcribe_corpus

SUBTITLE_SUFFIX_PRIORITY = ["", "-eng", ".eng", "_eng", "-en", ".en", "_en"]


@dataclass
class Inputs:
    flacs: list[Path] = field(default_factory=list)
    srts: list[Path] = field(default_factory=list)


@dataclass
class InputJob:
    flac: Path | None = None
    srt: Path | None = None
    ambiguous_srts: list[Path] = field(default_factory=list)


@dataclass
class InputDiscovery:
    jobs: list[InputJob] = field(default_factory=list)

    @property
    def flacs(self) -> list[Path]:
        return [j.flac for j in self.jobs if j.flac is not None]

    @property
    def srts(self) -> list[Path]:
        seen: set[Path] = set()
        out: list[Path] = []
        for j in self.jobs:
            if j.srt is not None and j.srt not in seen:
                seen.add(j.srt)
                out.append(j.srt)
        return out

    @property
    def paired_count(self) -> int:
        return sum(1 for j in self.jobs if j.flac is not None and j.srt is not None)

    @property
    def unpaired_flac_count(self) -> int:
        return sum(1 for j in self.jobs if j.flac is not None and j.srt is None)

    @property
    def unpaired_srt_count(self) -> int:
        return sum(1 for j in self.jobs if j.flac is None and j.srt is not None)

    @property
    def ambiguous_count(self) -> int:
        return sum(1 for j in self.jobs if j.ambiguous_srts)


def _subtitle_pair_options(path: Path) -> list[tuple[str, int]]:
    stem = path.stem
    options = [(stem, 0)]
    for priority, suffix in enumerate(SUBTITLE_SUFFIX_PRIORITY[1:], start=1):
        if stem.lower().endswith(suffix):
            options.append((stem[: -len(suffix)], priority))
    return options


def _subtitle_pair_key(path: Path) -> tuple[str, int]:
    return _subtitle_pair_options(path)[0]


def _choose_subtitle(candidates: list[tuple[int, Path]]) -> tuple[Path | None, list[Path]]:
    if not candidates:
        return None, []
    ranked = sorted(candidates, key=lambda x: (x[0], x[1].name.lower()))
    best_priority = ranked[0][0]
    best = [p for priority, p in ranked if priority == best_priority]
    if len(best) > 1:
        return None, best
    return best[0], []


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


def discover_input_jobs(root: Path, do_audio: bool = True, do_subs: bool = True,
                        limit: int | None = None) -> InputDiscovery:
    """Discover FLAC/SRT jobs, pairing subtitles by exact stem or language suffix."""
    root = Path(root)
    flacs: list[Path] = []
    srts: list[Path] = []

    if root.is_file():
        if root.suffix.lower() == ".flac" and do_audio:
            flacs.append(root)
            if do_subs:
                srts = sorted(p for p in root.parent.glob("*.srt") if p.is_file())
        elif root.suffix.lower() == ".srt" and do_subs:
            srts.append(root)
    else:
        if do_audio:
            flacs = sorted(p for p in root.rglob("*.flac") if p.is_file())
        if do_subs:
            srts = sorted(p for p in root.rglob("*.srt") if p.is_file())

    by_stem: dict[str, list[tuple[int, Path]]] = {}
    for srt in srts:
        for key, priority in _subtitle_pair_options(srt):
            by_stem.setdefault(key.lower(), []).append((priority, srt))

    used_srts: set[Path] = set()
    jobs: list[InputJob] = []
    for flac in flacs:
        candidates = by_stem.get(flac.stem.lower(), []) if do_subs else []
        srt, ambiguous = _choose_subtitle(candidates)
        if srt is not None:
            used_srts.add(srt)
        jobs.append(InputJob(flac=flac, srt=srt, ambiguous_srts=ambiguous))

    if do_subs:
        for srt in srts:
            if srt not in used_srts:
                if not flacs or (root.is_file() and root.suffix.lower() == ".srt"):
                    jobs.append(InputJob(srt=srt))
                elif root.is_dir():
                    jobs.append(InputJob(srt=srt))

    jobs = sorted(jobs, key=lambda j: str(j.flac or j.srt or Path("")))
    if limit is not None:
        jobs = jobs[:limit]
    return InputDiscovery(jobs=jobs)


def transcribe_inputs(flacs: list[Path], models: list[str] | None = None,
                      cache_dir: Path = CACHE_DIR, force: bool = False,
                      workers_by_model: dict[str, int] | None = None
                      ) -> dict[Path, dict[str, dict]]:
    """Return {flac: {model: raw_dict}} for the given flacs (cache-aware)."""
    if not flacs:
        return {}
    return transcribe_corpus(flacs, models or DEFAULT_MODELS, cache_dir=cache_dir,
                             force=force, workers_by_model=workers_by_model)
