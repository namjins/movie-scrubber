"""Stage 3 (apply): backups + ffmpeg audio mute + SRT write-back.

Output modes:
  - copy mode (default): originals untouched; clean copies written under --output, mirroring
    the input tree. A no-hit file is copied through unchanged.
  - in-place mode (--in-place): the original is backed up under backups/ (mirroring the input
    tree) BEFORE being overwritten. (Never mutate without a verified backup.)
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import audio as audio_mod
from .audio import AudioFormat
from .plan import AudioPlan
from .srt import SrtMaskResult, mask_srt_text, read_srt, write_srt
from .wordlists import Censor


@dataclass
class FileResult:
    src: Path
    dst: Path | None
    kind: str               # "audio" | "subtitle"
    changes: int            # muted windows / masked spans
    action: str             # "muted" | "masked" | "copied" | "skipped" | "would-mute" | "would-mask" | "error"
    detail: str = ""


def _rel(src: Path, input_root: Path) -> Path:
    try:
        return src.relative_to(input_root)
    except ValueError:
        return Path(src.name)


def backup_file(src: Path, backups_root: Path, input_root: Path) -> Path:
    rel = _rel(src, input_root)
    dst = backups_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def apply_audio(src: Path, dst: Path, plan: AudioPlan, fmt: AudioFormat, *,
                dry_run: bool, in_place: bool, backups_root: Path, input_root: Path,
                ffmpeg: str | None = None, ffprobe: str | None = None) -> FileResult:
    n = len(plan.intervals)
    if n == 0:
        if dry_run:
            return FileResult(src, None, "audio", 0, "skipped", "no profanity")
        if not in_place:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return FileResult(src, dst, "audio", 0, "copied")
        return FileResult(src, src, "audio", 0, "skipped", "no profanity")

    if dry_run:
        return FileResult(src, None, "audio", n, "would-mute", "; ".join(plan.samples[:4]))

    try:
        if in_place:
            backup_file(src, backups_root, input_root)
            tmp = src.with_name(src.name + ".scrub.tmp.flac")
            audio_mod.encode_muted(src, tmp, plan.intervals, fmt, ffmpeg=ffmpeg)
            audio_mod.assert_parity(src, tmp, ffprobe=ffprobe)
            os.replace(tmp, src)
            return FileResult(src, src, "audio", n, "muted")
        else:
            audio_mod.encode_muted(src, dst, plan.intervals, fmt, ffmpeg=ffmpeg)
            audio_mod.assert_parity(src, dst, ffprobe=ffprobe)
            return FileResult(src, dst, "audio", n, "muted")
    except Exception as exc:  # noqa: BLE001
        return FileResult(src, dst, "audio", n, "error", str(exc))


def apply_srt(src: Path, dst: Path, censor: Censor, *, dry_run: bool, in_place: bool,
              backups_root: Path, input_root: Path) -> FileResult:
    try:
        text, codec = read_srt(src)
    except OSError as exc:
        return FileResult(src, None, "subtitle", 0, "error", str(exc))
    res: SrtMaskResult = mask_srt_text(text, censor)
    n = res.n_masked

    if n == 0:
        if dry_run:
            return FileResult(src, None, "subtitle", 0, "skipped", "no profanity")
        if not in_place:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return FileResult(src, dst, "subtitle", 0, "copied")
        return FileResult(src, src, "subtitle", 0, "skipped", "no profanity")

    if dry_run:
        return FileResult(src, None, "subtitle", n, "would-mask", "; ".join(res.preview))

    target = src if in_place else dst
    if in_place:
        backup_file(src, backups_root, input_root)
    write_srt(target, res.new_text, codec)
    return FileResult(src, target, "subtitle", n, "masked", "; ".join(res.preview[:3]))
