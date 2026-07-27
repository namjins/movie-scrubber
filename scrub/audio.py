"""FLAC format probing, the ffmpeg all-channel mute, and the format-parity gate.

Adapted from universal_game_editor/scripts/audio/wildlands-audio-mute.py
(`_apply_mute_ffmpeg`), re-targeted from PCM-WAV to FLAC with full format preservation.

Pad/clamp note: the intervals handed to `build_mute_filter` / `encode_muted` are ALREADY
expanded by ±PAD and clamped to [0, duration] at the derive site (scrub/plan.py), exactly
as wildlands `_derive_lang_intervals` does — using the imported PAD_PRE_S/PAD_POST_S
constants, never hardcoded. The ffmpeg `between(t,a,b)` expression therefore renders only
final numbers. (clamp floor == real pad — mute-audio-with-monkeyplug.md INVARIANT.)
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ToolError(RuntimeError):
    """ffmpeg/ffprobe missing or failed."""


def resolve_tool(name: str, override: str | None = None) -> str:
    if override:
        return override
    found = shutil.which(name)
    if not found:
        raise ToolError(f"{name} not found on PATH (install ffmpeg, or pass an explicit path)")
    return found


@dataclass
class AudioFormat:
    sample_rate: int
    channels: int
    channel_layout: str | None
    sample_fmt: str | None
    bits_per_raw_sample: int | None
    duration: float

    def matches(self, other: "AudioFormat", duration_tol_s: float = 0.010) -> tuple[bool, str]:
        """Parity check: rate, channels, sample_fmt, and duration (±tol) must match."""
        if self.sample_rate != other.sample_rate:
            return False, f"sample_rate {self.sample_rate} != {other.sample_rate}"
        if self.channels != other.channels:
            return False, f"channels {self.channels} != {other.channels}"
        if self.sample_fmt != other.sample_fmt:
            return False, f"sample_fmt {self.sample_fmt} != {other.sample_fmt}"
        if abs(self.duration - other.duration) > duration_tol_s:
            return False, f"duration {self.duration:.3f} != {other.duration:.3f} (>{duration_tol_s}s)"
        return True, "ok"


def probe_format(path: Path, ffprobe: str | None = None) -> AudioFormat:
    """Read the first audio stream's format via ffprobe. Works on .flac (the WAV-only
    `wave`-module helpers in whisper_flatten CANNOT parse flac — never use them here)."""
    exe = resolve_tool("ffprobe", ffprobe)
    argv = [
        exe, "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,channel_layout,sample_fmt,bits_per_raw_sample",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"ffprobe failed on {path}: {exc}") from exc
    if proc.returncode != 0:
        raise ToolError(f"ffprobe rc={proc.returncode} on {path}: {proc.stderr.strip()[:200]}")
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise ToolError(f"no audio stream in {path}")
    s = streams[0]
    fmt = data.get("format", {})
    bits = s.get("bits_per_raw_sample")
    return AudioFormat(
        sample_rate=int(s["sample_rate"]),
        channels=int(s["channels"]),
        channel_layout=s.get("channel_layout"),
        sample_fmt=s.get("sample_fmt"),
        bits_per_raw_sample=int(bits) if bits not in (None, "N/A") else None,
        duration=float(fmt.get("duration", 0.0) or 0.0),
    )


def build_mute_filter(intervals: list[tuple[float, float]]) -> str:
    """One ffmpeg `volume=0` filter that mutes ALL channels over every interval.

    `+` is logical-OR in ffmpeg expressions; a single `volume` filter applies to every
    channel = monkeyplug's default all-channel mute. Intervals must already be padded +
    clamped to [0, duration]. Returns "" if there is nothing to mute.
    """
    if not intervals:
        return ""
    enable_expr = "+".join(f"between(t,{a:.4f},{b:.4f})" for a, b in intervals)
    return f"volume=enable='{enable_expr}':volume=0"


def encode_muted(
    src: Path,
    dst: Path,
    intervals: list[tuple[float, float]],
    fmt: AudioFormat,
    ffmpeg: str | None = None,
) -> None:
    """Re-encode `src` -> `dst` (FLAC) with `intervals` silenced, preserving format.

    Preserves sample rate, channels, and sample format (bit depth); copies metadata/tags.
    No normalization runs, so overall volume is unchanged except inside the muted windows.
    Caller is responsible for NOT calling this when intervals is empty (copy through instead).
    """
    exe = resolve_tool("ffmpeg", ffmpeg)
    dst.parent.mkdir(parents=True, exist_ok=True)
    af = build_mute_filter(intervals)
    if not af:
        raise ValueError("encode_muted called with no intervals; copy the file through instead")
    argv = [
        exe, "-y", "-loglevel", "error",
        "-i", str(src),
        "-af", af,
        "-ac", str(fmt.channels),
        "-ar", str(fmt.sample_rate),
        "-c:a", "flac",
        "-map_metadata", "0",
    ]
    if fmt.sample_fmt:
        argv += ["-sample_fmt", fmt.sample_fmt]
    argv += [str(dst)]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"ffmpeg mute failed on {src}: {exc}") from exc
    if proc.returncode != 0 or not dst.exists():
        raise ToolError(f"ffmpeg mute rc={proc.returncode} on {src}: {proc.stderr.strip()[:300]}")


def assert_parity(src: Path, dst: Path, ffprobe: str | None = None,
                  duration_tol_s: float = 0.010) -> AudioFormat:
    """Re-probe `dst` and assert it matches `src`'s format. Raises ToolError on drift.

    This is the feedback_audio_duration_rate_parity gate — it catches a silent resample
    (monkeyplug's 48 kHz default is exactly the bug we prevent)."""
    src_fmt = probe_format(src, ffprobe)
    dst_fmt = probe_format(dst, ffprobe)
    ok, why = dst_fmt.matches(src_fmt, duration_tol_s=duration_tol_s)
    if not ok:
        raise ToolError(f"format parity FAILED for {dst.name}: {why}")
    return dst_fmt
