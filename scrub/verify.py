"""Stage 4 (verify): residual re-scan + format parity on written outputs.

Audio parity is asserted inline at apply time (audio.assert_parity); this stage re-checks
the written SRT outputs for any residual profanity the mask missed — a broader net than the
apply (it re-masks and asserts zero remaining), per feedback_verify_net_broader_than_censor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio import AudioFormat, probe_format
from .srt import mask_srt_text, read_srt
from .wordlists import Censor


@dataclass
class VerifyIssue:
    path: Path
    kind: str
    detail: str


def verify_srt(path: Path, censor: Censor) -> list[VerifyIssue]:
    """Re-scan a written SRT; any residual maskable hit is a verification failure."""
    issues: list[VerifyIssue] = []
    try:
        text, _ = read_srt(path)
    except OSError as exc:
        return [VerifyIssue(path, "subtitle", f"unreadable: {exc}")]
    res = mask_srt_text(text, censor)
    if res.n_masked:
        issues.append(VerifyIssue(path, "subtitle",
                                  f"{res.n_masked} residual hit(s): {'; '.join(res.preview)}"))
    return issues


def verify_audio_parity(src: Path, dst: Path, ffprobe: str | None = None) -> list[VerifyIssue]:
    """Confirm a written audio output still matches its source format."""
    try:
        src_fmt: AudioFormat = probe_format(src, ffprobe)
        dst_fmt: AudioFormat = probe_format(dst, ffprobe)
    except Exception as exc:  # noqa: BLE001
        return [VerifyIssue(dst, "audio", f"probe failed: {exc}")]
    ok, why = dst_fmt.matches(src_fmt)
    return [] if ok else [VerifyIssue(dst, "audio", f"parity drift: {why}")]
