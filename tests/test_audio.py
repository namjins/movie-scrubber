"""Interval union + format-parity logic + (optional) a real ffmpeg encode round-trip."""
import shutil

import pytest

from scrub.audio import AudioFormat, build_mute_filter
from scrub.vendored.intervals import interval_union


def test_interval_union_merges_and_sorts():
    out = interval_union([(1.0, 2.0), (0.0, 0.5), (1.8, 3.0), (0.4, 0.6)])
    assert out == [(0.0, 0.6), (1.0, 3.0)]


def test_build_mute_filter_empty_is_blank():
    assert build_mute_filter([]) == ""


def test_build_mute_filter_uses_or_expression():
    f = build_mute_filter([(0.25, 1.80), (3.00, 3.50)])
    assert f.startswith("volume=enable='between(t,")
    assert "+between(t," in f          # logical OR
    assert f.endswith("':volume=0")


def test_format_parity_detects_resample():
    src = AudioFormat(48000, 2, "stereo", "s16", 16, 10.0)
    same = AudioFormat(48000, 2, "stereo", "s16", 16, 10.005)
    resampled = AudioFormat(44100, 2, "stereo", "s16", 16, 10.0)
    assert same.matches(src)[0] is True
    assert resampled.matches(src)[0] is False


# --- Optional: exercise the real ffmpeg encode if ffmpeg/ffprobe are present ----
ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")


@pytest.mark.skipif(not (ffmpeg and ffprobe), reason="ffmpeg/ffprobe not on PATH")
def test_real_encode_preserves_format(tmp_path):
    import subprocess

    from scrub.audio import assert_parity, encode_muted, probe_format

    src = tmp_path / "tone.flac"
    # 2s stereo 44.1k 16-bit sine
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=2:sample_rate=44100",
                    "-ac", "2", "-sample_fmt", "s16", "-c:a", "flac", str(src)],
                   check=True)
    fmt = probe_format(src)
    assert fmt.sample_rate == 44100 and fmt.channels == 2
    dst = tmp_path / "out.flac"
    encode_muted(src, dst, [(0.5, 1.0)], fmt)
    out = assert_parity(src, dst)  # raises on drift
    assert out.sample_rate == 44100 and out.channels == 2
