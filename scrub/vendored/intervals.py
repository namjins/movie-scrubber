# Vendored from universal_game_editor/scripts/audio/wildlands-audio-mute.py @ 2026-06-11
# (the `_interval_union` tuple merge that pairs with `_apply_mute_ffmpeg`). Kept tuple-based
# end-to-end to match the reviewed, shipped Wildlands muter exactly. The dict-shaped
# `interval_union` in phonetic_match.py is for the corpus phonetic pre-pass we deliberately
# descope (see plan § Deliberate descopes), so it is intentionally NOT vendored.
"""Overlap/adjacency merge for [start, end] mute intervals."""
from __future__ import annotations


def interval_union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent [start, end] intervals. Input need not be sorted.

    Merge when start_next <= end_cur (overlap OR exact touch); carry max-end so a
    contained interval is absorbed. This is what unions the mute windows produced by
    the 3 transcription models into one minimal set.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
