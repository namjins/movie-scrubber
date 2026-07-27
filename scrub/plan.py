"""Derive the audio mute window set for one .flac from its per-model RAW transcripts.

Mirrors universal_game_editor wildlands `_derive_lang_intervals`, generalized from
"per language" to "per model": derive_flat (whole-clip promotion + pad-aware clamp +
context exemptions), select spans with the SAME scan regex (NOT set membership — that
under-mutes hyphenated compounds), pad-expand by the imported PAD constants, clamp to
[0,duration], then union across all models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import MAX_WORD_DURATION_S, PAD_POST_S, PAD_PRE_S
from .vendored.intervals import interval_union
from .vendored.whisper_flatten import derive_flat_from_raw
from .wordlists import Censor


@dataclass
class AudioPlan:
    intervals: list[tuple[float, float]]          # final, unioned, padded, clamped
    per_model_counts: dict[str, int] = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)   # a few "[s-e] word" previews


def _intervals_for_model(raw: dict, censor: Censor, duration: float,
                         samples: list[str]) -> list[tuple[float, float]]:
    flat = derive_flat_from_raw(
        raw, censor.swears_set, duration, PAD_PRE_S, PAD_POST_S,
        context_permit=censor.permit,
    )
    intervals: list[tuple[float, float]] = []
    for e in flat:
        if not censor.pattern.search(e["word"]):
            continue
        raw_end = min(float(e["end"]), float(e["start"]) + MAX_WORD_DURATION_S)
        start = max(0.0, float(e["start"]) - PAD_PRE_S)
        end = min(duration, raw_end + PAD_POST_S)
        if end > start:
            intervals.append((start, end))
            if len(samples) < 6:
                samples.append(f'[{start:.2f}-{end:.2f}] {e["word"]!r}')
    return intervals


def derive_audio_plan(raws_by_model: dict[str, dict], censor: Censor,
                      duration: float) -> AudioPlan:
    pooled: list[tuple[float, float]] = []
    per_model: dict[str, int] = {}
    samples: list[str] = []
    for model, raw in raws_by_model.items():
        iv = _intervals_for_model(raw, censor, duration, samples)
        per_model[model] = len(iv)
        pooled.extend(iv)
    merged = interval_union(pooled)
    return AudioPlan(intervals=merged, per_model_counts=per_model, samples=samples)
