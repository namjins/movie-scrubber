"""Derive audio mute windows from Whisper transcripts, optionally aided by SRT cues."""
from __future__ import annotations

import math
import statistics
import string
from dataclasses import dataclass, field
from typing import Iterable

from .config import MAX_WORD_DURATION_S, PAD_POST_S, PAD_PRE_S
from .srt import (
    SrtCue,
    SubtitleHit,
    SubtitleToken,
    cue_tokens,
    extract_subtitle_hits,
    format_time_s,
)
from .vendored.whisper_flatten import derive_flat_from_raw
from .wordlists import Censor

SUBTITLE_SEARCH_PAD_S = 1.0
ANCHOR_SEARCH_PAD_S = 3.0
ALIGN_MIN_ANCHORS = 8
ALIGN_MIN_CUES = 4
ALIGN_MAX_MEDIAN_ABS_ERROR_S = 0.75
ALIGN_MAX_DRIFT_S = 0.50
LONG_FALLBACK_CUE_S = 6.0

_MODEL_PRIORITY = ["large-v3-turbo", "medium.en", "medium", "base.en", "base"]

_STOP_CHARS = string.punctuation + "\u2018\u2019\u201c\u201d"


@dataclass
class WindowSource:
    kind: str
    model: str | None = None
    cue_index: int | None = None
    cue_time: str | None = None
    preview: str | None = None


@dataclass
class MuteWindow:
    start: float
    end: float
    sources: list[WindowSource] = field(default_factory=list)


@dataclass
class ReportOnlyHit:
    kind: str
    cue_index: int | None
    cue_time: str
    preview: str
    reason: str


@dataclass
class AudioPlan:
    intervals: list[tuple[float, float]]
    per_model_counts: dict[str, int] = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)
    windows: list[MuteWindow] = field(default_factory=list)
    report_only: list[ReportOnlyHit] = field(default_factory=list)
    alignment: "AlignmentResult | None" = None

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for w in self.windows:
            kinds = {src.kind for src in w.sources}
            for kind in kinds:
                counts[kind] = counts.get(kind, 0) + 1
        return counts


@dataclass
class TimelineWord:
    word: str
    normalized: str
    start: float
    end: float
    index: int

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class Anchor:
    cue: SrtCue
    token: SubtitleToken
    word: TimelineWord
    cue_anchor_time: float
    offset: float


@dataclass
class AlignmentResult:
    passed: bool
    file_offset_s: float = 0.0
    median_abs_error_s: float = math.inf
    drift_s: float = math.inf
    anchors: list[Anchor] = field(default_factory=list)


def _normalize_token(text: str) -> str:
    token = text.lower().replace("\u2019", "'").strip(_STOP_CHARS)
    if token.endswith("'s") and len(token) > 2:
        token = token[:-2]
    elif token.endswith("s'") and len(token) > 2:
        token = token[:-1]
    return token.strip(_STOP_CHARS)


def _cue_time(cue: SrtCue) -> str:
    return f"{format_time_s(cue.start_s)} --> {format_time_s(cue.end_s)}"


def _preview(cue: SrtCue, limit: int = 80) -> str:
    text = " ".join(line.strip() for line in cue.text_lines).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _window(start: float, end: float, duration: float, source: WindowSource) -> MuteWindow | None:
    start = max(0.0, min(duration, start))
    end = max(0.0, min(duration, end))
    if end <= start:
        return None
    return MuteWindow(start, end, [source])


def _merge_windows(windows: Iterable[MuteWindow]) -> list[MuteWindow]:
    ordered = sorted((w for w in windows if w.end > w.start), key=lambda w: (w.start, w.end))
    merged: list[MuteWindow] = []
    for w in ordered:
        if not merged or w.start > merged[-1].end:
            merged.append(MuteWindow(w.start, w.end, list(w.sources)))
            continue
        merged[-1].end = max(merged[-1].end, w.end)
        merged[-1].sources.extend(w.sources)
    return merged


def _flat_for_model(raw: dict, censor: Censor, duration: float) -> list[dict]:
    return derive_flat_from_raw(
        raw, censor.swears_set, duration, PAD_PRE_S, PAD_POST_S,
        context_permit=censor.permit,
    )


def _windows_for_model(model: str, raw: dict, censor: Censor, duration: float,
                       samples: list[str]) -> list[MuteWindow]:
    windows: list[MuteWindow] = []
    for e in _flat_for_model(raw, censor, duration):
        if not censor.pattern.search(e["word"]):
            continue
        raw_end = min(float(e["end"]), float(e["start"]) + MAX_WORD_DURATION_S)
        start = max(0.0, float(e["start"]) - PAD_PRE_S)
        end = min(duration, raw_end + PAD_POST_S)
        source = WindowSource(kind=f"whisper:{model}", model=model)
        w = _window(start, end, duration, source)
        if w is None:
            continue
        windows.append(w)
        if len(samples) < 6:
            samples.append(f'[{start:.2f}-{end:.2f}] {e["word"]!r}')
    return windows


def _select_alignment_raw(raws_by_model: dict[str, dict]) -> tuple[str | None, dict | None]:
    for model in _MODEL_PRIORITY:
        if model in raws_by_model:
            return model, raws_by_model[model]
    for model, raw in raws_by_model.items():
        return model, raw
    return None, None


def _timeline_from_raw(raw: dict) -> list[TimelineWord]:
    timeline: list[TimelineWord] = []
    index = 0
    for seg in raw.get("segments", []):
        for w in seg.get("words", []):
            word = (w.get("word") or "").strip()
            norm = _normalize_token(word)
            if not norm:
                continue
            timeline.append(TimelineWord(
                word=word,
                normalized=norm,
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                index=index,
            ))
            index += 1
    return timeline


def _expected_time(cue: SrtCue, ordinal: int, n_tokens: int) -> float:
    if n_tokens <= 0:
        return cue.start_s
    return cue.start_s + ((ordinal + 0.5) / n_tokens) * (cue.end_s - cue.start_s)


def _assign_anchors(cue: SrtCue, tokens: list[SubtitleToken],
                    timeline: list[TimelineWord]) -> dict[int, TimelineWord]:
    eligible = [t for t in tokens if t.anchor_eligible]
    if not eligible:
        return {}
    n_tokens = len(tokens)
    candidates: dict[int, list[TimelineWord]] = {}
    for token in eligible:
        expected = _expected_time(cue, token.cue_token_ordinal, n_tokens)
        candidates[token.cue_token_ordinal] = [
            w for w in timeline
            if w.normalized == token.normalized
            and cue.start_s - ANCHOR_SEARCH_PAD_S <= w.midpoint <= cue.end_s + ANCHOR_SEARCH_PAD_S
        ]
        candidates[token.cue_token_ordinal].sort(key=lambda w: (abs(w.midpoint - expected), w.index))

    states: dict[int, tuple[float, dict[int, TimelineWord]]] = {-1: (0.0, {})}
    for token in eligible:
        ordinal = token.cue_token_ordinal
        expected = _expected_time(cue, ordinal, n_tokens)
        next_states = dict(states)
        for last_idx, (cost, assigned) in states.items():
            for word in candidates.get(ordinal, []):
                if word.index <= last_idx:
                    continue
                new_cost = cost + abs(word.midpoint - expected)
                prev = next_states.get(word.index)
                new_len = len(assigned) + 1
                if (prev is None or new_len > len(prev[1])
                        or (new_len == len(prev[1]) and new_cost < prev[0])):
                    new_assigned = dict(assigned)
                    new_assigned[ordinal] = word
                    next_states[word.index] = (new_cost, new_assigned)
        states = next_states

    best = min(states.values(), key=lambda item: (-len(item[1]), item[0]))
    return best[1]


def _alignment(cues: list[SrtCue], tokens_by_cue: dict[int, list[SubtitleToken]],
               timeline: list[TimelineWord]) -> tuple[AlignmentResult, dict[int, dict[int, TimelineWord]]]:
    assigned_by_cue: dict[int, dict[int, TimelineWord]] = {}
    anchors: list[Anchor] = []
    for cue_pos, cue in enumerate(cues):
        tokens = tokens_by_cue.get(cue_pos, [])
        assigned = _assign_anchors(cue, tokens, timeline)
        assigned_by_cue[cue_pos] = assigned
        n_tokens = len(tokens)
        for ordinal, word in assigned.items():
            token = next(t for t in tokens if t.cue_token_ordinal == ordinal)
            cue_anchor_time = _expected_time(cue, ordinal, n_tokens)
            anchors.append(Anchor(
                cue=cue,
                token=token,
                word=word,
                cue_anchor_time=cue_anchor_time,
                offset=word.midpoint - cue_anchor_time,
            ))

    if not anchors:
        return AlignmentResult(False), assigned_by_cue

    offsets = [a.offset for a in anchors]
    file_offset = statistics.median(offsets)
    median_abs_error = statistics.median(abs(o - file_offset) for o in offsets)
    distinct_cues = {id(a.cue) for a in anchors}
    ordered = sorted(anchors, key=lambda a: a.cue.start_s)
    half = max(1, len(ordered) // 2)
    first = ordered[:half]
    second = ordered[half:] or ordered[half - 1:]
    drift = abs(statistics.median(a.offset for a in first) -
                statistics.median(a.offset for a in second))
    passed = (
        len(anchors) >= ALIGN_MIN_ANCHORS
        and len(distinct_cues) >= ALIGN_MIN_CUES
        and median_abs_error <= ALIGN_MAX_MEDIAN_ABS_ERROR_S
        and drift <= ALIGN_MAX_DRIFT_S
    )
    return AlignmentResult(passed, file_offset, median_abs_error, drift, anchors), assigned_by_cue


def _candidate_hits(cues: list[SrtCue], censor: Censor
                    ) -> tuple[dict[int, list[SubtitleHit]], dict[int, list[SubtitleToken]]]:
    hits_by_cue: dict[int, list[SubtitleHit]] = {}
    tokens_by_cue: dict[int, list[SubtitleToken]] = {}
    for cue_pos, cue in enumerate(cues):
        hits = extract_subtitle_hits(cue, censor)
        hits_by_cue[cue_pos] = hits
        tokens_by_cue[cue_pos] = cue_tokens(cue, hits)
    return hits_by_cue, tokens_by_cue


def _confirmed_window(hit: SubtitleHit, cue: SrtCue, flats_by_model: dict[str, list[dict]],
                      censor: Censor, duration: float, search_pad: float,
                      alignment: AlignmentResult) -> MuteWindow | None:
    offset = alignment.file_offset_s if alignment.passed else 0.0
    start_bound = max(0.0, cue.start_s + offset - search_pad)
    end_bound = min(duration, cue.end_s + offset + search_pad)
    for model, flat in flats_by_model.items():
        for e in flat:
            if not censor.pattern.search(e["word"]):
                continue
            word_start = float(e["start"])
            word_end = min(float(e["end"]), word_start + MAX_WORD_DURATION_S)
            if word_start <= end_bound and word_end >= start_bound:
                source = WindowSource(
                    kind="subtitle-confirmed",
                    model=model,
                    cue_index=hit.cue_index,
                    cue_time=_cue_time(cue),
                    preview=_preview(cue),
                )
                return _window(word_start - PAD_PRE_S, word_end + PAD_POST_S, duration, source)
    return None


def _duration_for_hit(hit: SubtitleHit) -> float:
    n = max(1, len(hit.token_ordinals))
    return min(1.20, max(0.35 * n, 0.35))


def _centered_window(target_s: float, hit: SubtitleHit, cue: SrtCue, duration: float,
                     kind: str) -> MuteWindow | None:
    hit_duration = _duration_for_hit(hit)
    start = target_s - hit_duration / 2.0 - PAD_PRE_S
    end = target_s + hit_duration / 2.0 + PAD_POST_S
    return _window(start, end, duration, WindowSource(
        kind=kind,
        cue_index=hit.cue_index,
        cue_time=_cue_time(cue),
        preview=_preview(cue),
    ))


def _token_center(hit: SubtitleHit) -> tuple[int, int, float] | None:
    if not hit.token_ordinals:
        return None
    first = min(hit.token_ordinals)
    last = max(hit.token_ordinals)
    return first, last, (first + last + 1) / 2.0


def _inferred_window(hit: SubtitleHit, cue: SrtCue, tokens: list[SubtitleToken],
                     assigned: dict[int, TimelineWord], alignment: AlignmentResult,
                     duration: float) -> MuteWindow | None:
    center = _token_center(hit)
    if center is None:
        return None
    first, last, hit_center = center
    left_ord = max((o for o in assigned if o < first), default=None)
    right_ord = min((o for o in assigned if o > last), default=None)
    if left_ord is not None and right_ord is not None and right_ord > left_ord:
        left = assigned[left_ord]
        right = assigned[right_ord]
        alpha = (hit_center - left_ord) / (right_ord - left_ord)
        target = left.midpoint + alpha * (right.midpoint - left.midpoint)
        return _centered_window(target, hit, cue, duration, "subtitle-inferred")

    n_tokens = len(tokens)
    if n_tokens <= 0:
        return None
    adjusted_start = max(0.0, min(duration, cue.start_s + alignment.file_offset_s))
    adjusted_end = max(0.0, min(duration, cue.end_s + alignment.file_offset_s))
    target = adjusted_start + (hit_center / n_tokens) * (adjusted_end - adjusted_start)
    neighbor_ord = left_ord if left_ord is not None else right_ord
    if neighbor_ord is None:
        return None
    if first == 0 or last >= n_tokens - 1:
        if abs(target - assigned[neighbor_ord].midpoint) <= 1.0:
            return _centered_window(target, hit, cue, duration, "subtitle-inferred")
    return None


def _fallback_window(hit: SubtitleHit, cue: SrtCue, tokens: list[SubtitleToken],
                     alignment: AlignmentResult, duration: float) -> MuteWindow | None:
    center = _token_center(hit)
    if center is None or len(tokens) <= 0:
        return None
    if cue.end_s - cue.start_s > LONG_FALLBACK_CUE_S:
        return None
    _first, _last, hit_center = center
    adjusted_start = max(0.0, min(duration, cue.start_s + alignment.file_offset_s))
    adjusted_end = max(0.0, min(duration, cue.end_s + alignment.file_offset_s))
    target = adjusted_start + (hit_center / len(tokens)) * (adjusted_end - adjusted_start)
    return _centered_window(target, hit, cue, duration, "subtitle-fallback")


def _report_only(kind: str, cue: SrtCue, hit: SubtitleHit, reason: str) -> ReportOnlyHit:
    return ReportOnlyHit(kind, hit.cue_index, _cue_time(cue), _preview(cue), reason)


def derive_audio_plan(
    raws_by_model: dict[str, dict],
    censor: Censor,
    duration: float,
    subtitle_cues: list[SrtCue] | None = None,
    subtitle_hints: bool = True,
    subtitle_fallbacks: str = "report",
    subtitle_search_pad: float = SUBTITLE_SEARCH_PAD_S,
) -> AudioPlan:
    windows: list[MuteWindow] = []
    per_model: dict[str, int] = {}
    samples: list[str] = []
    flats_by_model: dict[str, list[dict]] = {}

    for model, raw in raws_by_model.items():
        flats_by_model[model] = _flat_for_model(raw, censor, duration)
        iv = _windows_for_model(model, raw, censor, duration, samples)
        per_model[model] = len(iv)
        windows.extend(iv)

    report_only: list[ReportOnlyHit] = []
    alignment: AlignmentResult | None = None

    if subtitle_hints and subtitle_cues:
        hits_by_cue, tokens_by_cue = _candidate_hits(subtitle_cues, censor)
        _model, align_raw = _select_alignment_raw(raws_by_model)
        timeline = _timeline_from_raw(align_raw or {})
        alignment, assigned_by_cue = _alignment(subtitle_cues, tokens_by_cue, timeline)

        for cue_pos, cue in enumerate(subtitle_cues):
            tokens = tokens_by_cue.get(cue_pos, [])
            for hit in hits_by_cue.get(cue_pos, []):
                if hit.exempted:
                    continue
                confirmed = _confirmed_window(hit, cue, flats_by_model, censor, duration,
                                              subtitle_search_pad, alignment)
                if confirmed is not None:
                    windows.append(confirmed)
                    continue
                if not alignment.passed:
                    report_only.append(_report_only("subtitle-fallback", cue, hit,
                                                    "alignment gate failed"))
                    continue
                inferred = _inferred_window(hit, cue, tokens, assigned_by_cue.get(cue_pos, {}),
                                            alignment, duration)
                if inferred is not None:
                    windows.append(inferred)
                    continue
                fallback = _fallback_window(hit, cue, tokens, alignment, duration)
                if fallback is None:
                    report_only.append(_report_only("subtitle-fallback", cue, hit,
                                                    "no usable token timing"))
                elif subtitle_fallbacks == "apply":
                    windows.append(fallback)
                else:
                    report_only.append(_report_only("subtitle-fallback", cue, hit,
                                                    "fallback mode is report"))

    merged = _merge_windows(windows)
    intervals = [(w.start, w.end) for w in merged]
    return AudioPlan(
        intervals=intervals,
        per_model_counts=per_model,
        samples=samples,
        windows=merged,
        report_only=report_only,
        alignment=alignment,
    )
