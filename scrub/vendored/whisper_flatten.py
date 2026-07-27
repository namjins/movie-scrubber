# Vendored from universal_game_editor/scripts/shared/whisper_flatten.py @ 2026-07-27
# (re-vendor of the 2026-06-11 baseline). Picks up two upstream changes since then:
#   - Rule 3b, the 2026-06-26 edge-extension clamp: a flagged word that is the FIRST/LAST spoken
#     word Whisper detected across the WHOLE input mutes to the file boundary regardless of timing
#     proximity (Whisper mis-times boundary-word onsets/tails). For movie-scrubber `audio_duration`
#     is the whole .flac's duration (scrub/plan.py passes the file's full length), so this only
#     fires if a curse word is literally the first/last word Whisper transcribes in the entire film
#     — rarer than the short-game-VO-clip case this was built for, but not zero (cold opens/closes).
#   - load_context_permit(), 2026-07-02: a convenience wrapper so a mute step that re-derives from
#     the raw cache restores the same context-rule exemptions the scan applied. ONE LOCAL FIX: its
#     `from context_filter import ...` (bare, assumes scripts/shared/ on PYTHONPATH in UGE) is
#     changed to `from .context_filter import ...` (relative) to match movie-scrubber's package
#     layout (scrub/vendored/ is a real package, not a PYTHONPATH-added dir). Not currently called —
#     scrub/wordlists.py + scrub/plan.py already wire context_permit themselves via Censor.permit —
#     kept for upstream parity / future use.
# See D:\repos\universal_game_editor\logs\movie-scrubber-audit-2026-07-27.md.
r"""Universal Whisper-result -> monkeyplug-flat derivation helpers.

# Game:        shared utility (any game)
# Mode:        N/A -- library module, not a standalone script
# Invocation:  `from whisper_flatten import (
#                  PAD_PRE_S, PAD_POST_S, WHISPER_INITIAL_PROMPT,
#                  normalize_possessive, derive_flat_from_raw,
#                  get_wav_duration, get_wav_format,
#                  stable_result_to_dict,   # stable-ts (import: stable_whisper) WhisperResult -> dict
#              )` (when scripts/shared/ is on PYTHONPATH)
#
# Why this exists
# ---------------
# The end-to-end audio mute pipeline (workflows/mute-audio-with-monkeyplug.md)
# applies a specific set of rules to convert raw Whisper segment+word output
# into the flat monkeyplug-compatible JSON array used to drive span muting:
#
#   1. Possessive normalization. Whisper may emit "bastard's" where the
#      swears list has bare "bastard"; we rewrite the trailing 's / s' to
#      restore the match.
#   2. Asymmetric padding constants. PAD_PRE_S=0.050, PAD_POST_S=0.200.
#      These are the values monkeyplug should be invoked with, and the
#      same values used here to decide whole-clip promotion. Validated
#      across BG3 (2026-05-21), RoboCop (2026-05-22), and Vanquish.
#   3. Clamp+both-pads whole-clip promotion. If a profanity word is too
#      close to BOTH ends of the clip for the asymmetric pad to fit,
#      promote to whole-clip mute. If only one end is over, clamp that
#      side. Otherwise keep the word as-is. RETRACTED legacy rules
#      ("every word is profanity" / ">50% domination") MUST NOT be
#      reintroduced -- see workflow doc.
#   3b. Edge-extension clamp (2026-06-26). A flagged word that is the FIRST
#      spoken word of a clip mutes from the clip START, and the LAST word
#      mutes to the clip END -- triggered by word POSITION, not just timing
#      proximity, because Whisper mis-times boundary-word onsets/tails and the
#      tight pad clamp would otherwise leave an audible edge bleed.
#   4. Standard initial_prompt for Whisper transcription that primes the
#      model to transcribe profanity verbatim instead of self-censoring.
#
# These rules were duplicated across baldurs-gate-3-audio-scan.py,
# baldurs-gate-3-audio-context-filter.py, and robocop-audio-scan.py with subtle
# implementation drift. The operator's standing rule (per the desperados-iii
# audio mute design discussion, 2026-05-24) is that **the BG3 clamping rules
# are the standard going forward and must be the single source of truth.**
# Any audio mute / scan / context-filter script in this repo that touches
# Whisper output MUST import these helpers rather than re-implement them
# (including `stable_result_to_dict` for scripts that use stable_whisper).
#
# Cross-references
# ----------------
# - workflows/mute-audio-with-monkeyplug.md § "Whole-clip promotion
#   (revised 2026-05-22)" and § "Possessive normalization (mandatory)"
#   -- canonical English-language documentation of these rules.
# - auto-memory `feedback-audio-mute-pipeline` -- pointer with fast-recall
#   anti-patterns.
# - rules/script-conventions.md § Tool paths -- "Import, don't copy" rule
#   for shared helpers.
"""
from __future__ import annotations

import string
import wave
from pathlib import Path
from typing import Any, Callable


# --- stable-ts result conversion ---------------------------------------------

def stable_result_to_dict(result: Any) -> dict:
    """Convert a stable_whisper WhisperResult to an openai-whisper-compatible dict.

    Use this in per-process worker initialisers that call
    `stable_whisper.load_model()` instead of `whisper.load_model()`. The returned
    dict has the canonical shape expected by `derive_flat_from_raw()` and by
    the sha256-keyed JSON cache:

        {"text": str, "segments": [
            {"text": str, "start": float, "end": float,
             "words": [{"word": str, "start": float, "end": float,
                        "probability": float}, ...]},
            ...
        ], "language": str}

    Reference: scripts/audio/fw-stable-whisper-scan.py `_to_standard_dict`.
    """
    segments = []
    for seg in result.segments:
        words = []
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end),
                    "probability": float(getattr(w, "probability", 0.9)),
                })
        segments.append({
            "text": seg.text,
            "start": float(seg.start),
            "end": float(seg.end),
            "words": words,
        })
    return {
        "text": result.text,
        "segments": segments,
        "language": getattr(result, "language", "en"),
    }


# --- Constants ----------------------------------------------------------------

# Asymmetric padding rule from workflows/mute-audio-with-monkeyplug.md § Padding.
# These constants serve two roles in the pipeline:
#   1. Passed to monkeyplug as --pad-milliseconds-pre / --pad-milliseconds-post
#      (after multiplying by 1000).
#   2. Used here to detect "pre_over" / "post_over" conditions for whole-clip
#      promotion (a word too close to either end of the clip for the asymmetric
#      pad to fit triggers clamp-or-promote).
#
# These are the canonical DEFAULTS. The asymmetric 50/200 was empirically tuned
# (BG3 16-sample benchmark, 88% full-silence rate vs 44% for symmetric-50) and is
# the right starting point for every game.
#
# A game MAY raise the pad (e.g. FC3 2026-05-30 raised pad_pre to 150ms to close a
# Whisper late-word-start onset leak on short words). When it does, it MUST pass the
# SAME value to BOTH the mute step (monkeyplug --pad-*-ms) AND derive_flat_from_raw's
# pad_pre_s/pad_post_s, so the authoritative clamp matches the pad actually applied.
# Never let the clamp floor and the mute pad diverge -- that produces negative ffmpeg
# windows (clamp < pad) or surviving onsets (clamp > pad). The clamp rule always wins;
# keep it equal to the real pad.
PAD_PRE_S: float = 0.050   # seconds
PAD_POST_S: float = 0.200  # seconds


# Whisper initial_prompt that primes the model to transcribe profanity
# verbatim. Without this, model='base' / 'base.en' often self-censors
# (returns "f***" or simply omits the word). With this prompt, recall on
# explicit-content audio is materially higher.
#
# Empirically validated on Vanquish 2026-05-20: `small` without prompt
# returned 0 of 778 transcripts containing 'motherfucker' despite it being
# the most-spoken combat callout; `base` + prompt fixed the recall.
WHISPER_INITIAL_PROMPT: str = (
    "The following audio includes explicit profanity. "
    "Transcribe every spoken word verbatim, including curse words, "
    "without censoring or substituting."
)


# --- WAV introspection (stdlib only) -----------------------------------------

def get_wav_duration(wav: Path) -> float:
    """Return WAV duration in seconds via the stdlib `wave` module.

    Returns 0.0 on parse error. Callers should treat 0.0 as "duration unknown"
    and skip the whole-clip promotion check (the rule needs a real duration
    to decide post_over).
    """
    try:
        with wave.open(str(wav), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate if rate else 0.0
    except (wave.Error, OSError, EOFError):
        return 0.0


def get_wav_format(wav: Path) -> tuple[int | None, int | None]:
    """Return (sample_rate_hz, channels). (None, None) on parse error.

    Used by callers that need to pass `-s <rate>` and `-c <channels>` to
    monkeyplug. monkeyplug's defaults (48 kHz, stereo) are usually wrong
    for game VO.
    """
    try:
        with wave.open(str(wav), "rb") as wf:
            return wf.getframerate(), wf.getnchannels()
    except (wave.Error, OSError, EOFError):
        return None, None


# --- Possessive normalization -------------------------------------------------

def normalize_possessive(word: str, swears_set: set[str]) -> str:
    """Strip trailing 's or s' if the bare base form is in the swears set.

    Per workflows/mute-audio-with-monkeyplug.md § Possessive normalization
    (mandatory). monkeyplug's scrubword() does edge-punctuation only;
    without this step, `bastard's` fails to match bare `bastard` in the
    swears list and the word leaks un-muted.

    Smart-quote U+2019 is normalized to ASCII U+0027 first so that
    "bastard's" (smart-quote) matches the same way as "bastard's" (ASCII).

    The case in the input `word` is preserved on the returned base form
    (so "Bastard's" returns "Bastard", not "bastard").
    """
    if not word:
        return word
    # Normalize smart-quote -> ASCII for the suffix-strip check; keep the
    # original `word` for the case-preserved return value.
    lower = word.lower().replace("’", "'")
    for suffix in ("'s", "s'"):
        if lower.endswith(suffix):
            base = lower[: -len(suffix)]
            if base in swears_set:
                return word[: -len(suffix)]
    return word


# --- Context-rule permit (the audio-mute counterpart of the text censor's rules) ----

def load_context_permit(
    per_game: Path | None = None,
    include_shared: bool = True,
) -> Callable[[str, str], bool]:
    """Build the `(word, context) -> bool` permit an audio MUTE passes to
    `derive_flat_from_raw(..., context_permit=...)`, so mute-time re-derivation applies the
    SAME religious exemptions the scan applied (scan net == censor net — religious-terms-policy).

    Convenience one-liner over `context_filter.load_merged_context_rules` + `build_permit`, so a
    per-game mute that MUST re-derive from the raw cache can restore the exemption in one call
    instead of dropping it (the bug this closes: most mute scripts re-derived without a permit and
    resurrected scan-time exemptions like "God of War"). The shared religious baseline is ON by
    default; pass `include_shared=False` for a per-game-only rule set, or a `per_game` delta path.

    Returns an always-False permit when no rules are found, so callers can pass the result
    unconditionally (equivalent to `context_permit=None` — a pure no-op).
    """
    from .context_filter import load_merged_context_rules, build_permit
    rules = load_merged_context_rules(per_game_path=per_game, include_shared=include_shared)
    return build_permit(rules)


# --- Per-load derivation: raw Whisper -> flat monkeyplug array ---------------

def derive_flat_from_raw(
    whisper_result: dict,
    swears_set: set[str],
    audio_duration: float,
    pad_pre_s: float = PAD_PRE_S,
    pad_post_s: float = PAD_POST_S,
    context_permit: Callable[[str, str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Convert a raw Whisper result dict -> monkeyplug-compatible flat array.

    Applies the universal pipeline rules in order:
      1. Walk every word in every segment of the raw Whisper output.
      2. Possessive-normalize each word against the swears set.
      3. Decide per-profanity-word whether to clamp boundaries (single-end
         padding overflow) or promote to whole-clip mute (both-end padding
         overflow on the same word).
      4. Emit a single whole-clip entry if promoted, else the adjusted
         flat list.

    AUTHORITATIVE PAD-AWARE CLAMP (universal rule, operator directive 2026-05-30,
    applies to EVERY game): the clamp/promote thresholds use the SAME pad values
    that the downstream mute step will apply. monkeyplug subtracts pad_pre from each
    span start and adds pad_post to each end; if the clamp floor here did not match
    the mute's actual pad, the post-pad window `[start-pad_pre, end+pad_post]` could
    go negative (ffmpeg `between(t,-0.1,...)` produces no output) or past EOF.
    Callers that raise the mute pad above the 50/200 default (e.g. FC3 onset-leak
    fix at 150ms) MUST pass the SAME pad_pre_s/pad_post_s here. Defaults preserve the
    canonical 50/200 so every existing per-game caller is unchanged.

    Returns
    -------
    list[dict]
        Each dict has the monkeyplug-compatible shape:
        {"word": str, "start": float, "end": float, "probability": float}.
        Non-profanity words are kept verbatim so a downstream wordlist scan
        (including re: regex entries) can still match the full transcript.

    Cache invariant
    ---------------
    This function MUST run on every load (cache miss or hit). Callers MUST
    cache the raw Whisper result, NOT the derived flat. Otherwise wordlist
    changes silently fail to take effect on previously-cached clips.
    See workflows/mute-audio-with-monkeyplug.md § Cache invariant.
    """
    flat: list[dict[str, Any]] = []

    for seg in whisper_result.get("segments", []):
        seg_text = (seg.get("text") or "").strip()
        words = seg.get("words", [])
        if not words:
            # Defensive fallback: malformed cache entry with no word
            # timestamps. Use segment boundaries; warn upstream if this
            # path fires (word_timestamps=True should make it impossible).
            if seg_text:
                words = [{
                    "word": seg_text,
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "probability": 1.0,
                }]

        for w in words:
            raw_word = (w.get("word") or "").strip()
            if not raw_word:
                continue
            normalized = normalize_possessive(raw_word, swears_set)
            # Context-rules exemption (opt-in): drop a profanity word whose
            # surrounding segment matches a high-precision allow pattern for its
            # token (e.g. "God of War" deity title vs "oh my God" exclamation).
            # Non-profanity words are untouched. context_permit=None (the default)
            # is a pure no-op, so every existing caller is byte-for-byte unchanged.
            if context_permit is not None:
                base_for_match = normalized.lower().strip(string.punctuation)
                if base_for_match in swears_set and context_permit(normalized, seg_text):
                    continue
            flat.append({
                "word": normalized,
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "probability": float(w.get("probability", 1.0)),
            })

    if not flat:
        return flat

    # Clamp+both-pads whole-clip promotion (workflows/mute-audio-with-monkeyplug.md
    # § Whole-clip promotion (revised 2026-05-22)).
    adjusted: list[dict[str, Any]] = []
    whole_clip = False
    placeholder_swear: str | None = None

    n_words = len(flat)
    for i, e in enumerate(flat):
        base = e["word"].lower().strip(string.punctuation)
        # KNOWN OPEN DEFECT -- this exact-membership test is NARROWER than the scan's net
        # (wordlist_match.build_pattern), so a scan-FLAGGED word can land here as not-profanity and get no
        # clamp / no whole-clip promotion / no rule-3b edge-extension -- i.e. the mute silently drops it.
        # Measured divergence classes: elongation (fuuuuck, shiiiit) and hyphen-compound (nasty-ass, kick-ass).
        # Possessives ARE handled (normalized above). This violates feedback_scan_and_mute_same_matcher in the
        # SSOT itself, so every compliant caller inherits it; run-checks' mute-canonical-clamp lint does NOT
        # catch it (that lint flags callers that never reach here).
        # Callers currently close it locally by augmenting swears_set with the scan-matched surface forms.
        # Fix + blast radius + why it is not landed yet: knowledge/_pipeline/mute-matcher-net-alignment.md
        is_prof = base in swears_set
        if not is_prof:
            adjusted.append(e)
            continue

        # Track the first real profanity word so a whole-clip emit can use
        # the real swear text (NOT a synthetic "PROFANITY" placeholder --
        # placeholder would fall out of downstream wordlist scans).
        if placeholder_swear is None:
            placeholder_swear = e["word"]

        # EDGE-EXTENSION CLAMP (universal rule, operator directive 2026-06-26): a flagged word that is
        # the FIRST spoken word of the clip mutes from the clip START, and the LAST spoken word mutes to
        # the clip END -- regardless of Whisper's timestamp. Whisper systematically mis-times the
        # onset/tail of boundary words, so the tight pad clamp (which only fires when start < pad_pre)
        # leaves an audible bleed of the word at the clip edge. Triggering on POSITION (first/last word),
        # not just timing proximity, closes it. Expressed via the SAME pad-aware clamp values
        # (start = pad_pre -> padded window starts at 0; end = duration - pad_post -> window ends at
        # duration), so the pad invariant above still holds for monkeyplug AND byte-splice games.
        is_first_word = (i == 0)
        is_last_word = (i == n_words - 1)
        pre_over = (e["start"] < pad_pre_s) or is_first_word
        post_over = audio_duration > 0 and ((e["end"] + pad_post_s > audio_duration) or is_last_word)

        if pre_over and post_over:
            whole_clip = True
            break

        if pre_over:
            e = dict(e)
            e["start"] = pad_pre_s
        if post_over:
            e = dict(e)
            e["end"] = max(e["start"] + 0.01, audio_duration - pad_post_s)

        adjusted.append(e)

    if whole_clip and audio_duration > 0 and placeholder_swear is not None:
        # Emit exactly one entry. monkeyplug's natural pre/post padding
        # extends this to [0, duration], silencing the full clip.
        return [{
            "word": placeholder_swear,
            "start": pad_pre_s,
            "end": max(pad_pre_s + 0.01, audio_duration - pad_post_s),
            "probability": 1.0,
        }]

    return adjusted
