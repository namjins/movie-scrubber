"""Repo paths, defaults, and the canonical pad constants (re-exported from vendored)."""
from __future__ import annotations

from pathlib import Path

from .vendored.whisper_flatten import PAD_PRE_S, PAD_POST_S, WHISPER_INITIAL_PROMPT

# scrub/config.py -> parents[1] = repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

WORDLISTS_DIR = REPO_ROOT / "wordlists"
DEFAULT_WORDLIST = WORDLISTS_DIR / "default.txt"
DEFAULT_CONTEXT_RULES = WORDLISTS_DIR / "religious-context-rules.txt"
# Movie-scrubber-local exemption delta, merged ON TOP of DEFAULT_CONTEXT_RULES (mirrors UGE's
# shared-baseline + per-game-delta pattern). This is the home for exemptions specific to this
# project that aren't part of the upstream shared file, so they survive a future re-vendor of
# DEFAULT_CONTEXT_RULES untouched (that file is marked "do not edit, re-sync from upstream").
# Optional: silently skipped if the file doesn't exist.
DEFAULT_LOCAL_CONTEXT_RULES = WORDLISTS_DIR / "local-context-rules.txt"

# RAW Whisper transcripts, keyed by (model, sha256(flac)). Cache RAW, never derived
# (so a wordlist change re-derives correctly — UGE cache invariant).
CACHE_DIR = REPO_ROOT / "cache" / "whisper" / "raw"

# In-place backups land here before any original is overwritten.
BACKUPS_DIR = REPO_ROOT / "backups"

# The three models you requested for the mute union. base.en/medium.en are English-only;
# large-v3-turbo is multilingual (extra recall + homophone diversity). NOTE: UGE's measured
# union is 2-model (turbo ∪ base.en); medium.en is an unmeasured addition — keep or drop via
# --models. See plan § "Note on the 3rd model".
DEFAULT_MODELS = ["base.en", "medium.en", "large-v3-turbo"]

# Per-model concurrent ProcessPool workers -- how many copies of ONE model transcribe
# different files at once. Models still run ONE AT A TIME (operator directive 2026-07-27: no
# concurrent models sharing the GPU); this is file-level parallelism WITHIN whichever model
# is currently running, not a cross-model total.
#
# Operator-confirmed allocation (2026-07-27) for this box's actual GPU -- nvidia-smi: RTX 3080
# Ti, 12 GB VRAM (the previous table assumed an RTX 5080 / ~16-17 GB and was wrong for this
# box). universal_game_editor's measured ground truth for this same card is large-v3-turbo:
# 2 workers fit alone (~8 GB), 3 OOM-thrash -- these per-model counts (base.en=4, medium.en=2,
# large-v3-turbo=1) are a deliberately conservative subset of that single-model-alone ceiling.
#
# NOT a formula -- this is THIS card's numbers (see universal_game_editor's system_probe.py
# convention: never hardcode a GPU-sized worker count across machines). Re-derive if the GPU
# changes.
WORKERS_BY_MODEL = {
    "base.en": 4,
    "base": 4,
    "medium.en": 2,
    "medium": 2,
    "large-v3-turbo": 1,
    "large-v3": 1,
}

DEFAULT_OUTPUT_DIRNAME = "scrubbed"

# Cap individual Whisper word-span durations before PAD expansion. Whisper sometimes
# stretches a word's end timestamp to fill a trailing silence or music cue — a 20-second
# "God." is the canonical example. Real spoken words don't exceed ~3 seconds.
MAX_WORD_DURATION_S = 3.0

__all__ = [
    "PAD_PRE_S", "PAD_POST_S", "WHISPER_INITIAL_PROMPT",
    "REPO_ROOT", "WORDLISTS_DIR", "DEFAULT_WORDLIST", "DEFAULT_CONTEXT_RULES",
    "DEFAULT_LOCAL_CONTEXT_RULES",
    "CACHE_DIR", "BACKUPS_DIR", "DEFAULT_MODELS", "WORKERS_BY_MODEL",
    "DEFAULT_OUTPUT_DIRNAME", "MAX_WORD_DURATION_S",
]
