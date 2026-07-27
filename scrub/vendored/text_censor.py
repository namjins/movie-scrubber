# Vendored from universal_game_editor/scripts/shared/text_censor_base.py @ 2026-06-11
# (the canonical masking primitives only — NOT the run() driver, which assumes the UGE
# game-pipeline shape). Per rules/script-conventions.md § Censorship rendering:
# replace the WHOLE matched word with `*`, length-preserving, no first-char hint.
"""Length-preserving full-word asterisk masking."""
from __future__ import annotations

from .wordlist_match import find_hits


def asterisk_censor(span: str) -> str:
    """Replace the whole span with asterisks, length-preserving. Full-word, no
    first-char hint: `fuck`->`****`, `fucking`->`*******`."""
    return "*" * len(span)


def censor_text(text: str, pattern, allow: "frozenset[str] | set[str]" = frozenset()) -> tuple[str, int]:
    """Asterisk-substitute every wordlist hit in `text` (except allow-listed words).

    Returns (new_text, n_replacements). Length-preserving (asserted). Allow-set is
    matched case-insensitively against the matched span. NOTE: this uses a flat allow
    SET, not the context-rules engine — for SRT we apply `is_context_permitted` per-hit
    ourselves (see scrub/srt.py); this function is kept for reference/tests.
    """
    hits = find_hits(text, pattern)
    if not hits:
        return text, 0
    out = list(text)
    n = 0
    for start, end, matched in hits:
        if matched.lower() in allow:
            continue
        masked = asterisk_censor(matched)
        for i, c in enumerate(masked):
            out[start + i] = c
        n += 1
    new = "".join(out)
    assert len(new) == len(text), f"length drift {len(text)} -> {len(new)}"
    return new, n
