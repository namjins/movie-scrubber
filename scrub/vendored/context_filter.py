# Vendored from universal_game_editor/scripts/shared/context_filter.py @ 2026-07-27
# (re-vendor of the 2026-06-11 baseline, REPO_ROOT/SHARED_DIR rebased for the flat movie-scrubber
# wordlists/ layout, same as before). Picks up the 2026-07-02 Phase 5 additions:
#   - `extra_shared` param on load_merged_context_rules() + the POLYTHEISTIC_CONTEXT_RULES bundle
#     (wordlists/polytheistic-context-rules.txt, also re-vendored) — pass
#     extra_shared=[POLYTHEISTIC_CONTEXT_RULES] to load_merged_context_rules() to opt a
#     fantasy/mythology title into "by the gods" / "the old gods" / "the hells" style exemptions.
# Deliberately NOT vendored: resolve_posture()/load_game_religious_rules()/load_gazetteer_rules()/
# PER_GAME_DIR/VALID_POSTURES/GAZETTEER_TOKENS — those assume UGE's per-game slug directory layout
# (wordlists/per-game/<slug>/posture.txt|gazetteer.txt), which movie-scrubber has no equivalent of
# (one library of titles, not one persistent config per game). If per-title config is added later,
# re-vendor those too and rebase PER_GAME_DIR to a wordlists/per-title/ layout. Until then, callers
# opt into the polytheistic bundle explicitly via extra_shared=[...], no posture.txt needed.
# See D:\repos\universal_game_editor\logs\movie-scrubber-audit-2026-07-27.md.
"""Shared context-rules engine: exempt a profanity hit when its surrounding text
matches a high-precision allow pattern for that token.

This is the durable home of the mechanism that used to live only inside
`scripts/audio/baldurs-gate-3-audio-context-filter.py`. Lifting it here lets every
scan/censor reuse ONE loader + matcher (compose-the-base, not copy) — and lets the
repo ship a SHARED, game-agnostic exemption set (`wordlists/shared/*.txt`) that every
game inherits, instead of re-deriving "God of X is a deity name, not blasphemy" by hand.

Posture (per [[feedback-blasphemy-in-scope]] + knowledge/_pipeline/religious-terms-policy.md):
context rules are an OPT-IN false-positive reducer. Without `--context-rules`, every hit
is censored (blasphemy in-scope). With it, a hit is exempted ONLY when a high-precision
pattern matches its context — favoring "false-positive censored" over "false-negative leaked".

Format (one rule per line, drop-in compatible with existing per-game files):
    <token>\t<context-regex>
  - `#` comments and blank lines ignored.
  - token: case-insensitive plain word (the wordlist hit to potentially exempt).
  - regex: case-insensitive (the `regex` lib, Unicode) applied to the hit's context text
    (the parent Whisper SEGMENT for audio; the full LOCA string for text). Multiple rules
    per token are OR'd — any match exempts the hit.

Malformed lines raise ValueError so a typo fails LOUD, never silently fail-open (which would
leak profanity in every game that inherits the shared file).
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import regex  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("context_filter requires the `regex` library (pip install regex)") from exc

# REBASED for the movie-scrubber vendored layout (scrub/vendored/context_filter.py):
# parents[2] = repo root; wordlists live at <repo>/wordlists/ (flat, no shared/ subdir).
# Callers should still pass shared_path= explicitly to load_merged_context_rules().
REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "wordlists"
#: Canonical game-agnostic exemption set for God/Lord/Jesus/Christ/hell/damn & derivatives.
RELIGIOUS_CONTEXT_RULES = SHARED_DIR / "religious-context-rules.txt"
#: Opt-in bundle for deity-lore titles (fantasy/mythology films): pantheon/plane/title exemptions.
POLYTHEISTIC_CONTEXT_RULES = SHARED_DIR / "polytheistic-context-rules.txt"


@dataclass
class ContextRule:
    token: str              # lower-cased plain token (the wordlist hit)
    pattern: "regex.Pattern"  # case-insensitive compiled regex
    raw_pattern: str        # original regex source (for reporting which rule fired)
    source: str = ""        # file the rule came from (shared vs per-game; for reporting)


def load_context_rules(path: Path, source_label: str | None = None) -> dict[str, list[ContextRule]]:
    """Parse a `<token>\\t<context-regex>` file -> {lower-token: [ContextRule, ...]}.

    Skips blank lines and `#` comments. Raises ValueError on a malformed line so a
    misconfigured rule fails loud rather than silently fail-open.
    """
    label = source_label or path.name
    by_token: dict[str, list[ContextRule]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.rstrip("\n").rstrip("\r")
            if not stripped.strip() or stripped.lstrip().startswith("#"):
                continue
            if "\t" not in stripped:
                raise ValueError(f"{path}:{lineno}: rule missing TAB separator: {stripped!r}")
            token_part, rx_part = stripped.split("\t", 1)
            token = token_part.strip().lower()
            rx_src = rx_part.strip()
            if not token or not rx_src:
                raise ValueError(f"{path}:{lineno}: empty token or regex: {stripped!r}")
            try:
                pat = regex.compile(rx_src, regex.IGNORECASE)
            except regex.error as e:
                raise ValueError(f"{path}:{lineno}: bad regex {rx_src!r}: {e}") from e
            by_token.setdefault(token, []).append(
                ContextRule(token=token, pattern=pat, raw_pattern=rx_src, source=label)
            )
    return by_token


def merge_rules(*rule_dicts: dict[str, list[ContextRule]]) -> dict[str, list[ContextRule]]:
    """Combine several rule dicts, concatenating the rule lists per token (all OR'd)."""
    merged: dict[str, list[ContextRule]] = {}
    for d in rule_dicts:
        for token, rules in d.items():
            merged.setdefault(token, []).extend(rules)
    return merged


def load_merged_context_rules(
    per_game_path: Path | None = None,
    include_shared: bool = True,
    shared_path: Path = RELIGIOUS_CONTEXT_RULES,
    extra_shared: "list[Path] | None" = None,
) -> dict[str, list[ContextRule]]:
    """Load the shared religious exemptions + an optional per-game delta file, merged.

    `include_shared=False` loads ONLY the per-game file (e.g. a game that wants to opt
    out of the shared baseline). Missing files are skipped silently EXCEPT a per_game_path
    that was explicitly requested but absent -> ValueError (a typo'd path must fail loud).

    `extra_shared`: additional always-shared bundles to merge (e.g. POLYTHEISTIC_CONTEXT_RULES for
    a deity-lore title). Each must exist if listed (fail loud on a typo, like per_game_path).
    """
    dicts: list[dict[str, list[ContextRule]]] = []
    if include_shared and shared_path.exists():
        dicts.append(load_context_rules(shared_path, source_label="shared"))
    for extra in (extra_shared or []):
        if not extra.exists():
            raise ValueError(f"extra_shared context-rules file not found: {extra}")
        dicts.append(load_context_rules(extra, source_label=extra.name))
    if per_game_path is not None:
        if not per_game_path.exists():
            raise ValueError(f"--context-rules file not found: {per_game_path}")
        dicts.append(load_context_rules(per_game_path, source_label=per_game_path.name))
    return merge_rules(*dicts) if dicts else {}


def is_context_permitted(
    word: str,
    context_text: str,
    rules_by_token: dict[str, list[ContextRule]],
) -> ContextRule | None:
    """Return the matching ContextRule if `word` is exempted in this context, else None.

    Looks up rules by the lower-cased word stripped of edge punctuation; if any rule's
    regex matches the context text, returns that rule (so callers can report which fired).
    """
    key = word.lower().strip(string.punctuation)
    rules = rules_by_token.get(key)
    if not rules:
        return None
    for rule in rules:
        if rule.pattern.search(context_text or ""):
            return rule
    return None


def build_permit(rules_by_token: dict[str, list[ContextRule]]) -> Callable[[str, str], bool]:
    """Return a picklable-free closure `(word, context_text) -> bool` for derive_flat_from_raw.

    Empty rules -> a callable that always returns False (nothing exempted), so callers can
    pass it unconditionally.
    """
    if not rules_by_token:
        return lambda _w, _t: False
    return lambda w, t: is_context_permitted(w, t, rules_by_token) is not None
