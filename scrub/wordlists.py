"""Build the swears set, scan pattern, and context-permit once; share across audio + SRT.

`censor net == scan net` (feedback_scan_and_mute_same_matcher): the SAME `build_pattern`
regex selects audio mute spans AND masks subtitle text. The swears *set* is used only where
`derive_flat_from_raw` needs membership (possessive normalization + whole-clip promotion).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import DEFAULT_CONTEXT_RULES, DEFAULT_WORDLIST
from .vendored.context_filter import build_permit, load_merged_context_rules
from .vendored.wordlist_match import build_pattern, load_wordlist


@dataclass
class Censor:
    swears_set: set[str]                  # lowercase plain words (no `re:` entries)
    pattern: "object"                     # compiled regex from build_pattern (incl. re: entries)
    permit: Callable[[str, str], bool]    # (word, context_text) -> True if exempted
    n_rules: int                          # number of loaded context rules (0 if disabled)


def load_censor(
    wordlist: Path = DEFAULT_WORDLIST,
    context_rules: Path | None = DEFAULT_CONTEXT_RULES,
    use_context_rules: bool = True,
) -> Censor:
    if not Path(wordlist).exists():
        raise FileNotFoundError(f"Wordlist not found: {wordlist}")
    words = load_wordlist(wordlist)
    plain = [w for w in words if not w.startswith("re:")]
    swears_set = {w.lower() for w in plain}
    pattern = build_pattern(words)

    rules: dict = {}
    if use_context_rules:
        shared = Path(context_rules) if context_rules else DEFAULT_CONTEXT_RULES
        # include_shared loads the shared file at shared_path; we keep per_game_path=None.
        rules = load_merged_context_rules(per_game_path=None, include_shared=True, shared_path=shared)
    permit = build_permit(rules)
    n_rules = sum(len(v) for v in rules.values())
    return Censor(swears_set=swears_set, pattern=pattern, permit=permit, n_rules=n_rules)
