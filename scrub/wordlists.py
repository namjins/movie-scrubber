"""Build the swears set, scan pattern, and context-permit once; share across audio + SRT.

`censor net == scan net` (feedback_scan_and_mute_same_matcher): the SAME `build_pattern`
regex selects audio mute spans AND masks subtitle text. The swears *set* is used only where
`derive_flat_from_raw` needs membership (possessive normalization + whole-clip promotion).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import DEFAULT_CONTEXT_RULES, DEFAULT_LOCAL_CONTEXT_RULES, DEFAULT_WORDLIST
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
    local_context_rules: Path | None = DEFAULT_LOCAL_CONTEXT_RULES,
    extra_shared: list[Path] | None = None,
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
        # `local_context_rules` is a movie-scrubber-specific delta merged on top of the shared
        # baseline (mirrors UGE's shared + per-game-delta pattern) -- optional, so a missing
        # file is treated as "no local exemptions" rather than the fail-loud ValueError
        # load_merged_context_rules raises for an explicitly-requested-but-absent per_game_path.
        local = Path(local_context_rules) if local_context_rules else None
        per_game_path = local if local and local.exists() else None
        # `extra_shared` opts a title into always-on bundles beyond the default baseline (e.g.
        # POLYTHEISTIC_CONTEXT_RULES for a mythology/fantasy title where god/gods/hell(s) are
        # lore, not blasphemy) -- inert (no rules merged) unless the caller passes one.
        rules = load_merged_context_rules(per_game_path=per_game_path, include_shared=True,
                                          shared_path=shared, extra_shared=extra_shared)
    permit = build_permit(rules)
    n_rules = sum(len(v) for v in rules.values())
    return Censor(swears_set=swears_set, pattern=pattern, permit=permit, n_rules=n_rules)
