"""Wordlist + context-rules load correctly (the fail-open guard) and exempt the right uses."""
from scrub.wordlists import load_censor
from scrub.vendored.context_filter import (
    is_context_permitted, load_merged_context_rules, POLYTHEISTIC_CONTEXT_RULES,
)
from scrub.config import DEFAULT_CONTEXT_RULES


def test_context_rules_load_nonzero():
    censor = load_censor()
    # The fail-open guard: a wrong shared_path would silently load 0 rules.
    assert censor.n_rules > 0


def test_wordlist_has_expected_size():
    censor = load_censor()
    # 872 non-comment entries were vendored; sanity-check it's in the right ballpark.
    assert len(censor.swears_set) >= 800
    assert "fuck" in censor.swears_set
    assert "god" in censor.swears_set


def test_god_of_war_exempt_but_oh_my_god_masked():
    rules = load_merged_context_rules(per_game_path=None, include_shared=True,
                                      shared_path=DEFAULT_CONTEXT_RULES)
    assert is_context_permitted("god", "god of war is great", rules) is not None
    assert is_context_permitted("god", "oh my god what now", rules) is None


def test_permit_callable_matches():
    censor = load_censor()
    assert censor.permit("god", "the god of war") is True
    assert censor.permit("god", "oh my god") is False


# --- Regression tests for the 2026-07-27 re-vendor (universal_game_editor's 2026-07-02 corpus-
# accuracy pass fixed these as real leaks in the file movie-scrubber had vendored; see
# universal_game_editor/logs/movie-scrubber-audit-2026-07-27.md). Each MUST censor (no rule fires).

def test_jesus_wept_no_longer_exempt():
    rules = load_merged_context_rules(per_game_path=None, include_shared=True,
                                      shared_path=DEFAULT_CONTEXT_RULES)
    assert is_context_permitted("jesus", "Jesus wept! I can't believe it.", rules) is None


def test_in_christs_name_censored_but_faith_in_christ_stays_exempt():
    rules = load_merged_context_rules(per_game_path=None, include_shared=True,
                                      shared_path=DEFAULT_CONTEXT_RULES)
    assert is_context_permitted("christ", "in Christ's name we pray", rules) is None
    assert is_context_permitted("christ", "I have faith in Christ", rules) is not None


def test_bare_lord_jesus_exclamation_censored():
    rules = load_merged_context_rules(per_game_path=None, include_shared=True,
                                      shared_path=DEFAULT_CONTEXT_RULES)
    assert is_context_permitted("jesus", "Lord Jesus! What happened?", rules) is None


def test_corpus_driven_respectful_exemptions_present():
    """Section 8 of the re-vendored file (worship/thanksgiving/blessing patterns) exists."""
    rules = load_merged_context_rules(per_game_path=None, include_shared=True,
                                      shared_path=DEFAULT_CONTEXT_RULES)
    assert is_context_permitted("god", "God is good, all the time", rules) is not None
    assert is_context_permitted("jesus", "Jesus changed my life", rules) is not None


def test_wordlist_re_vendored_compound_entries_present():
    """default.txt re-vendor (872 -> 999 entries) picked up general compound profanity."""
    censor = load_censor()
    for w in ("bullshitter", "chucklefuck", "dickass", "apeshit"):
        assert w in censor.swears_set


def test_polytheistic_bundle_available_as_opt_in():
    """Not wired in by default (extra_shared=[] unless a caller opts in), but loadable."""
    assert POLYTHEISTIC_CONTEXT_RULES.exists()
    rules = load_merged_context_rules(per_game_path=None, include_shared=True,
                                      shared_path=DEFAULT_CONTEXT_RULES,
                                      extra_shared=[POLYTHEISTIC_CONTEXT_RULES])
    assert is_context_permitted("gods", "by the gods, look at that", rules) is not None
    # Default load_censor() does NOT opt in -- the bundle must not fire unsolicited.
    default_rules = load_merged_context_rules(per_game_path=None, include_shared=True,
                                               shared_path=DEFAULT_CONTEXT_RULES)
    assert is_context_permitted("gods", "by the gods, look at that", default_rules) is None
