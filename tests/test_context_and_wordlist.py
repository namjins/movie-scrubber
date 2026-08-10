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


def test_load_censor_extra_shared_wires_polytheistic_bundle():
    """load_censor(extra_shared=[...]) -- the plumbing scrub/cli.py's --polytheistic flag
    uses -- opts in the bundle; the default call (no extra_shared) must not.

    Goes through find_hits(), not censor.permit() directly, and uses a phrase built on
    "hells" -- NOT "gods" -- because "gods" (plural) is not in wordlists/default.txt in this
    repo (only singular "god" is), so find_hits() can never emit a "gods" hit here and a test
    built on it would pass without ever exercising the real pipeline. "hells" IS a wordlist
    entry, and the bundle's "hells -> the hells" rule genuinely changes its outcome."""
    from scrub.vendored.wordlist_match import find_hits

    phrase = "go to the hells and back"
    poly_censor = load_censor(extra_shared=[POLYTHEISTIC_CONTEXT_RULES])
    hits = find_hits(phrase, poly_censor.pattern)
    assert [m for _s, _e, m in hits] == ["hells"]
    assert poly_censor.permit("hells", phrase) is True

    default_censor = load_censor()
    default_hits = find_hits(phrase, default_censor.pattern)
    assert [m for _s, _e, m in default_hits] == ["hells"]
    assert default_censor.permit("hells", phrase) is False


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


# --- Local delta (wordlists/local-context-rules.txt, 2026-07-27): movie-scrubber-specific
# exemptions merged on top of the shared baseline, auto-loaded by load_censor() by default.

def test_hellfire_variants_exempt_via_local_delta():
    censor = load_censor()
    assert censor.permit("hell", "hellfire") is True         # never a hit in the first place,
                                                               # but the permit call itself is safe
    assert censor.permit("hell", "hell-fire") is True
    assert censor.permit("hell", "hell fire") is True


def test_bare_hell_still_censored():
    censor = load_censor()
    assert censor.permit("hell", "go to hell") is False
    assert censor.permit("hell", "what the hell is that") is False


def test_scripture_quote_exempt_via_local_delta():
    """Corpus-driven (2026-08-05, Stranger Things S01E05 Pastor Charles eulogy): a
    verbatim Isaiah 41:10 paraphrase and its follow-on sermon line must be exempted,
    not censored."""
    censor = load_censor()
    assert censor.permit("god", "Be not dismayed, for I am your God.") is True
    assert censor.permit("god", "It would be easy to turn away from God...") is True


def test_scripture_quote_delta_does_not_leak_vain_god():
    """The new patterns are tight -- they must not fire on ordinary vain uses."""
    censor = load_censor()
    assert censor.permit("god", "oh my God") is False
    assert censor.permit("god", "God, this is annoying") is False
    assert censor.permit("god", "I swear to God") is False


def test_god_complex_idiom_exempt_but_bare_god_still_censored():
    """Corpus-driven (2026-08-05): "god complex" is a secular idiom (S05E07/S05E08), never a
    vain exclamation -- exempt it, but don't let the pattern over-match a plain "God" nearby."""
    censor = load_censor()
    assert censor.permit("god", "A psychopath with a serious god complex, but...") is True
    assert censor.permit("god", "That bitch has a God complex.") is True
    assert censor.permit("god", "oh my God, what a complex situation") is False


def test_god_has_a_plan_exempt_by_operator_override():
    """Operator-directed exemption (2026-08-05): "I know God has a plan…" (S04E05 #616) is a
    testimony/faith idiom -- exempt, per an explicit operator override of the initial
    ambiguous-delivery call. Ordinary vain uses of "God" must still censor."""
    censor = load_censor()
    assert censor.permit("god", "I know God has a plan…") is True
    assert censor.permit("god", "God has a plan for all of us.") is True
    assert censor.permit("god", "oh my God") is False
    assert censor.permit("god", "God, this is annoying") is False


def test_bat_god_exempt_but_bare_god_still_censored():
    """Corpus-driven (2026-08-10, Batman: Caped Crusader S02E04 "The Devil's Due" cult
    storyline): "bat god" (Barbatos) is an in-world deity reference, never a vain
    exclamation -- exempt it, but don't let the pattern over-match a plain "God" nearby."""
    censor = load_censor()
    assert censor.permit("god", "Barbatos is a demon bat god, who slumbers...") is True
    assert censor.permit("god", "a cult devoted to an evil bat god") is True
    assert censor.permit("god", "oh my God, what a god awful mess") is False


def test_hell_on_earth_exempt_by_operator_override():
    """Operator-directed exemption (2026-08-10): "A fire there will be hell on earth"
    (Batman: Caped Crusader S01E04 #264) is an idiom the moderate-posture default would
    NOT have exempted on its own (same shape as "hell of a" / "to hell with", which stay
    censored) -- exempt per explicit operator override. Ordinary vain/exclamatory uses of
    "hell" must still censor."""
    censor = load_censor()
    assert censor.permit("hell", "there will be hell on earth") is True
    assert censor.permit("hell", "go to hell") is False
    assert censor.permit("hell", "what the hell is that") is False


def test_local_delta_absent_does_not_break_load_censor(tmp_path):
    """load_censor() must not fail-loud when local_context_rules points at a missing file
    (mirrors the optional-file contract in config.py's DEFAULT_LOCAL_CONTEXT_RULES docstring)."""
    from scrub.config import DEFAULT_CONTEXT_RULES, DEFAULT_WORDLIST
    censor = load_censor(DEFAULT_WORDLIST, DEFAULT_CONTEXT_RULES, True,
                         local_context_rules=tmp_path / "does-not-exist.txt")
    assert censor.n_rules > 0                     # shared baseline still loaded
    assert censor.permit("hell", "hell fire") is False   # the local exemption is NOT applied
