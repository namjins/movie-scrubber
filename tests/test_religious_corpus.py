"""Accuracy harness: run the operator's labeled religious-terms corpus through the REAL
`scrub` censor stack (`load_censor()` -- shared baseline + `wordlists/local-context-rules.txt`,
the same wiring the `scrub` CLI uses by default, no --polytheistic).

Vendored from universal_game_editor/tests/test_religious_corpus.py @ 2026-08-05 and adapted to
movie-scrubber's simpler (posture-less, gazetteer-less) mechanism. The corpus fixture
(tests/fixtures/religious-terms-corpus.jsonl, 237 labeled examples) is copied verbatim from
universal_game_editor/tests/fixtures/religious-terms-corpus.jsonl -- that repo's
knowledge/_pipeline/religious-terms-corpus.md is the operator-authored source of truth; this repo
does not vendor the .md or the build script, so there is no `test_fixture_up_to_date` check here.
Re-vendor the .jsonl by hand (`cp` from UGE) if the operator corpus grows there.

Puts a number on "how accurate is the religious filter" and guards it against regression. The
filter is a CENSOR with a moderate posture (default-censor, exempt only high-precision legit
uses), NOT a two-class classifier -- so a corpus row "passes" when:

  * potentially_vain_use  -> the phrase is CENSORED (>=1 hit not exempted). A fully-exempted vain
    phrase is a LEAK == hard failure (the dangerous direction).
  * respectful_use        -> the phrase is EXEMPTED (no hit censored), OR it is an EXPECTED-CENSOR
    bare vocative ("God, forgive me") which the moderate posture deliberately censors (locked
    decision, matches UGE), OR it contains no active censor token (informational, not scored).

Scoping: rows whose phrase contains no active `wordlists/default.txt` token (e.g. bare-`lord`
prayers -- `lord`/`holy` are not in movie-scrubber's default wordlist at all) are INFORMATIONAL.
See knowledge/_pipeline/religious-terms-corpus.md + religious-terms-policy.md in
universal_game_editor for the corpus provenance and policy rationale.

Run as a script for the full per-row report:  python tests/test_religious_corpus.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # let `python tests/test_religious_corpus.py` run standalone
                                     # (pytest runs already get this via tests/conftest.py)

from scrub.vendored.wordlist_match import find_hits
from scrub.wordlists import load_censor

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "religious-terms-corpus.jsonl"

# EXPECTED-CENSOR override (locked decision, matches UGE): a bare sacred vocative opener
# ("God,"/"Jesus,"/"Christ," + anything) is structurally identical to the vain "God, shut up" and
# is censored under the moderate posture, even when the corpus labels it respectful.
VOCATIVE_EXPECTED_CENSOR = re.compile(r'^"?\s*(god|jesus|christ)\s*[,!]', re.IGNORECASE)

# Tone-dependent rows: the corpus marks delivery ("said sarcastically/jokingly/mockingly") as
# meta-commentary a real subtitle/transcript line would NOT contain. A text/audio filter can't
# read tone, so these are INFORMATIONAL.
TONE_META = re.compile(r"said\s+(sarcastically|jokingly|mockingly)", re.IGNORECASE)

# Policy-vain idioms censored REGARDLESS of reverent framing ("thank God", "God knows",
# "for God's sake", "I swear to God"). The corpus sometimes labels these respectful, but the
# moderate posture keeps them censored, so they are EXPECTED-CENSOR.
POLICY_VAIN_IDIOM = re.compile(
    r"(thank\s+(you,?\s+)?god|for\s+god'?s?\s+sake|swear\s+to\s+god|\bgod\s+knows\b|thank\s+christ)",
    re.IGNORECASE)

# Accuracy ratchet. Measured 2026-08-05 against movie-scrubber's real censor stack: 1.000 over
# 170 scored rows (zero vain leaks) -- matches UGE's own measured baseline, since the shared rule
# content is byte-identical and the local delta doesn't touch any corpus phrase. Floor set to
# 0.98 (a few rows of headroom), same as UGE. Never LOWER without a documented reason.
MIN_ACCURACY = 0.98


def _load_rows() -> list[dict]:
    return [json.loads(l) for l in FIXTURE.read_text(encoding="utf-8").splitlines() if l.strip()]


def _classify(phrase: str, censor) -> str:
    """'censored' (>=1 hit not exempted), 'exempted' (all hits exempted), or 'no_token' (no hit)."""
    hits = find_hits(phrase, censor.pattern)
    if not hits:
        return "no_token"
    for (_s, _e, matched) in hits:
        if not censor.permit(matched, phrase):
            return "censored"
    return "exempted"


def _score():
    """Return (results, accuracy). results: list of (row, outcome, ok) for active-token rows."""
    rows = _load_rows()
    censor = load_censor()
    results = []
    for row in rows:
        if TONE_META.search(row["phrase"]):
            continue  # tone-dependent, undetectable from text -- informational
        outcome = _classify(row["phrase"], censor)
        if outcome == "no_token":
            continue  # informational, not scored
        vain = row["classification"] == "potentially_vain_use"
        if vain:
            ok = outcome == "censored"
        else:
            expected_censor = bool(VOCATIVE_EXPECTED_CENSOR.match(row["phrase"])
                                   or POLICY_VAIN_IDIOM.search(row["phrase"]))
            ok = (outcome == "censored") if expected_censor else (outcome == "exempted")
        results.append((row, outcome, ok))
    accuracy = (sum(1 for _r, _o, ok in results if ok) / len(results)) if results else 1.0
    return results, accuracy


# --- hard assertions ----------------------------------------------------------

def test_corpus_fixture_present_and_shaped():
    """Sanity check the vendored fixture parses and has both classes represented."""
    rows = _load_rows()
    assert len(rows) > 100
    classes = {r["classification"] for r in rows}
    assert classes == {"potentially_vain_use", "respectful_use"}


def test_no_vain_use_leaks():
    """SAFETY: no potentially_vain_use phrase with an active token may be fully exempted (a leak)."""
    results, _acc = _score()
    leaks = [row["example_id"] for row, outcome, _ok in results
             if row["classification"] == "potentially_vain_use" and outcome != "censored"]
    assert not leaks, f"vain-use LEAKS (exempted when they should censor): {leaks}"


def test_expected_censor_vocatives_are_censored():
    """The locked-decision vocative openers ARE censored (moderate posture), not exempted."""
    rows = _load_rows()
    censor = load_censor()
    misses = []
    for row in rows:
        if row["classification"] != "respectful_use":
            continue
        if not VOCATIVE_EXPECTED_CENSOR.match(row["phrase"]):
            continue
        if _classify(row["phrase"], censor) == "exempted":
            misses.append(row["example_id"])
    assert not misses, f"vocative prayers wrongly EXEMPTED (should censor per policy): {misses}"


def test_accuracy_at_or_above_baseline():
    """Accuracy over active-token rows must not regress below the ratchet floor."""
    _results, accuracy = _score()
    assert accuracy >= MIN_ACCURACY, f"accuracy {accuracy:.3f} < floor {MIN_ACCURACY:.3f}"


# --- report (python tests/test_religious_corpus.py) ---------------------------

def _report() -> str:
    results, accuracy = _score()
    rows = _load_rows()
    n_active = len(results)
    n_info = len(rows) - n_active
    lines = ["# Religious-corpus accuracy report (movie-scrubber)", "",
             f"- rows: {len(rows)} total, {n_active} scored (active token), {n_info} informational (no token)",
             f"- accuracy (scored rows): {accuracy:.3f}  (floor {MIN_ACCURACY:.3f})", ""]
    misses = [(row, outcome) for row, outcome, ok in results if not ok]
    lines.append(f"## Misses ({len(misses)})")
    for row, outcome in misses:
        want = "censor" if row["classification"] == "potentially_vain_use" else "exempt"
        lines.append(f"- {row['example_id']}: got {outcome}, want {want} -- {row['phrase']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(_report())
