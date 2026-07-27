# Vendored verbatim from universal_game_editor/scripts/shared/wordlist_match.py @ 2026-06-11.
# Do not edit here; re-sync from upstream if the matcher changes.
"""Wordlist loader + elongation-normalizing whole-word matcher.

# Game:        shared utility (any game)
# Targets:     a text string to match, plus a wordlist file path
# Mode:        N/A — library helper, not a standalone script
# Invocation:  `from wordlist_match import load_wordlist, build_pattern, normalize_elongation`
#              `python3 scripts/shared/wordlist_match.py <text>` (debug mode — see __main__)
# Flags:       N/A (debug mode reads stdin or argv)
# Backup:      N/A — read-only library
# Idempotency: N/A — pure functions

Why this exists:
- Profanity detection needs whole-word matching (so "ass" doesn't fire inside "classic").
- It also needs to catch elongated variants ("fuuuuuck" → "fuck") without producing false
  positives on legitimate doubles like "good" or "off".
- The trick: normalize runs of 3+ identical characters to 1 char before matching, but
  leave runs of 1-2 untouched. Match against the normalized string with word-boundary
  regex against the literal wordlist. Map matches back to original spans if you need
  to replace them.

Functions:
- load_wordlist(path)             → list[str]
- normalize_elongation(text)      → (normalized: str, span_starts: list[int], span_ends: list[int])
- build_pattern(words)            → compiled regex
- find_hits(text, pattern)        → list[(start, end, match)] — matches against normalized
                                    text, with spans mapped back to the original

The `regex` library (not stdlib `re`) is used because it has full Unicode word boundary
support and is faster on alternation patterns of this size.

Originally ported from cp2077_profanity_pipeline/cp2077_profanity/scanner.py — the
WolvenKit-specific JSON parsing was left behind because it's REDengine-4-specific. This
module focuses purely on the matcher; per-format adapters should call into it.
"""

from pathlib import Path

import regex  # third-party: pip install regex


def load_wordlist(wordlist_path) -> list[str]:
    """Load profanity words from a wordlist file.

    Skips blank lines and lines starting with #. Returns words in source order
    (caller may want to sort for canonical hashing).

    Lines starting with `re:` are returned as-is — caller decides whether to
    treat them as regex (this loader doesn't compile them).
    """
    wordlist_path = Path(wordlist_path)
    if not wordlist_path.exists():
        raise FileNotFoundError(f"Wordlist not found: {wordlist_path}")

    words: list[str] = []
    with open(wordlist_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                words.append(stripped)
    return words


def normalize_elongation(text: str) -> tuple[str, list[int], list[int]]:
    """Collapse runs of 3+ identical characters down to a single character.

    Runs of 1-2 identical characters are left untouched, preserving legitimate
    double letters (e.g. "good" has "oo" = 2 chars, stays as-is).
    Runs of 3+ are elongation (e.g. "gooood" -> "god", "fuuuuuck" -> "fuck").

    The span mapping lets callers replace matched spans in the *original* text.
    When matching "fuck" inside normalized "fuuuuuck", the span mapping shows the
    match covers all 9 original characters, so a replacement could be "*********".

    Returns:
        normalized   — the collapsed string
        span_starts  — span_starts[i] is the original index where normalized char i begins
        span_ends    — span_ends[i] is the original index (exclusive) where the run ends
    """
    if not text:
        return text, [], []

    normalized: list[str] = []
    span_starts: list[int] = []
    span_ends: list[int] = []

    i = 0
    while i < len(text):
        char = text[i]
        run_start = i
        while i < len(text) and text[i] == char:
            i += 1
        run_len = i - run_start

        if run_len >= 3:
            normalized.append(char)
            span_starts.append(run_start)
            span_ends.append(i)
        else:
            for j in range(run_len):
                normalized.append(char)
                span_starts.append(run_start + j)
                span_ends.append(run_start + j + 1)

    return "".join(normalized), span_starts, span_ends


def build_pattern(words: list[str]) -> "regex.Pattern":
    """Compile a whole-word, case-insensitive regex over the given wordlist.

    Plain words are regex-escaped. Lines prefixed `re:` are passed through as
    user-authored regex fragments (caller is responsible for their soundness;
    detect-profanity's startup-test step is where you'd validate them).

    Sorted by length descending so longer phrases match before substrings
    (e.g. "fuck face" before "fuck").
    """
    if not words:
        raise ValueError("Wordlist is empty — cannot build a match pattern")

    fragments: list[str] = []
    for w in sorted(words, key=len, reverse=True):
        if w.startswith("re:"):
            fragments.append(w[3:])
        else:
            fragments.append(regex.escape(w))

    pattern_str = r"\b(?:" + "|".join(fragments) + r")\b"
    return regex.compile(pattern_str, regex.IGNORECASE)


def find_hits(text: str, pattern: "regex.Pattern") -> list[tuple[int, int, str]]:
    """Return [(orig_start, orig_end, matched_text), ...] for hits in `text`.

    `text` is normalized first; matches found on the normalized form are mapped
    back to original-text indices via the span tables.

    Literal escape sequences in string-table values (`\\n`, `\\r`, `\\t`) glue their
    letter onto the following word and break its word boundary, hiding line-start
    profanity (e.g. `...\\nMOTHERFUCKER`). They are replaced with spaces for MATCHING
    only — length-preserved (2 chars -> 2 chars), so spans still map back into `text`.
    """
    matchable = text.replace("\\n", "  ").replace("\\r", "  ").replace("\\t", "  ")
    normalized, span_starts, span_ends = normalize_elongation(matchable)
    hits: list[tuple[int, int, str]] = []
    for m in pattern.finditer(normalized):
        orig_start = span_starts[m.start()]
        orig_end = span_ends[m.end() - 1] if m.end() > m.start() else span_starts[m.start()]
        hits.append((orig_start, orig_end, text[orig_start:orig_end]))
    return hits


if __name__ == "__main__":
    # Debug mode: `python3 wordlist_match.py "test string"`
    # Loads ../../wordlists/default.txt and prints hits.
    import sys

    if len(sys.argv) < 2:
        sys.stderr.write("usage: wordlist_match.py <text-to-scan>\n")
        sys.exit(2)

    repo_root = Path(__file__).resolve().parent.parent.parent
    wordlist = repo_root / "wordlists" / "default.txt"
    words = load_wordlist(wordlist)
    plain_words = [w for w in words if not w.startswith("re:")]
    pattern = build_pattern(plain_words)
    for start, end, matched in find_hits(sys.argv[1], pattern):
        print(f"{start}-{end}: {matched!r}")
