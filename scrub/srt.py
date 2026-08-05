"""SRT subtitle masking: full-word asterisks on profanity, context-rules exemptions, with
timing lines / indices / blank separators / line endings / encoding preserved byte-for-byte.

Masking is length-preserving (`fuck`->`****`) and only ever touches characters inside a
cue's TEXT lines, so a no-hit file round-trips byte-identical and a hit file differs only in
text bytes. Context (deity/scripture) is judged against the cue's full joined text, exactly
like the audio side judges against the Whisper segment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .vendored.text_censor import asterisk_censor
from .vendored.wordlist_match import find_hits
from .wordlists import Censor

_BOM = b"\xef\xbb\xbf"


@dataclass
class _Line:
    content: str
    start: int        # offset of content start in the full text
    end: int          # offset of content end (exclusive of any \r / \n)


@dataclass
class SrtCue:
    index: int | None
    start_s: float
    end_s: float
    text_lines: list[str]
    line_spans: list[tuple[int, int]]
    text: str


@dataclass
class SubtitleToken:
    text: str
    normalized: str
    line_index: int
    line_span: tuple[int, int]
    abs_span: tuple[int, int]
    cue_token_ordinal: int
    anchor_eligible: bool


@dataclass
class SubtitleHit:
    cue_index: int | None
    start_s: float
    end_s: float
    matched: str
    line_index: int
    line_span: tuple[int, int]
    abs_span: tuple[int, int]
    cue_text: str
    exempted: bool
    token_ordinals: list[int] = field(default_factory=list)


@dataclass
class SrtMaskResult:
    new_text: str
    n_masked: int
    preview: list[str] = field(default_factory=list)   # a few "word -> ****" samples


_TIMING_RE = re.compile(
    r"(?P<sh>\d{2}):(?P<sm>\d{2}):(?P<ss>\d{2}),(?P<sms>\d{3})\s+-->\s+"
    r"(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2}),(?P<ems>\d{3})"
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)?")
_HTML_RE = re.compile(r"<[^>]+>")
_HTML_BLOCK_RE = re.compile(r"<(?P<tag>[A-Za-z][A-Za-z0-9]*)\b[^>]*>.*?</(?P=tag)>")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")

_STOPWORDS = {
    "about", "after", "again", "against", "also", "because", "been", "before", "being",
    "between", "could", "does", "doing", "down", "from", "have", "here", "hers", "him",
    "his", "into", "just", "like", "more", "most", "much", "must", "only", "other",
    "ours", "over", "same", "should", "some", "such", "than", "that", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "under", "very", "was",
    "were", "what", "when", "where", "which", "while", "with", "would", "your",
}


def _iter_lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    pos, n = 0, len(text)
    while pos < n:
        nl = text.find("\n", pos)
        if nl == -1:
            line_end, nxt = n, n
        else:
            line_end, nxt = nl, nl + 1
        content_end = line_end
        if content_end > pos and text[content_end - 1] == "\r":
            content_end -= 1
        lines.append(_Line(text[pos:content_end], pos, content_end))
        pos = nxt
    return lines


def _parse_time_s(match: re.Match[str], prefix: str) -> float:
    h = int(match.group(prefix + "h"))
    m = int(match.group(prefix + "m"))
    s = int(match.group(prefix + "s"))
    ms = int(match.group(prefix + "ms"))
    return h * 3600.0 + m * 60.0 + s + ms / 1000.0


def format_time_s(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _normalized_token(text: str) -> str:
    token = text.lower().replace("\u2019", "'").strip("'")
    if token.endswith("'s") and len(token) > 2:
        token = token[:-2]
    elif token.endswith("s'") and len(token) > 2:
        token = token[:-1]
    return token


def _suppressed_spans(line: str) -> list[tuple[int, int]]:
    spans = [(m.start(), m.end()) for m in _HTML_BLOCK_RE.finditer(line)]
    spans.extend((m.start(), m.end()) for m in _HTML_RE.finditer(line))
    spans.extend((m.start(), m.end()) for m in _BRACKET_RE.finditer(line))
    return spans


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def parse_srt_cues(text: str) -> list[SrtCue]:
    """Parse cue timing/text while preserving decoded-character offsets."""
    lines = _iter_lines(text)
    cues: list[SrtCue] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        timing = _TIMING_RE.search(ln.content)
        if not timing:
            i += 1
            continue

        idx: int | None = None
        if i > 0 and lines[i - 1].content.strip().isdigit():
            idx = int(lines[i - 1].content.strip())

        text_lines: list[str] = []
        line_spans: list[tuple[int, int]] = []
        j = i + 1
        while j < len(lines) and lines[j].content.strip() != "":
            text_lines.append(lines[j].content)
            line_spans.append((lines[j].start, lines[j].end))
            j += 1
        if text_lines:
            cues.append(SrtCue(
                index=idx,
                start_s=_parse_time_s(timing, "s"),
                end_s=_parse_time_s(timing, "e"),
                text_lines=text_lines,
                line_spans=line_spans,
                text="\n".join(text_lines),
            ))
        i = max(j + 1, i + 1)
    return cues


def cue_tokens(cue: SrtCue, hits: list[SubtitleHit] | None = None) -> list[SubtitleToken]:
    """Return spoken cue tokens with original spans; hit-overlapping tokens are not anchors."""
    non_exempt_hit_spans = [h.abs_span for h in (hits or []) if not h.exempted]
    tokens: list[SubtitleToken] = []
    ordinal = 0
    for line_index, (line, (abs_start, _abs_end)) in enumerate(zip(cue.text_lines, cue.line_spans)):
        suppressed = _suppressed_spans(line)
        for m in _TOKEN_RE.finditer(line):
            line_span = (m.start(), m.end())
            abs_span = (abs_start + m.start(), abs_start + m.end())
            normalized = _normalized_token(m.group(0))
            anchor_eligible = (
                len(normalized) >= 4
                and normalized not in _STOPWORDS
                and not any(_overlaps(line_span, sp) for sp in suppressed)
                and not any(_overlaps(abs_span, hit_span) for hit_span in non_exempt_hit_spans)
            )
            tokens.append(SubtitleToken(
                text=m.group(0),
                normalized=normalized,
                line_index=line_index,
                line_span=line_span,
                abs_span=abs_span,
                cue_token_ordinal=ordinal,
                anchor_eligible=anchor_eligible,
            ))
            ordinal += 1
    return tokens


def extract_subtitle_hits(cue: SrtCue, censor: Censor) -> list[SubtitleHit]:
    """Extract cue hits with the same regex/context net used by mask_srt_text."""
    hits: list[SubtitleHit] = []
    for line_index, (line, (abs_start, _abs_end)) in enumerate(zip(cue.text_lines, cue.line_spans)):
        for start, end, matched in find_hits(line, censor.pattern):
            exempted = censor.permit(matched, cue.text)
            hits.append(SubtitleHit(
                cue_index=cue.index,
                start_s=cue.start_s,
                end_s=cue.end_s,
                matched=matched,
                line_index=line_index,
                line_span=(start, end),
                abs_span=(abs_start + start, abs_start + end),
                cue_text=cue.text,
                exempted=exempted,
            ))
    if hits:
        toks = cue_tokens(cue, hits)
        for hit in hits:
            hit.token_ordinals = [
                t.cue_token_ordinal for t in toks if _overlaps(t.abs_span, hit.abs_span)
            ]
    return hits


def mask_srt_text(text: str, censor: Censor) -> SrtMaskResult:
    """Mask profanity in SRT text. Only cue TEXT lines (those after a '-->' timing line,
    until the next blank line) are eligible; index/timing/blank lines are never touched."""
    lines = _iter_lines(text)
    chars = list(text)
    n_masked = 0
    preview: list[str] = []

    cue_lines: list[_Line] = []     # text lines accumulated for the current cue
    in_text = False                 # have we passed the timing line of the current cue?

    def flush(cue: list[_Line]) -> None:
        nonlocal n_masked
        if not cue:
            return
        context_text = "\n".join(ln.content for ln in cue)
        for ln in cue:
            for start, end, matched in find_hits(ln.content, censor.pattern):
                if censor.permit(matched, context_text):
                    continue
                masked = asterisk_censor(matched)
                abs_start = ln.start + start
                for i, c in enumerate(masked):
                    chars[abs_start + i] = c
                n_masked += 1
                if len(preview) < 5:
                    preview.append(f'{matched!r} -> {masked!r}')

    for ln in lines:
        if ln.content.strip() == "":
            flush(cue_lines)
            cue_lines, in_text = [], False
            continue
        if "-->" in ln.content:
            in_text = True
            continue
        if in_text:
            cue_lines.append(ln)
        # else: index line (before timing) — ignore
    flush(cue_lines)

    return SrtMaskResult(new_text="".join(chars), n_masked=n_masked, preview=preview)


def read_srt(path: Path) -> tuple[str, str]:
    """Return (decoded_text, codec). UTF-8 BOM -> 'utf-8-sig' (preserved on write)."""
    data = path.read_bytes()
    codec = "utf-8-sig" if data.startswith(_BOM) else "utf-8"
    try:
        return data.decode(codec), codec
    except UnicodeDecodeError:
        # Some subtitle files ship as latin-1 / cp1252; fall back length-safely.
        return data.decode("latin-1"), "latin-1"


def write_srt(path: Path, text: str, codec: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode(codec))
