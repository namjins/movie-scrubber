"""SRT subtitle masking: full-word asterisks on profanity, context-rules exemptions, with
timing lines / indices / blank separators / line endings / encoding preserved byte-for-byte.

Masking is length-preserving (`fuck`->`****`) and only ever touches characters inside a
cue's TEXT lines, so a no-hit file round-trips byte-identical and a hit file differs only in
text bytes. Context (deity/scripture) is judged against the cue's full joined text, exactly
like the audio side judges against the Whisper segment.
"""
from __future__ import annotations

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
class SrtMaskResult:
    new_text: str
    n_masked: int
    preview: list[str] = field(default_factory=list)   # a few "word -> ****" samples


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
