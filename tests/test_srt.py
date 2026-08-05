"""SRT masking: byte-identity on no-hit, length-preserving masking, context exemption,
and timing/index/line-ending preservation."""
from scrub.srt import (
    cue_tokens,
    extract_subtitle_hits,
    mask_srt_text,
    parse_srt_cues,
    read_srt,
    write_srt,
)
from scrub.wordlists import load_censor

CENSOR = load_censor()

CLEAN = (
    "1\r\n"
    "00:00:01,000 --> 00:00:03,000\r\n"
    "Hello there, traveler.\r\n"
    "\r\n"
    "2\r\n"
    "00:00:04,000 --> 00:00:06,000\r\n"
    "The god of war watches over us.\r\n"
)

DIRTY = (
    "1\n"
    "00:00:01,000 --> 00:00:03,000\n"
    "This shit happens every week.\n"
    "\n"
    "2\n"
    "00:00:04,000 --> 00:00:06,000\n"
    "Oh my god, you bitch!\n"
)


def test_no_hit_is_byte_identical():
    res = mask_srt_text(CLEAN, CENSOR)
    assert res.n_masked == 0
    assert res.new_text == CLEAN  # byte-for-byte


def test_god_of_war_not_masked():
    # "god of war" is a deity title -> exempt even in a hit-bearing file
    res = mask_srt_text(CLEAN, CENSOR)
    assert "god of war" in res.new_text


def test_masking_is_length_preserving_and_full_word():
    res = mask_srt_text(DIRTY, CENSOR)
    assert res.n_masked >= 3  # shit, god (vain), bitch
    assert len(res.new_text) == len(DIRTY)
    assert "****" in res.new_text          # shit -> ****
    assert "*****" in res.new_text         # bitch -> *****
    # no first-char hint
    assert "s***" not in res.new_text
    assert "b****" not in res.new_text


def test_timing_and_index_lines_untouched():
    res = mask_srt_text(DIRTY, CENSOR)
    assert "00:00:01,000 --> 00:00:03,000" in res.new_text
    assert "00:00:04,000 --> 00:00:06,000" in res.new_text
    # line endings preserved (this file is \n only)
    assert "\r\n" not in res.new_text


def test_vain_god_is_masked():
    # "oh my god" is a multi-word wordlist phrase -> whole phrase masked (9 chars),
    # and the context rule (which keys on the single token "god") does NOT exempt it.
    res = mask_srt_text(DIRTY, CENSOR)
    assert "Oh my god" not in res.new_text
    assert "*********, you *****!" in res.new_text  # comma + rest preserved


def test_read_write_roundtrip_bom(tmp_path):
    p = tmp_path / "x.srt"
    p.write_bytes(b"\xef\xbb\xbf" + CLEAN.encode("utf-8"))
    text, codec = read_srt(p)
    assert codec == "utf-8-sig"
    out = tmp_path / "out.srt"
    write_srt(out, text, codec)
    assert out.read_bytes().startswith(b"\xef\xbb\xbf")


def test_parse_cues_preserves_absolute_spans_and_times():
    cues = parse_srt_cues(DIRTY)
    assert len(cues) == 2
    assert cues[0].index == 1
    assert cues[0].start_s == 1.0
    assert cues[0].end_s == 3.0
    assert DIRTY[cues[0].line_spans[0][0]:cues[0].line_spans[0][1]] == (
        "This shit happens every week."
    )


def test_extract_hits_uses_context_and_maps_multiword_tokens():
    text = (
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n"
        "Oh my god, what now?\n"
    )
    cue = parse_srt_cues(text)[0]
    hits = extract_subtitle_hits(cue, CENSOR)
    hit = next(h for h in hits if h.matched.lower() == "oh my god")
    assert hit.exempted is False
    assert hit.line_span == (0, 9)
    assert hit.token_ordinals == [0, 1, 2]


def test_anchor_tokens_suppress_bracketed_captions_and_profane_hits():
    text = (
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n"
        "[Michelle] alpha damn omega <i>noise</i>\n"
    )
    cue = parse_srt_cues(text)[0]
    hits = extract_subtitle_hits(cue, CENSOR)
    tokens = cue_tokens(cue, hits)
    by_text = {t.text.lower(): t for t in tokens}
    assert by_text["michelle"].anchor_eligible is False
    assert by_text["damn"].anchor_eligible is False
    assert by_text["noise"].anchor_eligible is False
    assert by_text["alpha"].anchor_eligible is True
    assert by_text["omega"].anchor_eligible is True
