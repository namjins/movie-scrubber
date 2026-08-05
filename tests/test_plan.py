"""Audio derivation: same-matcher span selection (catches hyphenated compounds),
whole-clip promotion, pad-aware clamp, context exemption, and cross-model union."""
from scrub.config import PAD_POST_S, PAD_PRE_S
from scrub.plan import derive_audio_plan
from scrub.srt import parse_srt_cues
from scrub.wordlists import load_censor

CENSOR = load_censor()


def _raw(words, text=None):
    text = text or " ".join(w[0] for w in words)
    return {"segments": [{"text": text, "start": words[0][1], "end": words[-1][2],
                          "words": [{"word": w, "start": s, "end": e, "probability": 0.9}
                                    for (w, s, e) in words]}],
            "language": "en"}


def test_hyphenated_compound_is_muted_via_pattern():
    # "nasty-ass" is NOT in the swears SET, but \bass\b matches inside it. The
    # pattern.search selection must catch it (the documented under-mute bug fix).
    raw = _raw([("you", 0.0, 0.3), ("nasty-ass", 0.3, 0.9), ("bastard", 1.0, 1.6)])
    plan = derive_audio_plan({"m": raw}, CENSOR, duration=2.0)
    assert plan.intervals, "expected at least one mute window"
    # the merged window should cover the nasty-ass region (start near 0.25)
    assert plan.intervals[0][0] <= 0.30


def test_whole_clip_promotion_when_word_hugs_both_ends():
    raw = _raw([("fuck", 0.02, 0.45)], text="fuck")
    plan = derive_audio_plan({"m": raw}, CENSOR, duration=0.5)
    assert len(plan.intervals) == 1
    s, e = plan.intervals[0]
    assert s == 0.0 and abs(e - 0.5) < 1e-6  # silenced full clip


def test_pad_aware_clamp_keeps_windows_in_bounds():
    raw = _raw([("hello", 0.0, 0.3), ("shit", 0.32, 0.6)])
    dur = 1.0
    plan = derive_audio_plan({"m": raw}, CENSOR, duration=dur)
    for s, e in plan.intervals:
        assert s >= 0.0
        assert e <= dur + 1e-9


def test_context_exempts_deity_title_in_audio():
    raw = _raw([("the", 0.0, 0.2), ("god", 0.2, 0.5), ("of", 0.5, 0.6), ("war", 0.6, 0.9)],
               text="the god of war")
    plan = derive_audio_plan({"m": raw}, CENSOR, duration=2.0)
    assert plan.intervals == []  # "god of war" exempt


def test_vain_god_is_muted_in_audio():
    raw = _raw([("oh", 0.0, 0.2), ("my", 0.2, 0.4), ("god", 0.4, 0.8)], text="oh my god")
    plan = derive_audio_plan({"m": raw}, CENSOR, duration=2.0)
    assert len(plan.intervals) == 1


def test_union_across_models_merges_overlap():
    # model A hears the curse a touch earlier than model B; union -> one window.
    a = _raw([("shit", 0.50, 0.80)])
    b = _raw([("shit", 0.55, 0.90)])
    plan = derive_audio_plan({"a": a, "b": b}, CENSOR, duration=2.0)
    assert len(plan.intervals) == 1
    assert plan.per_model_counts == {"a": 1, "b": 1}


def _srt(blocks):
    out = []
    for i, (start, end, text) in enumerate(blocks, start=1):
        out.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(out)


def test_subtitle_confirmed_adds_source_provenance():
    raw = _raw([("alpha", 0.20, 0.50), ("damn", 0.80, 1.00), ("omega", 1.30, 1.60)])
    cues = parse_srt_cues(_srt([("00:00:00,000", "00:00:02,000", "alpha damn omega")]))
    plan = derive_audio_plan({"large-v3-turbo": raw}, CENSOR, duration=2.0,
                             subtitle_cues=cues)
    assert plan.source_counts()["subtitle-confirmed"] == 1
    assert any(src.kind == "subtitle-confirmed" for w in plan.windows for src in w.sources)


def test_subtitle_inferred_uses_bracketing_anchor_words():
    cues = parse_srt_cues(_srt([
        ("00:00:00,000", "00:00:02,000", "alpha bravo"),
        ("00:00:03,000", "00:00:05,000", "charlie delta"),
        ("00:00:06,000", "00:00:08,000", "echo foxtrot"),
        ("00:00:09,000", "00:00:11,000", "alpha damn omega"),
    ]))
    raw = _raw([
        ("alpha", 0.45, 0.55), ("bravo", 1.45, 1.55),
        ("charlie", 3.45, 3.55), ("delta", 4.45, 4.55),
        ("echo", 6.45, 6.55), ("foxtrot", 7.45, 7.55),
        ("alpha", 9.35, 9.45), ("omega", 10.55, 10.65),
    ])
    plan = derive_audio_plan({"large-v3-turbo": raw}, CENSOR, duration=12.0,
                             subtitle_cues=cues)
    assert plan.alignment and plan.alignment.passed
    assert plan.source_counts()["subtitle-inferred"] == 1
    inferred = next(w for w in plan.windows if any(s.kind == "subtitle-inferred" for s in w.sources))
    assert inferred.start < 10.30 < inferred.end


def test_subtitle_fallback_is_report_only_by_default_and_apply_opt_in():
    cues = parse_srt_cues(_srt([
        ("00:00:00,000", "00:00:02,000", "alpha bravo"),
        ("00:00:03,000", "00:00:05,000", "charlie delta"),
        ("00:00:06,000", "00:00:08,000", "echo foxtrot"),
        ("00:00:09,000", "00:00:11,000", "gamma omega"),
        ("00:00:12,000", "00:00:14,000", "damn"),
    ]))
    raw = _raw([
        ("alpha", 0.45, 0.55), ("bravo", 1.45, 1.55),
        ("charlie", 3.45, 3.55), ("delta", 4.45, 4.55),
        ("echo", 6.45, 6.55), ("foxtrot", 7.45, 7.55),
        ("gamma", 9.45, 9.55), ("omega", 10.45, 10.55),
    ])
    report = derive_audio_plan({"large-v3-turbo": raw}, CENSOR, duration=15.0,
                               subtitle_cues=cues)
    assert report.alignment and report.alignment.passed
    assert len(report.report_only) == 1
    assert "subtitle-fallback" not in report.source_counts()

    apply = derive_audio_plan({"large-v3-turbo": raw}, CENSOR, duration=15.0,
                              subtitle_cues=cues, subtitle_fallbacks="apply")
    assert apply.source_counts()["subtitle-fallback"] == 1


def test_alignment_gate_fails_on_large_offset_error():
    cues = parse_srt_cues(_srt([
        ("00:00:00,000", "00:00:02,000", "alpha bravo"),
        ("00:00:03,000", "00:00:05,000", "charlie delta"),
        ("00:00:06,000", "00:00:08,000", "echo foxtrot"),
        ("00:00:09,000", "00:00:11,000", "gamma omega"),
        ("00:00:12,000", "00:00:14,000", "damn"),
    ]))
    raw = _raw([
        ("alpha", 0.45, 0.55), ("bravo", 1.45, 1.55),
        ("charlie", 8.45, 8.55), ("delta", 9.45, 9.55),
        ("echo", 6.45, 6.55), ("foxtrot", 7.45, 7.55),
        ("gamma", 14.45, 14.55), ("omega", 15.45, 15.55),
    ])
    plan = derive_audio_plan({"large-v3-turbo": raw}, CENSOR, duration=20.0,
                             subtitle_cues=cues)
    assert plan.alignment and not plan.alignment.passed
    assert plan.report_only
