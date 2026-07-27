"""Audio derivation: same-matcher span selection (catches hyphenated compounds),
whole-clip promotion, pad-aware clamp, context exemption, and cross-model union."""
from scrub.config import PAD_POST_S, PAD_PRE_S
from scrub.plan import derive_audio_plan
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
