from scrub import cli
from scrub.apply import FileResult
from scrub.audio import AudioFormat


RAW_ANCHORS = {
    "segments": [{
        "text": "alpha bravo charlie delta echo foxtrot gamma omega",
        "start": 0.0,
        "end": 11.0,
        "words": [
            {"word": "alpha", "start": 0.45, "end": 0.55, "probability": 0.9},
            {"word": "bravo", "start": 1.45, "end": 1.55, "probability": 0.9},
            {"word": "charlie", "start": 3.45, "end": 3.55, "probability": 0.9},
            {"word": "delta", "start": 4.45, "end": 4.55, "probability": 0.9},
            {"word": "echo", "start": 6.45, "end": 6.55, "probability": 0.9},
            {"word": "foxtrot", "start": 7.45, "end": 7.55, "probability": 0.9},
            {"word": "gamma", "start": 9.45, "end": 9.55, "probability": 0.9},
            {"word": "omega", "start": 10.45, "end": 10.55, "probability": 0.9},
        ],
    }],
    "language": "en",
}


SRT_WITH_FALLBACK = (
    "1\n00:00:00,000 --> 00:00:02,000\nalpha bravo\n\n"
    "2\n00:00:03,000 --> 00:00:05,000\ncharlie delta\n\n"
    "3\n00:00:06,000 --> 00:00:08,000\necho foxtrot\n\n"
    "4\n00:00:09,000 --> 00:00:11,000\ngamma omega\n\n"
    "5\n00:00:12,000 --> 00:00:14,000\ndamn\n"
)


def _write_pair(tmp_path):
    flac = tmp_path / "Movie.flac"
    srt = tmp_path / "Movie-eng.srt"
    flac.write_bytes(b"not really flac")
    srt.write_text(SRT_WITH_FALLBACK, encoding="utf-8")
    return flac, srt


def _patch_audio(monkeypatch, captured):
    monkeypatch.setattr(cli, "probe_format",
                        lambda *_a, **_k: AudioFormat(48000, 2, "stereo", "s16", 16, 15.0))
    monkeypatch.setattr(cli, "transcribe_inputs",
                        lambda flacs, *_a, **_k: {f: {"large-v3-turbo": RAW_ANCHORS} for f in flacs})

    def fake_apply_audio(src, dst, plan, fmt, **kwargs):
        captured.append(plan)
        action = "would-mute" if plan.intervals else "skipped"
        return FileResult(src, None, "audio", len(plan.intervals), action)

    monkeypatch.setattr(cli, "apply_audio", fake_apply_audio)


def test_no_subs_disables_subtitle_reading_and_hints(tmp_path, monkeypatch, capsys):
    _write_pair(tmp_path)
    captured = []
    _patch_audio(monkeypatch, captured)
    monkeypatch.setattr(cli, "read_srt", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))
    rc = cli.main([str(tmp_path), "--no-subs"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "subs=0" in out
    assert captured[0].report_only == []


def test_no_subtitle_hints_still_masks_subtitles(tmp_path, monkeypatch):
    _write_pair(tmp_path)
    captured = []
    srt_calls = []
    _patch_audio(monkeypatch, captured)

    def fake_apply_srt(src, dst, censor, **kwargs):
        srt_calls.append(src)
        return FileResult(src, None, "subtitle", 1, "would-mask")

    monkeypatch.setattr(cli, "apply_srt", fake_apply_srt)
    rc = cli.main([str(tmp_path), "--no-subtitle-hints"])
    assert rc == 0
    assert srt_calls
    assert captured[0].report_only == []


def test_subtitle_fallbacks_report_vs_apply(tmp_path, monkeypatch, capsys):
    _write_pair(tmp_path)
    captured = []
    _patch_audio(monkeypatch, captured)
    rc = cli.main([str(tmp_path), "--subtitle-fallbacks", "report"])
    out = capsys.readouterr().out
    assert rc == 0
    assert captured[0].report_only
    assert "subtitle-fallback report-only=1" in out

    captured.clear()
    rc = cli.main([str(tmp_path), "--subtitle-fallbacks", "apply"])
    assert rc == 0
    assert captured[0].source_counts()["subtitle-fallback"] == 1
