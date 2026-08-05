from pathlib import Path

from scrub.detect import discover_input_jobs


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_derry_girls_suffix_pairing(tmp_path):
    flac = tmp_path / "Derry Girls (2018) - S01E01 - Episode 1.flac"
    srt = tmp_path / "Derry Girls (2018) - S01E01 - Episode 1-eng.srt"
    _touch(flac)
    _touch(srt)
    d = discover_input_jobs(tmp_path)
    assert d.paired_count == 1
    assert d.jobs[0].flac == flac
    assert d.jobs[0].srt == srt


def test_exact_stem_subtitle_wins_over_language_suffix(tmp_path):
    flac = tmp_path / "Movie.flac"
    exact = tmp_path / "Movie.srt"
    eng = tmp_path / "Movie-eng.srt"
    for p in (flac, exact, eng):
        _touch(p)
    d = discover_input_jobs(tmp_path)
    paired = next(j for j in d.jobs if j.flac == flac)
    assert paired.srt == exact
    assert any(j.srt == eng and j.flac is None for j in d.jobs)


def test_same_priority_subtitle_ambiguity_disables_hints(tmp_path):
    flac = tmp_path / "Movie.flac"
    a = tmp_path / "Movie-eng.srt"
    b = tmp_path / "Movie-ENG.srt"
    for p in (flac, a, b):
        _touch(p)
    d = discover_input_jobs(tmp_path)
    paired = next(j for j in d.jobs if j.flac == flac)
    assert paired.srt is None
    assert sorted(p.name for p in paired.ambiguous_srts) == sorted([a.name, b.name])


def test_single_flac_discovers_matching_sidecar_only(tmp_path):
    flac = tmp_path / "Movie.flac"
    sidecar = tmp_path / "Movie-en.srt"
    other = tmp_path / "Other-en.srt"
    for p in (flac, sidecar, other):
        _touch(p)
    d = discover_input_jobs(flac)
    assert len(d.jobs) == 1
    assert d.jobs[0].flac == flac
    assert d.jobs[0].srt == sidecar


def test_limit_applies_after_pairing(tmp_path):
    for i in range(3):
        _touch(tmp_path / f"Movie {i}.flac")
        _touch(tmp_path / f"Movie {i}-eng.srt")
    d = discover_input_jobs(tmp_path, limit=2)
    assert len(d.jobs) == 2
    assert all(j.flac is not None and j.srt is not None for j in d.jobs)
