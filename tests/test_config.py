"""Pin the operator-confirmed, hardware-tuned constants in config.py so they don't silently
drift back to the wrong GPU's numbers (this box: RTX 3080 Ti / 12 GB, confirmed via
nvidia-smi 2026-07-27 -- the file previously carried RTX 5080 / ~16-17 GB assumptions)."""
from scrub.config import WORKERS_BY_MODEL


def test_workers_by_model_matches_operator_confirmed_allocation():
    assert WORKERS_BY_MODEL["base.en"] == 4
    assert WORKERS_BY_MODEL["medium.en"] == 2
    assert WORKERS_BY_MODEL["large-v3-turbo"] == 1
    # Alias keys (bare model family names) stay in sync with their .en counterparts.
    assert WORKERS_BY_MODEL["base"] == WORKERS_BY_MODEL["base.en"]
    assert WORKERS_BY_MODEL["medium"] == WORKERS_BY_MODEL["medium.en"]


def test_workers_by_model_never_exceeds_the_measured_safe_ceiling():
    # universal_game_editor's measured ground truth for THIS card: large-v3-turbo tops out
    # at 2 workers alone (3 OOM-thrash). Our per-model count must stay under that ceiling.
    assert WORKERS_BY_MODEL["large-v3-turbo"] < 2
