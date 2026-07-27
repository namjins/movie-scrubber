# movie-scrubber

Automated profanity muting for movie **`.flac`** audio and masking for **`.srt`** subtitles.
Point one command at a folder; get back clean copies — flagged words silenced in the audio,
masked in the subtitles — with the original audio characteristics (sample rate, channels,
bit depth, duration, volume) preserved and the originals backed up.

It inherits the hard-won correctness rules from `universal_game_editor` (the clamp/pad math,
stable-whisper DTW transcription, the blasphemy context-rules, the shared wordlist) as
**vendored** modules under `scrub/vendored/` — there is no runtime dependency on that repo.

## What it does

- **Audio (`.flac`)** — transcribes each file with three stable-whisper models
  (`base.en`, `medium.en`, `large-v3-turbo`), unions the profanity windows, and silences
  them with a single ffmpeg `volume=0` filter across **all channels** (the way monkeyplug
  mutes by default). Mute windows are derived with the canonical asymmetric pad
  (50 ms pre / 200 ms post) and pad-aware clamp, and selected with the **same** whole-word
  regex used to scan — so hyphenated compounds (`nasty-ass`) aren't missed. Output format is
  re-probed and asserted equal to the source (catches any silent resample).
- **Subtitles (`.srt`)** — masks flagged words with length-preserving full-word asterisks
  (`fuck` → `****`, no first-character hint), judged against the **blasphemy context-rules**
  (`god of war` stays, vain `oh my god` is masked). Timing lines, indices, blank separators,
  line endings, and encoding (incl. UTF-8 BOM) are preserved byte-for-byte.

Both surfaces are processed independently, from the same wordlist and context rules, in one
`scrub` invocation — point it at a folder and both `.flac` and `.srt` files underneath get
scrubbed together; no separate command or flag needed for subtitles.

An optional **polytheistic bundle** (`wordlists/polytheistic-context-rules.txt`) is shipped
but not wired in by default — see [Context rules](#context-rules) below if you're scrubbing a
fantasy/mythology title where "god"/"gods"/"hell(s)" are in-world setting nouns.

## Requirements

- **Linux** (or WSL2) with an NVIDIA GPU for the audio pass — Triton DTW word-timestamps are
  Linux-only. The subtitle pass and the CLI run anywhere.
- **ffmpeg + ffprobe** on `PATH`.
- A CUDA build of PyTorch (install it **before** the rest, or whisper falls back to CPU).

## Setup (Linux CUDA venv)

```bash
python3 -m venv ~/movie-scrubber-venv
source ~/movie-scrubber-venv/bin/activate
# CUDA torch FIRST (pick the cu* index matching your driver):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .            # installs deps from pyproject + the `scrub` command
# verify:
python -c "import torch, stable_whisper, triton; print(torch.cuda.is_available())"  # True
```

## Usage

```bash
scrub /movies/the-film               # DRY-RUN: report what would change, write nothing
scrub /movies/the-film --apply       # write clean copies to ./scrubbed/ (mirrors input tree)
scrub /movies/the-film --apply --in-place   # overwrite originals (backs up to backups/ first)

scrub film.srt --no-audio            # subtitles only (runs without a GPU)
scrub /movies --apply --models base.en,large-v3-turbo   # drop medium.en (2-model union)
scrub /movies --limit 3              # smoke-test on the first few files
scrub /movies --apply --workers 1    # force every model to 1 worker (see GPU worker sizing)
```

Exit codes: **0** ok · **1** ran but a file errored / verify failed · **2** did not run
(bad invocation, missing wordlist, no inputs, missing ffmpeg).

## GPU worker sizing

The 3 models transcribe **one at a time** (base.en fully finishes across every file that
needs it, then medium.en, then large-v3-turbo) — never concurrently, so there's no risk of
two models sharing GPU memory at once. What's tunable is how many *files* the current model
processes in parallel, via `WORKERS_BY_MODEL` in `scrub/config.py`:

```python
WORKERS_BY_MODEL = {"base.en": 4, "medium.en": 2, "large-v3-turbo": 1, ...}
```

These are **hardware-specific**, not a formula — they're sized for the maintainer's RTX 3080
Ti (12 GB VRAM), just under the measured point where `large-v3-turbo` starts OOM-thrashing
(3 workers thrashes; these numbers stay at 1). If you're on a different card, check your VRAM
(`nvidia-smi --query-gpu=memory.total --format=csv`) and re-derive conservative counts for it
— start low and watch `nvidia-smi` during a multi-file run before raising a number.

A single model's pool never uses more workers than there are files for it to process, so on a
1-2 movie run you'll see 1 worker per model regardless of the table — the table only matters
once you're batch-processing enough files that a model has more than a few to split across.

`--workers N` on the CLI overrides **every** model to exactly `N` workers, discarding the
per-model table above. Useful for a deliberate uniform cap (e.g. `--workers 1` to be maximally
conservative on VRAM), but don't reach for it as your default — it undoes the per-model tuning.

## Context rules

Five pieces determine whether a hit is censored, in load order:

1. `wordlists/default.txt` — the master censor wordlist. Censored by default.
2. `wordlists/religious-context-rules.txt` — the always-on baseline exemption set for
   God/Lord/Jesus/Christ/hell/damn (e.g. "God of War" stays, "oh my God" doesn't). Loaded
   automatically; `--no-context-rules` disables it for a strict censor-everything run. Vendored
   from `universal_game_editor` — don't hand-edit it, add project-specific exemptions to
   `wordlists/local-context-rules.txt` instead (see below) so they survive a re-vendor.
3. `wordlists/local-context-rules.txt` — a **movie-scrubber-local delta**, merged on top of the
   shared baseline automatically (no flag needed; silently skipped if absent). This is where
   this project's own exemptions live — e.g. "hellfire"/"hell-fire"/"hell fire" as a common
   idiom/proper noun, not a vain "hell" exclamation.
4. `wordlists/polytheistic-context-rules.txt` — an **opt-in** bundle for fantasy/mythology
   titles where "god"/"gods"/"hell(s)" are in-world setting nouns ("by the gods", "the old
   gods"). Shipped but **not yet wired to a CLI flag** — to use it today, concatenate it onto
   a copy of `religious-context-rules.txt` and pass that combined file via `--context-rules`.
5. `--context-rules PATH` — point at a custom/replacement rules file instead of the shipped
   baseline (same `<token>\t<context-regex>` format). Note this replaces the *shared* baseline
   path, not the local delta — `local-context-rules.txt` still merges on top either way.

**Caveat, applies to every rule in every one of these files:** an exemption is checked against
the *whole* line (subtitle cue) or Whisper segment, not the specific occurrence of the word. If
two hits of the same token land in the same line/segment — one legitimately exempt, one a vain
use — an exemption for either one exempts both (e.g. "hell fire, go to hell" would wrongly keep
both). Narrow in practice, but real; keep new rules as tight as possible.

## Notes

- **The third model.** `medium.en` is included because it was requested, but UGE's measured
  union is 2-model (`large-v3-turbo` ∪ `base.en`); `medium.en` has no measured recall benefit
  and costs a full extra transcription pass. Measure it on one movie and drop it via
  `--models` if it isn't earning its keep.
- **Caching.** RAW transcripts are cached at `cache/whisper/raw/<model>/<sha256>.json`. A
  re-run reuses them (no GPU work); changing the wordlist re-derives correctly because only
  the RAW result is cached, never the derived windows.
- **Deliberate descopes.** No phonetic-homophone recovery layer (acceptable for clean studio
  VO) and no cross-referencing of subtitle timings against audio. See the project plan.

## Layout

```
scrub/            cli, detect, plan, apply, verify, transcribe, audio, srt, wordlists, config
scrub/vendored/   re-vendored UGE modules (whisper_flatten, wordlist_match,
                  context_filter [rebased, incl. polytheistic-bundle support], intervals,
                  text_censor)
wordlists/        default.txt (999 entries), religious-context-rules.txt (always-on baseline),
                  polytheistic-context-rules.txt (opt-in bundle, see Context rules above)
tests/            pytest suite (run: pytest)
```

## Development

```bash
pytest            # 29 tests: SRT round-trip, clamp/promotion, parity gate, context rules
                   # (incl. the religious-exemption regression tests), worker-count sizing
```

The vendored modules under `scrub/vendored/` are periodically re-synced from
`universal_game_editor` as that repo's shared audio-mute/context-rules logic evolves. Each
vendored file's header comment states its source date and what changed since the last sync.
