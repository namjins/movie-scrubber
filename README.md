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

Both surfaces are processed independently; context-rules apply to both.

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
```

Exit codes: **0** ok · **1** ran but a file errored / verify failed · **2** did not run
(bad invocation, missing wordlist, no inputs, missing ffmpeg).

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
scrub/vendored/   verbatim-vendored UGE modules (whisper_flatten, wordlist_match,
                  context_filter [rebased], intervals, text_censor)
wordlists/        default.txt (872 entries) + religious-context-rules.txt
tests/            pytest suite (run: pytest)
```

## Development

```bash
pytest            # 21 tests: SRT round-trip, clamp/promotion, parity gate, context rules
```
