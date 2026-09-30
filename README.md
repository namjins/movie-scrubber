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

Both surfaces use the same wordlist and context rules in one `scrub` invocation. When a
paired `.srt` is available, subtitle timings are also used as conservative hints for the
audio plan: Whisper-confirmed and tightly inferred subtitle hits can add mute windows, while
rough subtitle-only fallback hits are report-only unless explicitly opted into.

An optional **polytheistic bundle** (`wordlists/polytheistic-context-rules.txt`), opt-in via
`--polytheistic`, is available if you're scrubbing a fantasy/mythology title where "god"/
"hell(s)" are in-world setting nouns — see [Context rules](#context-rules) below. Note: it
only affects **`god`/`hell`/`hells`** in this repo; 12 of its 24 rules key on the plural
`gods`, which isn't in `wordlists/default.txt` here and so can never be hit.

## Requirements

- **Linux** (or WSL2) with an NVIDIA GPU for the audio pass — Triton DTW word-timestamps are
  Linux-only. The subtitle pass and the CLI run anywhere.
- **ffmpeg + ffprobe** on `PATH`.
- A CUDA build of PyTorch (install it **before** the rest, or whisper falls back to CPU).

## Setup (Linux CUDA venv)

On native Linux, run these directly. On Windows, run them inside WSL2 (see the WSL2
subsection below first) - there is no native-Windows GPU path, since Triton DTW word
timestamps require Linux.

```bash
# ffmpeg + ffprobe (system packages, not pip):
sudo apt update && sudo apt install -y ffmpeg          # Debian/Ubuntu
# or: sudo dnf install -y ffmpeg / brew install ffmpeg

python3 -m venv ~/movie-scrubber-venv
source ~/movie-scrubber-venv/bin/activate
# CUDA torch FIRST (pick the cu* index matching your driver):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .            # installs deps from pyproject + the `scrub` command
# verify GPU torch actually landed:
python -c "import torch, stable_whisper, triton; print(torch.cuda.is_available())"  # True
```

If that last line prints `False`, torch fell back to CPU - see Troubleshooting below
before running anything on real files.

### WSL2 (Windows host)

1. Enable WSL2 and install an Ubuntu distro: `wsl --install -d Ubuntu` (PowerShell,
   admin). Requires a recent NVIDIA driver on the **Windows** side with WSL support
   (already the case for any driver from the last few years) - do **not** install a
   separate Linux NVIDIA driver inside WSL2, it uses the Windows host's driver directly.
2. Confirm the GPU is visible inside WSL2 before doing anything else:
   `wsl -d Ubuntu -- nvidia-smi` - if this fails, fix the Windows-side driver/WSL
   kernel update first; nothing below will work until it succeeds.
3. Run the setup block above from inside that Ubuntu distro
   (`wsl -d Ubuntu -- bash -lc '...'`, or open a WSL shell first), not from PowerShell.
4. Reach repo files under `D:\...` from WSL2 at `/mnt/d/...` (adjust the drive letter to
   wherever this repo is cloned).

### Troubleshooting

- **`torch.cuda.is_available()` prints `False`, or logs say
  `FP16 is not supported on CPU`.** Torch installed the CPU build. Fix: `pip uninstall
  torch` inside the venv, then reinstall with the `--index-url` CUDA line above
  *before* anything else touches `torch` (a plain `pip install -e .` run first will
  silently pull in CPU torch as a transitive dependency and it needs to be replaced,
  not layered on top).
- **`nvidia-smi` not found, or found but WSL2's copy can't see the GPU.** The Windows
  host's NVIDIA driver needs updating (WSL2 GPU passthrough support), not a driver
  install inside WSL2 itself.
- **First `scrub` run is slow / downloads several GB.** `openai-whisper` fetches each
  model (`base.en`, `medium.en`, `large-v3-turbo`) to `~/.cache/whisper/` on first use.
  This is a one-time cost per model, not per file; subsequent runs reuse the cached
  weights.
- **`ffmpeg: command not found`.** It's a system package, not a pip dependency - see
  the `apt install` line above (or your distro's equivalent).

## Usage

```bash
scrub /movies/the-film               # DRY-RUN: report what would change, write nothing
scrub /movies/the-film --apply       # write clean copies to ./scrubbed/ (mirrors input tree)
scrub /movies/the-film --apply --in-place   # overwrite originals (backs up to backups/ first)

scrub film.srt --no-audio            # subtitles only (runs without a GPU)
scrub /movies --apply --models base.en,large-v3-turbo   # drop medium.en (2-model union)
scrub /movies --limit 3              # smoke-test on the first few files
scrub /movies --apply --workers 1    # force every model to 1 worker (see GPU worker sizing)
scrub /movies --no-subtitle-hints     # mask subtitles, but don't use SRT timings for audio
scrub /movies --subtitle-fallbacks apply  # opt into rough subtitle-only fallback mutes
scrub /movies --polytheistic         # fantasy/mythology title: "god of war"-style lore stays
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
   titles where "god"/"hell(s)" are in-world setting nouns, enabled via `--polytheistic`.
   Only reaches `god`/`hell`/`hells` in this repo — the bundle's `gods`-keyed rules ("by the
   gods", "the old gods") are inert here because plural `gods` isn't in `default.txt`; add it
   there first if you need those to fire.
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
- **Subtitle hints.** Pairing prefers `Movie.srt` for `Movie.flac`, then language suffixes
  like `Movie-eng.srt` / `Movie.en.srt`. Dry-run reports paired/unpaired counts plus
  subtitle-confirmed, subtitle-inferred, and report-only subtitle-fallback hits.
- **Caching.** RAW transcripts are cached at `cache/whisper/raw/<model>/<sha256>.json`. A
  re-run reuses them (no GPU work); changing the wordlist re-derives correctly because only
  the RAW result is cached, never the derived windows.
- **Deliberate descopes.** No phonetic-homophone recovery layer (acceptable for clean studio
  VO).

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
pytest            # SRT round-trip, pairing, subtitle hints, clamp/promotion, parity gate,
                   # context rules, worker-count sizing
```

The vendored modules under `scrub/vendored/` are periodically re-synced from
`universal_game_editor` as that repo's shared audio-mute/context-rules logic evolves. Each
vendored file's header comment states its source date and what changed since the last sync.
