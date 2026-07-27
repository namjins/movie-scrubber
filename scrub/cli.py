"""`scrub` — one command: folder of .flac/.srt -> clean copies + backups + report.

Stages: detect (discover + transcribe) -> plan (derive mute windows) -> apply -> verify.
Dry-run is the default; pass --apply to write. Exit codes: 0 ok / 1 ran-with-failures /
2 did-not-run (bad invocation, missing wordlist, no inputs, missing tool).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .apply import FileResult, apply_audio, apply_srt
from .audio import ToolError, probe_format
from .config import (BACKUPS_DIR, DEFAULT_CONTEXT_RULES, DEFAULT_MODELS,
                     DEFAULT_OUTPUT_DIRNAME, DEFAULT_WORDLIST)
from .detect import find_inputs, transcribe_inputs
from .plan import derive_audio_plan
from .verify import verify_audio_parity, verify_srt
from .wordlists import load_censor


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scrub", description=__doc__)
    p.add_argument("input", type=Path, help="a .flac/.srt file or a folder of them")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True,
                   help="report what would change, write nothing (default)")
    g.add_argument("--apply", dest="dry_run", action="store_false", help="write changes")
    p.add_argument("--output", type=Path, default=None,
                   help=f"output dir (default ./{DEFAULT_OUTPUT_DIRNAME}); ignored with --in-place")
    p.add_argument("--in-place", action="store_true",
                   help="overwrite originals (backs up to backups/ first)")
    p.add_argument("--audio", dest="audio", action="store_true", default=True)
    p.add_argument("--no-audio", dest="audio", action="store_false")
    p.add_argument("--subs", dest="subs", action="store_true", default=True)
    p.add_argument("--no-subs", dest="subs", action="store_false")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                   help="comma-separated whisper models for the mute union")
    p.add_argument("--wordlist", type=Path, default=DEFAULT_WORDLIST)
    p.add_argument("--context-rules", type=Path, default=DEFAULT_CONTEXT_RULES)
    p.add_argument("--no-context-rules", dest="use_context_rules",
                   action="store_false", default=True)
    p.add_argument("--workers", type=int, default=None,
                   help="override per-model worker count")
    p.add_argument("--limit", type=int, default=None, help="process first N files (smoke)")
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--ffprobe", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.input.exists():
        sys.stderr.write(f"ERROR: input not found: {args.input}\n")
        return 2
    if not args.wordlist.exists():
        sys.stderr.write(f"ERROR: wordlist not found: {args.wordlist}\n")
        return 2

    try:
        censor = load_censor(args.wordlist, args.context_rules, args.use_context_rules)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2
    if args.use_context_rules and censor.n_rules == 0:
        sys.stderr.write("ERROR: context rules enabled but 0 rules loaded "
                         f"(check {args.context_rules})\n")
        return 2

    inputs = find_inputs(args.input, do_audio=args.audio, do_subs=args.subs, limit=args.limit)
    if not inputs.flacs and not inputs.srts:
        sys.stderr.write("ERROR: no .flac or .srt inputs found\n")
        return 2

    input_root = args.input if args.input.is_dir() else args.input.parent
    out_root = (args.output or Path.cwd() / DEFAULT_OUTPUT_DIRNAME)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    mode = "DRY-RUN" if args.dry_run else ("APPLY in-place" if args.in_place else "APPLY")
    print(f"# movie-scrubber  mode={mode}  audio={len(inputs.flacs)} subs={len(inputs.srts)}  "
          f"models={','.join(models)}  context-rules={censor.n_rules}")

    results: list[FileResult] = []

    # --- Audio -------------------------------------------------------------
    if inputs.flacs:
        workers_override = ({m: args.workers for m in models} if args.workers else None)
        try:
            raws = transcribe_inputs(inputs.flacs, models, workers_by_model=workers_override)
        except SystemExit:
            raise
        except ToolError as exc:
            sys.stderr.write(f"ERROR: {exc}\n")
            return 2

        for flac in inputs.flacs:
            try:
                fmt = probe_format(flac, args.ffprobe)
            except ToolError as exc:
                sys.stderr.write(f"ERROR: {exc}\n")
                return 2
            plan = derive_audio_plan(raws.get(flac, {}), censor, fmt.duration)
            rel = flac.relative_to(input_root) if flac != input_root else Path(flac.name)
            dst = out_root / rel
            r = apply_audio(flac, dst, plan, fmt, dry_run=args.dry_run,
                            in_place=args.in_place, backups_root=BACKUPS_DIR,
                            input_root=input_root, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
            results.append(r)
            print(f"  [audio] {rel}: {r.action} ({r.changes})"
                  + (f"  {r.detail}" if r.detail else ""))

    # --- Subtitles ---------------------------------------------------------
    for srt in inputs.srts:
        rel = srt.relative_to(input_root) if srt != input_root else Path(srt.name)
        dst = out_root / rel
        r = apply_srt(srt, dst, censor, dry_run=args.dry_run, in_place=args.in_place,
                      backups_root=BACKUPS_DIR, input_root=input_root)
        results.append(r)
        print(f"  [subs ] {rel}: {r.action} ({r.changes})"
              + (f"  {r.detail}" if r.detail else ""))

    # --- Verify (post-apply only) -----------------------------------------
    issues = []
    if not args.dry_run:
        for r in results:
            if r.kind == "subtitle" and r.action in ("masked", "copied") and r.dst:
                issues += verify_srt(r.dst, censor)
            if r.kind == "audio" and r.action == "muted" and r.dst:
                issues += verify_audio_parity(r.src, r.dst, args.ffprobe)

    # --- Summary -----------------------------------------------------------
    errors = [r for r in results if r.action == "error"]
    muted = sum(r.changes for r in results if r.kind == "audio" and r.action in ("muted", "would-mute"))
    masked = sum(r.changes for r in results if r.kind == "subtitle" and r.action in ("masked", "would-mask"))
    verb = "would mute/mask" if args.dry_run else "muted/masked"
    print(f"\n{verb}: {muted} audio window(s), {masked} subtitle span(s) "
          f"across {len(results)} file(s).")
    if args.dry_run:
        print("DRY-RUN -- pass --apply to write.")
    for iss in issues:
        sys.stderr.write(f"VERIFY FAIL: [{iss.kind}] {iss.path}: {iss.detail}\n")
    for r in errors:
        sys.stderr.write(f"ERROR: [{r.kind}] {r.src}: {r.detail}\n")

    if errors or issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
