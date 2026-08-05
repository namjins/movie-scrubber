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
from .detect import discover_input_jobs, transcribe_inputs
from .plan import derive_audio_plan
from .srt import parse_srt_cues, read_srt
from .vendored.context_filter import POLYTHEISTIC_CONTEXT_RULES
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
    p.add_argument("--no-subtitle-hints", dest="subtitle_hints",
                   action="store_false", default=True,
                   help="mask subtitles normally, but do not use SRT timings for audio")
    p.add_argument("--subtitle-fallbacks", choices=("report", "apply"), default="report",
                   help="whether pure subtitle fallback windows are report-only or applied")
    p.add_argument("--subtitle-search-pad", type=float, default=1.0,
                   help="seconds around a cue to search for Whisper-confirmed profanity")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                   help="comma-separated whisper models for the mute union")
    p.add_argument("--wordlist", type=Path, default=DEFAULT_WORDLIST)
    p.add_argument("--context-rules", type=Path, default=DEFAULT_CONTEXT_RULES)
    p.add_argument("--no-context-rules", dest="use_context_rules",
                   action="store_false", default=True)
    p.add_argument("--polytheistic", action="store_true", default=False,
                   help="opt into the polytheistic exemption bundle (\"by the gods\", "
                        "\"the old gods\", \"the hells\") for a mythology/fantasy title "
                        "where god/gods/hell(s) are lore, not blasphemy")
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

    extra_shared = [POLYTHEISTIC_CONTEXT_RULES] if args.polytheistic else None
    try:
        censor = load_censor(args.wordlist, args.context_rules, args.use_context_rules,
                             extra_shared=extra_shared)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2
    if args.use_context_rules and censor.n_rules == 0:
        sys.stderr.write("ERROR: context rules enabled but 0 rules loaded "
                         f"(check {args.context_rules})\n")
        return 2

    discovery = discover_input_jobs(args.input, do_audio=args.audio, do_subs=args.subs,
                                    limit=args.limit)
    if not discovery.flacs and not discovery.srts:
        sys.stderr.write("ERROR: no .flac or .srt inputs found\n")
        return 2

    input_root = args.input if args.input.is_dir() else args.input.parent
    out_root = (args.output or Path.cwd() / DEFAULT_OUTPUT_DIRNAME)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    mode = "DRY-RUN" if args.dry_run else ("APPLY in-place" if args.in_place else "APPLY")
    print(f"# movie-scrubber  mode={mode}  audio={len(discovery.flacs)} subs={len(discovery.srts)}  "
          f"models={','.join(models)}  context-rules={censor.n_rules}")
    print(f"  pairs={discovery.paired_count} unpaired-audio={discovery.unpaired_flac_count} "
          f"unpaired-subs={discovery.unpaired_srt_count} ambiguous-subs={discovery.ambiguous_count}")

    results: list[FileResult] = []

    # --- Audio -------------------------------------------------------------
    if discovery.flacs:
        workers_override = ({m: args.workers for m in models} if args.workers else None)
        try:
            raws = transcribe_inputs(discovery.flacs, models, workers_by_model=workers_override)
        except SystemExit:
            raise
        except ToolError as exc:
            sys.stderr.write(f"ERROR: {exc}\n")
            return 2

        for job in discovery.jobs:
            flac = job.flac
            if flac is None:
                continue
            try:
                fmt = probe_format(flac, args.ffprobe)
            except ToolError as exc:
                sys.stderr.write(f"ERROR: {exc}\n")
                return 2
            cues = None
            if args.subs and args.subtitle_hints and job.srt is not None and not job.ambiguous_srts:
                try:
                    srt_text, _codec = read_srt(job.srt)
                    cues = parse_srt_cues(srt_text)
                except OSError as exc:
                    sys.stderr.write(f"WARN: subtitle hints disabled for {flac.name}: {exc}\n")
            plan = derive_audio_plan(
                raws.get(flac, {}),
                censor,
                fmt.duration,
                subtitle_cues=cues,
                subtitle_hints=bool(cues) and args.subtitle_hints,
                subtitle_fallbacks=args.subtitle_fallbacks,
                subtitle_search_pad=args.subtitle_search_pad,
            )
            rel = flac.relative_to(input_root) if flac != input_root else Path(flac.name)
            dst = out_root / rel
            r = apply_audio(flac, dst, plan, fmt, dry_run=args.dry_run,
                            in_place=args.in_place, backups_root=BACKUPS_DIR,
                            input_root=input_root, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
            results.append(r)
            source_counts = ", ".join(f"{k}={v}" for k, v in sorted(plan.source_counts().items()))
            details = [d for d in (r.detail, source_counts) if d]
            if plan.report_only:
                previews = "; ".join(
                    f"{h.cue_time} {h.preview}" for h in plan.report_only[:3]
                )
                details.append(f"subtitle-fallback report-only={len(plan.report_only)}"
                               + (f" [{previews}]" if previews else ""))
            if job.ambiguous_srts:
                details.append("subtitle hints disabled: ambiguous subtitles")
            print(f"  [audio] {rel}: {r.action} ({r.changes})"
                  + (f"  {'; '.join(details)}" if details else ""))

    # --- Subtitles ---------------------------------------------------------
    for srt in discovery.srts:
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
