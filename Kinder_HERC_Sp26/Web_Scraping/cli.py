# Kinder_HERC_Sp26/Web_Scraping/cli.py
"""
Kinder_HERC_Sp26.Web_Scraping.cli

Command-line interface for downloading school board meeting audio.

This CLI is intentionally simple:
- You provide one or more sources as "District|URL"
- Optionally provide a text file containing one "District|URL" per line
- The pipeline processes each source and outputs WAV files to:

    School Board Meetings/<District>/YYYY-MM-DD_title.wav

Examples
--------
A) Provide sources inline (repeat --source):
    python -m Kinder_HERC_Sp26.Web_Scraping.cli ^
        --source "Spring Branch ISD|https://springbranchisdtx.new.swagit.com/videos/364571/download" ^
        --source "Alief ISD|https://video.aliefisd.net/show-videos?g=..." ^
        --cutoff 2024-09-01

B) Provide sources via text file (one per line: District|URL):
    python -m Kinder_HERC_Sp26.Web_Scraping.cli --sources-file sources.txt --cutoff 2024-09-01

Notes
-----
- Run with `python -m ...` so relative imports work properly.
- ffmpeg must be installed and available on PATH for WAV extraction.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from .models import Source
from .pipeline import PipelineConfig, process_source


def _parse_source_spec(spec: str) -> Source:
    """
    Parse a single "District|URL" spec into a Source.

    Inputs
    ------
    spec : str
        String in the form "District|URL".

    Outputs
    -------
    Source

    Effects
    -------
    None.
    """
    if "|" not in spec:
        raise ValueError(f'Invalid source spec: "{spec}" (expected "District|URL")')
    district, url = spec.split("|", 1)
    district = district.strip()
    url = url.strip()
    if not district or not url:
        raise ValueError(f'Invalid source spec: "{spec}" (district and url must be non-empty)')
    return Source(district=district, url=url)


def _read_sources_file(path: Path) -> List[Source]:
    """
    Read sources from a text file.

    File format
    -----------
    One source per line:
        District|URL

    Blank lines and lines starting with '#' are ignored.

    Inputs
    ------
    path : Path
        Path to sources file.

    Outputs
    -------
    List[Source]

    Effects
    -------
    File I/O: reads the file.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[Source] = []
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(_parse_source_spec(line))
        except ValueError as ex:
            raise ValueError(f"{path}:{i}: {ex}") from ex
    return out


def _dedup_sources(sources: List[Source]) -> List[Source]:
    """
    Deduplicate sources by (district, url) while preserving input order.

    Inputs
    ------
    sources : List[Source]

    Outputs
    -------
    List[Source]

    Effects
    -------
    None.
    """
    seen: Dict[Tuple[str, str], None] = {}
    out: List[Source] = []
    for s in sources:
        key = (s.district, s.url)
        if key in seen:
            continue
        seen[key] = None
        out.append(s)
    return out


def main() -> None:
    """
    Entry point for the CLI.

    Parses arguments, builds PipelineConfig, and processes each source in a thread pool.

    Effects
    -------
    - Prints status to stdout
    - Network + disk I/O via pipeline
    """
    ap = argparse.ArgumentParser(description="Download school board meeting audio as WAV files.")
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        help='Repeatable. Format: "District|URL"',
    )
    ap.add_argument(
        "--sources-file",
        type=Path,
        default=None,
        help='Text file with one "District|URL" per line (comments allowed with #).',
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("School Board Meetings"),
        help='Output root directory (default: "School Board Meetings").',
    )
    ap.add_argument(
        "--cutoff",
        type=str,
        default="2024-09-01",
        help=(
            "Only keep meetings with date >= cutoff (YYYY-MM-DD). "
            "Applied to iframe candidate filtering and final download filtering. "
            "Default: 2024-09-01"
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of concurrent sources to process (default: min(8, CPU count)).",
    )
    ap.add_argument(
        "--max-candidates",
        type=int,
        default=60,
        help="Max candidate links to try when HTML scraping (default: 60).",
    )
    ap.add_argument(
        "--frag-workers",
        type=int,
        default=8,
        help="Concurrent fragment downloads for streaming (default: 8).",
    )
    ap.add_argument(
        "--strict-dates",
        action="store_true",
        help="If set, skip downloads when date cannot be determined (instead of keeping with 0000-00-00).",
    )
    ap.add_argument(
        "--include-title",
        type=str,
        default=r"(?i)\b(board\s*meeting|meeting\s*of\s*the\s*board|board\s*of\s*trustees|trustee\s*meeting|regular\s*board\s*meeting|special\s*board\s*meeting)\b",
        help="Regex: only download videos whose title matches this pattern. Default targets board meetings.",
    )
    ap.add_argument(
        "--exclude-title",
        type=str,
        default="",
        help="Regex: skip videos whose title matches this pattern (optional).",
    )
    ap.add_argument(
        "--include-anchor-label",
        type=str,
        default=r"\bboard\W*meetings?\b",
        help="Regex: for Swagit iframe anchors, keep only links whose visible label matches this pattern.",
    )
    ap.add_argument(
        "--ffmpeg-location",
        type=Path,
        default=None,
        help="Path to ffmpeg 'bin' directory (or ffmpeg.exe). Passed to yt-dlp.",
    )

    args = ap.parse_args()

    # Parse cutoff date
    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d").date()

    # Collect sources
    sources: List[Source] = []
    if args.sources_file is not None:
        sources.extend(_read_sources_file(args.sources_file))
    if args.source:
        sources.extend(_parse_source_spec(s) for s in args.source)

    sources = _dedup_sources(sources)

    if not sources:
        raise SystemExit("No sources provided. Use --source or --sources-file.")

    # Ensure output directory exists
    args.out.mkdir(parents=True, exist_ok=True)

    # Decide worker count
    if args.workers and args.workers > 0:
        workers = args.workers
    else:
        workers = min(8, (os.cpu_count() or 4))

    cfg = PipelineConfig(
        out_root=args.out,
        cutoff=cutoff,
        fragment_workers=args.frag_workers,
        max_candidates=args.max_candidates,
        keep_unknown_date=(not args.strict_dates),
        include_title_regex=args.include_title,
        exclude_title_regex=(args.exclude_title or None),
        include_anchor_label_regex=(args.include_anchor_label or None),
        min_date=cutoff,
        ffmpeg_location=args.ffmpeg_location,
    )

    # Run sources concurrently
    def _run_one(src: Source) -> Tuple[bool, str]:
        def _status(line: str) -> None:
            tqdm.write(f"[{src.district}] {line}")

        _status("queued")
        ok, msg = process_source(src, cfg, status_cb=_status)
        return ok, f"[{src.district}] {msg}"

    results: List[Tuple[bool, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_one, s) for s in sources]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Sources"):
            results.append(fut.result())

    # Summarize
    ok_count = sum(1 for ok, _ in results if ok)
    print(f"\nDone. Success={ok_count}/{len(results)}")

    # Print failures for easy debugging
    for ok, msg in results:
        if not ok:
            print("FAIL:", msg)


if __name__ == "__main__":
    main()
