"""
Kinder_HERC_Sp26.Web_Scraping.ytdlp_runner

yt-dlp + ffmpeg integration for downloading meeting videos and extracting audio.

Module does:
  - probing metadata without downloading
  - downloading videos/playlists using yt-dlp
  - extracting audio to WAV via ffmpeg (yt-dlp postprocessor)
  - renaming extracted WAVs to the required format:
        School Board Meetings/<District>/YYYY-MM-DD_title.wav

Notes
-----
- ffmpeg must be installed and available on PATH.
- For YouTube playlists, we can filter entries *before download* using yt-dlp's
  matchtitle / rejecttitle options (fastest way to skip unwanted videos).
"""

from __future__ import annotations

import re
import random
import time
from datetime import datetime, date
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import yt_dlp

from .dates import parse_date_from_text
from .config import USER_AGENT

MAX_TRANSIENT_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 5.0
BACKOFF_JITTER_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 300.0
UUID_TITLE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def is_transient_throttle_error(msg: str) -> bool:
    """
    Return True for likely temporary throttling / verification failures.
    """
    m = (msg or "").lower()
    if "429" in m or "too many requests" in m:
        return True
    if "sign in to confirm you" in m and "not a bot" in m:
        return True
    if "captcha" in m:
        return True
    if "http error 403" in m and (
        "rate limit" in m
        or "too many requests" in m
        or "captcha" in m
        or "not a bot" in m
        or "temporar" in m
    ):
        return True
    return False


# ---------------------------
# Filename helpers
# ---------------------------

def safe_filename(s: str, max_len: int = 140) -> str:
    """
    Cleans a filename by removing weird characters and normalizing whitespace.

    Inputs
    ------
    s : str
        Input string (video title).
    max_len : int
        Max length to keep (helps avoid Windows path length issues).

    Outputs
    -------
    str
        Cleaned filename component.

    Effects
    -------
    None.
    """
    s = (s or "").strip()
    s = re.sub(r"[^\w\s.-]", "", s)   # remove weird characters
    s = re.sub(r"\s+", "_", s)        # spaces -> underscores
    return s[:max_len] if len(s) > max_len else s


# ---------------------------
# yt-dlp option builders
# ---------------------------

def build_ydl_opts(
    tmp_dir: Path,
    cutoff: date,
    fragment_workers: int = 8,
    quiet: bool = True,
    keep_unknown_date: bool = True,
    include_title_regex: Optional[str] = None,
    exclude_title_regex: Optional[str] = None,
    ffmpeg_location: Optional[Path] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> Dict:
    """
    Build yt-dlp options.

    Inputs
    ------
    tmp_dir : Path
        Directory where yt-dlp should place intermediate outputs.
    cutoff : date
        Only keep meetings with date >= cutoff (when date is known).
    fragment_workers : int
        Concurrency for HLS/DASH fragments (helps speed up .m3u8 downloads).
    quiet : bool
        Reduce console output.
    keep_unknown_date : bool
        If True, keep entries even when date cannot be determined.
        If False, skip unknown-date entries.
    include_title_regex : Optional[str]
        If provided, yt-dlp will only download entries whose title matches this regex.
        (This is the fastest way to filter playlists.)
    exclude_title_regex : Optional[str]
        If provided, yt-dlp will skip entries whose title matches this regex.

    Outputs
    -------
    Dict
        Options dictionary to pass into yt_dlp.YoutubeDL(...)

    Effects
    -------
    - Ensures tmp_dir exists.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    include_pat = re.compile(include_title_regex) if include_title_regex else None
    exclude_pat = re.compile(exclude_title_regex) if exclude_title_regex else None

    def _match_filter(info_dict, *, incomplete):
        # Filter at selection time so non-matching videos are skipped before media download.
        title = (info_dict.get("title") or "").strip()

        # yt-dlp may invoke match_filter with partial metadata first.
        # In that phase, avoid strict include filtering to prevent false negatives.
        if incomplete:
            if exclude_pat and title and exclude_pat.search(title):
                return "title matches exclude filter"
            return None

        if include_pat and not include_pat.search(title):
            return "title does not match include filter"
        if exclude_pat and exclude_pat.search(title):
            return "title matches exclude filter"

        fallback_text = (info_dict.get("webpage_url") or "").strip()
        dt = entry_meeting_date(info_dict, fallback_text=fallback_text)
        if dt is None and not keep_unknown_date:
            return "date unknown and strict date filtering enabled"
        if dt is not None and dt < cutoff:
            return f"meeting date before cutoff ({cutoff.isoformat()})"
        return None

    progress_buckets: Dict[str, int] = {}

    def _progress_hook(hook_data: Dict) -> None:
        if not status_cb:
            return
        state = (hook_data.get("status") or "").strip().lower()
        info = hook_data.get("info_dict") or {}
        vid = (info.get("id") or "unknown").strip()
        title = (info.get("title") or vid).strip()

        if state == "downloading":
            total = hook_data.get("total_bytes") or hook_data.get("total_bytes_estimate")
            done = hook_data.get("downloaded_bytes")
            if total and done:
                pct = int((done / total) * 100)
                bucket = (pct // 10) * 10
                prev = progress_buckets.get(vid, -10)
                if bucket > prev:
                    progress_buckets[vid] = bucket
                    status_cb(f"downloading: {title} ({pct}%)")
        elif state == "finished":
            status_cb(f"download finished: {title}")
        elif state == "error":
            status_cb(f"download error: {title}")

    opts: Dict = {
        # Logging/noise control
        "quiet": quiet,
        "no_warnings": quiet,
        "noprogress": quiet,

        # Keep going if one entry fails in a playlist
        "ignoreerrors": True,
        "retries": 3,
        "fragment_retries": 3,

        # Speed for segmented streams (.m3u8)
        "concurrent_fragment_downloads": max(1, fragment_workers),

        # Look like a browser
        "http_headers": {"User-Agent": USER_AGENT},

        # Deterministic temp naming
        "outtmpl": str(tmp_dir / "%(id)s.%(ext)s"),

        # Prefer true audio-only streams to avoid unnecessary video transfer.
        "format": "bestaudio[acodec!=none]/bestaudio/best[acodec!=none]/best",

        # Extract WAV via ffmpeg after download
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],

        # Apply title/date filtering before downloading media.
        "match_filter": _match_filter,
    }
    if ffmpeg_location:
        opts["ffmpeg_location"] = str(ffmpeg_location)
    if status_cb:
        opts["progress_hooks"] = [_progress_hook]

    return opts


# ---------------------------
# Metadata probing
# ---------------------------

def probe_metadata(url: str) -> Optional[Dict]:
    """
    Try to extract metadata for a URL without downloading.

    Inputs
    ------
    url : str

    Outputs
    -------
    Optional[dict]
        yt-dlp info dict on success, else None.

    Effects
    -------
    Network I/O (metadata requests), but no file downloads.
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
        "http_headers": {"User-Agent": USER_AGENT},
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception:
        return None


def iter_entries(info: Dict) -> Iterable[Dict]:
    """
    Normalize a yt-dlp info dict into a stream of per-video entries.

    Inputs
    ------
    info : dict
        yt-dlp info dict returned by extract_info()

    Outputs
    -------
    Iterable[dict]
        Each dict represents one downloadable entry.

    Effects
    -------
    None.
    """
    if not info:
        return []
    if "entries" in info and info["entries"]:
        return (e for e in info["entries"] if e)
    return [info]


def entry_meeting_date(entry: Dict, fallback_text: str = "") -> Optional[date]:
    """
    Determine the meeting date for a video entry.

    Tries:
      1) entry['upload_date'] if it is YYYYMMDD
      2) parse date from entry['title']
      3) parse date from fallback_text (often URL)

    Inputs
    ------
    entry : dict
        yt-dlp entry dict.
    fallback_text : str
        Extra string to parse dates from if title lacks a date.

    Outputs
    -------
    Optional[date]
        Meeting date if discovered, else None.
    """
    up = (entry.get("upload_date") or "").strip()
    if re.fullmatch(r"\d{8}", up):
        try:
            return datetime.strptime(up, "%Y%m%d").date()
        except Exception:
            pass

    title = (entry.get("title") or "").strip()
    return parse_date_from_text(title) or parse_date_from_text(fallback_text)


# ---------------------------
# Download + rename
# ---------------------------

def download_source_to_wav(
    url: str,
    district_dir: Path,
    district_name: str,
    cutoff: date,
    fragment_workers: int = 8,
    keep_unknown_date: bool = True,
    include_title_regex: Optional[str] = None,
    exclude_title_regex: Optional[str] = None,
    override_title: Optional[str] = None,
    override_meeting_date: Optional[date] = None,
    ffmpeg_location: Optional[Path] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """
    Download a URL (video or playlist) and extract WAV audio files into district_dir.

    Output naming convention:
        DISTRICT-YYYY-MM-DD-Title.wav

    Inputs
    ------
    url : str
        Source URL (playlist, vendor page, etc.) to feed into yt-dlp.
    district_dir : Path
        Output directory for the district.
    district_name : str
        District name used as filename prefix.
    cutoff : date
        Only keep meetings with date >= cutoff (when date is known).
    fragment_workers : int
        Concurrent fragment download workers for segmented streams.
    keep_unknown_date : bool
        If True: keep files whose date cannot be determined (named "0000-00-00_...").
        If False: skip/delete entries with unknown dates.
    include_title_regex : Optional[str]
        If provided, yt-dlp will only download entries whose title matches this regex.
    exclude_title_regex : Optional[str]
        If provided, yt-dlp will skip entries whose title matches this regex.
    override_title : Optional[str]
        If provided, use this title for output naming (iframe HTML label source).
    override_meeting_date : Optional[date]
        If provided, use this date for output naming/filtering (iframe HTML row source).

    Outputs
    -------
    (bool, str)
        Success flag and a status message.

    Effects
    -------
    - Network downloads via yt-dlp
    - Disk writes (WAV output files)
    - ffmpeg execution (via yt-dlp postprocessor)
    """
    district_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = district_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = build_ydl_opts(
        tmp_dir=tmp_dir,
        cutoff=cutoff,
        fragment_workers=fragment_workers,
        quiet=True,
        keep_unknown_date=keep_unknown_date,
        include_title_regex=include_title_regex,
        exclude_title_regex=exclude_title_regex,
        ffmpeg_location=ffmpeg_location,
        status_cb=status_cb,
    )

    if status_cb:
        status_cb(f"starting yt-dlp: {url}")

    try:
        info = None
        for attempt in range(1, MAX_TRANSIENT_ATTEMPTS + 1):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                break
            except Exception as ex:
                err = str(ex)
                should_retry = is_transient_throttle_error(err) and attempt < MAX_TRANSIENT_ATTEMPTS
                if not should_retry:
                    return False, f"yt-dlp failed: {ex}"

                sleep_s = min(
                    BACKOFF_MAX_SECONDS,
                    BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0.0, BACKOFF_JITTER_SECONDS),
                )
                if status_cb:
                    status_cb(f"transient block detected on attempt {attempt}/{MAX_TRANSIENT_ATTEMPTS}; retrying")
                    status_cb(f"backoff sleep: {sleep_s:.1f}s")
                time.sleep(sleep_s)
    except Exception as ex:
        return False, f"yt-dlp failed: {ex}"

    if info is None:
        return False, f"yt-dlp failed: no info extracted for {url}"

    # Map video id -> entry metadata (to rename <id>.wav -> YYYY-MM-DD_title.wav)
    id_to_entry: Dict[str, Dict] = {}
    for e in iter_entries(info):
        vid = (e.get("id") or "").strip()
        if vid:
            id_to_entry[vid] = e

    wav_files = list(tmp_dir.glob("*.wav"))
    if not wav_files:
        return False, f"No WAV files produced for: {url}"

    moved = 0
    skipped = 0
    include_pat = re.compile(include_title_regex) if include_title_regex else None
    exclude_pat = re.compile(exclude_title_regex) if exclude_title_regex else None

    for wav_path in wav_files:
        vid = wav_path.stem
        entry = id_to_entry.get(vid, {})

        title = (override_title or "").strip() or (entry.get("title") or "").strip()
        if not title or UUID_TITLE_RE.fullmatch(title):
            title = "Board Meeting"
        dt = override_meeting_date if override_meeting_date is not None else entry_meeting_date(entry, fallback_text=url)

        # Enforce title filters again as a final safety net.
        if include_pat and not include_pat.search(title):
            wav_path.unlink(missing_ok=True)
            skipped += 1
            continue
        if exclude_pat and exclude_pat.search(title):
            wav_path.unlink(missing_ok=True)
            skipped += 1
            continue

        # Apply cutoff if we have a date.
        if dt is not None and dt < cutoff:
            wav_path.unlink(missing_ok=True)
            skipped += 1
            continue

        # If date unknown, either keep (0000-00-00) or skip depending on config.
        if dt is None and not keep_unknown_date:
            wav_path.unlink(missing_ok=True)
            skipped += 1
            continue

        date_str = dt.strftime("%Y-%m-%d") if dt else "0000-00-00"
        district_slug = safe_filename(district_name or district_dir.name)
        title_slug = safe_filename(title)
        final_name = f"{district_slug}-{date_str}-{title_slug}.wav"
        final_path = district_dir / final_name

        # Avoid overwriting; append _2, _3, ...
        if final_path.exists():
            k = 2
            while True:
                cand = district_dir / f"{final_path.stem}_{k}.wav"
                if not cand.exists():
                    final_path = cand
                    break
                k += 1

        wav_path.replace(final_path)
        moved += 1

    # Cleanup temp dir if empty
    try:
        if not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except Exception:
        pass

    return True, f"downloaded={moved}, skipped={skipped} from {url}"
