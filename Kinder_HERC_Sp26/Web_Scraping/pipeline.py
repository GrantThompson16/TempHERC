"""
Swagit-only pipeline orchestration.

Flow:
1) If source URL is a direct Swagit video URL, try it directly (including normalized
   watch -> download variant).
2) Otherwise treat source URL as a webpage, scrape Swagit video links, then try each.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .models import Source
from .html_scrape import (
    fetch_html,
    fetch_html_with_final_url,
    is_swagit_video_url,
    normalize_swagit,
    scrape_candidate_links,
    scrape_swagit_paginated,
    scrape_swagit_table_rows,
    scrape_swagit_tabs_single_page,
)
from .ytdlp_runner import download_source_to_wav


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """
    Configuration for pipeline behavior.

    Parameters
    ----------
    out_root : Path
        Root directory for outputs, ex: "School Board Meetings"
    cutoff : date
        Only keep meetings with date >= cutoff (when date is known).
    fragment_workers : int
        Fragment concurrency for segmented streams
    max_candidates : int
        Maximum number of candidate links to attempt from HTML scraping
        Lower = faster, higher = more thorough
    keep_unknown_date : bool
        If True, keep downloads even when date cannot be determined (named 0000-00-00_...).
        If False, skip unknown-date items.
    """
    out_root: Path
    cutoff: date
    fragment_workers: int = 8
    max_candidates: int = 60
    keep_unknown_date: bool = True
    include_title_regex: str | None = None
    exclude_title_regex: str | None = None
    include_anchor_label_regex: str | None = r"\bboard\W*meetings?\b"
    min_date: date | None = None
    ffmpeg_location: Optional[Path] = None


def scrape_swagit_candidates_with_iframes(
    page_url: str,
    html: str,
    max_links: int,
    fetcher: Callable[[str], str] = fetch_html,
    status_cb: Optional[Callable[[str], None]] = None,
    include_anchor_label_regex: str | None = r"\bboard\W*meetings?\b",
    min_date: date | None = None,
    iframe_fetcher: Callable[[str], Tuple[str, str]] = fetch_html_with_final_url,
) -> Tuple[List[str], bool, Dict[str, Tuple[str, date]]]:
    """
    Scrape Swagit candidates from the page, with one-level Swagit iframe fallback.

    Returns
    -------
    Tuple[List[str], bool, Dict[str, Tuple[str, date]]]
        (candidate_urls, came_from_iframe_label_filter, iframe_metadata_by_candidate_url)
    """
    if max_links <= 0:
        return [], False, {}
    _ = fetcher  # Backward-compatible arg retained; iframe fetch uses iframe_fetcher.

    candidates = scrape_candidate_links(page_url=page_url, html=html, max_links=max_links)
    if candidates:
        return candidates, False, {}

    if status_cb:
        status_cb("no direct swagit candidates on page; checking swagit iframes")

    label_pat = re.compile(include_anchor_label_regex, re.IGNORECASE) if include_anchor_label_regex else None

    soup = BeautifulSoup(html, "lxml")
    seen_iframes: set[str] = set()
    seen_candidates: set[str] = set()
    all_iframe_candidates: List[str] = []
    iframe_metadata: Dict[str, Tuple[str, date]] = {}
    for iframe in soup.select("iframe"):
        raw_src = (
            (iframe.get("src") or "").strip()
            or (iframe.get("data-src") or "").strip()
            or (iframe.get("data-original-src") or "").strip()
        )
        if not raw_src:
            continue

        iframe_src = urljoin(page_url, raw_src)
        iframe_host = (urlparse(iframe_src).netloc or "").lower()
        if not iframe_host or "swagit.com" not in iframe_host:
            continue
        if iframe_src in seen_iframes:
            continue
        seen_iframes.add(iframe_src)

        if status_cb:
            status_cb(f"scraping swagit iframe: {iframe_src}")

        try:
            final_iframe_url, iframe_html = iframe_fetcher(iframe_src)
        except Exception:
            continue

        iframe_soup = BeautifulSoup(iframe_html, "lxml")
        has_next_pagination = iframe_soup.select_one('a[rel="next"][href]') is not None
        has_tab_panes = iframe_soup.select_one('a[data-toggle="tab"][href^="#"]') is not None

        remaining = max_links - len(all_iframe_candidates)
        if remaining <= 0:
            break

        if has_next_pagination:
            if status_cb:
                status_cb("iframe mode: rel=next pagination")
            table_candidates = scrape_swagit_paginated(
                page_url=final_iframe_url,
                fetch_html_with_final_url=iframe_fetcher,
                max_links=remaining,
            )
        elif has_tab_panes:
            if status_cb:
                status_cb("iframe mode: bootstrap tabs")
            table_candidates = scrape_swagit_tabs_single_page(
                page_url=final_iframe_url,
                html=iframe_html,
                max_links=remaining,
            )
        else:
            if status_cb:
                status_cb("iframe mode: single page rows")
            table_candidates = scrape_swagit_table_rows(
                page_url=final_iframe_url,
                html=iframe_html,
                max_links=remaining,
            )

        for candidate_url, label_text, meeting_date in table_candidates:
            if label_pat and not label_pat.search(label_text or ""):
                continue
            if min_date is not None and meeting_date < min_date:
                continue
            if candidate_url in seen_candidates:
                continue
            seen_candidates.add(candidate_url)
            all_iframe_candidates.append(candidate_url)
            iframe_metadata[candidate_url] = (label_text, meeting_date)
            if len(all_iframe_candidates) >= max_links:
                return all_iframe_candidates[:max_links], True, iframe_metadata

    if all_iframe_candidates:
        return all_iframe_candidates[:max_links], True, iframe_metadata
    return [], False, {}


def process_source(
    source: Source,
    cfg: PipelineConfig,
    status_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """
    Process one Source end-to-end.

    Inputs
    ------
    source : Source
        District + URL to process.
    cfg : PipelineConfig
        Pipeline settings.

    Outputs
    -------
    (bool, str)
        success flag and a human-readable message.

    Effects
    -------
    - Network I/O: yt-dlp downloads, optional HTML fetching
    - Disk I/O: WAV file writes + renames
    """
    district_dir = cfg.out_root / source.district
    district_dir.mkdir(parents=True, exist_ok=True)
    if status_cb:
        status_cb("starting source")

    raw_url = source.url.strip()
    normalized_url = normalize_swagit(raw_url)
    source_is_swagit = is_swagit_video_url(raw_url) or is_swagit_video_url(normalized_url)
    if status_cb:
        if source_is_swagit:
            status_cb("source type: swagit")
        else:
            status_cb("source type: webpage (scrape for swagit links)")
        status_cb(f"effective fragment workers: {max(1, cfg.fragment_workers)}")
        if normalized_url != raw_url:
            status_cb(f"normalized swagit watch -> download: {raw_url} -> {normalized_url}")
    # 1 Fast path for Swagit URLs only
    if source_is_swagit:
        if status_cb:
            status_cb("trying direct swagit URL")
        to_try: List[str] = []
        if is_swagit_video_url(raw_url):
            to_try.append(raw_url)
        if normalized_url != raw_url and is_swagit_video_url(normalized_url):
            to_try.append(normalized_url)

        direct_msg: Optional[str] = None
        for u in to_try:
            ok, msg = download_source_to_wav(
                url=u,
                district_dir=district_dir,
                district_name=source.district,
                cutoff=cfg.cutoff,
                fragment_workers=max(1, cfg.fragment_workers),
                keep_unknown_date=cfg.keep_unknown_date,
                # Direct Swagit URLs are explicit user intent; do not gate on title regex.
                include_title_regex=None,
                exclude_title_regex=None,
                ffmpeg_location=cfg.ffmpeg_location,
                status_cb=status_cb,
            )
            if ok:
                return True, f"[direct] {msg}"
            direct_msg = msg

        if direct_msg is None:
            direct_msg = "no direct swagit URL variants to try"
        return False, f"[direct failed] {direct_msg}"
    else:
        direct_msg = "not a direct swagit video URL"

    # 2 Webpage mode only: scrape HTML candidates
    if status_cb:
        status_cb("scraping HTML for swagit candidate links")
    try:
        html = fetch_html(source.url)
    except Exception as ex:
        return False, f"[direct failed] {direct_msg} | [html fetch failed] {ex}"

    candidates, iframe_label_filtered, iframe_metadata = scrape_swagit_candidates_with_iframes(
        page_url=source.url,
        html=html,
        max_links=cfg.max_candidates,
        include_anchor_label_regex=cfg.include_anchor_label_regex,
        min_date=cfg.min_date,
        status_cb=status_cb,
    )

    if not candidates:
        return False, f"[direct failed] {direct_msg} | [no candidates found from HTML]"

    # Try all candidate links up to max_candidates.
    last_err: Optional[str] = None
    successes = 0
    attempted = 0
    for cand in candidates[: cfg.max_candidates]:
        attempted += 1
        if status_cb:
            status_cb(f"trying candidate {cand}")
        override_title = None
        override_meeting_date = None
        if iframe_label_filtered:
            meta = iframe_metadata.get(cand)
            if meta:
                override_title, override_meeting_date = meta
        ok2, msg2 = download_source_to_wav(
            url=cand,
            district_dir=district_dir,
            district_name=source.district,
            cutoff=cfg.cutoff,
            fragment_workers=max(1, cfg.fragment_workers),
            keep_unknown_date=cfg.keep_unknown_date,
            include_title_regex=None if iframe_label_filtered else cfg.include_title_regex,
            exclude_title_regex=None if iframe_label_filtered else cfg.exclude_title_regex,
            override_title=override_title,
            override_meeting_date=override_meeting_date,
            ffmpeg_location=cfg.ffmpeg_location,
            status_cb=status_cb,
        )
        if ok2:
            successes += 1
            if status_cb:
                status_cb(f"candidate success: {cand}")
            continue
        last_err = msg2

    if successes > 0:
        return True, f"[candidates] succeeded={successes}, attempted={attempted}"
    return False, f"[direct failed] {direct_msg} | [candidates tried={attempted}] last={last_err}"
