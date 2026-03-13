"""
Docstring for Kinder_HERC_Sp26.Web_Scraping

HTML fetching and link extraction for the scraper.

fetch_html(url): HTTP GET --> HTML string
scrape_candidate_links(page_url, html, cap): extraction of likely video urls
"""



from __future__ import annotations
import re
from datetime import date, datetime
from typing import Callable, List, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import USER_AGENT

URL_IN_TEXT_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
SWAGIT_WATCH_RE = re.compile(
    r"^/videos/(?P<vid>\d+)/?$",
    re.IGNORECASE,
)
SWAGIT_VIDEO_RE = re.compile(
    r"^/videos/\d+(/download)?/?$",
    re.IGNORECASE,
)
SWAGIT_TABLE_HREF_RE = re.compile(
    r"^/videos/\d+/?$",
    re.IGNORECASE,
)
SWAGIT_ROW_DATE_LIKE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)


def _parse_table_row_date(text: str) -> date | None:
    """
    Parse date text commonly used in Swagit table rows, ex: "Jan 27, 2026".
    """
    if not text:
        return None

    cleaned = " ".join(text.replace("\xa0", " ").split())
    cleaned = cleaned.replace(" ,", ",")
    for fmt in ("%b %d, %Y", "%b %d %Y", "%B %d, %Y", "%B %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None
def fetch_html(url: str, timeout_s: int = 60) -> str:
    """
    Download HTML from a URL using a browser-like User-Agent.

    Inputs
    ------
    url : str
        Page URL to fetch.
    timeout_s : int
        Requests timeout in seconds.

    Outputs
    -------
    str
        Raw HTML response text.

    Effects
    -------
    - Network I/O: performs an HTTP GET request.
    - Raises requests exceptions for non-2xx responses/timeouts.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    return r.text

def fetch_html_with_final_url(url: str, timeout_s: int = 60) -> tuple[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=timeout_s, allow_redirects=True)
    r.raise_for_status()
    return r.url, r.text


def _normalize_url(base_url: str, href: str) -> str:
    """
    Resolve a possibly-relative href into an absolute URL and strip fragments (#...).

    Inputs
    ------
    base_url : str
        Page URL where href was found.
    href : str
        Raw href attribute value.

    Outputs
    -------
    str
        Absolute URL without fragment component.

    Effects
    -------
    None.
    """
    abs_url = urljoin(base_url, (href or "").strip())

    # Strip fragment: lowers duplicates and doesn't affect downloads for our use-case.
    parsed = urlparse(abs_url)
    if parsed.fragment:
        abs_url = abs_url.split("#", 1)[0]

    abs_url = normalize_swagit(abs_url)
    return abs_url


def normalize_swagit(url: str) -> str:
    """
    Normalize Swagit watch URLs to their direct downloadable endpoint.

    Example:
      https://district.new.swagit.com/videos/12345
      -> https://district.new.swagit.com/videos/12345/download
    """
    raw = (url or "").strip()
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    if not host.endswith(".swagit.com"):
        return raw

    m = SWAGIT_WATCH_RE.match(path)
    if not m:
        return raw

    if path.lower().endswith("/download"):
        return raw.split("#", 1)[0]

    return f"{parsed.scheme}://{parsed.netloc}{path.rstrip('/')}/download"


def is_swagit_video_url(url: str) -> bool:
    """Return True for Swagit video/watch/download URLs."""
    raw = (url or "").strip()
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    if not host.endswith(".swagit.com"):
        return False
    return bool(SWAGIT_VIDEO_RE.match(parsed.path or ""))


def _is_candidate_url(u: str) -> bool:
    """
    Swagit-only candidate filter.
    """
    return bool(u and is_swagit_video_url(u))


def scrape_candidate_links(page_url: str, html: str, max_links: int = 100) -> List[str]:
    """
    Extract up to `max_links` likely meeting video URLs from HTML.

    Approach:
      1) Scan <a href> links.
      2) Scan <iframe src> embeds.
      3) Scan <script> text for embedded URLs.
      4) Keep only Swagit video URLs and stop once we have max_links.

    Returns
    -------
    List[str]
        A deduplicated list of URLs (best effort).
    """
    if max_links <= 0:
        return []

    soup = BeautifulSoup(html, "lxml")

    seen: Set[str] = set()
    out: List[str] = []

    def add_url(u: str) -> None:
        """Add URL `u` into the right bucket if it passes filters and is new."""
        if not u or u in seen:
            return
        if not _is_candidate_url(u):
            return

        seen.add(u)
        out.append(u)

    # 1: Anchor links
    for a in soup.select("a[href]"):
        add_url(_normalize_url(page_url, a.get("href")))
        if len(out) >= max_links:
            break

    # 2: Iframe embeds
    if len(out) < max_links:
        for f in soup.select("iframe[src]"):
            add_url(_normalize_url(page_url, f.get("src")))
            if len(out) >= max_links:
                break

    # 3: Script text 
    if len(out) < max_links:
        for sc in soup.select("script"):
            txt = sc.string
            if not txt:
                continue
            for m in URL_IN_TEXT_RE.finditer(txt):
                add_url(_normalize_url(page_url, m.group(0)))
                if len(out) >= max_links:
                    break
            if len(out) >= max_links:
                break

    return out[:max_links]


def scrape_labeled_swagit_links(page_url: str, html: str, max_links: int = 100) -> List[Tuple[str, str]]:
    """
    Extract labeled Swagit video links from anchors in HTML.

    Returns
    -------
    List[Tuple[str, str]]
        List of (normalized_candidate_url, anchor_label_text).
    """
    if max_links <= 0:
        return []

    soup = BeautifulSoup(html, "lxml")

    seen: Set[str] = set()
    out: List[Tuple[str, str]] = []

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        candidate_url = _normalize_url(page_url, href)
        if not _is_candidate_url(candidate_url):
            continue
        if candidate_url in seen:
            continue

        label_text = " ".join(a.stripped_strings).strip()
        seen.add(candidate_url)
        out.append((candidate_url, label_text))
        if len(out) >= max_links:
            break

    return out[:max_links]


def scrape_swagit_table_candidates(page_url: str, html: str, max_links: int) -> List[Tuple[str, str, date]]:
    """
    Extract (candidate_url, label_text, meeting_date) from Swagit archive table rows.
    """
    return scrape_swagit_table_rows(page_url=page_url, html=html, max_links=max_links)


def scrape_swagit_table_rows(page_url: str, html: str, max_links: int) -> List[Tuple[str, str, date]]:
    """
    Extract (candidate_url, label_text, meeting_date) from Swagit table rows.
    """
    soup = BeautifulSoup(html, "lxml")
    return _scrape_swagit_table_rows_from_soup(
        soup=soup,
        page_url=page_url,
        max_links=max_links,
        seen=set(),
    )


def _scrape_swagit_table_rows_from_soup(
    soup: BeautifulSoup,
    page_url: str,
    max_links: int,
    seen: Set[str] | None = None,
) -> List[Tuple[str, str, date]]:
    if max_links <= 0:
        return []
    seen = seen if seen is not None else set()

    out: List[Tuple[str, str, date]] = []
    for row in soup.select("tr"):
        anchor = None
        for a in row.select("a[href]"):
            href = (a.get("href") or "").strip()
            if SWAGIT_TABLE_HREF_RE.match(href):
                anchor = a
                break
        if anchor is None:
            continue

        href = (anchor.get("href") or "").strip()
        abs_url = normalize_swagit(urljoin(page_url, href))
        if not is_swagit_video_url(abs_url):
            continue
        if abs_url in seen:
            continue

        meeting_date: date | None = None
        for td in row.select("td"):
            td_text = " ".join(td.stripped_strings).strip()
            if not td_text or not SWAGIT_ROW_DATE_LIKE_RE.search(td_text):
                continue
            meeting_date = _parse_table_row_date(td_text)
            if meeting_date is not None:
                break
        if meeting_date is None:
            continue

        label_text = " ".join(anchor.stripped_strings).strip()
        seen.add(abs_url)
        out.append((abs_url, label_text, meeting_date))
        if len(out) >= max_links:
            break
    return out[:max_links]


def scrape_swagit_paginated(
    page_url: str,
    fetch_html_with_final_url: Callable[[str], tuple[str, str]],
    max_links: int,
) -> List[Tuple[str, str, date]]:
    """
    Follow rel="next" pages and aggregate Swagit table rows.
    """
    if max_links <= 0:
        return []

    results: List[Tuple[str, str, date]] = []
    seen_urls: Set[str] = set()
    seen_pages: Set[str] = set()
    current_url: str | None = page_url

    while current_url and len(results) < max_links:
        if current_url in seen_pages:
            break
        seen_pages.add(current_url)

        final_url, html = fetch_html_with_final_url(current_url)
        soup = BeautifulSoup(html, "lxml")

        remaining = max_links - len(results)
        rows = _scrape_swagit_table_rows_from_soup(
            soup=soup,
            page_url=final_url,
            max_links=remaining,
            seen=seen_urls,
        )
        results.extend(rows)
        if len(results) >= max_links:
            break

        next_a = soup.select_one('a[rel="next"][href]')
        if next_a is None:
            break
        next_href = (next_a.get("href") or "").strip()
        if not next_href:
            break
        current_url = urljoin(final_url, next_href)

    return results[:max_links]


def scrape_swagit_tabs_single_page(page_url: str, html: str, max_links: int) -> List[Tuple[str, str, date]]:
    """
    Scrape Swagit table rows across Bootstrap tab panes in a single HTML page.
    """
    if max_links <= 0:
        return []

    soup = BeautifulSoup(html, "lxml")
    seen_urls: Set[str] = set()
    results: List[Tuple[str, str, date]] = []

    tab_anchors = soup.select('a[data-toggle="tab"][href^="#"]')
    if not tab_anchors:
        return _scrape_swagit_table_rows_from_soup(
            soup=soup,
            page_url=page_url,
            max_links=max_links,
            seen=seen_urls,
        )

    seen_panes: Set[str] = set()
    for anchor in tab_anchors:
        href = (anchor.get("href") or "").strip()
        pane_id = href.lstrip("#")
        if not pane_id or pane_id in seen_panes:
            continue
        seen_panes.add(pane_id)

        pane = soup.find(id=pane_id)
        if pane is None:
            continue

        remaining = max_links - len(results)
        if remaining <= 0:
            break
        pane_soup = BeautifulSoup(str(pane), "lxml")
        pane_rows = _scrape_swagit_table_rows_from_soup(
            soup=pane_soup,
            page_url=page_url,
            max_links=remaining,
            seen=seen_urls,
        )
        results.extend(pane_rows)

    return results[:max_links]
