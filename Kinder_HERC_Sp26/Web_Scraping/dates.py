"""
Docstring for Kinder_HERC_Sp26.Web_Scraping.dates

Date parsing utilities for the school baord meeting scraper.

Purpose: Need to filer for meetings after August 2024 and name output files as:
    YYY-MM-DD_title.wav
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from .config import DATE_PATTERNS, MONTH_MAP

def parse_date_from_text(text: str) -> Optional[date]:
    """
    Attempt to parse a calendar date from an arbitrary text string.

    Inputs
    ------
    text : str (the title of the video)

    Outputs
    -------
    Optional[date]
        - Returns a `datetime.date` if a date is detected and valid.
        - Returns None if no supported date pattern is found.

    Parsing rules
    ------------------------
    Function tries the regex patterns in `config.DATE_PATTERNS` in order.
    The first successful match wins.

    Supported date formats (examples)
    ---------------------------------
    1) YYYY-MM-DD (also supports '_' or '/' separators):
        - 2024-09-12
        - 2024_09_12
        - 2024/09/12

    2) MM-DD-YYYY (also supports '/' separators):
        - 09-12-2024
        - 09/12/2024

    3) Month name formats:
        - Sep 12 2024
        - Sept 12, 2024
        - September 12 2024
    """
    t = (text or "").strip()
    if not t:
        return None

    for pat in DATE_PATTERNS:
        m = pat.search(t)
        if not m:
            continue

        gd = m.groupdict()

        try:
            # Month name format: mon + day + year
            if "mon" in gd and gd.get("mon"):
                month = MONTH_MAP[gd["mon"].lower()]
                day = int(gd["d"])
                year = int(gd["y"])
                return date(year, month, day)

            # Numeric formats: year/month/day or month/day/year
            year = int(gd["y"])
            month = int(gd["m"])
            day = int(gd["d"])
            return date(year, month, day)

        except Exception:
            # match is malformed
            continue

    return None