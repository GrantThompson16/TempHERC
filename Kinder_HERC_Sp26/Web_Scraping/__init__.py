# Kinder_HERC_Sp26/Web_Scraping/__init__.py
"""
Kinder_HERC_Sp26.Web_Scraping

Package for downloading school board meeting videos and extracting audio files.

Public entry points
-------------------
- CLI:
    python -m Kinder_HERC_Sp26.Web_Scraping.cli --source "District|URL" --cutoff 2024-09-01

Core modules
------------
- config.py        : constants + regex patterns
- models.py        : Source dataclass
- dates.py         : date parsing utilities
- html_scrape.py   : HTML fetch + candidate URL extraction (fallback)
- ytdlp_runner.py  : yt-dlp + ffmpeg download/extract/rename
- pipeline.py      : orchestration per Source
- cli.py           : command-line runner

This package is designed to be:
- Swagit-focused (direct Swagit URLs + webpage Swagit discovery)
- Extendable (add district/vendor-specific extractors later)
"""

__all__ = []
