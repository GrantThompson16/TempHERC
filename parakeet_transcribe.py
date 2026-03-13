import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# NeMo / Parakeet
import nemo.collections.asr as nemo_asr

import textwrap

def fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

def wrap_paragraphs(text: str, width: int = 100) -> str:
    """
    Wrap text to a max line width while preserving paragraph breaks.
    """
    paras = [p.strip() for p in text.split("\n")]

    wrapped = []
    for p in paras:
        if not p:
            wrapped.append("")
        else:
            wrapped.append(textwrap.fill(p, width=width))
    return "\n".join(wrapped)

###############################################################################
# Overlap deduplication — suffix/prefix matching (replaces LCS approach)
###############################################################################

def _normalize_word(w: str) -> str:
    """Lowercase and strip punctuation for comparison purposes."""
    return w.lower().strip(".,!?;:\"'()-")


def find_suffix_prefix_overlap(
    tail: List[str],
    head: List[str],
    min_len: int = 2,
    max_check: int = 40,
) -> int:
    """
    Find the longest suffix of `tail` that matches a prefix of `head`
    (case-insensitive, punctuation-tolerant).

    Returns the number of words to trim from the START of `head` (i.e. the
    overlap length).  Returns 0 when no overlap of at least `min_len` is found.

    Why suffix-prefix instead of LCS
    ---------------------------------
    A 1-second chunk overlap means the same audio is transcribed twice, so the
    repeated words appear at the END of the previous transcript and the START of
    the next one — a contiguous suffix/prefix pattern.  LCS finds the longest
    common *subsequence* (non-contiguous), which can match scattered words far
    from the boundary and trim far too aggressively.
    """
    tail_norm = [_normalize_word(w) for w in tail]
    head_norm = [_normalize_word(w) for w in head]

    max_overlap = min(len(tail), len(head), max_check)
    for length in range(max_overlap, min_len - 1, -1):
        if tail_norm[-length:] == head_norm[:length]:
            return length
    return 0


def merge_all_chunks_global(
    chunk_texts: List[str],
    min_overlap: int = 2,
    max_check: int = 40,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Sequentially merge *all* chunk transcripts in order, removing the
    duplicated overlap words at each boundary via suffix-prefix matching.

    Returns
    -------
    merged_words : List[str]
        The single deduplicated word list for the whole audio.
    chunk_word_ranges : List[Tuple[int, int]]
        For each chunk i, (start_idx, end_idx) into merged_words indicating
        which words came from that chunk (used to assign words to time sections).
    """
    if not chunk_texts:
        return [], []

    merged_words: List[str] = []
    chunk_word_ranges: List[Tuple[int, int]] = []

    for i, text in enumerate(chunk_texts):
        new_words = text.split()
        if not new_words:
            chunk_word_ranges.append((len(merged_words), len(merged_words)))
            continue

        if i == 0:
            # First chunk — no overlap to remove
            start_idx = 0
            merged_words.extend(new_words)
        else:
            # Window the comparison to keep it fast
            tail = merged_words[-max_check:] if len(merged_words) > max_check else merged_words[:]
            head = new_words[:max_check]

            trim = find_suffix_prefix_overlap(tail, head, min_len=min_overlap, max_check=max_check)

            start_idx = len(merged_words)
            merged_words.extend(new_words[trim:])

        chunk_word_ranges.append((start_idx, len(merged_words)))

    return merged_words, chunk_word_ranges


def merge_transcripts_with_lcs(
    transcripts: List[str],
    min_overlap: int = 2,
    max_check: int = 40,
) -> str:
    """
    Public helper kept for the --no_sections code path.
    Merges a list of transcript strings using suffix-prefix overlap detection.
    """
    merged_words, _ = merge_all_chunks_global(
        transcripts, min_overlap=min_overlap, max_check=max_check
    )
    return " ".join(merged_words)


def build_sectioned_transcript(
    chunks: List["Chunk"],
    chunk_texts: List[str],
    section_s: int = 30,
    wrap_width: int = 100,
) -> str:
    """
    Build a readable transcript with [start–end] headers every section_s seconds.

    FIX: Previously, chunks were grouped by section *first* and LCS-merged only
    within each section bucket.  This meant overlapping words at section
    boundaries (e.g. a chunk ending at 35 s vs. one starting at 34 s) were
    never deduplicated, producing repeated words at every section seam.

    Correct approach:
      1. Merge ALL chunks globally (sequential suffix-prefix dedup) → one clean
         word list + a per-chunk word-range index.
      2. Assign each word to a section using its originating chunk's start time.
      3. Emit sections in order.
    """
    if not chunks or not chunk_texts:
        return ""

    # Step 1 — global merge across ALL chunks (no section boundaries in merge)
    merged_words, chunk_word_ranges = merge_all_chunks_global(chunk_texts)

    if not merged_words:
        return ""

    # Step 2 — assign each word to a section via its originating chunk's time
    word_section: List[int] = [0] * len(merged_words)
    for ch, (start_idx, end_idx) in zip(chunks, chunk_word_ranges):
        sec_idx = int(ch.start_s // section_s)
        for j in range(start_idx, end_idx):
            word_section[j] = sec_idx

    # Step 3 — bucket words by section and emit
    buckets: dict = {}
    for word, sec in zip(merged_words, word_section):
        buckets.setdefault(sec, []).append(word)

    lines: List[str] = []
    for sec_idx in sorted(buckets.keys()):
        start = sec_idx * section_s
        end = start + section_s
        header = f"[{fmt_ts(start)}–{fmt_ts(end)}]"
        text = wrap_paragraphs(" ".join(buckets[sec_idx]), width=wrap_width)
        lines.append(header)
        lines.append(text)
        lines.append("")

    return "\n".join(lines).rstrip()


###############################################################################
# Audio preprocessing via ffmpeg
###############################################################################

def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg and make sure it's available as `ffmpeg`."
        )
    return ffmpeg

def ffprobe_duration_seconds(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0

    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        return float(out)
    except Exception:
        return 0.0

def preprocess_audio_to_wav(
    input_path: Path,
    output_wav: Path,
    sample_rate: int = 16000,
    mono: bool = True,
    pcm: str = "pcm_s16le",
    loudnorm: bool = False,
    highpass_hz: Optional[int] = 80,
    lowpass_hz: Optional[int] = 7500,
) -> None:
    """
    Converts arbitrary audio -> mono 16kHz PCM WAV, optionally applying:
      - loudness normalization (EBU R128 loudnorm)
      - highpass / lowpass filters
    """
    ffmpeg = require_ffmpeg()

    filters = []
    if highpass_hz is not None:
        filters.append(f"highpass=f={int(highpass_hz)}")
    if lowpass_hz is not None:
        filters.append(f"lowpass=f={int(lowpass_hz)}")
    if loudnorm:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    filter_chain = ",".join(filters) if filters else None

    cmd = [ffmpeg, "-y", "-i", str(input_path)]
    if filter_chain:
        cmd += ["-af", filter_chain]

    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(sample_rate)]
    cmd += ["-c:a", pcm]
    cmd += ["-f", "wav", str(output_wav)]

    subprocess.check_call(cmd)

def extract_wav_segment(
    input_wav: Path,
    start_s: float,
    duration_s: float,
    output_wav: Path,
) -> None:
    """Extract a WAV segment (mono 16kHz PCM) using ffmpeg."""
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration_s:.3f}",
        "-i", str(input_wav),
        "-c:a", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav),
    ]
    subprocess.check_call(cmd)

###############################################################################
# Chunk planning: hour blocks, then 30–40s chunks with 1s overlap
###############################################################################

@dataclass
class Chunk:
    start_s: float
    dur_s: float
    path: Path

def plan_chunks(
    wav_path: Path,
    chunk_s: float = 35.0,
    overlap_s: float = 1.0,
    hour_block_s: float = 3600.0,
    tmp_dir: Path = Path("."),
) -> List[List[Chunk]]:
    """
    Returns list of blocks; each block is a list of chunks.
    If duration > 1 hour, split into hour blocks.
    """
    total = ffprobe_duration_seconds(wav_path)
    if total <= 0:
        # If ffprobe fails, assume one block; extraction still works until EOF.
        total = hour_block_s

    num_blocks = max(1, math.ceil(total / hour_block_s))
    blocks: List[List[Chunk]] = []

    for b in range(num_blocks):
        block_start = b * hour_block_s
        block_end = min((b + 1) * hour_block_s, total)
        block_len = max(0.0, block_end - block_start)

        chunks: List[Chunk] = []
        t = 0.0
        idx = 0
        step = max(0.01, chunk_s - overlap_s)

        while t < block_len:
            dur = min(chunk_s, block_len - t)
            out = tmp_dir / f"chunk_b{b:03d}_{idx:05d}.wav"
            chunks.append(Chunk(start_s=block_start + t, dur_s=dur, path=out))
            idx += 1
            t += step

        blocks.append(chunks)

    return blocks

###############################################################################
# Transcription
###############################################################################

def transcribe_chunks(
    model,
    chunks: List[Chunk],
    batch_size: int = 8,
    timestamps: bool = False,
) -> Tuple[List[str], Optional[List[dict]]]:
    texts: List[str] = []
    stamps: List[dict] = []

    paths = [str(c.path) for c in chunks]
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        out = model.transcribe(batch_paths, timestamps=timestamps)
        for hyp in out:
            texts.append(getattr(hyp, "text", "") if hyp is not None else "")
            if timestamps:
                stamps.append(getattr(hyp, "timestamp", {}) if hyp is not None else {})

    return texts, (stamps if timestamps else None)

def is_audio_file(p: Path) -> bool:
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm"}
    return p.suffix.lower() in exts

def main():
    ap = argparse.ArgumentParser(
        description="Parakeet long-form transcription with preprocessing + overlap chunking + suffix-prefix merge."
    )
    ap.add_argument("--input", required=True, help="Path to an audio file OR a directory of audio files.")
    ap.add_argument("--output_dir", required=True, help="Directory to write transcripts.")
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3", help="HF model name for NeMo ASRModel.")
    ap.add_argument("--timestamps", action="store_true", help="Also emit timestamps JSON (may be larger/slower).")

    # preprocessing toggles
    ap.add_argument("--loudnorm", action="store_true", help="Apply EBU R128 loudness normalization.")
    ap.add_argument("--no_highpass", action="store_true", help="Disable high-pass filter.")
    ap.add_argument("--no_lowpass", action="store_true", help="Disable low-pass filter.")
    ap.add_argument("--highpass_hz", type=int, default=80, help="High-pass cutoff Hz (default 80).")
    ap.add_argument("--lowpass_hz", type=int, default=7500, help="Low-pass cutoff Hz (default 7500).")

    # chunking params
    ap.add_argument("--chunk_s", type=float, default=35.0, help="Chunk length seconds (default 35).")
    ap.add_argument("--overlap_s", type=float, default=1.0, help="Overlap seconds (default 1).")
    ap.add_argument("--hour_block_s", type=float, default=3600.0, help="Hour block seconds (default 3600).")
    ap.add_argument("--batch_size", type=int, default=8, help="Transcribe batch size (paths per call).")

    ap.add_argument("--wrap_width", type=int, default=100, help="Max characters per line in output transcript.")
    ap.add_argument("--section_s", type=int, default=30, help="Section size in seconds for timestamp headers.")
    ap.add_argument("--no_sections", action="store_true", help="Disable [MM:SS–MM:SS] section headers.")

    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect input audio files
    if in_path.is_dir():
        audio_files = sorted([p for p in in_path.rglob("*") if p.is_file() and is_audio_file(p)])
    elif in_path.is_file():
        audio_files = [in_path]
    else:
        raise FileNotFoundError(f"Input not found: {in_path}")

    if not audio_files:
        print("No audio files found.")
        sys.exit(0)

    print(f"Loading model: {args.model}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)

    with tempfile.TemporaryDirectory(prefix="parakeet_tmp_") as tmp:
        tmp_dir = Path(tmp)

        for src in audio_files:
            print(f"\n=== Processing: {src.name} ===")

            standardized = tmp_dir / f"{src.stem}__std.wav"
            preprocess_audio_to_wav(
                input_path=src,
                output_wav=standardized,
                sample_rate=16000,
                mono=True,
                pcm="pcm_s16le",
                loudnorm=args.loudnorm,
                highpass_hz=None if args.no_highpass else args.highpass_hz,
                lowpass_hz=None if args.no_lowpass else args.lowpass_hz,
            )

            blocks = plan_chunks(
                wav_path=standardized,
                chunk_s=args.chunk_s,
                overlap_s=args.overlap_s,
                hour_block_s=args.hour_block_s,
                tmp_dir=tmp_dir,
            )

            all_block_texts: List[str] = []
            all_block_meta: List[dict] = []

            for b_idx, chunks in enumerate(blocks):
                for ch in chunks:
                    extract_wav_segment(standardized, ch.start_s, ch.dur_s, ch.path)

                chunk_texts, chunk_stamps = transcribe_chunks(
                    asr_model, chunks, batch_size=args.batch_size, timestamps=args.timestamps
                )

                # Build readable transcript for this block
                if args.no_sections:
                    merged_text = merge_transcripts_with_lcs(chunk_texts)
                    merged_text = wrap_paragraphs(merged_text, width=args.wrap_width)
                else:
                    merged_text = build_sectioned_transcript(
                        chunks=chunks,
                        chunk_texts=chunk_texts,
                        section_s=args.section_s,
                        wrap_width=args.wrap_width,
                    )

                all_block_texts.append(merged_text)

                if args.timestamps:
                    all_block_meta.append({
                        "block_index": b_idx,
                        "block_start_s": chunks[0].start_s if chunks else 0.0,
                        "chunking": {
                            "chunk_s": args.chunk_s,
                            "overlap_s": args.overlap_s,
                            "num_chunks": len(chunks),
                        },
                        "chunks": [
                            {
                                "chunk_index": i,
                                "start_s": chunks[i].start_s,
                                "dur_s": chunks[i].dur_s,
                                "text": chunk_texts[i],
                                "timestamps": (chunk_stamps[i] if chunk_stamps else {}),
                            }
                            for i in range(len(chunks))
                        ]
                    })

            final_text = "\n\n".join(all_block_texts).strip()
            base = out_dir / src.stem
            txt_path = Path(str(base) + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(final_text + "\n")
            print(f"Wrote transcript: {txt_path}")

            if args.timestamps:
                json_path = Path(str(base) + ".timestamps.json")
                payload = {
                    "source_file": str(src),
                    "model": args.model,
                    "preprocess": {
                        "mono": True,
                        "sample_rate_hz": 16000,
                        "pcm": "pcm_s16le",
                        "loudnorm": bool(args.loudnorm),
                        "highpass_hz": (None if args.no_highpass else args.highpass_hz),
                        "lowpass_hz": (None if args.no_lowpass else args.lowpass_hz),
                    },
                    "longform_strategy": {
                        "hour_block_s": args.hour_block_s,
                        "chunk_s": args.chunk_s,
                        "overlap_s": args.overlap_s,
                        "merge": "suffix-prefix word overlap",
                    },
                    "blocks": all_block_meta,
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                print(f"Wrote timestamps: {json_path}")

if __name__ == "__main__":
    main()