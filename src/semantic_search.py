# src/semantic_search.py

import os
import glob
import json
import argparse
from pathlib import Path

import numpy as np

try:
    import faiss  # faiss-cpu
except ImportError as e:
    raise ImportError(
        "FAISS is not installed. Run: pip install faiss-cpu"
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is not installed. Run: pip install sentence-transformers"
    ) from e


# ✅ SBERT checkpoint (fast + good baseline)
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_words: int = 300, overlap_words: int = 50) -> list[str]:
    """
    Split long transcript text into overlapping word chunks.
    Overlap helps keep context across boundaries.
    """
    words = text.split()
    chunks = []
    i = 0

    step = max(1, chunk_words - overlap_words)
    while i < len(words):
        chunk = words[i : i + chunk_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += step

    return chunks


def load_transcripts(input_dir: str) -> list[dict]:
    """
    Load every .txt file in input_dir and return a list of:
    {source_file, full_text}
    """
    txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    if not txt_files:
        raise ValueError(f"No .txt files found in: {input_dir}")

    docs = []
    for fp in txt_files:
        text = Path(fp).read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append({"source_file": os.path.basename(fp), "full_text": text})

    if not docs:
        raise ValueError(f"Found .txt files but they were empty in: {input_dir}")

    return docs


def build_index(
    input_dir: str,
    out_dir: str,
    model_name: str = DEFAULT_MODEL,
    chunk_words: int = 300,
    overlap_words: int = 50,
):
    """
    Build FAISS index from transcript chunks.
    Saves:
      - out_dir/index.faiss
      - out_dir/records.json   (metadata + chunk text)
      - out_dir/meta.json      (model + chunking settings)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading transcripts from: {input_dir}")
    docs = load_transcripts(input_dir)

    # 1) Chunk all transcripts
    records = []
    for d in docs:
        chunks = chunk_text(d["full_text"], chunk_words=chunk_words, overlap_words=overlap_words)
        for ci, ch in enumerate(chunks):
            records.append(
                {
                    "source_file": d["source_file"],
                    "chunk_id": ci,
                    "text": ch,
                }
            )

    if not records:
        raise ValueError("No chunks were created. Check input files and chunk settings.")

    print(f"Total chunks: {len(records)}")
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # 2) Embed chunks (normalize_embeddings=True makes cosine similarity work nicely)
    texts = [r["text"] for r in records]
    print("Embedding chunks (this can take a bit on first run)...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    # 3) Build FAISS index for fast nearest-neighbor search
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (works like cosine because we normalized)
    index.add(embeddings)

    # 4) Save everything
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    Path(os.path.join(out_dir, "records.json")).write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    Path(os.path.join(out_dir, "meta.json")).write_text(
        json.dumps(
            {
                "model_name": model_name,
                "chunk_words": chunk_words,
                "overlap_words": overlap_words,
                "num_chunks": len(records),
                "embedding_dim": dim,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n✅ Index built successfully!")
    print(f"Saved to: {out_dir}/index.faiss, records.json, meta.json")


def search_index(out_dir: str, query: str, k: int = 5, model_name: str = DEFAULT_MODEL):
    """
    Search an existing FAISS index with a natural-language query.
    Prints top-k best matching chunks with file + chunk + snippet.
    """
    index_path = os.path.join(out_dir, "index.faiss")
    records_path = os.path.join(out_dir, "records.json")

    if not os.path.exists(index_path) or not os.path.exists(records_path):
        raise ValueError(
            f"Missing index files in {out_dir}. Run build first:\n"
            f"python src/semantic_search.py build --input_dir data --out_dir {out_dir}"
        )

    index = faiss.read_index(index_path)
    records = json.loads(Path(records_path).read_text(encoding="utf-8"))

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")

    scores, ids = index.search(q_emb, k)

    print(f"\nQuery: {query}\nTop {k} matches:")
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        r = records[int(idx)]
        snippet = r["text"][:700].replace("\n", " ")
        if len(r["text"]) > 700:
            snippet += "..."

        print("\n" + "-" * 60)
        print(f"{rank}) Score: {float(score):.3f}")
        print(f"File: {r['source_file']} | Chunk: {r['chunk_id']}")
        print(f"Snippet: {snippet}")


def main():
    parser = argparse.ArgumentParser(description="SBERT + FAISS semantic search for transcript .txt files")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build FAISS index from transcripts")
    b.add_argument("--input_dir", required=True, help="Folder containing .txt transcripts")
    b.add_argument("--out_dir", default="semantic_index", help="Where to save the index")
    b.add_argument("--model", default=DEFAULT_MODEL, help="SBERT model checkpoint")
    b.add_argument("--chunk_words", type=int, default=300, help="Words per chunk")
    b.add_argument("--overlap_words", type=int, default=50, help="Overlap words between chunks")

    s = sub.add_parser("search", help="Search the built index")
    s.add_argument("--out_dir", default="semantic_index", help="Folder containing the saved index")
    s.add_argument("--model", default=DEFAULT_MODEL, help="SBERT model checkpoint (must match build for best results)")
    s.add_argument("--query", required=True, help="Search query, e.g. 'research used to justify decision'")
    s.add_argument("--k", type=int, default=5, help="How many results to return")

    args = parser.parse_args()

    if args.cmd == "build":
        build_index(
            input_dir=args.input_dir,
            out_dir=args.out_dir,
            model_name=args.model,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
        )
    elif args.cmd == "search":
        search_index(out_dir=args.out_dir, query=args.query, k=args.k, model_name=args.model)


if __name__ == "__main__":
    main()
