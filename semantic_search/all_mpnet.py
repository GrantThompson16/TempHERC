"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""

from sentence_transformers import SentenceTransformer, util
import re
import nltk
from nltk.tokenize import sent_tokenize
from TempHERC.semantic_search.test_transcript import raw_text

# Download the sentence tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

def preprocess_transcript(text):
    # 1. Remove timestamps like [00:00:01]
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
    
    # 2. Remove bracketed headers like [Call to Order]
    text = re.sub(r'\[.*?\]', '', text)
    
    # 3. Clean up extra whitespace/newlines
    text = " ".join(text.split())

    # 4. Split into actual sentences
    sentences = sent_tokenize(text)
    
    # Optional: Group sentences into small chunks (e.g., 2 sentences per chunk)
    # This gives the semantic search more "context" to work with.
    chunk_size = 3 
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    
    return chunks

#applying the preprocessing function to the raw text
corpus = preprocess_transcript(raw_text)

# Check the first few entries
for i, chunk in enumerate(corpus[:3]):
    print(f"Chunk {i}: {chunk}\n")

embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") #might change the model


# Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = [
    "What research was used for this policy?",
    "What university are the researchers from?",
    "What data does the study use?",
    "What did the study find?",
    "The findings are according to which study?",
    "Where is the evidence from?",
    "research",
    "Houston",
    "data",
    "findings",
    "evidence",
    "according to",
    "researchers",
    "university",
]

#embedding all queries at once
query_embedding = embedder.encode_query(queries, convert_to_tensor=True)
all_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)

for i, query in enumerate(queries):
    hits = all_hits[i]      #Get the hits for the i-th query
    print(f"Query: {query}")
    for hit in hits:
        print(f"Score: {hit['score']:.4f} | Text: {corpus[hit['corpus_id']]}")
    