from sentence_transformers import CrossEncoder, SentenceTransformer, util
import nltk
import re
from nltk.tokenize import sent_tokenize
from TempHERC.semantic_search.test_transcript import raw_text

# Download the sentence tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')


queries = [
    "Find specific mentions of academic studies, university researchers, or statistical evidence from reports used to support decisions.",
    "Find mentions of the Kinder Institute for Urban Research at Rice University.",
]
keywords = ["research", "university", "findings", "evidence", "Kinder Institute", "report", "HERC", "Rice", "impact"]


def preprocess_transcript(text):
    # 1. Remove timestamps like [00:00:01]
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
    
    # 2. Remove bracketed headers like [Call to Order]
    text = re.sub(r'\[.*?\]', '', text)
    
    # 3. Clean up extra whitespace/newlines
    text = " ".join(text.split())

    # 4. Split into actual sentences
    sentences = sent_tokenize(text)
    
    #  For chunking, will create a slider for overlap for better context per chunk
    chunks = []
    chunk_size = 3
    overlap = 1
    #slide = chunk_size - overlap
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i : i + chunk_size])
        chunks.append(chunk)
        
        # Stop if we've reached the end of the transcript
        if i + chunk_size >= len(sentences):
            break
            
    return chunks

#applying the preprocessing function to the raw text
corpus = preprocess_transcript(raw_text)

#embedding using mpnet still
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") #might change the model
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

#keep my normal bi-encoder for the queries
query_embedding = embedder.encode_query(queries, convert_to_tensor=True)
all_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5) 

print("Finished initial semantic search")
#get pre-trained cross-encoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


#new code for re-ranking the results, one query at a time
for i, query in enumerate(queries):
    hits = all_hits[i]
    #create pairs of query and top 5 chunks the normal encoder found
    sentence_pairs = [[query, corpus[hit['corpus_id']]] for hit in hits]
    #get relevant scores per pair
    rerank_scores = reranker.predict(sentence_pairs)
    #add new scores to existing hits
    for j in range(len(hits)):
        final_score = rerank_scores[j]
        #will try boosting so that if we find the keywords, we "boost" its score
        text = corpus[hits[j]['corpus_id']].lower()
        if any(word in text for word in keywords):
            # Give it a tiny nudge so the re-ranker definitely sees it
            final_score += 1.5 #boosting score, might need to modify

        hits[j]['rerank_score'] = final_score
    
    #now sort the scores so we can get highest
    hits = sorted(hits, key=lambda x: x['rerank_score'], reverse=True)

    #now print the top re-ranked results for each query
    print(f"\n--- RE-RANKED RESULTS FOR: {query} ---")
    for hit in hits:
        print(f"Rerank Score: {hit['rerank_score']:.4f} | Text: {corpus[hit['corpus_id']]}")