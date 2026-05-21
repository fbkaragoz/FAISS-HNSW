from sentence_transformers import CrossEncoder, SentenceTransformer
import faiss
import numpy as np
import time


doc_texts = [
    "Docker Compose lets you define and run multi-container applications using a YAML file.",
    "PostgreSQL is a relational database often used with web applications.",
    "FAISS is a library for fast vector similarity search.",
    "Cross-encoders rerank query-document pairs with higher precision than embedding search alone.",
    "Ekşi Sözlük entries are informal Turkish user-generated texts with slang and subjective tone.",
    "Ottoman Turkish NLP requires careful normalization, transliteration, and historical spelling handling.",
]

encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Encoding documents...")

doc_embs = encoder.encode(
    doc_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=32,
)

d = doc_embs.shape[1]
index = faiss.indexHNSWFlat(d, 16)
index.hnsw.efConstruction = 100
index.hnsw.efSearch = 64
index.add(doc_embs.astype("float32"))

def search(query: str, k_retrieve: int = 5):
    t0 = time.time()

    q_emb = encoder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    D, I = index.search(q_emb, k_retrieve) 

    candidates = [doc_texts[i] for i in I[0]]
    pairs = [(query, doc) for doc in candidates]

    rerank_scores = reranker.predict(pairs, batch_size=16)
    order = np.argsort(-rerank_scores)

    elapsed_ms = (time.time() - t0) * 1000

    print(f"Query: {query}")
    print(f"Elapsed time: {elapsed_ms:.2f} ms")

    for rank, idx in enumerate(order, start=1):
        print(f"{rank}. score={rerank_scores[idx]:.4f} | {candidates[idx]}")

    search("What is FAISS?")
    search("How to run multi-container applications?")
    search("What is Ottoman Turkish?")
    search("How do I improve retrieval quality after embedding search?")
    search("What makes Ekşi Sözlük text hard for NLP?")