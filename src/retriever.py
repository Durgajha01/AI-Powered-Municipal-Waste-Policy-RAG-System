"""
retriever.py — Builds a FAISS vector index over policy chunks and retrieves
the top-k most relevant clauses for a query.

NOTE ON EMBEDDINGS:
This uses TF-IDF vectors (scikit-learn) turned dense + L2-normalized, indexed
with faiss.IndexFlatIP (== cosine similarity on normalized vectors). This is
fully offline and dependency-light, and works well on policy text because
clause language is keyword-dense (fines, sections, thresholds).

For production / better semantic recall on paraphrased questions, swap
`Embedder` below for a sentence-transformer
(e.g. `sentence-transformers/all-MiniLM-L6-v2`) or an IBM Granite embedding
model via watsonx.ai — the rest of the pipeline (FAISS index, retrieval,
generation) is unchanged. See `Embedder.encode()` for the swap point.
"""

import json
import pickle
import pathlib
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder:
    """Swap this class's internals to change the embedding backend."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
        self._fitted = False

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        # --- swap point for sentence-transformers / Granite embeddings ---
        vecs = self.vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return vecs / norms


class PolicyRetriever:
    def __init__(self):
        self.embedder = Embedder()
        self.index = None
        self.chunks = []

    def build(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.embedder.fit(texts)
        vecs = self.embedder.encode(texts)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)

    def search(self, query: str, k: int = 3) -> list[dict]:
        qvec = self.embedder.encode([query])
        scores, idxs = self.index.search(qvec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def save(self, out_dir: str):
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "policy.index"))
        with open(out / "retriever.pkl", "wb") as f:
            pickle.dump({"embedder": self.embedder, "chunks": self.chunks}, f)

    @classmethod
    def load(cls, out_dir: str) -> "PolicyRetriever":
        out = pathlib.Path(out_dir)
        r = cls()
        r.index = faiss.read_index(str(out / "policy.index"))
        with open(out / "retriever.pkl", "rb") as f:
            data = pickle.load(f)
        r.embedder = data["embedder"]
        r.chunks = data["chunks"]
        return r


def build_index_from_chunks_file(chunks_path: str, index_dir: str):
    chunks = json.loads(pathlib.Path(chunks_path).read_text(encoding="utf-8"))
    retriever = PolicyRetriever()
    retriever.build(chunks)
    retriever.save(index_dir)
    print(f"Built FAISS index over {len(chunks)} chunks -> {index_dir}")
    return retriever


if __name__ == "__main__":
    build_index_from_chunks_file("data/chunks.json", "data/index")
