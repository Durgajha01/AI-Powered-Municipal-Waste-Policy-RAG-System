"""
rag.py — Ties retrieval to answer generation, with citations back to the
exact policy clause.

Two generation backends are provided:

  - "extractive"  : no API key needed. Composes a templated answer directly
                     from the retrieved clauses. Good for offline demos and
                     as a safety fallback if the LLM call fails.
  - "anthropic"   : calls Claude (or swap in IBM Granite via watsonx.ai /
                     OpenAI the same way) to synthesize a fluent answer from
                     the retrieved clauses. Requires ANTHROPIC_API_KEY.

Swap GRANITE in by replacing `generate_anthropic()` with a call to your
watsonx.ai deployment endpoint for `granite-3-*-instruct` — the retrieval
and prompt-construction logic stays identical.
"""

import os
from retriever import PolicyRetriever


SYSTEM_PROMPT = (
    "You are a municipal waste policy assistant. Answer ONLY using the "
    "provided policy clauses. Always cite the document and section for "
    "every claim. If the clauses don't contain the answer, say so plainly "
    "instead of guessing."
)


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"[{c['doc_title']} — {c['section']}] ({c['jurisdiction']}, "
            f"effective {c['effective_date']})\n{c['text']}"
        )
    return "\n\n".join(blocks)


def generate_extractive(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant policy clause was found for this query."
    lines = [f"Based on {len(chunks)} matching policy clause(s):\n"]
    for c in chunks:
        lines.append(
            f"• {c['doc_title']} — Section \"{c['section']}\" "
            f"(relevance {c['score']:.2f}):\n  {c['text']}"
        )
    return "\n\n".join(lines)


def generate_anthropic(query: str, chunks: list[dict]) -> str:
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    context = format_context(chunks)
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Policy clauses:\n\n{context}\n\nQuestion: {query}",
            }
        ],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


class WastePolicyRAG:
    def __init__(self, index_dir: str = "data/index", backend: str = "extractive"):
        self.retriever = PolicyRetriever.load(index_dir)
        self.backend = backend

    def answer(self, query: str, k: int = 3) -> dict:
        chunks = self.retriever.search(query, k=k)
        if self.backend == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                text = generate_anthropic(query, chunks)
            except Exception as e:  # fall back gracefully
                text = generate_extractive(query, chunks) + f"\n\n[LLM call failed, showing extractive answer: {e}]"
        else:
            text = generate_extractive(query, chunks)
        return {
            "query": query,
            "answer": text,
            "sources": [
                {"doc": c["doc_title"], "section": c["section"], "score": round(c["score"], 3)}
                for c in chunks
            ],
        }


if __name__ == "__main__":
    rag = WastePolicyRAG(backend="extractive")
    for q in [
        "What is the fine for not segregating wet and dry waste at home?",
        "How often should e-waste collection points operate?",
        "What triggers a mandatory awareness drive in a ward?",
    ]:
        result = rag.answer(q)
        print("Q:", result["query"])
        print(result["answer"])
        print("Sources:", result["sources"])
        print("-" * 80)
