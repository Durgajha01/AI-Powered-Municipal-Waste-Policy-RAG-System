# AI-Powered Municipal Waste Policy RAG System

Intelligent Waste Pattern Analyzer with Policy-Grounded Recommendations.

A retrieval-augmented system that answers questions about municipal solid
waste policy — segregation rules, fines, collection schedules, escalation
thresholds — grounded in clause-level citations, and demonstrates how it
plugs into a waste-pattern-detection pipeline for the AI Waste Pattern
Analyzer project.

## How it works

1. **`src/ingest.py`** — parses policy documents (`data/policies/*.md`) into
   clause-level chunks (one `##` section = one chunk), preserving document
   title, jurisdiction, effective date, and section name as metadata. This
   is what lets every answer cite an exact clause instead of a vague
   page reference.

2. **`src/retriever.py`** — embeds each chunk (TF-IDF, offline/no external
   model download) and indexes them in FAISS (`IndexFlatIP` = cosine
   similarity on normalized vectors). Swap `Embedder.encode()` for a
   sentence-transformer or IBM Granite embedding model to improve recall
   on paraphrased questions — nothing else in the pipeline changes.

3. **`src/rag.py`** — retrieves top-k clauses for a question and generates
   an answer. Two backends:
   - `extractive` (default, no API key) — templates a cited answer directly
     from retrieved clauses.
   - `anthropic` — calls an LLM to synthesize a fluent, cited answer from
     the same retrieved context. Swap this function for a watsonx.ai call
     to `granite-3-*-instruct` to use IBM Granite instead.

4. **`src/pattern_to_policy_demo.py`** — the part that ties this into the
   *pattern analyzer* half of the project: synthetic ward audit data is
   run through a threshold rule (stand-in for your real pattern-detection
   model), and each flagged ward is automatically handed to the RAG system
   to produce a policy-grounded recommendation with citations — rather than
   RAG being a standalone chatbot bolted onto the side.

## Run it

```bash
pip install -r requirements.txt
cd src
python3 ingest.py               # parse policy docs -> data/chunks.json
python3 -c "from retriever import build_index_from_chunks_file as b; b('../data/chunks.json', '../data/index')"
python3 rag.py                  # ask sample questions, print cited answers
python3 pattern_to_policy_demo.py   # flagged-ward -> policy recommendation demo
```

## Using a real LLM backend

Set `ANTHROPIC_API_KEY` and instantiate `WastePolicyRAG(backend="anthropic")`
in `rag.py`, or replace `generate_anthropic()` with a watsonx.ai request to
a Granite model — the prompt-construction and retrieval logic is unchanged.

## Extending with real municipal data

Drop additional `.md` files into `data/policies/` following the same
`# Title` / `Jurisdiction:` / `Effective Date:` / `## Section N.N: Name`
structure, then re-run `ingest.py` and rebuild the index. PDF bylaws can be
converted to this structure with `pdfplumber` or similar before ingestion.
"# AI-Powered-Municipal-Waste-Policy-RAG-System" 
