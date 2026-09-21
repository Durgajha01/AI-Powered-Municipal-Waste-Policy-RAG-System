"""
ingest.py — Parses municipal policy documents (markdown, clause-structured)
into retrieval-ready chunks with metadata.

Design choice: policy/legal text is chunked by CLAUSE (## Section heading),
not by fixed token windows. This keeps each chunk semantically whole (one
rule = one chunk) and lets us cite "Section X.Y" directly in answers.
"""

import re
import json
import pathlib
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    jurisdiction: str
    effective_date: str
    section: str
    text: str
    source_file: str


HEADER_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


def parse_doc(path: pathlib.Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    # Pull title / jurisdiction / effective date from the top matter (# and metadata lines)
    doc_title = lines[0].lstrip("# ").strip() if lines else path.stem
    jurisdiction = _extract_field(raw, "Jurisdiction")
    effective_date = _extract_field(raw, "Effective Date")

    chunks = []
    matches = list(HEADER_RE.finditer(raw))
    for i, m in enumerate(matches):
        section_title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        if not body:
            continue
        chunk_id = f"{path.stem}__{i:02d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_title=doc_title,
                jurisdiction=jurisdiction,
                effective_date=effective_date,
                section=section_title,
                text=body,
                source_file=path.name,
            )
        )
    return chunks


def _extract_field(raw: str, field: str) -> str:
    m = re.search(rf"^{field}:\s*(.+)$", raw, re.MULTILINE)
    return m.group(1).strip() if m else ""


def load_corpus(policy_dir: str) -> list[Chunk]:
    all_chunks = []
    for path in sorted(pathlib.Path(policy_dir).glob("*.md")):
        all_chunks.extend(parse_doc(path))
    return all_chunks


if __name__ == "__main__":
    chunks = load_corpus("data/policies")
    print(f"Parsed {len(chunks)} clause-level chunks from policy documents.\n")
    for c in chunks[:3]:
        print(f"[{c.chunk_id}] {c.doc_title} — {c.section}")
        print(c.text[:120].replace("\n", " ") + "...\n")

    out = pathlib.Path("data/chunks.json")
    out.write_text(json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks -> {out}")
