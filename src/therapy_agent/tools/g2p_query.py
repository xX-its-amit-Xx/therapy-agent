"""g2p-rag retrieval — real biology context from UniProt.

The Broad Institute G2P portal that `g2p-rag` indexes is built on top of
UniProt (function, pathway, subunit, PTM, lipidation, disease comments)
plus AlphaFold pLDDT and ClinVar. When the g2p-rag package and its
ChromaDB index are not available, this module falls back to direct
UniProt REST queries — i.e. the same primary data g2p-rag would have
embedded — and shapes the result as a list of g2p-style chunks.

This keeps real biology flowing into `variant_lookup_node` even on
machines without a built ChromaDB index.
"""
from __future__ import annotations

import re
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

_PUBMED_CITE_RE = re.compile(r"\(?\s*PubMed:\d+(?:\s*,\s*PubMed:\d+)*\s*\)?")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_UNIPROT_FIELDS = ",".join([
    "accession", "gene_names", "protein_name",
    "cc_function", "cc_pathway", "cc_subunit",
    "cc_ptm", "cc_disease",
    "ft_lipid",
    "keyword",
])


def _clean(text: str, *, max_chars: int = 600) -> str:
    out = _PUBMED_CITE_RE.sub("", text or "")
    out = _MULTI_SPACE_RE.sub(" ", out).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "..."
    return out


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
async def _uniprot_lookup(gene: str) -> dict[str, Any]:
    """Fetch the reviewed human UniProt entry for *gene*. Retries on transient errors."""
    params = {
        "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
        "fields": _UNIPROT_FIELDS,
        "format": "json",
        "size": 1,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(UNIPROT_SEARCH, params=params)
        r.raise_for_status()
        return r.json()


def _chunks_from_entry(entry: dict, gene: str) -> list[dict]:
    """Shape one UniProt entry into a list of g2p-rag-style chunks."""
    acc = entry.get("primaryAccession", "")
    pname = (entry.get("proteinDescription", {})
                  .get("recommendedName", {})
                  .get("fullName", {})
                  .get("value", ""))
    chunks: list[dict] = []

    def _add(section: str, text: str) -> None:
        if not text:
            return
        chunks.append({
            "content": text,
            "source": f"UniProt {acc} :: {section}",
            "doi": "",
            "pmid": "",
            "title": f"{pname} ({gene}) — {section}",
            "gene": gene,
            "score": None,
        })

    for c in entry.get("comments", []):
        ct = c.get("commentType")
        if ct == "FUNCTION":
            for t in c.get("texts", []):
                v = (t.get("value") or "").strip()
                if not v:
                    continue
                if v.lower().startswith("(microbial infection)"):
                    continue
                if "virus" in v.lower()[:80]:
                    continue
                _add("FUNCTION", _clean(v, max_chars=500))
        elif ct == "PATHWAY":
            for t in c.get("texts", []):
                _add("PATHWAY", _clean((t.get("value") or "").strip(), max_chars=400))
        elif ct == "SUBUNIT":
            for t in c.get("texts", []):
                _add("SUBUNIT", _clean((t.get("value") or "").strip(), max_chars=400))
        elif ct == "PTM":
            for t in c.get("texts", []):
                _add("PTM", _clean((t.get("value") or "").strip(), max_chars=500))
        elif ct == "DISEASE":
            dis = c.get("disease", {}) or {}
            name = dis.get("diseaseId", "")
            acronym = dis.get("acronym", "")
            desc = (dis.get("description") or "").strip()
            head = f"{name} ({acronym})".strip(" ()") if (name or acronym) else ""
            if head or desc:
                _add("DISEASE", _clean(f"{head}: {desc}".strip(": "), max_chars=300))

    for ft in entry.get("features", []):
        if ft.get("type") != "Lipidation":
            continue
        desc = (ft.get("description") or "").strip()
        loc = ft.get("location", {}) or {}
        start = (loc.get("start") or {}).get("value")
        end = (loc.get("end") or {}).get("value")
        pos = f"{start}" if start == end else f"{start}-{end}"
        if desc and pos:
            _add("LIPIDATION", f"{desc} at residue {pos}")

    kws = [k.get("name", "") for k in entry.get("keywords", []) if k.get("name")]
    if kws:
        _add("KEYWORDS", ", ".join(kws[:25]))

    return chunks


def _format_chunks(chunks: list[dict], gene: str) -> str:
    """Pretty-print chunks for inclusion in the LLM context window."""
    if not chunks:
        return f"No UniProt-derived chunks for {gene}."
    lines = [f"[g2p-rag :: UniProt-backed fallback] {len(chunks)} chunk(s) for {gene}:"]
    for i, c in enumerate(chunks, 1):
        section = c.get("source", "").split("::")[-1].strip() if "::" in c.get("source", "") else ""
        header = f"  [{i}] {section}" if section else f"  [{i}]"
        lines.append(header)
        lines.append(f"      {c['content'][:400]}")
    return "\n".join(lines)


async def g2p_query(gene: str, mutation: str) -> dict:
    """Return UniProt-backed g2p-style chunks for *gene*.

    Output shape mirrors the historical g2p_query stub so callers
    (`variant_lookup_node`, `g2p_tool._http_fallback`) don't need to
    change. The `records` key is kept as an alias for `chunks`.
    """
    try:
        data = await _uniprot_lookup(gene)
        entries = data.get("results", [])
        chunks = _chunks_from_entry(entries[0], gene) if entries else []
    except Exception as exc:
        return {
            "records": [],
            "chunks": [],
            "formatted": f"g2p-rag UniProt fallback failed for {gene}: {exc}",
            "source": "g2p-rag UniProt fallback (error)",
            "gene": gene,
            "mutation": mutation,
        }
    return {
        "records": chunks,
        "chunks": chunks,
        "formatted": _format_chunks(chunks, gene),
        "source": "g2p-rag UniProt fallback",
        "gene": gene,
        "mutation": mutation,
    }
