"""g2p-rag retrieval tool — real integration with the g2p-rag package.

Lazy-initializes G2PRetrieverLangChain on first call so the import cost
(model load, index warm-up) is paid only when the tool is actually used.

Falls back to the HTTP stub in g2p_query.py if the package is not installed,
so the rest of the pipeline degrades gracefully in environments where
g2p-rag is not available.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from therapy_agent.config import G2P_RETRIEVAL_K, get_g2p_index_dir

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Module-level singleton — populated on first call to _get_retriever()
_retriever = None
_retriever_lock = asyncio.Lock()


def _get_retriever():
    """Return the module-level G2PRetriever, initializing it if needed.

    Uses g2p-rag's native API directly (not the LangChain adapter) so we can
    pass a per-call gene_filter -- disease-gene queries should return
    chunks for THAT gene, not a similarity-ranked mix across the index.

    Raises ImportError if g2p-rag isn't installed; the caller in
    g2p_retrieve() catches that and falls back to the UniProt-direct path.
    """
    global _retriever
    if _retriever is None:
        from g2p_rag import G2PRetriever  # type: ignore[import]
        persist_dir = get_g2p_index_dir()
        logger.info("Initializing G2PRetriever(persist_dir=%r) k=%d",
                     persist_dir, G2P_RETRIEVAL_K)
        _retriever = G2PRetriever(persist_dir=persist_dir)
        logger.info("g2p-rag retriever ready.")
    return _retriever


def _chunk_to_dict(chunk) -> dict:
    """Serialize a g2p_rag RetrievedChunk to a plain dict for state storage."""
    return {
        "content": chunk.text,
        "source": f"g2p-rag :: {chunk.chunk_type} {chunk.residue_range}".strip(),
        "doi": "",
        "pmid": "",
        "title": f"{chunk.gene} ({chunk.uniprot_id}) {chunk.chunk_type}",
        "gene": chunk.gene,
        "score": chunk.score,
        "uniprot_id": chunk.uniprot_id,
        "chunk_type": chunk.chunk_type,
        "residue_range": chunk.residue_range,
    }


def _format_chunks(chunks: list[dict]) -> str:
    """Format raw chunk dicts into a readable string for the LLM context window."""
    if not chunks:
        return "No g2p-rag results found."
    lines = [f"[g2p-rag] {len(chunks)} retrieved chunk(s):\n"]
    for i, c in enumerate(chunks, 1):
        header = f"  [{i}]"
        if c.get("title"):
            header += f" {c['title']}"
        if c.get("doi"):
            header += f" (DOI: {c['doi']})"
        lines.append(header)
        lines.append(f"      {c['content'][:300].strip()}")
    return "\n".join(lines)


async def g2p_retrieve(gene: str, mutation: str) -> dict:
    """Query g2p-rag with (gene, mutation) and return formatted + raw results.

    Returns:
        {
            "formatted": str,          # human-readable for LLM context
            "chunks":    list[dict],   # raw serialized Documents for citation tracking
            "source":    str,
            "gene":      str,
            "mutation":  str,
        }
    """
    query = f"{gene} {mutation}".strip()
    try:
        retriever = _get_retriever()

        # G2PRetriever.retrieve is synchronous (ChromaDB call + numpy fusion).
        # Run it in a worker thread so the LangGraph event loop doesn't stall.
        loop = asyncio.get_event_loop()
        retrieved = await loop.run_in_executor(
            None,
            lambda: retriever.retrieve(query, k=G2P_RETRIEVAL_K,
                                         gene_filter=[gene] if gene else None),
        )
        chunks = [_chunk_to_dict(c) for c in retrieved]
        return {
            "formatted": _format_chunks(chunks),
            "chunks": chunks,
            "source": "g2p-rag (package)",
            "gene": gene,
            "mutation": mutation,
        }

    except ImportError:
        logger.info("g2p-rag package / ChromaDB index not available; using "
                    "UniProt-backed g2p fallback (real biology, no embeddings).")
        return await _http_fallback(gene, mutation)
    except Exception as exc:
        logger.warning("g2p-rag retrieval failed (%s); using UniProt-backed g2p fallback.", exc)
        return await _http_fallback(gene, mutation)


async def _http_fallback(gene: str, mutation: str) -> dict:
    """Fall back to the UniProt-backed g2p_query when the ChromaDB index isn't
    available. g2p_query returns chunks in the same shape g2p-rag would have
    returned, derived from UniProt directly (which is what g2p-rag indexes)."""
    from therapy_agent.tools.g2p_query import g2p_query as _http_query
    result = await _http_query(gene, mutation)
    chunks = result.get("chunks") or result.get("records", [])
    return {
        "formatted": result.get("formatted") or f"[g2p-rag fallback] {len(chunks)} chunks",
        "chunks": chunks,
        "source": result.get("source", "g2p-rag UniProt fallback"),
        "gene": gene,
        "mutation": mutation,
    }
