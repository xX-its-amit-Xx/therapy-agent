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

from therapy_agent.config import G2P_RETRIEVAL_K

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Module-level singleton — populated on first call to _get_retriever()
_retriever = None
_retriever_lock = asyncio.Lock()


def _get_retriever():
    """Return the module-level G2PRetrieverLangChain, initializing it if needed.

    Raises ImportError if g2p-rag is not installed (caller should handle).
    """
    global _retriever
    if _retriever is None:
        from g2p_rag import G2PRetrieverLangChain  # type: ignore[import]
        logger.info("Initializing G2PRetrieverLangChain (k=%d)…", G2P_RETRIEVAL_K)
        _retriever = G2PRetrieverLangChain(k=G2P_RETRIEVAL_K)
        logger.info("G2PRetrieverLangChain ready.")
    return _retriever


def _chunk_to_dict(doc) -> dict:
    """Serialize a LangChain Document to a plain dict for state storage."""
    return {
        "content": doc.page_content,
        "source": doc.metadata.get("source", ""),
        "doi": doc.metadata.get("doi", ""),
        "pmid": doc.metadata.get("pmid", ""),
        "title": doc.metadata.get("title", ""),
        "gene": doc.metadata.get("gene", ""),
        "score": doc.metadata.get("score"),
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
    query = f"{gene} {mutation}"
    try:
        retriever = _get_retriever()

        # Use async variant when available, fall back to sync in executor
        if hasattr(retriever, "aget_relevant_documents"):
            docs = await retriever.aget_relevant_documents(query)
        else:
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                None, retriever.get_relevant_documents, query
            )

        chunks = [_chunk_to_dict(d) for d in docs]
        return {
            "formatted": _format_chunks(chunks),
            "chunks": chunks,
            "source": "g2p-rag (package)",
            "gene": gene,
            "mutation": mutation,
        }

    except ImportError:
        logger.warning("g2p-rag package not installed — falling back to HTTP stub.")
        return await _http_fallback(gene, mutation)
    except Exception as exc:
        logger.warning("g2p-rag retrieval failed (%s) — falling back to HTTP stub.", exc)
        return await _http_fallback(gene, mutation)


async def _http_fallback(gene: str, mutation: str) -> dict:
    """Fall back to the original HTTP-based g2p_query stub."""
    from therapy_agent.tools.g2p_query import g2p_query as _http_query
    result = await _http_query(gene, mutation)
    # Normalise to the new schema
    return {
        "formatted": f"[g2p-rag HTTP stub] records={len(result.get('records', []))}",
        "chunks": result.get("records", []),
        "source": result.get("source", "g2p-rag (HTTP stub)"),
        "gene": gene,
        "mutation": mutation,
    }
