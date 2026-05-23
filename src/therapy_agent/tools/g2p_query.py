"""g2p-rag query tool — stubs to the sibling g2p-rag project."""
import os
import httpx

G2P_BASE = os.environ.get("G2P_RAG_URL", "http://localhost:8000")

async def g2p_query(gene: str, mutation: str) -> dict:
    """Query the g2p-rag sibling project for variant-disease associations."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{G2P_BASE}/query",
                params={"gene": gene, "mutation": mutation},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception:
        # Graceful fallback with minimal mock data
        return {
            "records": [],
            "source": "g2p-rag (unavailable — using fallback)",
            "gene": gene,
            "mutation": mutation,
        }
