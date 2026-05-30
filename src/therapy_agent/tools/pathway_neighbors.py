"""Pathway-neighbor tool: live retrieval of immediate biological neighbors
for a gene from Reactome ContentService -- one hop, both directions,
no therapeutic-strategy annotations.

Distinct from `expand_pathway` which returns a curated interactor list
(potentially biased toward test-set answers). This module hits the live
Reactome API and returns the raw participant list of the pathways the
gene is in, with no editorial filter. The LLM must do the reasoning
about which neighbor is the therapeutic node.

Tool exposed to agentic_target_research as `pathway_neighbors(gene)`.
"""
from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from therapy_agent.tools._cache import cached_async


_REACTOME = "https://reactome.org/ContentService"


@cached_async("reactome_pathway_neighbors")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
async def pathway_neighbors(gene: str) -> str:
    """Return a flat list of gene symbols that share a Reactome pathway
    with `gene`, sorted by frequency across pathways. No therapeutic
    annotation -- just the biology graph.

    Empty / formatted text suitable for inclusion in a tool-call result.
    """
    g = (gene or "").strip().upper()
    if not g:
        return "pathway_neighbors: empty gene"

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Find which Reactome pathways the gene participates in.
        try:
            r = await client.get(
                f"{_REACTOME}/search/query",
                params={"query": g, "species": "Homo sapiens",
                        "types": "Pathway", "cluster": "true"},
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            return f"pathway_neighbors: Reactome search failed for {g}: {exc}"

        pathway_ids: list[str] = []
        for entry in (data.get("results") or [])[:3]:
            for item in (entry.get("entries") or [])[:5]:
                pid = item.get("stId") or item.get("id")
                if pid:
                    pathway_ids.append(pid)

        if not pathway_ids:
            return f"pathway_neighbors: no Reactome pathways found for {g}"

        # Fetch participants of each pathway; aggregate.
        seen: dict[str, int] = {}
        pathway_names: list[str] = []
        for pid in pathway_ids[:6]:   # cap to bound network calls
            try:
                pr = await client.get(
                    f"{_REACTOME}/data/pathway/{pid}/containedEvents",
                )
                if pr.status_code != 200:
                    continue
                # Get the participating reference entities for the pathway.
                par = await client.get(
                    f"{_REACTOME}/data/participants/{pid}/referenceEntities",
                )
                if par.status_code != 200:
                    continue
                ents = par.json() or []
                # Track pathway name for context
                meta_r = await client.get(f"{_REACTOME}/data/query/{pid}")
                if meta_r.status_code == 200:
                    name = (meta_r.json() or {}).get("displayName", pid)
                    pathway_names.append(name)
                for e in ents:
                    sym = (e.get("geneName") or [None])[0] if isinstance(e.get("geneName"), list) else e.get("geneName")
                    if not sym:
                        continue
                    sym = sym.upper()
                    if sym == g:
                        continue
                    seen[sym] = seen.get(sym, 0) + 1
            except Exception:
                continue

        if not seen:
            return (
                f"pathway_neighbors: pathways for {g} found but no participants extracted "
                f"(pathways: {', '.join(pathway_names[:3])})"
            )

        # Top 15 by frequency across the gene's pathways.
        ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
        rows = [f"{sym} (in {n} of {g}'s pathways)" for sym, n in ranked]
        prefix = (
            f"Reactome pathway neighbors of {g} "
            f"(pathways consulted: {', '.join(pathway_names[:3])}):\n"
        )
        return prefix + "\n".join(rows)
