import asyncio, os, json
from typing import Any
from therapy_agent.state import AgentState
from therapy_agent.tools.reactome_query import reactome_query


async def pathway_expansion_node(state: AgentState) -> dict:
    gene = state.get("gene_symbol") or state["gene"]
    mechanism = state.get("molecular_mechanism", "lof")

    try:
        reactome_data = await reactome_query(gene)
        pathway_genes = reactome_data.get("interactors", [])
        pathway_context = reactome_data.get("pathway_context", "")
        pathways = reactome_data.get("pathways", [])

        trace_lines = [
            f"Reactome: {len(pathways)} pathways for {gene}",
            f"Interactors/pathway members: {', '.join(pathway_genes[:10])}",
        ]
        if mechanism in ("lof",):
            trace_lines.append(f"LoF mechanism: prioritizing downstream effectors and compensatory targets")
        elif mechanism in ("misfolding", "mislocalization"):
            trace_lines.append(f"{mechanism} mechanism: prioritizing protein homeostasis machinery (ERAD, lysosome, chaperones, TMED)")

        return {
            "pathway_genes": pathway_genes,
            "pathway_context": pathway_context,
            "reasoning_trace": trace_lines,
        }
    except Exception as e:
        return {
            "pathway_genes": [],
            "pathway_context": "",
            "errors": [f"pathway_expansion error: {e}"],
            "reasoning_trace": [f"pathway_expansion failed: {e}"],
        }
