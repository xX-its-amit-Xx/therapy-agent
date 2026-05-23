import asyncio, os, json
from typing import Any
from therapy_agent.state import AgentState
from therapy_agent.tools.clinvar_query import clinvar_query
from therapy_agent.tools.g2p_query import g2p_query


async def variant_lookup_node(state: AgentState) -> dict:
    """Query ClinVar and g2p-rag for variant-level features and pathogenicity."""
    gene = state.get("gene_symbol") or state["gene"]
    mutation = state["mutation"]

    clinvar_task = asyncio.create_task(clinvar_query(gene, mutation))
    g2p_task = asyncio.create_task(g2p_query(gene, mutation))

    clinvar_data, g2p_data = await asyncio.gather(clinvar_task, g2p_task, return_exceptions=True)

    errors = []
    if isinstance(clinvar_data, Exception):
        errors.append(f"ClinVar error: {clinvar_data}")
        clinvar_data = {}
    if isinstance(g2p_data, Exception):
        errors.append(f"g2p error: {g2p_data}")
        g2p_data = {}

    variants = clinvar_data.get("variants", [])

    trace = [f"ClinVar: found {len(variants)} variants for {gene}"]
    if g2p_data.get("records"):
        trace.append(f"g2p-rag: found {len(g2p_data['records'])} records")

    return {
        "clinvar_variants": variants,
        "g2p_data": g2p_data,
        "reasoning_trace": trace,
        "errors": errors,
    }
