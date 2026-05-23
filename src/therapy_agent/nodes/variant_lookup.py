import asyncio
from therapy_agent.state import AgentState
from therapy_agent.tools.clinvar_query import clinvar_query
from therapy_agent.tools.g2p_tool import g2p_retrieve


async def variant_lookup_node(state: AgentState) -> dict:
    """Query ClinVar and g2p-rag for variant-level features and pathogenicity."""
    gene = state.get("gene_symbol") or state["gene"]
    mutation = state["mutation"]

    clinvar_task = asyncio.create_task(clinvar_query(gene, mutation))
    g2p_task = asyncio.create_task(g2p_retrieve(gene, mutation))

    clinvar_data, g2p_result = await asyncio.gather(clinvar_task, g2p_task, return_exceptions=True)

    errors = []
    if isinstance(clinvar_data, Exception):
        errors.append(f"ClinVar error: {clinvar_data}")
        clinvar_data = {}
    if isinstance(g2p_result, Exception):
        errors.append(f"g2p error: {g2p_result}")
        g2p_result = {"formatted": "", "chunks": [], "source": "error"}

    variants = clinvar_data.get("variants", [])
    chunks: list[dict] = g2p_result.get("chunks", [])

    trace = [f"ClinVar: found {len(variants)} variants for {gene}"]
    if chunks:
        trace.append(f"g2p-rag: retrieved {len(chunks)} chunk(s) via {g2p_result.get('source', '?')}")
    else:
        trace.append(f"g2p-rag: no chunks returned ({g2p_result.get('source', '?')})")

    # g2p_data carries the human-readable formatted text for LLM context;
    # g2p_chunks carries raw serialized Documents for downstream citation tracking.
    return {
        "clinvar_variants": variants,
        "g2p_data": {
            "formatted": g2p_result.get("formatted", ""),
            "source": g2p_result.get("source", ""),
        },
        "g2p_chunks": chunks,
        "reasoning_trace": trace,
        "errors": errors,
    }
