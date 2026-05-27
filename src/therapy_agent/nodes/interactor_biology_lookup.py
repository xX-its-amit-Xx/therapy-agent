"""interactor_biology_lookup_node — fetch g2p-rag chunks for the top-K
druggable candidate interactors, so the LLM sees real biology for each
candidate target (not just the disease gene).

Without this node, the agent only ever sees UniProt biology for the
disease gene. For cases where the FDA target is a different protein
(BRD4780 UMOD->TMED9; Ekterly SERPING1->KLKB1; Givlaari HMBS->ALAS1;
SMA SMN1->SMN2; Obesity POMC->MC4R) the agent has to choose among
~10 pathway interactors with no information beyond their gene
symbols. This node closes that gap by retrieving UniProt FUNCTION /
PATHWAY / SUBUNIT / PTM / DISEASE / LIPIDATION text for each top
candidate.

Lives between druggable_target_search and strategy_synthesis in the
LangGraph.
"""
from __future__ import annotations

import asyncio

from therapy_agent.state import AgentState
from therapy_agent.tools.g2p_query import g2p_query


_MAX_INTERACTOR_LOOKUPS = 5  # extra UniProt fetches beyond the disease gene


async def interactor_biology_lookup_node(state: AgentState) -> dict:
    """For the top-K druggable candidate interactors, fetch g2p-rag chunks.

    Adds an ``interactor_g2p_data`` field to the state mapping gene_name ->
    formatted UniProt context. strategy_synthesis renders these into the
    prompt under "Candidate target biology".
    """
    disease_gene = state.get("gene_symbol") or state["gene"]
    candidates = state.get("candidate_targets") or []

    # De-duplicate, exclude the disease gene (already fetched in variant_lookup),
    # cap at K to keep network traffic / token use bounded.
    seen: set[str] = {disease_gene.upper()}
    targets: list[str] = []
    for c in candidates:
        g = (c.get("gene_name") or "").upper()
        if not g or g in seen:
            continue
        seen.add(g)
        targets.append(g)
        if len(targets) >= _MAX_INTERACTOR_LOOKUPS:
            break

    if not targets:
        return {
            "interactor_g2p_data": {},
            "reasoning_trace": ["interactor_biology_lookup: no candidate interactors to look up"],
        }

    tasks = [asyncio.create_task(g2p_query(g, mutation="")) for g in targets]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    interactor_data: dict[str, str] = {}
    errors: list[str] = []
    for g, r in zip(targets, results):
        if isinstance(r, Exception):
            errors.append(f"interactor_biology_lookup error for {g}: {r}")
            continue
        formatted = (r.get("formatted") or "").strip()
        if formatted and "No UniProt-derived chunks" not in formatted:
            interactor_data[g] = formatted

    return {
        "interactor_g2p_data": interactor_data,
        "reasoning_trace": [
            f"interactor_biology_lookup: retrieved UniProt chunks for "
            f"{len(interactor_data)}/{len(targets)} candidate interactors"
        ],
        "errors": errors,
    }
