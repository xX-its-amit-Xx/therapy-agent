import asyncio, os, json
from typing import Any
from therapy_agent.state import AgentState
from therapy_agent.tools.chembl_query import chembl_query
from therapy_agent.tools.drugbank_query import drugbank_query


async def druggable_target_search_node(state: AgentState) -> dict:
    pathway_genes = state.get("pathway_genes") or []
    gene = state.get("gene_symbol") or state["gene"]
    mechanism = state.get("molecular_mechanism", "lof")

    # Always include the primary gene and its direct interactors. We
    # deliberately do NOT inject a curated ER-quality-control gene list
    # (TMED9/TMED2/TMED10 etc.) when the mechanism is misfolding. That
    # short-circuited the BRD4780/UMOD benchmark case by hand-placing
    # the answer in the candidate set. The Reactome interactor list for
    # the disease gene already surfaces these partners when they're
    # biologically real.
    search_genes = list(dict.fromkeys([gene] + pathway_genes[:14]))

    candidate_targets = []
    errors = []

    # Query ChEMBL (human-only, druggability count) and DrugBank
    # (druggability flag only — no specific approved-drug names) in
    # parallel. Both tools now return information about TRACTABILITY,
    # not about which approved drug already targets the gene; that
    # avoids handing the answer to strategy_synthesis.
    async def query_gene(g):
        try:
            chembl_result = await chembl_query(g)
            db_result = await drugbank_query(g)
            druggable = bool(chembl_result.get("druggable") or db_result.get("druggable"))
            if druggable:
                return {
                    "gene_name": g,
                    "druggable": True,
                    "chembl_n_active": chembl_result.get("n_active_compounds", 0),
                    "chembl_target_id": chembl_result.get("target_id"),
                }
        except Exception:
            return None
        return None

    tasks = [asyncio.create_task(query_gene(g)) for g in search_genes[:15]]
    results = await asyncio.gather(*tasks)

    for r in results:
        if r and r.get("druggable"):
            candidate_targets.append(r)

    trace = [
        f"Searched {len(search_genes)} genes for druggability",
        f"Found {len(candidate_targets)} druggable candidates (no specific drug names retrieved — blinded)",
    ]
    if candidate_targets:
        names = [t["gene_name"] for t in candidate_targets[:6]]
        trace.append(f"Top druggable candidates: {', '.join(names)}")

    return {
        "candidate_targets": candidate_targets,
        # approved_drugs intentionally left empty: passing specific
        # approved-drug names into the LLM prompt was the largest source
        # of test-set leakage in the previous benchmark configuration.
        "approved_drugs": [],
        "reasoning_trace": trace,
        "errors": errors,
    }
