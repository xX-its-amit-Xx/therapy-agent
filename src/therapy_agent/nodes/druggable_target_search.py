import asyncio, os, json
from typing import Any
from therapy_agent.state import AgentState
from therapy_agent.tools.chembl_query import chembl_query
from therapy_agent.tools.drugbank_query import drugbank_query


async def druggable_target_search_node(state: AgentState) -> dict:
    pathway_genes = state.get("pathway_genes") or []
    gene = state.get("gene_symbol") or state["gene"]
    mechanism = state.get("molecular_mechanism", "lof")

    # Always include the primary gene and its direct interactors
    search_genes = list(dict.fromkeys([gene] + pathway_genes[:12]))

    # For misfolding: also search ER quality control proteins
    if mechanism in ("misfolding", "mislocalization"):
        qc_genes = ["HSPA5", "CANX", "CALR", "TMED9", "TMED2", "TMED10", "VCP", "HSP90B1"]
        search_genes = list(dict.fromkeys(search_genes + qc_genes))

    candidate_targets = []
    approved_drugs_list = []
    errors = []

    # Query ChEMBL and DrugBank in parallel for top genes
    async def query_gene(g):
        try:
            chembl_result = await chembl_query(g)
            db_result = await drugbank_query(g)
            compounds = chembl_result.get("compounds", [])
            drugs = db_result.get("drugs", [])
            if compounds or drugs:
                return {"gene_name": g, "chembl_compounds": compounds[:3], "drugbank_drugs": drugs[:3], "druggable": bool(compounds or drugs)}
        except Exception as ex:
            return None
        return None

    tasks = [asyncio.create_task(query_gene(g)) for g in search_genes[:15]]
    results = await asyncio.gather(*tasks)

    for r in results:
        if r and r.get("druggable"):
            candidate_targets.append(r)
            for d in r.get("drugbank_drugs", []):
                if d.get("approved"):
                    approved_drugs_list.append(d)

    trace = [
        f"Searched {len(search_genes)} genes for druggable targets",
        f"Found {len(candidate_targets)} druggable targets, {len(approved_drugs_list)} with approved drugs",
    ]
    if candidate_targets:
        names = [t["gene_name"] for t in candidate_targets[:6]]
        trace.append(f"Top targets: {', '.join(names)}")

    return {
        "candidate_targets": candidate_targets,
        "approved_drugs": approved_drugs_list,
        "reasoning_trace": trace,
        "errors": errors,
    }
