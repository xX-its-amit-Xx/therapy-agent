"""ChEMBL REST API query — druggability evidence only.

Returns a coarse "is this gene a known druggable target" count rather
than specific compound identities. The agent should learn that a target
has tractable chemistry without seeing which approved drug already hits
it (that would be test-set answer leakage).

We also restrict the target search to organism = Homo sapiens so that
non-human paralogs don't poison the candidate ranking.
"""
from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from therapy_agent.tools._cache import cached_async


CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

_HUMAN_TAX_ID = 9606


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
async def _http_get(url: str, params: dict) -> httpx.Response:
    async with httpx.AsyncClient(timeout=15.0) as client:
        return await client.get(url, params=params)


@cached_async("chembl")
async def chembl_query(gene_name: str) -> dict:
    """Return druggability evidence for *gene_name* (human).

    Schema:
        {
            "compounds": [],                 # always empty (no names)
            "n_active_compounds": int,       # # compounds at pchembl >= 6
            "druggable": bool,               # n_active_compounds > 0
            "gene": str,
            "target_id": str | None,
            "error": str (optional),
        }
    """
    try:
        target_resp = await _http_get(
            f"{CHEMBL_BASE}/target/search",
            params={"q": gene_name, "format": "json", "limit": 5},
        )
        if target_resp.status_code != 200:
            return {"compounds": [], "n_active_compounds": 0, "druggable": False, "gene": gene_name, "target_id": None}

        all_targets = target_resp.json().get("targets", []) or []
        # Filter to human SINGLE PROTEIN targets. Other types (NUCLEIC-ACID,
        # PROTEIN-PROTEIN INTERACTION, PROTEIN FAMILY) often appear first in
        # the ranked list but carry zero pchembl-scored small-molecule
        # activities, so picking them would underestimate druggability.
        human_proteins = [
            t for t in all_targets
            if int(t.get("tax_id", 0) or 0) == _HUMAN_TAX_ID
            and (t.get("target_type") or "").upper() == "SINGLE PROTEIN"
        ]
        if not human_proteins:
            # Fall back to any human target (some druggable entries are
            # PROTEIN COMPLEX or PROTEIN FAMILY).
            human_proteins = [t for t in all_targets
                              if int(t.get("tax_id", 0) or 0) == _HUMAN_TAX_ID]
        if not human_proteins:
            return {"compounds": [], "n_active_compounds": 0, "druggable": False, "gene": gene_name, "target_id": None}

        target_chembl_id = human_proteins[0].get("target_chembl_id", "")
        if not target_chembl_id:
            return {"compounds": [], "n_active_compounds": 0, "druggable": False, "gene": gene_name, "target_id": None}

        activity_resp = await _http_get(
            f"{CHEMBL_BASE}/activity",
            params={
                "target_chembl_id": target_chembl_id,
                "format": "json",
                "limit": 50,
                "pchembl_value__gte": 6,  # IC50 < 1 uM
            },
        )
        if activity_resp.status_code != 200:
            return {"compounds": [], "n_active_compounds": 0, "druggable": False,
                    "gene": gene_name, "target_id": target_chembl_id}

        activities = activity_resp.json().get("activities", []) or []
        unique_mol_ids = {a.get("molecule_chembl_id") for a in activities if a.get("molecule_chembl_id")}
        n_active = len(unique_mol_ids)
        return {
            "compounds": [],                              # intentionally empty
            "n_active_compounds": n_active,
            "druggable": n_active > 0,
            "gene": gene_name,
            "target_id": target_chembl_id,
        }
    except Exception as e:
        return {"compounds": [], "n_active_compounds": 0, "druggable": False,
                "gene": gene_name, "error": str(e)}
