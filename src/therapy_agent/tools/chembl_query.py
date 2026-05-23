"""ChEMBL REST API query for drug-target pairs."""
import httpx

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

async def chembl_query(gene_name: str) -> dict:
    """Return approved/clinical compounds targeting a gene from ChEMBL."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Search target by gene name
            target_resp = await client.get(
                f"{CHEMBL_BASE}/target/search",
                params={"q": gene_name, "format": "json", "limit": 3},
            )
            if target_resp.status_code != 200:
                return {"compounds": [], "gene": gene_name}

            targets = target_resp.json().get("targets", [])
            if not targets:
                return {"compounds": [], "gene": gene_name}

            target_chembl_id = targets[0].get("target_chembl_id", "")
            if not target_chembl_id:
                return {"compounds": [], "gene": gene_name}

            # Get approved drugs for this target
            activity_resp = await client.get(
                f"{CHEMBL_BASE}/activity",
                params={
                    "target_chembl_id": target_chembl_id,
                    "format": "json",
                    "limit": 10,
                    "pchembl_value__gte": 6,  # IC50 < 1 µM
                },
            )
            if activity_resp.status_code != 200:
                return {"compounds": [], "gene": gene_name}

            activities = activity_resp.json().get("activities", [])
            seen = set()
            compounds = []
            for act in activities:
                mol_id = act.get("molecule_chembl_id", "")
                if mol_id and mol_id not in seen:
                    seen.add(mol_id)
                    compounds.append({
                        "chembl_id": mol_id,
                        "pchembl_value": act.get("pchembl_value"),
                        "activity_type": act.get("standard_type"),
                        "target": gene_name,
                    })
                if len(compounds) >= 5:
                    break

            return {"compounds": compounds, "gene": gene_name, "target_id": target_chembl_id}
    except Exception as e:
        return {"compounds": [], "gene": gene_name, "error": str(e)}
