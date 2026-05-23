"""OpenFDA drug label query."""
import httpx

OPENFDA_BASE = "https://api.fda.gov/drug/label.json"

async def openfda_query(drug_name: str) -> dict:
    """Return FDA drug label info for a given drug name."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                OPENFDA_BASE,
                params={"search": f'openfda.brand_name:"{drug_name}"', "limit": 3},
            )
            if resp.status_code != 200:
                return {"labels": [], "drug": drug_name}

            data = resp.json()
            results = data.get("results", [])
            labels = []
            for r in results[:3]:
                labels.append({
                    "brand_name": r.get("openfda", {}).get("brand_name", [drug_name]),
                    "generic_name": r.get("openfda", {}).get("generic_name", []),
                    "manufacturer": r.get("openfda", {}).get("manufacturer_name", []),
                    "indications": (r.get("indications_and_usage") or [""])[0][:300],
                    "mechanism": (r.get("mechanism_of_action") or [""])[0][:300],
                })
            return {"labels": labels, "drug": drug_name}
    except Exception as e:
        return {"labels": [], "drug": drug_name, "error": str(e)}
