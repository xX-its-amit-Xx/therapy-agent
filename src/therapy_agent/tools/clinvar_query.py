"""ClinVar query via NCBI E-utilities."""
import os
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")

async def clinvar_query(gene: str, mutation: str | None = None) -> dict:
    """Return ClinVar variant summaries for a gene (and optional mutation)."""
    try:
        query = f"{gene}[gene]"
        if mutation:
            keywords = mutation.split()[:3]
            query += " AND (" + " OR ".join(f"{kw}[variant]" for kw in keywords) + ")"

        params = {
            "db": "clinvar",
            "term": query,
            "retmax": 20,
            "retmode": "json",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        async with httpx.AsyncClient(timeout=15.0) as client:
            search = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
            search.raise_for_status()
            ids = search.json().get("esearchresult", {}).get("idlist", [])

            if not ids:
                return {"variants": [], "total": 0, "gene": gene}

            summary_params = {
                "db": "clinvar",
                "id": ",".join(ids[:10]),
                "retmode": "json",
            }
            if NCBI_API_KEY:
                summary_params["api_key"] = NCBI_API_KEY

            summary = await client.get(f"{EUTILS_BASE}/esummary.fcgi", params=summary_params)
            summary.raise_for_status()
            result = summary.json().get("result", {})

            variants = []
            for uid in ids[:10]:
                v = result.get(uid, {})
                if v:
                    variants.append({
                        "id": uid,
                        "title": v.get("title", ""),
                        "significance": v.get("clinical_significance", {}).get("description", ""),
                        "gene": v.get("genes", [{}])[0].get("symbol", gene) if v.get("genes") else gene,
                        "condition": v.get("trait_set", [{}])[0].get("trait_name", "") if v.get("trait_set") else "",
                    })

            return {"variants": variants, "total": len(ids), "gene": gene}
    except Exception as e:
        return {"variants": [], "total": 0, "gene": gene, "error": str(e)}
