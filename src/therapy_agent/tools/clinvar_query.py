"""ClinVar query via NCBI E-utilities.

Now properly retries on transient HTTP errors using ``tenacity`` (the
decorators were previously imported but never applied). NCBI's free
tier rate-limits unauthenticated requests to 3/sec; setting
NCBI_API_KEY lifts that to 10/sec.
"""
import os
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from therapy_agent.tools._cache import cached_async

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.ReadTimeout)),
    reraise=True,
)
async def _eutils_get(url: str, params: dict) -> httpx.Response:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r


@cached_async("clinvar")
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

        search = await _eutils_get(f"{EUTILS_BASE}/esearch.fcgi", params)
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

        summary = await _eutils_get(f"{EUTILS_BASE}/esummary.fcgi", summary_params)
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
