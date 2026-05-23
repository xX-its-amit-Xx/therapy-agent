"""PubMed search via NCBI E-utilities."""
import os
import httpx

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")

async def pubmed_search(query: str, max_results: int = 5) -> dict:
    """Search PubMed and return abstract summaries."""
    try:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        async with httpx.AsyncClient(timeout=15.0) as client:
            search = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
            search.raise_for_status()
            ids = search.json().get("esearchresult", {}).get("idlist", [])

            if not ids:
                return {"articles": [], "query": query}

            summary_params = {
                "db": "pubmed",
                "id": ",".join(ids[:max_results]),
                "retmode": "json",
            }
            if NCBI_API_KEY:
                summary_params["api_key"] = NCBI_API_KEY

            summary = await client.get(f"{EUTILS_BASE}/esummary.fcgi", params=summary_params)
            summary.raise_for_status()
            result = summary.json().get("result", {})

            articles = []
            for uid in ids[:max_results]:
                a = result.get(uid, {})
                if a:
                    articles.append({
                        "pmid": uid,
                        "title": a.get("title", ""),
                        "authors": [au.get("name", "") for au in a.get("authors", [])[:3]],
                        "journal": a.get("source", ""),
                        "pubdate": a.get("pubdate", ""),
                        "doi": next((i.get("value") for i in a.get("articleids", []) if i.get("idtype") == "doi"), ""),
                    })

            return {"articles": articles, "query": query, "total": len(ids)}
    except Exception as e:
        return {"articles": [], "query": query, "error": str(e)}
