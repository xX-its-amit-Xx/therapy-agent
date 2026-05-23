"""Unit tests for therapy-agent tools with mocked HTTP."""
from __future__ import annotations
import json
import pytest
import httpx
import respx


# ── drugbank_query (static, no mocking needed) ────────────────────────
@pytest.mark.asyncio
async def test_drugbank_query_klkb1():
    """drugbank_query should return sebetralstat and berotralstat for KLKB1."""
    from therapy_agent.tools.drugbank_query import drugbank_query

    result = await drugbank_query("KLKB1")
    assert result["gene"] == "KLKB1"
    drugs = result["drugs"]
    assert len(drugs) >= 2
    names = [d["name"] for d in drugs]
    assert any("sebetralstat" in n or "Ekterly" in n for n in names)
    assert any("berotralstat" in n or "Orladeyo" in n for n in names)
    assert all(d["approved"] for d in drugs)


@pytest.mark.asyncio
async def test_drugbank_query_tmed9():
    """drugbank_query should return BRD4780 for TMED9."""
    from therapy_agent.tools.drugbank_query import drugbank_query

    result = await drugbank_query("TMED9")
    drugs = result["drugs"]
    assert len(drugs) >= 1
    assert any("BRD4780" in d["name"] for d in drugs)
    assert not drugs[0]["approved"]  # preclinical


@pytest.mark.asyncio
async def test_drugbank_query_sod1():
    """drugbank_query should return tofersen for SOD1."""
    from therapy_agent.tools.drugbank_query import drugbank_query

    result = await drugbank_query("SOD1")
    drugs = result["drugs"]
    assert any("tofersen" in d["name"].lower() for d in drugs)


@pytest.mark.asyncio
async def test_drugbank_query_unknown_gene():
    """drugbank_query should return empty list for unknown gene."""
    from therapy_agent.tools.drugbank_query import drugbank_query

    result = await drugbank_query("UNKNOWNGENE999")
    assert result["drugs"] == []
    assert result["total"] == 0


# ── reactome_query (curated fallback) ────────────────────────────────
@pytest.mark.asyncio
async def test_reactome_query_serping1():
    """reactome_query should return KLKB1 and F12 in interactors for SERPING1."""
    from therapy_agent.tools.reactome_query import reactome_query

    result = await reactome_query("SERPING1")
    interactors = result.get("interactors", [])
    assert "KLKB1" in interactors
    assert "F12" in interactors
    assert "kallikrein" in result.get("pathway_context", "").lower()


@pytest.mark.asyncio
async def test_reactome_query_umod():
    """reactome_query should return TMED9 in interactors for UMOD."""
    from therapy_agent.tools.reactome_query import reactome_query

    result = await reactome_query("UMOD")
    interactors = result.get("interactors", [])
    assert "TMED9" in interactors
    assert "TMED2" in interactors or "TMED10" in interactors


# ── g2p_query (stub/fallback) ─────────────────────────────────────────
@pytest.mark.asyncio
async def test_g2p_query_returns_fallback():
    """g2p_query should return fallback dict when service is unavailable."""
    from therapy_agent.tools.g2p_query import g2p_query

    # With no service running, should not raise
    result = await g2p_query("SERPING1", "frameshift")
    assert "gene" in result or "records" in result
    assert result.get("gene") == "SERPING1" or result.get("records") is not None


# ── openfda_query ─────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_openfda_query_handles_404():
    """openfda_query should return empty labels list on error."""
    from therapy_agent.tools.openfda_query import openfda_query

    with respx.mock:
        respx.get("https://api.fda.gov/drug/label.json").mock(
            return_value=httpx.Response(404, json={"error": "not found"})
        )
        result = await openfda_query("nonexistent_drug_xyz")

    assert result["labels"] == []


# ── clinvar_query ─────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_clinvar_query_handles_api_error():
    """clinvar_query should return empty variants list on API failure."""
    from therapy_agent.tools.clinvar_query import clinvar_query

    with respx.mock:
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        result = await clinvar_query("SERPING1", "frameshift")

    assert result["variants"] == []
    assert "error" in result


# ── pubmed_search ─────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_pubmed_search_handles_empty_results():
    """pubmed_search should return empty articles on no results."""
    from therapy_agent.tools.pubmed_search import pubmed_search

    search_response = {"esearchresult": {"idlist": []}}

    with respx.mock:
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            return_value=httpx.Response(200, json=search_response)
        )
        result = await pubmed_search("nonexistent rare disease xyz 2099")

    assert result["articles"] == []
