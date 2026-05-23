"""Shared pytest fixtures for therapy-agent tests."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Minimal state fixture ─────────────────────────────────────────────
@pytest.fixture
def base_state():
    return {
        "gene": "SERPING1",
        "mutation": "frameshift causing haploinsufficiency",
        "disease_phenotype": "hereditary angioedema",
        "reasoning_trace": [],
        "citations": [],
        "errors": [],
        "gene_symbol": "SERPING1",
        "mutation_type": "frameshift",
        "phenotype_terms": ["hereditary angioedema", "HAE", "angioedema"],
        "clinvar_variants": [],
        "g2p_data": {},
        "molecular_mechanism": "lof",
        "mechanism_confidence": 0.95,
        "mechanism_reasoning": "SERPING1 frameshift → C1-INH haploinsufficiency → uncontrolled kallikrein",
        "pathway_genes": ["KLKB1", "F12", "BDKRB2", "KNG1", "F11"],
        "pathway_context": "Contact activation / kallikrein-kinin system",
        "candidate_targets": [
            {"gene_name": "KLKB1", "druggable": True, "drugbank_drugs": [
                {"name": "sebetralstat (Ekterly)", "approved": True, "year": 2025}
            ]},
        ],
        "approved_drugs": [
            {"name": "sebetralstat (Ekterly)", "approved": True, "indication": "HAE"}
        ],
        "strategy": None,
        "critique_notes": None,
        "final_strategy": None,
        "retry_count": 0,
    }


@pytest.fixture
def misfolding_state():
    return {
        "gene": "UMOD",
        "mutation": "frameshift causing ER retention",
        "disease_phenotype": "autosomal dominant tubulointerstitial kidney disease",
        "reasoning_trace": [],
        "citations": [],
        "errors": [],
        "gene_symbol": "UMOD",
        "mutation_type": "frameshift",
        "phenotype_terms": ["ADTKD", "tubulointerstitial kidney disease", "ER retention"],
        "clinvar_variants": [],
        "g2p_data": {},
        "molecular_mechanism": "misfolding",
        "mechanism_confidence": 0.92,
        "mechanism_reasoning": "UMOD frameshift → ER retention → UPR → tubular damage",
        "pathway_genes": ["TMED9", "TMED2", "TMED10", "HSPA5", "CANX", "VCP"],
        "pathway_context": "ER quality control / TMED cargo receptors",
        "candidate_targets": [
            {"gene_name": "TMED9", "druggable": True, "drugbank_drugs": [
                {"name": "BRD4780", "approved": False, "year": 2019}
            ]},
        ],
        "approved_drugs": [],
        "strategy": None,
        "critique_notes": None,
        "final_strategy": None,
        "retry_count": 0,
    }


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client that returns plausible responses."""
    client = MagicMock()
    return client


def make_claude_response(text: str):
    """Helper to create a mock Anthropic messages response."""
    content = MagicMock()
    content.text = text
    response = MagicMock()
    response.content = [content]
    return response
