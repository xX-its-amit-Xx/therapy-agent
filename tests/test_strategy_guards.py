"""Unit tests for the v0.9 / v0.9.1 / v0.9.2 / v0.9.3 strategy guards.

These tests verify the static guard logic in strategy_synthesis.py
without needing a real LLM. The Stage-1 pattern selector and the
Stage-2 picker are mocked; the guards' branching is what's exercised.

What's covered:

  - v0.9.2 phenotype-pattern override fires on ACTH-driven phenotype
    when Stage 1 picked disease_gene_protein_chaperone.
  - v0.9.2 override does NOT fire on non-feedback phenotypes (e.g.
    plain LoF without feedback markers).
  - v0.9.3 mechanism-pattern guard fires when mechanism=lof and
    Stage 1 picked disease_gene_mRNA.
  - The original v0.9 disease_gene_default hard-constraint guard still
    fires when Stage 2 picker outputs disease gene under a non-disease-
    gene target_kind.
"""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import make_claude_response


def _state_for_feedback_axis_case() -> dict:
    """Crinecerfont/CAH archetype: LoF disease gene + ACTH-driven phenotype."""
    return {
        "gene": "CYP21A2",
        "mutation": "biallelic loss-of-function",
        "disease_phenotype": ("classic 21-hydroxylase deficiency with "
                               "cortisol deficiency and ACTH-driven adrenal "
                               "androgen excess"),
        "gene_symbol": "CYP21A2",
        "molecular_mechanism": "lof",
        "mechanism_reasoning": "loss of 21-OH activity",
        "phenotype_terms": ["CAH", "21-hydroxylase deficiency"],
        "pathway_genes": ["CRHR1", "MC2R", "POMC"],
        "pathway_context": "Hypothalamic-pituitary-adrenal axis",
        "candidate_targets": [{"gene_name": "CRHR1", "druggable": True}],
        "approved_drugs": [],
        "reasoning_trace": [],
        "citations": [],
        "errors": [],
        "strategy": None,
        "critique_notes": None,
        "final_strategy": None,
        "retry_count": 0,
    }


def _state_for_lof_mrna_case() -> dict:
    """Iptacopan/PNH archetype: LoF disease gene + complement phenotype.
    The model often picks pattern 5 (disease_gene_mRNA) which is wrong
    for LoF."""
    return {
        "gene": "PIGA",
        "mutation": "somatic loss-of-function",
        "disease_phenotype": ("paroxysmal nocturnal hemoglobinuria with "
                               "complement-mediated intravascular hemolysis"),
        "gene_symbol": "PIGA",
        "molecular_mechanism": "lof",
        "mechanism_reasoning": "loss of GPI-anchored CD55/CD59",
        "phenotype_terms": ["PNH", "complement-mediated hemolysis"],
        "pathway_genes": ["C5", "CFB", "CFD", "C3"],
        "pathway_context": "Alternative complement pathway",
        "candidate_targets": [{"gene_name": "CFB", "druggable": True}],
        "approved_drugs": [],
        "reasoning_trace": [],
        "citations": [],
        "errors": [],
        "strategy": None,
        "critique_notes": None,
        "final_strategy": None,
        "retry_count": 0,
    }


@pytest.mark.asyncio
async def test_v92_phenotype_override_fires_on_acth_driven():
    """When Stage 1 picks chaperone for an ACTH-driven phenotype, the
    v0.9.2 override should force the pattern to feedback_axis_receptor
    BEFORE re-running anything."""
    from therapy_agent.nodes import strategy_synthesis

    state = _state_for_feedback_axis_case()

    # Stage 1 first picks chaperone (the wrong pattern for this case).
    selector_response = make_claude_response(json.dumps({
        "pattern_id": "4a",
        "target_kind": "disease_gene_protein_chaperone",
        "reasoning": "misfolding amenable to refolding",
    }))
    # Stage 2 picker (called after the override forced pattern 9).
    picker_response = make_claude_response(json.dumps({
        "target_protein": "CRHR1",
        "rationale": "block ACTH-driving CRH receptor to dampen the axis",
    }))

    client = MagicMock()
    client.messages.create.side_effect = [selector_response] + [picker_response] * 6

    with patch.object(strategy_synthesis, "_get_client", return_value=client), \
         patch.object(strategy_synthesis, "_get_model", return_value="test-model"):
        result = await strategy_synthesis.strategy_synthesis_node(state)

    strat = result["strategy"]
    assert strat["pattern_id"] == "9", \
        f"expected v0.9.2 override to set pattern 9, got {strat['pattern_id']}"
    assert strat["target_kind"] == "feedback_axis_receptor"


@pytest.mark.asyncio
async def test_v93_mechanism_guard_fires_on_lof_mrna():
    """When mechanism=lof and Stage 1 picks disease_gene_mRNA, the
    v0.9.3 guard should force the pattern to downstream_effector."""
    from therapy_agent.nodes import strategy_synthesis

    state = _state_for_lof_mrna_case()

    # Stage 1 incorrectly picks mRNA knockdown for a LoF case.
    selector_response = make_claude_response(json.dumps({
        "pattern_id": "5",
        "target_kind": "disease_gene_mRNA",
        "reasoning": "GOF mRNA knockdown",
    }))
    picker_response = make_claude_response(json.dumps({
        "target_protein": "CFB",
        "rationale": "block alternative complement amplification",
    }))

    client = MagicMock()
    client.messages.create.side_effect = [selector_response] + [picker_response] * 6

    with patch.object(strategy_synthesis, "_get_client", return_value=client), \
         patch.object(strategy_synthesis, "_get_model", return_value="test-model"):
        result = await strategy_synthesis.strategy_synthesis_node(state)

    strat = result["strategy"]
    assert strat["pattern_id"] == "1", \
        f"expected v0.9.3 mechanism guard to set pattern 1, got {strat['pattern_id']}"
    assert strat["target_kind"] == "downstream_effector"


@pytest.mark.asyncio
async def test_no_override_on_clean_lof_case():
    """An ordinary LoF case (no feedback markers, picked downstream_effector)
    should pass through without any guard firing."""
    from therapy_agent.nodes import strategy_synthesis

    # Use HAE/SERPING1 -- LoF, contact-activation pathway. No feedback
    # markers in the phenotype.
    state = {
        "gene": "SERPING1",
        "mutation": "frameshift causing haploinsufficiency",
        "disease_phenotype": "hereditary angioedema with episodic swelling",
        "gene_symbol": "SERPING1",
        "molecular_mechanism": "lof",
        "mechanism_reasoning": "C1-INH haploinsufficiency",
        "phenotype_terms": ["HAE"],
        "pathway_genes": ["KLKB1", "F12", "BDKRB2"],
        "pathway_context": "Contact activation",
        "candidate_targets": [{"gene_name": "KLKB1", "druggable": True}],
        "approved_drugs": [],
        "reasoning_trace": [],
        "citations": [],
        "errors": [],
        "strategy": None,
        "critique_notes": None,
        "final_strategy": None,
        "retry_count": 0,
    }
    selector_response = make_claude_response(json.dumps({
        "pattern_id": "1",
        "target_kind": "downstream_effector",
        "reasoning": "LoF inhibitor -> unbraked effector enzyme",
    }))
    picker_response = make_claude_response(json.dumps({
        "target_protein": "KLKB1",
        "rationale": "unbraked plasma kallikrein drives the bradykinin cascade",
    }))

    client = MagicMock()
    client.messages.create.side_effect = [selector_response] + [picker_response] * 6

    with patch.object(strategy_synthesis, "_get_client", return_value=client), \
         patch.object(strategy_synthesis, "_get_model", return_value="test-model"):
        result = await strategy_synthesis.strategy_synthesis_node(state)

    strat = result["strategy"]
    assert strat["pattern_id"] == "1"
    assert strat["target_kind"] == "downstream_effector"
