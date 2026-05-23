"""Unit tests for LangGraph nodes with mocked Claude and tool calls."""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import make_claude_response


# ── parse_input ───────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_input_extracts_gene_symbol(base_state):
    """parse_input should extract gene symbol and mutation type."""
    from therapy_agent.nodes.parse_input import parse_input_node

    parsed_json = json.dumps({
        "gene_symbol": "SERPING1",
        "mutation_type": "frameshift",
        "phenotype_terms": ["hereditary angioedema", "HAE", "angioedema"],
        "notes": "Haploinsufficiency of C1-esterase inhibitor",
    })

    with patch("therapy_agent.nodes.parse_input._get_client") as mock_gc:
        mock_gc.return_value.messages.create.return_value = make_claude_response(parsed_json)
        result = await parse_input_node(base_state)

    assert result["gene_symbol"] == "SERPING1"
    assert result["mutation_type"] == "frameshift"
    assert "hereditary angioedema" in result["phenotype_terms"]
    assert len(result.get("reasoning_trace", [])) > 0


@pytest.mark.asyncio
async def test_parse_input_fallback_on_error(base_state):
    """parse_input should fall back gracefully when Claude fails."""
    from therapy_agent.nodes.parse_input import parse_input_node

    with patch("therapy_agent.nodes.parse_input._get_client") as mock_gc:
        mock_gc.return_value.messages.create.side_effect = Exception("API error")
        result = await parse_input_node(base_state)

    assert result["gene_symbol"] == "SERPING1"
    assert len(result.get("errors", [])) > 0


# ── mechanism_classifier ──────────────────────────────────────────────
@pytest.mark.asyncio
async def test_mechanism_classifier_lof(base_state):
    """mechanism_classifier should return lof for SERPING1 frameshift."""
    from therapy_agent.nodes.mechanism_classifier import mechanism_classifier_node

    mech_json = json.dumps({
        "mechanism": "lof",
        "confidence": 0.95,
        "reasoning": "Frameshift → haploinsufficiency → uncontrolled kallikrein"
    })

    with patch("therapy_agent.nodes.mechanism_classifier._get_client") as mock_gc:
        mock_gc.return_value.messages.create.return_value = make_claude_response(mech_json)
        result = await mechanism_classifier_node(base_state)

    assert result["molecular_mechanism"] == "lof"
    assert result["mechanism_confidence"] == pytest.approx(0.95)
    assert len(result.get("mechanism_reasoning", "")) > 10


@pytest.mark.asyncio
async def test_mechanism_classifier_misfolding(misfolding_state):
    """mechanism_classifier should return misfolding for UMOD frameshift."""
    from therapy_agent.nodes.mechanism_classifier import mechanism_classifier_node

    mech_json = json.dumps({
        "mechanism": "misfolding",
        "confidence": 0.92,
        "reasoning": "UMOD frameshift → ER retention → UPR activation"
    })

    with patch("therapy_agent.nodes.mechanism_classifier._get_client") as mock_gc:
        mock_gc.return_value.messages.create.return_value = make_claude_response(mech_json)
        result = await mechanism_classifier_node(misfolding_state)

    assert result["molecular_mechanism"] == "misfolding"


# ── strategy_synthesis ────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_strategy_synthesis_hae(base_state):
    """strategy_synthesis should target KLKB1 for SERPING1/HAE."""
    from therapy_agent.nodes.strategy_synthesis import strategy_synthesis_node

    strat_json = json.dumps({
        "target_protein": "KLKB1 (plasma kallikrein)",
        "target_pathway": "Kallikrein-kinin system",
        "modulation_type": "inhibitor",
        "supporting_evidence": ["SERPING1 LoF removes brake on KLKB1"],
        "precedent_drugs": ["sebetralstat (Ekterly)", "berotralstat (Orladeyo)"],
        "confidence_score": 0.95,
        "citations": ["Webb DJ et al. KONFIDENT trial, KalVista 2025", "FDA approves Ekterly July 2025"],
        "rationale": "Targeting KLKB1 restores the inhibitory control lost by SERPING1 LoF."
    })

    with patch("therapy_agent.nodes.strategy_synthesis._get_client") as mock_gc:
        mock_gc.return_value.messages.create.return_value = make_claude_response(strat_json)
        result = await strategy_synthesis_node(base_state)

    assert result["strategy"] is not None
    strat = result["strategy"]
    assert "KLKB1" in strat["target_protein"]
    assert strat["modulation_type"] == "inhibitor"
    assert strat["confidence_score"] == pytest.approx(0.95)
    assert any("sebetralstat" in d for d in strat["precedent_drugs"])


@pytest.mark.asyncio
async def test_strategy_synthesis_adtkd(misfolding_state):
    """strategy_synthesis should target TMED9 for UMOD/ADTKD."""
    from therapy_agent.nodes.strategy_synthesis import strategy_synthesis_node

    strat_json = json.dumps({
        "target_protein": "TMED9 (cargo receptor)",
        "target_pathway": "ER quality control / COPI vesicle trafficking",
        "modulation_type": "inhibitor",
        "supporting_evidence": ["TMED9 retains misfolded UMOD in ER", "BRD4780 releases trapped protein"],
        "precedent_drugs": ["BRD4780 (preclinical, Dvela-Levitt et al. Cell 2019)"],
        "confidence_score": 0.88,
        "citations": ["Dvela-Levitt M et al. Cell 2019;178(3):521-535"],
        "rationale": "TMED9 modulation releases ER-retained mutant uromodulin for lysosomal degradation."
    })

    with patch("therapy_agent.nodes.strategy_synthesis._get_client") as mock_gc:
        mock_gc.return_value.messages.create.return_value = make_claude_response(strat_json)
        result = await strategy_synthesis_node(misfolding_state)

    assert result["strategy"] is not None
    strat = result["strategy"]
    assert "TMED9" in strat["target_protein"]
    assert any("BRD4780" in d for d in strat["precedent_drugs"])
    assert any("Dvela-Levitt" in c for c in strat["citations"])


# ── self_critique ─────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_self_critique_accepts_good_strategy(base_state):
    """self_critique should accept a well-supported strategy."""
    from therapy_agent.nodes.self_critique import self_critique_node

    base_state["strategy"] = {
        "target_protein": "KLKB1",
        "target_pathway": "Kallikrein-kinin system",
        "modulation_type": "inhibitor",
        "supporting_evidence": ["validated by 3 approved drugs"],
        "precedent_drugs": ["sebetralstat", "berotralstat", "lanadelumab"],
        "confidence_score": 0.95,
        "citations": ["Webb DJ et al. KalVista 2025"],
        "rationale": "Well-supported target",
    }

    critique_json = json.dumps({
        "verdict": "accept",
        "confidence_adjustment": 0.0,
        "critique_notes": ["Target well-validated by 3 approved drugs", "Logic is sound"],
        "unsupported_claims": [],
        "alternative_targets": ["F12 (Factor XIIa)"],
        "revised_confidence": 0.95,
    })

    with patch("therapy_agent.nodes.self_critique._get_client") as mock_gc:
        mock_gc.return_value.messages.create.return_value = make_claude_response(critique_json)
        result = await self_critique_node(base_state)

    assert result["final_strategy"] is not None
    assert result["final_strategy"]["confidence_score"] == pytest.approx(0.95)
    assert len(result.get("critique_notes", [])) > 0


@pytest.mark.asyncio
async def test_self_critique_flags_unsupported_claims(base_state):
    """self_critique should flag and record unsupported claims."""
    from therapy_agent.nodes.self_critique import self_critique_node

    base_state["strategy"] = {
        "target_protein": "KLKB1",
        "target_pathway": "Kallikrein-kinin system",
        "modulation_type": "inhibitor",
        "supporting_evidence": [],
        "precedent_drugs": ["invented_drug_xyz_2030"],
        "confidence_score": 0.9,
        "citations": ["Fake citation journal 2099"],
        "rationale": "Based on minimal evidence",
    }

    critique_json = json.dumps({
        "verdict": "revise",
        "confidence_adjustment": -0.3,
        "critique_notes": ["Drug 'invented_drug_xyz_2030' does not exist"],
        "unsupported_claims": ["invented_drug_xyz_2030 citation fabricated"],
        "alternative_targets": ["F12"],
        "revised_confidence": 0.6,
    })

    with patch("therapy_agent.nodes.self_critique._get_client") as mock_gc:
        mock_gc.return_value.messages.create.return_value = make_claude_response(critique_json)
        result = await self_critique_node(base_state)

    notes = result.get("critique_notes", [])
    assert any("UNSUPPORTED" in n or "unsupported" in n.lower() or "invented" in n.lower() for n in notes)
    assert result["final_strategy"]["confidence_score"] < 0.9
