"""Deterministic post-processor: if the rationale heavily features a gene
that isn't the target_protein, override the target_protein to match.

Closes the v0.7 Sotatercept failure mode where the LLM wrote BMPR2 (the
disease gene) into target_protein 3/3 self-consistency rounds, while the
rationale text explicitly cited ACVR2B as the better answer. self_critique
uses an LLM to detect this and didn't catch it. This node uses a regex +
frequency check that doesn't depend on a second LLM call.

Position in graph: AFTER self_critique, BEFORE END.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from therapy_agent.state import AgentState


_HGNC_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")

# Tokens that look like HGNC symbols but aren't (or are too generic to anchor
# a target on). Same list the scoring layer uses, kept in sync deliberately.
_NON_HGNC_BLOCKLIST = {
    "DNA", "RNA", "MRNA", "MIRNA", "SIRNA", "ASO", "PMO", "ATP", "GTP",
    "ADP", "GDP", "CAMP", "AAV", "AAV9", "LNP", "GALNAC", "PCR", "PNS",
    "CNS", "UPR", "ER", "GOLGI", "CAAX", "RT", "PTM", "PK", "PD", "FDA",
    "ICH", "ICMT", "NEJM", "PMID", "DOI", "HGNC",
    # Generic / discourse:
    "FOR", "AND", "OR", "WITH", "TO", "OF", "BY", "ON", "THE", "A", "IS",
    "AS", "WAS", "IF", "WHO", "I", "II", "III", "IV", "ALL",
}


def _hgnc_symbols(text: str) -> list[str]:
    raw = _HGNC_RE.findall(text or "")
    return [s for s in raw if s.upper() not in _NON_HGNC_BLOCKLIST]


def _dominant_gene_in_rationale(rationale: str) -> Optional[str]:
    """Return the most-mentioned HGNC symbol in the rationale, if any.

    Frequency >= 2 to count -- a single mention isn't strong enough.
    """
    symbols = _hgnc_symbols(rationale)
    if not symbols:
        return None
    counts = Counter(symbols)
    top, n = counts.most_common(1)[0]
    if n >= 2:
        return top
    # Single-mention case: still useful if the symbol appears in a strong
    # position (e.g., "ACVR2B is a paralogous receptor..."). Heuristic:
    # if the symbol appears in the first sentence, accept.
    first_sentence = rationale.split(".")[0]
    fs_symbols = _hgnc_symbols(first_sentence)
    if fs_symbols:
        return fs_symbols[0]
    return None


def _canonical(text: str) -> str:
    """Pull the first HGNC-shaped symbol out of an arbitrary target string."""
    if not text:
        return ""
    m = _HGNC_RE.search(text)
    return (m.group(1) if m else text.strip()).upper()


async def field_rationale_align_node(state: AgentState) -> dict:
    """Realign target_protein when the agentic research has converged on a
    different (non-disease-gene) target.

    v19 found that the previous frequency-based rationale-dominance check
    over-realigned in 3 dev cases (DMD->UTRN, GLA->M6PR, PCSK9->LDLR -- all
    cases where the rationale legitimately mentions a paralog/regulator in
    passing) and under-realigned in 3 others (POMC->MC4R, PIGA->C5,
    SMN1->SMN2 -- where the rationale clearly argued for the right target
    but target_protein wrote the disease gene). The frequency heuristic was
    wrong both ways.

    v20 simplifies to a single rule: if `research_proposed_target` differs
    from `target_protein` AND from the disease gene (i.e. the research has
    explicitly committed to a non-disease-gene target), realign. This
    trusts the research's explicit conclusion rather than heuristic text
    parsing.
    """
    strat = state.get("final_strategy") or state.get("strategy") or {}
    target = (strat.get("target_protein") or "").strip()
    research_target = (state.get("research_proposed_target") or "").strip()
    disease_gene = (state.get("gene_symbol") or state.get("gene") or "").strip()

    target_canonical = _canonical(target)
    research_canonical = _canonical(research_target)
    disease_canonical = _canonical(disease_gene)

    if not research_canonical:
        return {
            "reasoning_trace": ["field_rationale_align: no research proposal; skipping"],
        }
    if research_canonical == target_canonical:
        return {
            "reasoning_trace": [
                f"field_rationale_align: target already matches research ({target_canonical})"
            ],
        }
    if research_canonical == disease_canonical:
        # Research proposed the disease gene -- don't trust over the picker.
        return {
            "reasoning_trace": [
                f"field_rationale_align: research proposed the disease gene "
                f"({disease_canonical}); not overriding picker"
            ],
        }

    # Realign to the research's explicit non-disease-gene proposal.
    final_strategy = dict(strat)
    old_target = final_strategy.get("target_protein", "")
    final_strategy["target_protein"] = research_target
    final_strategy["rationale"] = (
        f"[field_rationale_align: realigned to research proposal "
        f"({research_target}); original target was {old_target!r}] "
        + (final_strategy.get("rationale") or "")
    )

    revised_strategy = dict(state.get("strategy") or {})
    revised_strategy["target_protein"] = research_target

    return {
        "strategy": revised_strategy,
        "final_strategy": final_strategy,
        "reasoning_trace": [
            f"field_rationale_align: REALIGNED {old_target!r} -> {research_target!r} "
            f"(research had explicitly proposed this non-disease-gene target)"
        ],
        "critique_notes": [
            f"REALIGN: target_protein {old_target!r} -> {research_target!r} "
            f"per agentic_target_research's explicit proposal"
        ],
    }


def _count_mentions(text: str, gene: str) -> int:
    return len(re.findall(rf"\b{re.escape(gene)}\b", text or ""))
