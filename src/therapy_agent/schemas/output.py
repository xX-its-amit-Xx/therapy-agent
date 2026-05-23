"""Versioned output schema for the therapy-agent pipeline.

This is the contract consumed by bio-rag-eval. Bump schema_version when
any field is added, renamed, or has its type changed.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from therapy_agent.config import BENCHMARK_SCHEMA_VERSION


# ── Sub-models ────────────────────────────────────────────────────────────────

class TargetProtein(BaseModel):
    name: str
    uniprot_id: Optional[str] = None
    gene_symbol: Optional[str] = None


class EvidenceClaim(BaseModel):
    claim: str
    source_url: Optional[str] = None
    doi: Optional[str] = None


class PrecedentDrug(BaseModel):
    name: str
    drugbank_id: Optional[str] = None
    approved: bool = False
    indication: Optional[str] = None
    year: Optional[int] = None


class ReasoningStep(BaseModel):
    node: str
    content: str
    timestamp: Optional[str] = None


# ── Canonical modulation type -------------------------------------------
ModulationType = Literal[
    "inhibitor",
    "agonist",
    "chaperone",
    "modulator",
    "ASO",
    "gene_therapy",
    "enzyme_replacement",
    "other",
]

# Map from agent-internal strings (permissive) → canonical ModulationType
_MODULATION_MAP: dict[str, ModulationType] = {
    "inhibitor": "inhibitor",
    "activator": "agonist",
    "agonist": "agonist",
    "chaperone": "chaperone",
    "modulator": "modulator",
    "replacement": "enzyme_replacement",
    "enzyme_replacement": "enzyme_replacement",
    "gene_therapy": "gene_therapy",
    "splice_modifier": "ASO",
    "sirna_aso": "ASO",     # lower-cased version of siRNA_ASO
    "aso": "ASO",
    "sirna": "ASO",
    "other": "other",
}


def normalize_modulation(raw: str | None) -> ModulationType:
    """Map an agent-produced modulation string to the canonical enum value."""
    if not raw:
        return "other"
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    return _MODULATION_MAP.get(key, "other")


# ── Top-level output ──────────────────────────────────────────────────────────

class TherapyStrategyOutput(BaseModel):
    schema_version: str = BENCHMARK_SCHEMA_VERSION

    # ── provenance ───────────────────────────────────────────────────────
    case_id: Optional[str] = None
    gene: str
    mutation: str
    disease_phenotype: str

    # ── strategy ─────────────────────────────────────────────────────────
    target_protein: TargetProtein
    target_pathway: Optional[str] = None
    modulation_type: ModulationType
    supporting_evidence: list[EvidenceClaim] = Field(default_factory=list)
    precedent_drugs: list[PrecedentDrug] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)

    # ── transparency ─────────────────────────────────────────────────────
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)

    # ── run metadata (populated by benchmark runner) ──────────────────────
    model_used: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    wall_clock_seconds: Optional[float] = None
    timestamp: Optional[str] = None

    @field_validator("confidence_score")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# ── Builder ───────────────────────────────────────────────────────────────────

def _extract_gene_symbol(target_name: str) -> str | None:
    """Pull the leading ALLCAPS word from strings like 'KLKB1 (plasma kallikrein)'."""
    m = re.match(r'^([A-Z][A-Z0-9]+)', target_name)
    return m.group(1) if m else None


def _parse_precedent_drugs(raw: list[str]) -> list[PrecedentDrug]:
    drugs = []
    for entry in raw:
        # Extract approval status hint
        approved = any(kw in entry.lower() for kw in ("fda approved", "ema approved", "approved"))
        # Extract year hint
        year_m = re.search(r'\b(19|20)\d{2}\b', entry)
        year = int(year_m.group()) if year_m else None
        drugs.append(PrecedentDrug(name=entry, approved=approved, year=year))
    return drugs


def build_output(
    *,
    case_id: str | None,
    final_state: dict,
    model: str,
    wall_clock: float,
) -> TherapyStrategyOutput:
    """Construct a TherapyStrategyOutput from a completed agent state dict."""
    strategy = final_state.get("final_strategy") or final_state.get("strategy") or {}
    token_usage = final_state.get("token_usage", [])

    total_in = sum(t.get("input_tokens", 0) for t in token_usage) or None
    total_out = sum(t.get("output_tokens", 0) for t in token_usage) or None

    target_name = strategy.get("target_protein") or "Unknown"

    evidence = [
        EvidenceClaim(claim=e)
        for e in strategy.get("supporting_evidence", [])
        if e
    ]
    # Attach DOIs from citations list where they look like DOI strings
    for cit in strategy.get("citations", []):
        doi_m = re.search(r'10\.\d{4,}/\S+', cit)
        evidence.append(EvidenceClaim(
            claim=cit,
            doi=doi_m.group() if doi_m else None,
        ))

    reasoning = [
        ReasoningStep(node="pipeline", content=step)
        for step in final_state.get("reasoning_trace", [])
        if step
    ]

    return TherapyStrategyOutput(
        case_id=case_id,
        gene=final_state.get("gene", ""),
        mutation=final_state.get("mutation", ""),
        disease_phenotype=final_state.get("disease_phenotype", ""),
        target_protein=TargetProtein(
            name=target_name,
            gene_symbol=_extract_gene_symbol(target_name),
        ),
        target_pathway=strategy.get("target_pathway"),
        modulation_type=normalize_modulation(strategy.get("modulation_type")),
        supporting_evidence=evidence,
        precedent_drugs=_parse_precedent_drugs(strategy.get("precedent_drugs", [])),
        confidence_score=float(strategy.get("confidence_score", 0.0)),
        reasoning_trace=reasoning,
        model_used=model,
        input_tokens=total_in,
        output_tokens=total_out,
        wall_clock_seconds=round(wall_clock, 3),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
