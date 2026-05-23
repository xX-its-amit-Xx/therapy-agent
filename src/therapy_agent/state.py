from typing import TypedDict, Optional, Annotated
import operator


class TherapeuticStrategy(TypedDict):
    target_protein: str
    target_pathway: str
    modulation_type: str       # inhibitor / activator / replacement / gene_therapy / splice_modifier / chaperone / siRNA_ASO
    supporting_evidence: list[str]
    precedent_drugs: list[str]
    confidence_score: float    # 0.0 – 1.0
    citations: list[str]
    rationale: str


class AgentState(TypedDict):
    # ── input ─────────────────────────────────────────────────────────
    gene: str
    mutation: str
    disease_phenotype: str
    # ── accumulate (reducer = operator.add) ──────────────────────────
    reasoning_trace: Annotated[list[str], operator.add]
    citations: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    # ── node outputs (replaced each time) ────────────────────────────
    gene_symbol: Optional[str]
    mutation_type: Optional[str]      # frameshift / missense / nonsense / splice / cnv / other
    phenotype_terms: Optional[list[str]]
    clinvar_variants: Optional[list[dict]]
    g2p_data: Optional[dict]
    molecular_mechanism: Optional[str]      # lof / gof / dominant_negative / misfolding / mislocalization
    mechanism_confidence: Optional[float]
    mechanism_reasoning: Optional[str]
    pathway_genes: Optional[list[str]]
    pathway_context: Optional[str]
    candidate_targets: Optional[list[dict]]
    approved_drugs: Optional[list[dict]]
    strategy: Optional[TherapeuticStrategy]
    critique_notes: Optional[list[str]]
    final_strategy: Optional[TherapeuticStrategy]
    retry_count: int


def make_initial_state(gene: str, mutation: str, disease_phenotype: str) -> AgentState:
    """Return a fresh AgentState with all lists initialized to [] and Optional fields to None."""
    return AgentState(
        gene=gene,
        mutation=mutation,
        disease_phenotype=disease_phenotype,
        reasoning_trace=[],
        citations=[],
        errors=[],
        gene_symbol=None,
        mutation_type=None,
        phenotype_terms=None,
        clinvar_variants=None,
        g2p_data=None,
        molecular_mechanism=None,
        mechanism_confidence=None,
        mechanism_reasoning=None,
        pathway_genes=None,
        pathway_context=None,
        candidate_targets=None,
        approved_drugs=None,
        strategy=None,
        critique_notes=None,
        final_strategy=None,
        retry_count=0,
    )
