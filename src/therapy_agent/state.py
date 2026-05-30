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
    # token usage per LLM call: {"node": str, "input_tokens": int, "output_tokens": int}
    token_usage: Annotated[list[dict], operator.add]
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
    g2p_chunks: Optional[list[dict]]
    # gene -> formatted UniProt biology text for top candidate interactors
    # (populated by interactor_biology_lookup_node)
    interactor_g2p_data: Optional[dict]
    # ReAct-style agentic-research output (populated by agentic_target_research_node)
    research_history: Optional[list[dict]]
    research_proposed_target: Optional[str]
    research_proposed_rationale: Optional[str]
    strategy: Optional[TherapeuticStrategy]
    critique_notes: Optional[list[str]]
    critique_pass_done: bool
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
        token_usage=[],
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
        g2p_chunks=None,
        interactor_g2p_data=None,
        research_history=None,
        research_proposed_target=None,
        research_proposed_rationale=None,
        strategy=None,
        critique_notes=None,
        critique_pass_done=False,
        final_strategy=None,
        retry_count=0,
    )
