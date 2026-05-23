import asyncio, os, json
from typing import Any
from therapy_agent.state import AgentState
import anthropic


def _get_client():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _get_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


SYSTEM = """You are a molecular geneticist specializing in rare-disease mechanisms.
Classify the molecular consequence of the given mutation.

MECHANISM TYPES:
- lof: Loss-of-function — protein absent, truncated, or non-functional (haploinsufficiency, biallelic)
- gof: Gain-of-function — hyperactive or neomorphic protein activity
- dominant_negative: Mutant protein interferes with wild-type activity
- misfolding: Protein misfolds, often leading to ER retention, aggregation, or UPR
- mislocalization: Protein reaches correct conformation but wrong cellular compartment

FEW-SHOT EXAMPLES:
Input: SERPING1 frameshift, C1-inhibitor deficiency, hereditary angioedema
Output: {"mechanism": "lof", "confidence": 0.95, "reasoning": "Frameshift in SERPING1 causes haploinsufficiency of C1-esterase inhibitor, reducing inhibition of plasma kallikrein and Factor XIIa, leading to bradykinin overproduction and angioedema attacks."}

Input: UMOD frameshift, protein misfolding ER retention, ADTKD
Output: {"mechanism": "misfolding", "confidence": 0.92, "reasoning": "Uromodulin frameshift mutations cause protein misfolding and ER retention in kidney tubule epithelial cells, activating UPR and leading to tubular dysfunction."}

Input: MUC1 frameshift (fs), ADTKD-MUC1
Output: {"mechanism": "misfolding", "confidence": 0.90, "reasoning": "MUC1 frameshift creates a toxic mutant protein that misfolds and accumulates in the ER of kidney collecting duct cells."}

Input: HBB p.Glu6Val (sickle), hemolytic anemia, vasoocclusion
Output: {"mechanism": "misfolding", "confidence": 0.88, "reasoning": "HbS polymerizes under deoxygenation due to Val6 hydrophobic patch; disease is driven by sickling rather than classic LoF."}

Input: DMD exon deletion, Duchenne muscular dystrophy
Output: {"mechanism": "lof", "confidence": 0.97, "reasoning": "Out-of-frame deletion eliminates dystrophin, causing complete protein absence and progressive myonecrosis."}

Return ONLY valid JSON."""


async def mechanism_classifier_node(state: AgentState) -> dict:
    client = _get_client()

    gene = state.get("gene_symbol") or state["gene"]
    mut_type = state.get("mutation_type", "unknown")
    phenotype = state["disease_phenotype"]
    mutation = state["mutation"]

    # Include ClinVar summary if available
    clinvar_summary = ""
    variants = state.get("clinvar_variants") or []
    if variants:
        pathogenic = [v for v in variants if "pathogenic" in str(v.get("significance", "")).lower()]
        clinvar_summary = f"\nClinVar: {len(variants)} variants found, {len(pathogenic)} pathogenic."

    user_msg = f"Gene: {gene}\nMutation: {mutation} ({mut_type})\nDisease: {phenotype}{clinvar_summary}\n\nClassify the molecular mechanism. Return JSON only."

    try:
        response = client.messages.create(
            model=_get_model(),
            max_tokens=512,
            system=SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        data = json.loads(response.content[0].text.strip())
        return {
            "molecular_mechanism": data["mechanism"],
            "mechanism_confidence": float(data["confidence"]),
            "mechanism_reasoning": data["reasoning"],
            "reasoning_trace": [f"Mechanism: {data['mechanism']} (confidence={data['confidence']:.2f}): {data['reasoning']}"],
        }
    except Exception as e:
        return {
            "molecular_mechanism": "lof",
            "mechanism_confidence": 0.5,
            "mechanism_reasoning": f"Defaulted to lof due to error: {e}",
            "errors": [f"mechanism_classifier error: {e}"],
            "reasoning_trace": ["mechanism_classifier: defaulted to lof"],
        }
