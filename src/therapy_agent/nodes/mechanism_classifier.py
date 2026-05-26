import asyncio, os, json, re
from typing import Any
from therapy_agent.state import AgentState
from therapy_agent.llm import get_backend


from therapy_agent.config import get_model


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _robust_json_parse(text: str):
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    m = _JSON_BLOCK_RE.search(text)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _get_client():
    return get_backend()


def _get_model() -> str:
    return get_model()


SYSTEM = """You are a molecular geneticist specializing in rare-disease mechanisms.
Classify the molecular consequence of the given mutation.

MECHANISM TYPES:
- lof: Loss-of-function — protein absent, truncated, or non-functional (haploinsufficiency, biallelic)
- gof: Gain-of-function — hyperactive or neomorphic protein activity
- dominant_negative: Mutant protein interferes with wild-type activity
- misfolding: Protein misfolds, often leading to ER retention, aggregation, or UPR
- mislocalization: Protein reaches correct conformation but wrong cellular compartment

OUTPUT SCHEMA (no test-case-specific examples — apply your knowledge to
the input directly):

{"mechanism": "lof|gof|dominant_negative|misfolding|mislocalization",
 "confidence": <float in [0,1]>,
 "reasoning": "<2-3 sentences naming the protein-level consequence and how
                it produces the disease phenotype>"}

HEURISTICS:
- Frameshift, nonsense, large deletion in a non-toxic protein → usually lof.
- Frameshift/missense that yields a protein-product retained in the ER,
  with kidney/muscle/liver phenotype involving cellular stress → misfolding.
- Missense that causes intracellular polymerization or aggregation under
  physiological conditions → misfolding (toxic aggregation subtype).
- Repeat expansion that yields a toxic RNA or protein → gof.
- Missense in a transmembrane domain altering channel gating → can be
  gof (constitutive activation) or lof (loss of conductance); use the
  phenotype to disambiguate.

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
        data = _robust_json_parse(response.content[0].text.strip())
        if not data or "mechanism" not in data:
            raise ValueError("mechanism_classifier could not parse JSON")
        return {
            "molecular_mechanism": data["mechanism"],
            "mechanism_confidence": float(data.get("confidence", 0.5)),
            "mechanism_reasoning": data.get("reasoning", ""),
            "reasoning_trace": [f"Mechanism: {data['mechanism']} (confidence={data['confidence']:.2f}): {data['reasoning']}"],
            "token_usage": [{"node": "mechanism_classifier", "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}],
        }
    except Exception as e:
        return {
            "molecular_mechanism": "lof",
            "mechanism_confidence": 0.5,
            "mechanism_reasoning": f"Defaulted to lof due to error: {e}",
            "errors": [f"mechanism_classifier error: {e}"],
            "reasoning_trace": ["mechanism_classifier: defaulted to lof"],
        }
