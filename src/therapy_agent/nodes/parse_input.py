import asyncio, os, json
from typing import Any
from therapy_agent.state import AgentState
import anthropic


def _get_client():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _get_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


async def parse_input_node(state: AgentState) -> dict:
    """Extract gene symbol, mutation type, and phenotype terms from free-text input."""
    client = _get_client()

    prompt = f"""Extract structured information from this genetic variant input.

Input:
- Gene: {state['gene']}
- Mutation: {state['mutation']}
- Disease phenotype: {state['disease_phenotype']}

Return a JSON object with:
- gene_symbol: str (HGNC symbol, uppercase)
- mutation_type: str (one of: frameshift, missense, nonsense, splice, deletion, duplication, expansion, other)
- phenotype_terms: list[str] (key phenotype keywords for searching, 3-6 terms)
- notes: str (any important observations)

Return ONLY valid JSON, no markdown."""

    try:
        response = client.messages.create(
            model=_get_model(),
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(response.content[0].text.strip())
        return {
            "gene_symbol": data.get("gene_symbol", state["gene"].upper()),
            "mutation_type": data.get("mutation_type", "other"),
            "phenotype_terms": data.get("phenotype_terms", [state["disease_phenotype"]]),
            "reasoning_trace": [f"Parsed: gene={data.get('gene_symbol')}, mutation_type={data.get('mutation_type')}, notes={data.get('notes', '')}"],
        }
    except Exception as e:
        return {
            "gene_symbol": state["gene"].upper(),
            "mutation_type": "other",
            "phenotype_terms": [state["disease_phenotype"]],
            "errors": [f"parse_input error: {e}"],
            "reasoning_trace": ["parse_input: fell back to raw input due to error"],
        }
