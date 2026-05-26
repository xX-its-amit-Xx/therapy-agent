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
        data = _robust_json_parse(response.content[0].text.strip()) or {}
        return {
            "gene_symbol": data.get("gene_symbol", state["gene"].upper()),
            "mutation_type": data.get("mutation_type", "other"),
            "phenotype_terms": data.get("phenotype_terms", [state["disease_phenotype"]]),
            "reasoning_trace": [f"Parsed: gene={data.get('gene_symbol')}, mutation_type={data.get('mutation_type')}, notes={data.get('notes', '')}"],
            "token_usage": [{"node": "parse_input", "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}],
        }
    except Exception as e:
        return {
            "gene_symbol": state["gene"].upper(),
            "mutation_type": "other",
            "phenotype_terms": [state["disease_phenotype"]],
            "errors": [f"parse_input error: {e}"],
            "reasoning_trace": ["parse_input: fell back to raw input due to error"],
        }
