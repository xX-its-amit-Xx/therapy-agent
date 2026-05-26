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


SYSTEM = """You are a critical reviewer of therapeutic strategy proposals.

Review the given strategy for:
1. Are the cited drugs/trials real and correctly attributed?
2. Is the target-mechanism logic sound?
3. Are there unsupported claims?
4. What alternative targets should be considered?
5. What are the main risks or failure modes?

Return JSON:
{
  "verdict": "accept" | "revise",
  "confidence_adjustment": float (e.g. -0.1, 0.0, +0.05),
  "critique_notes": [list of specific issues or confirmations],
  "unsupported_claims": [list of specific claims lacking evidence],
  "alternative_targets": [list of reasonable alternatives],
  "revised_confidence": float
}

Be honest. If the strategy is well-supported, say so. If it confabulates drugs or trials, flag it."""


async def self_critique_node(state: AgentState) -> dict:
    client = _get_client()
    strategy = state.get("strategy")
    if not strategy:
        return {
            "critique_notes": ["No strategy to critique"],
            "reasoning_trace": ["self_critique: no strategy found"],
        }

    gene = state.get("gene_symbol") or state["gene"]
    phenotype = state["disease_phenotype"]

    user_msg = f"""Review this therapeutic strategy for {gene} / {phenotype}:

{json.dumps(strategy, indent=2)}

Gene mechanism: {state.get('molecular_mechanism', 'unknown')}
Mechanism reasoning: {state.get('mechanism_reasoning', '')}

Be rigorous. Flag any confabulated citations or unlikely claims."""

    try:
        response = client.messages.create(
            model=_get_model(),
            max_tokens=1000,
            system=SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        data = _robust_json_parse(text) or {}

        verdict = data.get("verdict", "accept")
        adj = float(data.get("confidence_adjustment", 0.0))
        revised_conf = float(data.get("revised_confidence", strategy.get("confidence_score", 0.5)))
        notes = data.get("critique_notes", [])
        unsupported = data.get("unsupported_claims", [])
        alternatives = data.get("alternative_targets", [])

        # Build final strategy
        final_strategy = dict(strategy)
        final_strategy["confidence_score"] = revised_conf

        if unsupported:
            notes += [f"UNSUPPORTED: {c}" for c in unsupported]

        trace = [
            f"Critique verdict: {verdict} (confidence adj: {adj:+.2f} → {revised_conf:.2f})",
            f"Critique notes: {'; '.join(notes[:3])}",
        ]
        if alternatives:
            trace.append(f"Alternative targets: {', '.join(alternatives[:3])}")

        return {
            "critique_notes": notes,
            "final_strategy": final_strategy,
            "reasoning_trace": trace,
            "token_usage": [{"node": "self_critique", "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}],
        }
    except Exception as e:
        return {
            "critique_notes": [f"Self-critique error: {e}"],
            "final_strategy": strategy,
            "errors": [f"self_critique error: {e}"],
            "reasoning_trace": [f"self_critique failed: {e}; passing strategy through"],
        }
