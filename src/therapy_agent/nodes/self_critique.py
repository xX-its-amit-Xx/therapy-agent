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

Your primary job is to check ALIGNMENT between the rationale and the
target_protein. A common failure mode is the rationale correctly
identifies a target X (e.g. "TMED9 is the cargo receptor that retains
mutant UMOD") but the target_protein field says something different
(e.g. UMOD itself). When that happens, FIX the target_protein to match
the rationale's conclusion.

Also check:
1. Are cited drugs / trials real and correctly attributed?
2. Is the target-mechanism logic sound?
3. Are there unsupported claims?
4. What alternative targets should be considered?

Return JSON:
{
  "verdict": "accept" | "revise",
  "target_rationale_aligned": true | false,
  "corrected_target_protein": "...",      // if not aligned, the gene the rationale actually argues for; else copy original
  "confidence_adjustment": float,
  "critique_notes": [list of specific issues],
  "unsupported_claims": [list of specific claims lacking evidence],
  "alternative_targets": [list of reasonable alternatives],
  "revised_confidence": float
}

Be honest. If the strategy is well-supported, say so. If it confabulates
drugs or trials, flag it. If the target_protein doesn't match the
rationale, REVISE with corrected_target_protein."""


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
        notes = data.get("critique_notes", []) or []
        unsupported = data.get("unsupported_claims", []) or []
        alternatives = data.get("alternative_targets", []) or []
        aligned = bool(data.get("target_rationale_aligned", True))
        corrected = (data.get("corrected_target_protein") or "").strip()

        # Build final strategy
        final_strategy = dict(strategy)
        final_strategy["confidence_score"] = revised_conf

        # v0.5 used to rewrite target_protein here based on the LLM critique's
        # `corrected_target_protein` output. v0.7 disabled this because the
        # critique LLM was systematically pulling the answer back toward the
        # disease gene -- e.g. on Sotatercept, the deterministic
        # field_rationale_align node had correctly set target=ACVR2B, but
        # self_critique then "realigned" it back to BMPR2. Target rewriting
        # is now exclusively done by `field_rationale_align`, which uses a
        # deterministic check and is therefore not subject to LLM-prior
        # collapse. Self_critique here just records confidence + notes.
        if not aligned and corrected:
            notes.append(
                f"LLM critique flagged misalignment to {corrected!r}; "
                "deferring to deterministic field_rationale_align for any "
                "actual target_protein rewrite (this note is informational)"
            )

        if unsupported:
            notes += [f"UNSUPPORTED: {c}" for c in unsupported]

        trace = [
            f"Critique verdict: {verdict} (confidence adj: {adj:+.2f} -> {revised_conf:.2f})",
            f"Target/rationale aligned: {aligned}" + (f"; corrected to {corrected}" if not aligned and corrected else ""),
            f"Critique notes: {'; '.join(notes[:3])}",
        ]
        if alternatives:
            trace.append(f"Alternative targets: {', '.join(alternatives[:3])}")

        # Preserve `strategy.target_protein` exactly as field_rationale_align
        # set it. Only update confidence + carry through final_strategy.
        revised_strategy = dict(strategy)
        revised_strategy["confidence_score"] = revised_conf

        return {
            "strategy": revised_strategy,
            "final_strategy": final_strategy,
            "critique_notes": notes,
            # Mark that we've done one critique pass so should_revise won't
            # loop forever; the revise-loop cap (retry_count < 2) is the
            # secondary guard.
            "critique_pass_done": True,
            "reasoning_trace": trace,
            "token_usage": [{"node": "self_critique", "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}],
        }
    except Exception as e:
        return {
            "critique_notes": [f"Self-critique error: {e}"],
            "final_strategy": strategy,
            "critique_pass_done": True,
            "errors": [f"self_critique error: {e}"],
            "reasoning_trace": [f"self_critique failed: {e}; passing strategy through"],
        }
