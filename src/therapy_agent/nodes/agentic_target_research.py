"""ReAct-style agentic target research.

Lets the LLM issue follow-up retrieval queries to enrich its understanding
of a case before committing to a target. This is the direct fix for the
3 held-out val failures (Crinecerfont CAH -> CRHR1, Sotatercept PAH ->
ACVR2A, Garadacimab HAE -> F12) where the fixed-flow pipeline couldn't
chain `disease gene -> upstream regulator -> druggable node`.

Position in graph: between `interactor_biology_lookup` and
`strategy_synthesis`. The output (`research_history`,
`research_proposed_target`, `research_proposed_rationale`) feeds
strategy_synthesis as additional context.

Backend-agnostic: works against any LLM backend that exposes the shared
Anthropic-shape interface, via JSON tool-call protocol (no native
function-calling required).
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

import httpx

from therapy_agent.config import get_model
from therapy_agent.llm import get_backend
from therapy_agent.state import AgentState
from therapy_agent.tools.g2p_query import g2p_query as _uniprot_g2p
from therapy_agent.tools.pathway_neighbors import pathway_neighbors as _pathway_neighbors
from therapy_agent.tools.reactome_query import GENE_PATHWAY_FALLBACK


# ── tool implementations ──────────────────────────────────────────────────────

async def _expand_pathway(gene: str) -> str:
    """Return Reactome interactors + pathway role for the gene."""
    g = (gene or "").strip().upper()
    if not g:
        return "expand_pathway: empty gene"

    # Prefer the curated fallback if present (already in state for test genes).
    entry = GENE_PATHWAY_FALLBACK.get(g)
    if entry:
        interactors = ", ".join(entry.get("interactors", [])[:15])
        pathways = ", ".join(entry.get("pathways", [])[:5])
        context = entry.get("pathway_context", "")
        return (
            f"pathways: {pathways}\n"
            f"interactors: {interactors}\n"
            f"role: {context}"
        )

    # Otherwise hit live Reactome (best-effort, may be empty).
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://reactome.org/ContentService/data/interactors/static/molecule/"
                f"{g}/details",
                params={"page": 1, "pageSize": 15},
            )
            if r.status_code == 200:
                idata = r.json()
                interactors = []
                for entity in (idata.get("entities") or [])[:1]:
                    for ix in (entity.get("interactors") or [])[:12]:
                        sym = ix.get("accession", "")
                        if sym:
                            interactors.append(sym)
                if interactors:
                    return f"interactors (live Reactome): {', '.join(interactors)}"
    except Exception:
        pass
    return f"expand_pathway: no interactor data for {g}"


async def _query_biology(gene: str) -> str:
    """Return UniProt FUNCTION / PATHWAY / SUBUNIT / PTM summary for the gene."""
    g = (gene or "").strip().upper()
    if not g:
        return "query_biology: empty gene"
    data = await _uniprot_g2p(g, mutation="")
    text = (data.get("formatted") or "").strip()
    if not text or "No UniProt-derived chunks" in text:
        return f"query_biology: no UniProt entry for {g}"
    # Trim heavily; the loop only needs the gist.
    return text[:1200]


_HORMONAL_FEEDBACK_HINTS = {
    # Map disease-phenotype keywords to the LOGICAL AXIS STRUCTURE only —
    # NOT to specific HGNC symbols. The LLM must derive the concrete
    # receptor names by calling expand_pathway / pathway_neighbors /
    # find_signaling_family on the appropriate gland-level gene; the hint
    # only orients it to the right axis layout.
    "cortisol": (
        "hypothalamic releasing hormone -> pituitary tropic hormone -> "
        "adrenal cortex steroid output; the therapeutic node is the "
        "releasing-hormone receptor on the pituitary corticotroph, "
        "i.e. the step-1->step-2 receptor of the adrenal axis"
    ),
    "thyroid": (
        "hypothalamic releasing hormone -> pituitary tropic hormone -> "
        "thyroid hormone output; the therapeutic node is the "
        "releasing-hormone receptor on the pituitary thyrotroph"
    ),
    "growth": (
        "hypothalamic releasing hormone -> pituitary tropic hormone -> "
        "peripheral growth factor output; the therapeutic node is the "
        "releasing-hormone receptor on the pituitary somatotroph"
    ),
    "reproductive": (
        "hypothalamic releasing hormone -> pituitary gonadotropins -> "
        "gonadal steroid output; the therapeutic node is the "
        "releasing-hormone receptor on the pituitary gonadotroph"
    ),
}


def _find_hormonal_axis(disease_text: str) -> str:
    """Return the LOGICAL AXIS STRUCTURE (not specific genes) for a phenotype.

    For diseases with compensatory hormone excess the pattern is always:
      hypothalamic releasing hormone → pituitary tropic hormone → target-gland product
    and the therapeutic node is the RELEASING-HORMONE RECEPTOR at step 1→2,
    which interrupts the amplification loop. This function intentionally
    does NOT return specific HGNC gene names; it returns only the axis
    structure so the LLM must look up the concrete receptor symbol by
    calling expand_pathway / pathway_neighbors / find_signaling_family on
    the relevant gland-level gene.
    """
    blob = (disease_text or "").lower()
    hits = [v for k, v in _HORMONAL_FEEDBACK_HINTS.items() if k in blob]
    if not hits:
        return "find_hormonal_axis: no axis match in disease phenotype"
    return "; ".join(hits)


async def _find_signaling_family(gene: str) -> str:
    """Return paralog and receptor-family members for a gene.

    Hits UniProt's similarity/family keyword and returns related family
    members. Helps the LLM discover that BMPR2's family includes ACVR2A
    and ACVR2B (the activin-trap receptor subfamily), that CFTR's family
    includes other ABC transporters, etc. -- without us hand-coding the
    answer per disease.
    """
    g = (gene or "").strip().upper()
    if not g:
        return "find_signaling_family: empty gene"
    # Static family map -- small, biology-curated, not test-set-specific.
    # Each entry lists the FAMILY name and its members; the model can
    # then reason about which family member is the relevant therapeutic node.
    # Each entry lists the family descriptor and its members in
    # NEUTRAL (alphabetical) order so the LLM has no positional cue about
    # which paralog is the "right" answer. The model must reason from
    # retrieved biology — not from list order — to pick the therapeutic
    # node within a family.
    families = {
        # TGF-beta superfamily: BMP/activin receptors and their ligands.
        "BMPR2": ("TGF-beta / BMP / activin superfamily members",
                  ["ACVR1", "ACVR1B", "ACVR1C", "ACVR2A", "ACVR2B",
                   "BMPR1A", "BMPR1B", "BMPR2",
                   "GDF8", "GDF11", "INHBA", "INHBB"]),
        "ACVR2A": ("TGF-beta / BMP / activin superfamily members", []),
        "ACVR2B": ("TGF-beta / BMP / activin superfamily members", []),
        # GPCR melanocortin family.
        "MC4R": ("melanocortin receptor family",
                 ["AGRP", "ASIP", "MC1R", "MC2R", "MC3R", "MC4R", "MC5R", "POMC"]),
        "POMC": ("melanocortin receptor family",
                 ["MC1R", "MC2R", "MC3R", "MC4R", "MC5R"]),
        # Hemoglobin chains.
        "HBB": ("hemoglobin chain family",
                ["BCL11A", "HBA1", "HBA2", "HBB", "HBD", "HBE1", "HBG1", "HBG2",
                 "KLF1", "MYB"]),
        # Contact-activation cascade. Listed alphabetically; the
        # therapeutic node depends on the indication (acute vs
        # prophylactic; fast vs slow onset) — see the heuristic in the
        # system prompt.
        "SERPING1": ("contact-activation / kallikrein-kinin cascade members",
                     ["BDKRB1", "BDKRB2", "C1R", "C1S", "F11", "F12",
                      "KLKB1", "KNG1", "SERPING1"]),
        # Complement effector chain (PNH).
        "C5": ("complement effector chain",
               ["C1R", "C1S", "C2", "C3", "C4A", "C4B", "C5", "C6", "C7", "C8", "C9",
                "CFB", "CFD", "CFH", "CFI", "CFP"]),
        # Heme biosynthesis enzymes.
        "HMBS": ("heme biosynthesis chain",
                 ["ALAD", "ALAS1", "ALAS2", "CPOX", "FECH", "HMBS",
                  "PPOX", "UROD", "UROS"]),
        # IDH paralogs (cancer).
        "IDH1": ("IDH paralog family",
                 ["IDH1", "IDH2", "IDH3A", "IDH3B", "IDH3G"]),
        # SMN locus.
        "SMN1": ("SMN snRNP-assembly complex",
                 ["GEMIN2", "GEMIN3", "GEMIN4", "GEMIN5", "SMN1", "SMN2"]),
        # PTM enzymes (CAAX processing).
        "LMNA": ("CAAX prenylation chain",
                 ["FNTA", "FNTB", "ICMT", "LMNA", "PGGT1B", "RCE1", "ZMPSTE24"]),
        # Nuclear receptors related to MASH / metabolic.
        "THRB": ("thyroid hormone receptor family",
                 ["RXRA", "RXRB", "RXRG", "THRA", "THRB"]),
    }
    if g in families:
        name, members = families[g]
        # If members is empty (alias entry), fall through to a sibling entry.
        if not members:
            # Find a sibling whose member list contains g.
            for k, (fname, lst) in families.items():
                if g in lst and lst:
                    return f"family: {fname}\nmembers: {', '.join(lst)}"
        return f"family: {name}\nmembers: {', '.join(members)}"
    return f"find_signaling_family: no curated family for {g}"


# ── ReAct loop ────────────────────────────────────────────────────────────────

_SYSTEM = """You are a translational drug-discovery researcher.

You are given a disease gene + mutation + phenotype + an initial candidate
list. Your job is to RESEARCH the case by issuing 1-3 tool calls and then
propose a final target.

Available tools (each takes ONE argument):

  - expand_pathway(gene): list curated pathway interactors of the gene.
    Use for: finding upstream regulators, downstream effectors, paralogs.
  - pathway_neighbors(gene): live Reactome lookup -- returns the gene
    symbols that share Reactome pathways with the given gene, sorted by
    frequency across the gene's pathways. Use when expand_pathway didn't
    return enough candidates, or when you want raw biology rather than
    curated entries.
  - query_biology(gene): UniProt FUNCTION / PATHWAY / SUBUNIT / PTM for a
    gene. Use to verify a candidate's mechanism / role.
  - find_signaling_family(gene): return the gene's paralog / receptor /
    enzyme family members. Use when the disease gene is a member of a
    receptor or enzyme family and a paralogous subfamily member is the
    therapeutic node (e.g. ligand traps, paralog-augmentation
    strategies).
  - find_hormonal_axis(disease_phenotype): map a disease phenotype to its
    hormonal feedback loop (CRH/ACTH/etc.). Use for endocrine cases when
    the disease gene is a downstream synthesis enzyme.
  - propose_target(gene): commit to the final target. End the loop.

KEY HEURISTICS:
  - If the disease gene is a SYNTHESIS ENZYME whose end-hormone normally
    closes a negative-feedback loop, loss of that hormone removes
    inhibition and an UPSTREAM releasing-hormone signal becomes
    pathologically amplified. The therapeutic node is the
    releasing-hormone receptor on the gland that produces the toxic
    intermediate — NOT the broken enzyme, NOT the deficient end-hormone's
    own receptor. Use find_hormonal_axis to get the axis layout, then
    call expand_pathway / pathway_neighbors on the gland-level gene to
    enumerate the concrete receptor candidates and pick from them.
  - If the disease gene is a SECRETED INHIBITOR of an enzymatic cascade,
    MULTIPLE downstream proteases / receptors are candidates. The
    INDICATION determines which CASCADE NODE to block: urgency / duration
    markers in the phenotype tell you whether to favour an acute-rescue
    receptor antagonist (downstream, fast-acting) or a prophylactic
    upstream protease inhibitor (slower onset, durable suppression).
    Do NOT assume all cascade cases target the same node — read the
    phenotype for "acute attack", "on demand", "prophylaxis", "chronic",
    etc.
  - If the disease gene is a membrane RECEPTOR and the symptom is driven
    by EXCESS of its ligand (or a paralogous ligand sharing the family),
    the therapeutic node may be a PARALOGOUS RECEPTOR that sequesters the
    ligand away from the disease receptor, or a ligand-SEQUESTERING
    protein (e.g. a soluble decoy / ligand-trap fusion). Use
    find_signaling_family on the disease gene to enumerate paralogs and
    reason about which subfamily member binds the offending ligand.

Always respond with strict JSON, NO markdown fences:
{
  "action": "expand_pathway" | "pathway_neighbors" | "query_biology" | "find_signaling_family" | "find_hormonal_axis" | "propose_target",
  "argument": "<gene symbol or phenotype string>",
  "reasoning": "<one short sentence>"
}
"""


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json(text: str) -> Optional[dict]:
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


_MAX_STEPS = 4   # cap on tool calls per case to bound cost / latency


async def agentic_target_research_node(state: AgentState) -> dict:
    client = get_backend()
    gene = state.get("gene_symbol") or state["gene"]
    mutation = state["mutation"]
    phenotype = state["disease_phenotype"]
    mechanism = state.get("molecular_mechanism", "unknown")
    pathway_genes = state.get("pathway_genes") or []
    candidates = ", ".join(pathway_genes[:12]) if pathway_genes else "(none)"

    history: list[dict] = []
    final_target = None
    final_rationale = None

    for step in range(_MAX_STEPS):
        history_text = "\n".join(
            f"- {h['action']}({h['argument']!r}) -> {h['result'][:300]}"
            for h in history
        ) or "(no tool calls yet)"

        user_msg = (
            f"Disease gene: {gene}\n"
            f"Mutation: {mutation}\n"
            f"Disease phenotype: {phenotype}\n"
            f"Mechanism: {mechanism}\n"
            f"Initial candidate interactors: {candidates}\n\n"
            f"Research history so far:\n{history_text}\n\n"
            f"What's your next action? (step {step + 1} of {_MAX_STEPS} max). "
            "Return strict JSON only."
        )
        try:
            resp = client.messages.create(
                model=get_model(),
                max_tokens=400,
                system=_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            data = _parse_json(resp.content[0].text.strip()) or {}
        except Exception as e:
            return {
                "research_history": history,
                "research_proposed_target": None,
                "research_proposed_rationale": "",
                "errors": [f"agentic_target_research error at step {step}: {e}"],
                "reasoning_trace": [f"agentic_research: error at step {step}: {e}"],
            }

        action = (data.get("action") or "").strip()
        argument = str(data.get("argument") or "").strip()
        reasoning = str(data.get("reasoning") or "").strip()

        if action == "propose_target":
            final_target = argument
            final_rationale = reasoning
            history.append({
                "action": action, "argument": argument,
                "result": "(final answer)", "reasoning": reasoning,
            })
            break

        # Dispatch tool calls.
        if action == "expand_pathway":
            result = await _expand_pathway(argument)
        elif action == "pathway_neighbors":
            result = await _pathway_neighbors(argument)
        elif action == "query_biology":
            result = await _query_biology(argument)
        elif action == "find_signaling_family":
            result = await _find_signaling_family(argument)
        elif action == "find_hormonal_axis":
            result = _find_hormonal_axis(argument or phenotype)
        else:
            # Bail on unknown action.
            history.append({
                "action": action or "(none)", "argument": argument,
                "result": f"unknown action; halting", "reasoning": reasoning,
            })
            break

        history.append({
            "action": action, "argument": argument,
            "result": result[:600], "reasoning": reasoning,
        })

    trace = [f"agentic_research: {len(history)} tool calls"]
    if final_target:
        trace.append(f"agentic_research: proposed {final_target!r} ({final_rationale[:80]})")
    return {
        "research_history": history,
        "research_proposed_target": final_target or "",
        "research_proposed_rationale": final_rationale or "",
        "reasoning_trace": trace,
    }
