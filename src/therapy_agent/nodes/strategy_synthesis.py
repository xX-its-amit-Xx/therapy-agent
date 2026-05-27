import asyncio, os, json, re
from typing import Any, Optional
from therapy_agent.state import AgentState
from therapy_agent.llm import get_backend


from therapy_agent.config import get_model


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _robust_json_parse(text: str) -> Optional[dict]:
    """Try several parsing strategies; return the first dict we can build."""
    candidates: list[str] = [text.strip()]
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
    # Last-ditch: strip trailing commas before braces/brackets, which is the
    # most common Llama JSON error.
    repaired = re.sub(r",(\s*[}\]])", r"\1", text)
    m2 = _JSON_BLOCK_RE.search(repaired)
    if m2:
        try:
            obj = json.loads(m2.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _get_client():
    return get_backend()


def _get_model() -> str:
    return get_model()


SYSTEM = """You are an expert translational medicine scientist specializing in rare-disease drug discovery.

Given:
- A gene with a defined molecular mechanism
- Downstream pathway proteins
- Druggable candidates from ChEMBL/DrugBank

Synthesize a therapeutic strategy following this reasoning framework. The
ONLY information available about the disease is in the user message —
use the pathway/PTM/interactor evidence there and the categorical
patterns below to choose a target. Do not assume the answer.

MECHANISM-TO-STRATEGY PATTERNS (categorical, no test-case names):

  1. LoF of an INHIBITORY protein (a brake on a protease, a tumor
     suppressor, a regulator of a signalling cascade):
       → Target the enzyme or effector the inhibitor normally suppresses.
       → The disease gene is not the target — the unbraked downstream
         protein is.

  2. LoF of a STRUCTURAL or TRANSPORT protein with a functional paralog:
       → Augment the paralog: splice modulation, ASO, gene therapy,
         transcriptional upregulation.
       → The target is the paralog, not the broken gene, when a paralog
         exists and is silenced/under-expressed in adults.

  3. LoF of an ENZYME in a biosynthetic pathway, causing UPSTREAM
     SUBSTRATE accumulation that is itself toxic:
       → Knock down the rate-limiting UPSTREAM enzyme (often the first
         committed step) to drain flux into the pathway.
       → The target is upstream, not the broken gene.

  4. PROTEIN MISFOLDING / ER RETENTION:
       → Three sub-strategies, ranked by retrieved evidence:
         (a) pharmacological CHAPERONE to refold the mutant protein
             (target = the mutant protein itself; works when there are
             amenable residues)
         (b) modulate the CARGO RECEPTOR that retains the misfolded
             protein in the ER (target = the cargo receptor, often a
             TMED-family p24 protein) to redirect mutant protein to
             lysosomal degradation
         (c) proteostasis enhancers
       → Use the retrieved SUBUNIT/INTERACTOR/PTM evidence to choose.

  5. GOF or TOXIC GAIN (mutant protein gains a damaging activity):
       → Knock down the disease gene's mRNA (ASO or siRNA), or block
         the toxic downstream interaction.
       → The target IS the disease gene's mRNA in this case.

  6. PERMANENT POST-TRANSLATIONAL MODIFICATION causing toxicity:
       → If retrieved PTM/LIPIDATION evidence names a SPECIFIC
         transferase that performs the modification (e.g. a
         farnesyltransferase, prenyltransferase, palmitoyltransferase),
         that transferase is the target — not the disease gene.

  7. SPLICING DEFECT amenable to exon skipping or splice modulation:
       → ASO that restores the reading frame, or splice-switching ASO
         that promotes inclusion of a normally-skipped exon.

  8. TRANSCRIPTIONAL REPRESSOR controls a paralog that could
     compensate (e.g. fetal vs adult isoform):
       → Target the repressor's binding element to reactivate the
         paralog.

REASONING DISCIPLINE:
  - Read the retrieved pathway/PTM/interactor evidence FIRST.
  - Identify the dominant mechanism category from the rules above.
  - Within that category, name the SPECIFIC molecular target. Avoid
    proposing "the disease gene itself" unless category 5 or 7 applies.
  - When multiple categories are plausible, propose the one most
    consistent with the retrieved evidence and lower the confidence.

OUTPUT — strict JSON, no markdown fences:

{
  "target_protein": "HGNC symbol or descriptive name (e.g. 'PCSK9 (mRNA)', 'BCL11A erythroid enhancer')",
  "target_pathway": "name of the pathway the target sits in",
  "modulation_type": "inhibitor | activator | chaperone | siRNA | ASO | gene_therapy | crispr | modulator | replacement",
  "supporting_evidence": ["claim 1", "claim 2", "claim 3"],
  "precedent_drugs": ["any precedent compounds named in retrieved evidence (may be empty)"],
  "confidence_score": 0.0,
  "citations": ["citations grounded in the retrieved evidence"],
  "rationale": "1-3 sentence explanation of why this target follows from the mechanism category and the retrieved evidence"
}

If evidence is weak, lower confidence_score and say so in rationale."""


async def strategy_synthesis_node(state: AgentState) -> dict:
    client = _get_client()

    gene = state.get("gene_symbol") or state["gene"]
    mechanism = state.get("molecular_mechanism", "unknown")
    mechanism_reasoning = state.get("mechanism_reasoning", "")
    pathway_genes = state.get("pathway_genes") or []
    pathway_context_oneliner = (state.get("pathway_context") or "").strip()
    candidate_targets = state.get("candidate_targets") or []
    g2p_data = state.get("g2p_data") or {}
    interactor_g2p = state.get("interactor_g2p_data") or {}
    phenotype = state["disease_phenotype"]
    mutation = state["mutation"]
    retry = state.get("retry_count", 0)

    # Render candidate-target druggability without ANY specific drug names.
    # Reads the v0.3 schema (n_active_compounds via chembl_n_active) — not
    # the old chembl_compounds / drugbank_drugs which were silently always
    # empty lists in the previous version.
    sanitized_targets = []
    for ct in candidate_targets[:10]:
        sanitized_targets.append({
            "gene_name": ct.get("gene_name", ""),
            "druggable": True,
            "chembl_active_compounds_n": int(ct.get("chembl_n_active", 0) or 0),
            "chembl_target_id": ct.get("chembl_target_id"),
        })
    targets_text = json.dumps(sanitized_targets, indent=2) if sanitized_targets else "No druggable targets found"
    pathway_text = ", ".join(pathway_genes[:20]) if pathway_genes else "Not found"
    g2p_text = (g2p_data.get("formatted") if isinstance(g2p_data, dict) else "") or "No g2p-rag chunks retrieved for the disease gene."

    # Render g2p-rag biology for the top candidate interactors so the LLM
    # can compare biology across plausible targets rather than picking
    # blind from a list of gene symbols. Capped per case to keep tokens
    # bounded.
    if interactor_g2p:
        interactor_blocks = []
        for g, text in list(interactor_g2p.items())[:5]:
            interactor_blocks.append(f"--- candidate {g} ---\n{text}")
        interactor_text = "\n\n".join(interactor_blocks)
    else:
        interactor_text = "No interactor biology retrieved."

    # On retry, add critique context
    critique_ctx = ""
    if retry > 0 and state.get("critique_notes"):
        critique_ctx = f"\n\nPREVIOUS CRITIQUE FEEDBACK (fix these issues):\n" + "\n".join(state["critique_notes"])

    user_msg = f"""Synthesize a therapeutic strategy for:
Gene: {gene}
Mutation: {mutation}
Disease: {phenotype}
Molecular mechanism: {mechanism}
Mechanism reasoning: {mechanism_reasoning}

Pathway role of disease gene (one-liner):
{pathway_context_oneliner or "(none retrieved)"}

Pathway interactors / members (Reactome):
{pathway_text}

g2p-rag biology chunks for the DISEASE GENE (UniProt FUNCTION / PATHWAY /
SUBUNIT / PTM / LIPIDATION / DISEASE):
{g2p_text}

g2p-rag biology chunks for the TOP CANDIDATE INTERACTORS — compare these
against the disease-gene biology above when picking the target:
{interactor_text}

Druggability of candidate targets (booleans + active-compound counts from
ChEMBL human SINGLE PROTEIN targets only; NO specific drug names):
{targets_text}
{critique_ctx}

REASONING DISCIPLINE — apply IN ORDER:

  STEP 1 — Which mechanism-to-strategy pattern applies? Pick ONE of
  patterns 1-8 and name it. Use the molecular mechanism (lof / gof /
  dominant_negative / misfolding), the mutation text, AND the disease
  phenotype together. Quick triage table:

      Mutation contains "out-of-frame", "exon deletion", or
      "frameshift...amenable to exon skipping":
          → pattern 7 (exon-skipping ASO). Target = disease gene
            (an adjacent exon is skipped to restore reading frame).

      Mechanism is dominant_negative or gof AND the mutant protein
      aggregates / polymerizes / has new toxic activity:
          → pattern 5 (knock down mRNA). Target = disease gene's mRNA.

      Mechanism is gof in a regulatory enzyme that degrades or
      modifies another protein, causing the symptom:
          → pattern 5 (knock down regulator mRNA) OR pattern 1-style
            (block the regulator-substrate interaction). Target =
            disease gene (the regulatory enzyme itself).

      Mechanism is misfolding of an enzyme amenable to refolding:
          → pattern 4(a) chaperone. Target = disease gene.

      Mechanism is misfolding without a refoldable conformer:
          → pattern 4(b) cargo receptor modulation. Target =
            cargo receptor protein (not disease gene).

      Mechanism is lof in an inhibitor / brake whose downstream
      effector drives the symptom:
          → pattern 1. Target = downstream effector enzyme (not the
            disease gene).

      Mechanism is lof in a hormone precursor whose downstream
      receptor still works:
          → variant of pattern 1. Target = downstream RECEPTOR with an
            agonist that bypasses the missing ligand.

      Mechanism is lof in a structural / transport protein with a
      silent functional paralog:
          → pattern 2. Target = the silent paralog (often via splice
            modulation or augmentation).

      Mechanism is lof in an enzyme downstream of a rate-limiting step
      whose upstream substrate accumulates and is toxic:
          → pattern 3. Target = the upstream rate-limiting enzyme
            (not the disease gene).

  STEP 2 — Identify the THERAPEUTIC TARGET — the protein, RNA, or
  genomic element a drug would BIND OR MODIFY to correct the disease.
  Match your chosen pattern to one of the columns below:

    TARGET = DISEASE GENE ITSELF when:
      * Toxic gain-of-function or dominant-negative protein that
        aggregates / polymerizes / has new toxic activity → siRNA or
        ASO knockdown of the disease gene's mRNA.
      * Out-of-frame deletion or amenable splice mutation → ASO
        targeting an exon of the disease gene to restore reading
        frame.
      * Stable but mis-conforming enzyme or receptor with druggable
        surface (amenable missense) → pharmacological chaperone
        binding the disease gene's own protein.
      * Aggregation-prone or polymerization-prone protein with a
        small-molecule stabilizer binding-site → small-molecule
        stabilizer of the disease gene's protein.

    TARGET = A DIFFERENT PROTEIN / RNA / element when:
      * LoF of an inhibitor whose downstream effector is the symptom
        driver → target the unbraked DOWNSTREAM EFFECTOR enzyme.
      * LoF of a structural / transport protein with a silent or
        under-expressed PARALOG → target the paralog (or its splicing).
      * Downstream enzyme LoF with toxic UPSTREAM substrate buildup
        → target the rate-limiting UPSTREAM enzyme.
      * Misfolding + ER retention via a cargo receptor, WHERE the
        misfolded mutant has no refoldable conformer and chaperones
        do not help → target the CARGO RECEPTOR (e.g. a TMED-family
        p24 protein).
      * LoF of a hormone precursor with a downstream RECEPTOR axis
        → target the downstream RECEPTOR with an agonist that
        bypasses the missing ligand.
      * Permanent toxic PTM → target the MODIFYING ENZYME.
      * Compensatory PARALOG silenced by a REPRESSOR → target the
        repressor or its DNA element.

  Write that protein's name into the `target_protein` field. Do not
  default to the disease gene if the case clearly matches one of the
  "different protein" rows. Do not default to a downstream partner if
  the case clearly matches the "disease gene itself" rows.

  STEP 3 — Sanity-check the precedent_drugs and citations against your
  internal knowledge. If you cannot name a real approved drug or
  reference that targets your chosen protein, leave those fields empty
  rather than confabulate. Do NOT invent drug-target attributions
  (e.g. do not claim a replacement therapy "targets" the protein it
  replaces).

Return ONLY valid JSON matching the strategy schema."""

    try:
        response = client.messages.create(
            model=_get_model(),
            max_tokens=1500,
            system=SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        data = _robust_json_parse(text)
        if data is None:
            raise ValueError("Could not parse a JSON object from the LLM output.")

        strategy = {
            "target_protein": data.get("target_protein", "Unknown"),
            "target_pathway": data.get("target_pathway", "Unknown"),
            "modulation_type": data.get("modulation_type", "inhibitor"),
            "supporting_evidence": data.get("supporting_evidence", []),
            "precedent_drugs": data.get("precedent_drugs", []),
            "confidence_score": float(data.get("confidence_score", 0.5)),
            "citations": data.get("citations", []),
            "rationale": data.get("rationale", ""),
        }

        new_citations = data.get("citations", [])

        return {
            "strategy": strategy,
            "reasoning_trace": [f"Strategy: target={strategy['target_protein']}, modulation={strategy['modulation_type']}, confidence={strategy['confidence_score']:.2f}"],
            "citations": new_citations,
            "retry_count": retry + 1,
            "token_usage": [{"node": "strategy_synthesis", "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}],
        }
    except Exception as e:
        return {
            "strategy": None,
            "errors": [f"strategy_synthesis error: {e}"],
            "reasoning_trace": [f"strategy_synthesis failed: {e}"],
            "retry_count": retry + 1,
        }
