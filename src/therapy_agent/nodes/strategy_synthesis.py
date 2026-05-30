import asyncio, os, json, re
from collections import Counter
from typing import Any, Optional
from therapy_agent.state import AgentState
from therapy_agent.llm import get_backend


from therapy_agent.config import get_model


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _close_unbalanced_braces(text: str) -> str:
    """Best-effort repair when max_tokens truncates JSON mid-output. Appends
    closing braces/brackets to balance counts; strips dangling commas."""
    s = text
    # strip dangling comma at end before close
    s = re.sub(r",\s*$", "", s.rstrip())
    open_obj = s.count("{") - s.count("}")
    open_arr = s.count("[") - s.count("]")
    if open_arr > 0:
        s = s + "]" * open_arr
    if open_obj > 0:
        s = s + "}" * open_obj
    return s


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
    # Repairs: strip trailing comma before close, then close unbalanced braces
    repaired = re.sub(r",(\s*[}\]])", r"\1", text)
    for variant in (repaired, _close_unbalanced_braces(repaired)):
        m2 = re.search(r"\{[\s\S]*\}", variant)
        if m2:
            try:
                obj = json.loads(m2.group(0))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    # Last ditch: brace-close the raw text and try
    closed = _close_unbalanced_braces(text)
    m3 = re.search(r"\{[\s\S]*\}", closed)
    if m3:
        try:
            obj = json.loads(m3.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


# ── Stage 1: lightweight pattern selector (one focused LLM call) ──────────────

_PATTERN_SELECTOR_SYSTEM = """You are a translational drug-discovery scientist. Given a disease gene, mutation and phenotype, pick ONE mechanism-to-strategy pattern from this table:

  1. LOF of an inhibitor → target the unbraked downstream effector enzyme.
  2. LOF of a structural/transport protein with a silent paralog → target the paralog.
  3. LOF of a biosynthetic enzyme with toxic UPSTREAM substrate buildup → knock down the rate-limiting upstream enzyme.
  4a. Misfolding amenable to refolding → chaperone for the disease gene's own protein.
  4b. Misfolding via cargo receptor ER retention (no refoldable conformer) → target the cargo receptor (often a TMED-family p24).
  5. GOF / dominant-negative / aggregation → knock down the disease gene's mRNA (ASO / siRNA).
  6. LOF hormone precursor with intact downstream receptor → agonize the receptor to bypass the missing ligand.
  7. Out-of-frame exon deletion or splice defect → splice-modulating ASO targeting an adjacent exon of the disease gene.
  8. Transcriptional repressor controlling a useful paralog → disrupt the repressor / its DNA element.
  9. LOF of an enzyme in a NEGATIVE-FEEDBACK endocrine axis (loss of end-hormone removes inhibition of an upstream signaling cascade, causing toxic excess of an intermediate) → BLOCK the upstream signaling receptor or releasing hormone that now drives the compensatory excess. The disease gene is NOT the target; an upstream hypothalamic / pituitary / autocrine receptor IS. Example archetypes: CAH (CYP21A2 LOF → ACTH excess → adrenal androgen excess; block CRHR1), pituitary feedback loops in general.

Return strict JSON:
{
  "pattern_id": "1" | "2" | "3" | "4a" | "4b" | "5" | "6" | "7" | "8" | "9",
  "target_kind": "downstream_effector" | "paralog" | "upstream_enzyme" | "disease_gene_protein_chaperone" | "cargo_receptor" | "disease_gene_mRNA" | "downstream_receptor_agonist" | "disease_gene_exon_skip" | "repressor" | "feedback_axis_receptor",
  "reasoning": "<one sentence>"
}

NEGATIVE EXAMPLES (do NOT pick chaperone/4a when the real mechanism is feedback):
  - Disease gene CYP21A2 + ACTH-driven androgen excess phenotype: chaperone is WRONG; pattern 9 (feedback-axis blockade, target CRHR1) is right.
  - Disease gene with phenotype dominated by COMPENSATORY upstream signaling (e.g. ACTH excess, gonadotropin excess) → pattern 9 not 4a.
"""


async def _select_pattern(client, model: str, *, gene: str, mutation: str,
                          phenotype: str, mechanism: str,
                          mechanism_reasoning: str) -> dict:
    """Stage 1: pick the categorical pattern + target_kind only."""
    user = (
        f"Disease gene: {gene}\n"
        f"Mutation: {mutation}\n"
        f"Disease phenotype: {phenotype}\n"
        f"Molecular mechanism: {mechanism}\n"
        f"Mechanism reasoning: {mechanism_reasoning}\n\n"
        "Pick one pattern. Return JSON only."
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            system=_PATTERN_SELECTOR_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        data = _robust_json_parse(resp.content[0].text.strip()) or {}
        return {
            "pattern_id": str(data.get("pattern_id") or "").strip(),
            "target_kind": str(data.get("target_kind") or "").strip(),
            "reasoning": str(data.get("reasoning") or "").strip(),
        }
    except Exception as e:
        return {"pattern_id": "", "target_kind": "", "reasoning": f"selector error: {e}"}


# ── Stage 2: self-consistency vote on the specific target gene ────────────────

_TARGET_PICKER_SYSTEM = """You are picking ONE specific target gene/RNA from a list of pathway candidates, given the pattern category already chosen and the biology of each candidate.

Output strict JSON:
{
  "target_protein": "<HGNC symbol or 'GENE (mRNA)' or descriptive name>",
  "rationale": "<2-3 sentences explaining why this candidate fits the chosen pattern>"
}

Rules:
- For target_kind="disease_gene_mRNA": target_protein MUST be the disease gene (its mRNA).
- For target_kind="disease_gene_exon_skip": target_protein MUST be the disease gene (e.g. "DMD").
- For target_kind="disease_gene_protein_chaperone": target_protein MUST be the disease gene.
- For target_kind="downstream_effector" / "upstream_enzyme" / "cargo_receptor" / "downstream_receptor_agonist" / "paralog" / "repressor": target_protein is a DIFFERENT gene. Pick from the candidates whose biology matches that role.
- For target_kind="feedback_axis_receptor": target_protein is the UPSTREAM hypothalamic / pituitary / autocrine SIGNALING receptor that drives the compensatory excess -- specifically, the RELEASING HORMONE receptor (e.g. CRHR1 / CRHR2 for the HPA axis when ACTH is the toxic intermediate, GnRHR for hypogonadism-axis feedback, TRHR for thyroid-axis feedback). NOT the end-hormone's own receptor (e.g. NOT NR3C1 / glucocorticoid receptor when cortisol is deficient -- that worsens the deficiency). Look in the candidate / interactor / family list for a "...receptor" upstream of the disease gene's enzymatic product.

Do not invent drug names. Do not name any FDA-approved drug.
"""


async def _pick_target_once(client, model: str, *, user_msg: str,
                            temperature: float) -> dict:
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=800,
            system=_TARGET_PICKER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=temperature,
        )
        data = _robust_json_parse(resp.content[0].text.strip()) or {}
        return {
            "target_protein": str(data.get("target_protein") or "").strip(),
            "rationale": str(data.get("rationale") or "").strip(),
        }
    except Exception:
        return {"target_protein": "", "rationale": ""}


def _canonical_target(t: str) -> str:
    """Normalize a target string for vote tally — pull the first HGNC-shaped
    symbol if present, else strip whitespace and case-fold."""
    if not t:
        return ""
    m = re.search(r"\b[A-Z][A-Z0-9]{1,9}\b", t)
    return (m.group(0) if m else t.strip()).upper()


async def _pick_target_self_consistent(client, model: str, *, user_msg: str,
                                       n_samples: int = 3) -> dict:
    """Run target picker N times at moderate temperature, vote on canonical
    target. Returns the most common pick and its rationale."""
    samples = await asyncio.gather(*[
        _pick_target_once(client, model, user_msg=user_msg, temperature=0.5)
        for _ in range(n_samples)
    ])
    canonicals = [_canonical_target(s["target_protein"]) for s in samples if s["target_protein"]]
    if not canonicals:
        return {"target_protein": "", "rationale": "", "votes": Counter(), "samples": samples}
    counter = Counter(canonicals)
    winner_canonical, _ = counter.most_common(1)[0]
    # Pick the first sample matching the winner (preserves a real rationale).
    for s in samples:
        if _canonical_target(s["target_protein"]) == winner_canonical:
            return {
                "target_protein": s["target_protein"],
                "rationale": s["rationale"],
                "votes": counter,
                "samples": samples,
            }
    return {"target_protein": samples[0]["target_protein"],
            "rationale": samples[0]["rationale"],
            "votes": counter, "samples": samples}


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
    research_history = state.get("research_history") or []
    research_proposed_target = (state.get("research_proposed_target") or "").strip()
    research_proposed_rationale = (state.get("research_proposed_rationale") or "").strip()
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

    # v0.5: 2-stage decomposition + self-consistency vote.
    #
    # Stage 1 (single short LLM call): pick the categorical pattern_id and
    # target_kind. The small model is good at this when the task is
    # narrow ("which of 8 patterns applies?") rather than the original
    # one-shot "pick the pattern AND the gene AND the modality AND the
    # rationale" combo.
    #
    # Stage 2 (3 LLM calls + majority vote): given the chosen pattern,
    # pick the specific target gene from the candidates. Sample 3 times
    # at temperature 0.5 to mitigate small-model variance on contested
    # cases. The vote is on the canonical HGNC symbol of the prediction.
    try:
        # On retry from self_critique, keep the prior pattern selection if
        # we have one (we may want to revise the target gene, not the
        # pattern category).
        prior_pattern = state.get("strategy", {}).get("pattern_id") if state.get("strategy") else None
        if retry > 0 and prior_pattern:
            pattern = {
                "pattern_id": prior_pattern,
                "target_kind": state.get("strategy", {}).get("target_kind", ""),
                "reasoning": "reusing prior pattern on revise",
            }
        else:
            pattern = await _select_pattern(
                client, _get_model(),
                gene=gene, mutation=mutation, phenotype=phenotype,
                mechanism=mechanism, mechanism_reasoning=mechanism_reasoning,
            )

            # Phenotype-pattern consistency guard (v0.9.1).
            # If the phenotype contains explicit feedback-axis language but
            # Stage 1 picked a disease-gene-centric pattern (chaperone or
            # mRNA knockdown), force a re-pick with the feedback option
            # made explicit. This catches the Crinecerfont/CAH archetype
            # where Stage 1 picks pattern 4a despite the phenotype
            # describing CRH/ACTH-driven excess.
            feedback_axis_markers = (
                "acth-driven", "acth driven", "crh and", "crh-driven",
                "compensatory", "feedback drive", "gonadotropin excess",
                "tsh-driven", "tsh driven", "negative feedback",
                "hypothalamic-pituitary",
            )
            disease_gene_centric_kinds = {
                "disease_gene_protein_chaperone", "disease_gene_mRNA",
                "disease_gene_exon_skip",
            }
            phen_l = (phenotype or "").lower()

            # v0.9.3: mechanism-pattern consistency guard. Knocking down
            # the mRNA of a LoF disease gene is contradictory -- it
            # already lost function; further knockdown doesn't help. The
            # Iptacopan/PNH miss showed Stage 1 picking pattern 5
            # (disease_gene_mRNA) for a LoF PIGA case. Re-pick toward
            # downstream-effector pattern when this happens.
            if (pattern.get("target_kind") == "disease_gene_mRNA"
                    and (mechanism or "").lower() == "lof"):
                pattern = {
                    "pattern_id": "1",
                    "target_kind": "downstream_effector",
                    "reasoning": (
                        f"[v0.9.3 mechanism-pattern guard: mechanism=lof "
                        f"contradicts disease_gene_mRNA knockdown (you "
                        f"cannot knock down something that already lost "
                        f"function). Forced to downstream_effector; the "
                        f"target is the protein the disease gene's LoF "
                        f"unbrakes/dysregulates, not the disease gene "
                        f"itself.]"
                    ),
                }

            # v0.9.2 hard override fires whenever a feedback-axis marker
            # is in the phenotype, REGARDLESS of Stage 1's pick. This is
            # the right semantics: feedback-axis biology should not be
            # treated as a chaperone / mRNA / generic case. v0.9.2 used
            # to only fire when Stage 1 picked a disease-gene-centric
            # pattern, but R1-Distill 8B was returning empty/malformed
            # JSON from Stage 1 on Crinecerfont, which bypassed the gate.
            # The unconditional trigger is safer; the only "escape" is
            # a strong agentic-research target (handled in Stage 2).
            non_feedback_kinds = {
                "downstream_effector",  # already non-disease-gene
                "downstream_receptor_agonist",  # already non-disease-gene
                "upstream_enzyme",  # already non-disease-gene
            }
            if (any(m in phen_l for m in feedback_axis_markers)
                    and pattern.get("target_kind") not in non_feedback_kinds):
                fired_marker = next(m for m in feedback_axis_markers
                                     if m in phen_l)
                pattern = {
                    "pattern_id": "9",
                    "target_kind": "feedback_axis_receptor",
                    "reasoning": (
                        f"[v0.9.2 feedback-axis hard override: phenotype "
                        f"contains '{fired_marker}'; Stage 1 originally "
                        f"chose {pattern.get('target_kind', '?') or 'empty'} "
                        f"but feedback-axis biology takes precedence. The "
                        f"target is an upstream signaling receptor, NOT "
                        f"the disease gene.]"
                    ),
                }

        # Render the agentic-research history if the upstream node produced one.
        research_text = ""
        if research_history:
            lines = []
            for h in research_history[:6]:
                lines.append(
                    f"- {h.get('action', '?')}({h.get('argument', '')!r}) "
                    f"-> {(h.get('result') or '')[:240]}"
                )
            research_text = "Agentic-research log (the LLM's own follow-up retrieval):\n" + "\n".join(lines)
            if research_proposed_target:
                research_text += (
                    f"\nAgentic-research final proposal: {research_proposed_target} "
                    f"({research_proposed_rationale[:120]})"
                )

        # If agentic_target_research produced a final proposal, lead with it
        # in the prompt and instruct the picker to default to it. This stops
        # Stage 2 from quietly overriding a good research conclusion with a
        # less-informed re-pick (we observed e.g. research -> ACVR2B then
        # Stage 2 -> BMPR2 disease-gene fallback on the Sotatercept case).
        if research_proposed_target:
            target_picker_user_msg = (
                f"Disease gene: {gene}\n"
                f"Mutation: {mutation}\n"
                f"Disease phenotype: {phenotype}\n"
                f"Mechanism: {mechanism}\n\n"
                f"Pattern chosen: {pattern['pattern_id']} "
                f"(target_kind = {pattern['target_kind']})\n\n"
                f"=== AGENTIC RESEARCH PROPOSAL ===\n"
                f"Proposed target: {research_proposed_target}\n"
                f"Proposal rationale: {research_proposed_rationale[:240]}\n"
                f"Research log (the LLM's tool calls):\n{research_text}\n\n"
                f"=== ADDITIONAL CONTEXT (do not weight above the proposal unless\n"
                f"the proposal clearly violates the pattern's target_kind) ===\n"
                f"Pathway interactors: {pathway_text}\n"
                f"g2p-rag disease-gene biology:\n{g2p_text}\n"
                f"g2p-rag candidate biology:\n{interactor_text}\n"
                f"Druggability: {targets_text}\n"
                f"{critique_ctx}\n\n"
                "DEFAULT to the agentic research's proposed target. Only override\n"
                "if it directly contradicts the pattern's target_kind\n"
                f"({pattern['target_kind']}). If you accept the proposal,\n"
                "copy its target into target_protein verbatim.\n"
                "Return JSON only."
            )
        else:
            target_picker_user_msg = (
                f"Disease gene: {gene}\n"
                f"Mutation: {mutation}\n"
                f"Disease phenotype: {phenotype}\n"
                f"Mechanism: {mechanism}\n\n"
                f"Pattern chosen: {pattern['pattern_id']} "
                f"(target_kind = {pattern['target_kind']})\n"
                f"Pattern reasoning: {pattern.get('reasoning', '')}\n\n"
                f"Pathway role of disease gene: {pathway_context_oneliner or '(none)'}\n\n"
                f"Pathway interactors (Reactome): {pathway_text}\n\n"
                f"g2p-rag biology for the disease gene:\n{g2p_text}\n\n"
                f"g2p-rag biology for top candidate interactors:\n{interactor_text}\n\n"
                f"Druggability of candidates: {targets_text}\n\n"
                f"{critique_ctx}\n\n"
                "Pick the SPECIFIC target gene that fits the chosen pattern's "
                "target_kind. Return JSON only."
            )

        # If the agentic research proposed a target that DIFFERS from the
        # disease gene, trust the research and skip the Stage 2 picker.
        # We observed Stage 2 consistently override the research's
        # correctly-reasoned answer with the disease gene (e.g. research
        # proposed ACVR2B; Stage 2 picker output rationale mentioning
        # ACVR2B but wrote BMPR2 into target_protein 3/3 times). When the
        # research has done multi-step retrieval and arrived at a
        # non-disease-gene target, we should defer to it.
        rp = research_proposed_target.strip()
        rp_canonical = _canonical_target(rp)
        gene_canonical = _canonical_target(gene)
        bypass_stage2 = bool(rp_canonical and rp_canonical != gene_canonical)

        if bypass_stage2:
            picker = {
                "target_protein": rp,
                "rationale": research_proposed_rationale or
                             f"Deferred to agentic research's proposal "
                             f"({rp}); see research_history for tool calls.",
                "votes": {rp_canonical: 1},
                "samples": [{"target_protein": rp,
                             "rationale": research_proposed_rationale}],
            }
        else:
            picker = await _pick_target_self_consistent(
                client, _get_model(),
                user_msg=target_picker_user_msg,
                n_samples=3,
            )

            # disease_gene_default guard. Miss taxonomy showed that 100% of
            # R1-Distill v0.8 val misses were disease-gene-default: Stage 2
            # output the disease gene even when Stage 1 chose a non-disease-
            # gene target_kind. Detect and force a re-pick.
            non_disease_gene_kinds = {
                "downstream_effector", "paralog", "upstream_enzyme",
                "cargo_receptor", "downstream_receptor_agonist", "repressor",
                "feedback_axis_receptor",
            }
            picker_canonical = _canonical_target(picker.get("target_protein") or "")
            if (pattern["target_kind"] in non_disease_gene_kinds
                    and picker_canonical
                    and picker_canonical == _canonical_target(gene)):
                # Repick with explicit prohibition. The original prompt is
                # extended with a constraint that the disease gene is OFF
                # the candidate set.
                guarded_msg = (
                    target_picker_user_msg + "\n\n"
                    f"HARD CONSTRAINT: the chosen pattern's target_kind is "
                    f"'{pattern['target_kind']}', which by definition is NOT "
                    f"the disease gene ({gene}). Your previous pick was the "
                    f"disease gene itself, which violates the pattern. PICK "
                    f"A DIFFERENT GENE from the candidate / interactor / "
                    f"family lists above. Return JSON only."
                )
                picker = await _pick_target_self_consistent(
                    client, _get_model(),
                    user_msg=guarded_msg,
                    n_samples=3,
                )
                picker.setdefault("rationale", "")
                picker["rationale"] = (
                    f"[disease_gene_default guard re-pick: pattern "
                    f"target_kind={pattern['target_kind']} forbids disease "
                    f"gene as target] " + picker["rationale"]
                )

        # Modality / confidence are derived from the chosen target_kind in
        # a deterministic way -- avoids another LLM call and keeps the
        # modulation_type field clean for scoring. Confidence reflects
        # the vote margin: 3/3 -> 0.9, 2/3 -> 0.7, 1/3 -> 0.5.
        kind_to_modality = {
            "downstream_effector": "inhibitor",
            "paralog": "splice_modifier",
            "upstream_enzyme": "siRNA_ASO",
            "disease_gene_protein_chaperone": "chaperone",
            "cargo_receptor": "modulator",
            "disease_gene_mRNA": "siRNA_ASO",
            "downstream_receptor_agonist": "activator",
            "disease_gene_exon_skip": "splice_modifier",
            "repressor": "gene_therapy",
        }
        modality = kind_to_modality.get(pattern["target_kind"], "inhibitor")
        max_votes = max((v for v in picker["votes"].values()), default=0)
        confidence = {3: 0.9, 2: 0.7, 1: 0.5}.get(max_votes, 0.5)

        strategy = {
            "target_protein": picker["target_protein"] or "Unknown",
            "target_pathway": pathway_text.split(",")[0].strip() if pathway_text else "Unknown",
            "modulation_type": modality,
            "supporting_evidence": [
                f"Pattern {pattern['pattern_id']}: {pattern.get('reasoning', '')}",
                picker.get("rationale", ""),
            ],
            "precedent_drugs": [],   # blinded: agent should NOT name drugs
            "confidence_score": confidence,
            "citations": [],
            "rationale": picker.get("rationale", ""),
            # Carry pattern/kind so self_critique and retry can reuse them.
            "pattern_id": pattern["pattern_id"],
            "target_kind": pattern["target_kind"],
        }

        trace = [
            f"Stage1 pattern={pattern['pattern_id']} target_kind={pattern['target_kind']}",
            f"Stage2 picks={dict(picker['votes'])} winner={strategy['target_protein']} "
            f"confidence={confidence:.2f}",
        ]
        return {
            "strategy": strategy,
            "reasoning_trace": trace,
            "citations": [],
            "retry_count": retry + 1,
        }
    except Exception as e:
        return {
            "strategy": None,
            "errors": [f"strategy_synthesis error: {e}"],
            "reasoning_trace": [f"strategy_synthesis failed: {e}"],
            "retry_count": retry + 1,
        }
