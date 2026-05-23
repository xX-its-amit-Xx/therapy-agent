import asyncio, os, json
from typing import Any
from therapy_agent.state import AgentState
import anthropic


def _get_client():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _get_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


SYSTEM = """You are an expert translational medicine scientist specializing in rare-disease drug discovery.

Given:
- A gene with a defined molecular mechanism
- Downstream pathway proteins
- Druggable candidates from ChEMBL/DrugBank

Synthesize a therapeutic strategy following this reasoning framework:

MECHANISM-TO-STRATEGY RULES:
1. LoF of inhibitory protein (e.g. protease inhibitor, tumor suppressor):
   → Target the enzyme/pathway the inhibitor normally suppresses (downstream effector)
   → Example: SERPING1 LoF (C1-inhibitor deficiency) → inhibit KLKB1 (plasma kallikrein) downstream
   → Drug class: small-molecule inhibitor of the effector enzyme

2. LoF of structural/transport protein:
   → Replacement therapy (gene therapy, enzyme replacement) OR augment paralog
   → Example: SMN1 LoF → augment SMN2 via splice-switching ASO (nusinersen) or gene therapy (onasemnogene)

3. LoF causing toxic metabolite accumulation:
   → Inhibit the enzyme producing the toxic metabolite
   → Example: ALAS1 overactivity in AHP → silence ALAS1 with siRNA (givosiran)

4. Protein misfolding / ER retention:
   → Options: (a) pharmacological chaperone to refold, (b) TMED cargo receptor modulation to release
     trapped protein for lysosomal degradation, (c) proteostasis enhancement
   → Example: UMOD/MUC1 misfolding (ADTKD) → modulate TMED9 (cargo receptor) with BRD4780
     to divert trapped mutant protein from ER to lysosome for degradation (Dvela-Levitt Cell 2019)
   → Example: GLA misfolding (Fabry) → migalastat chaperone to restore GLA trafficking

5. GoF or toxic gain:
   → Silence the gene (siRNA, ASO) or inhibit the protein directly
   → Example: SOD1 ALS → tofersen ASO to reduce SOD1

6. Splicing defect / frameshift amenable to exon skipping:
   → Antisense oligonucleotide exon skipping to restore reading frame
   → Example: DMD exon 51 deletion → eteplirsen exon 51 skipping

FEW-SHOT EXAMPLES (full strategy objects):

Example 1 — SERPING1/HAE:
Input: gene=SERPING1, mechanism=lof, pathway includes KLKB1 F12 BDKRB2, candidate targets include KLKB1
Output JSON:
{
  "target_protein": "KLKB1 (plasma kallikrein)",
  "target_pathway": "Kallikrein-kinin system / contact activation pathway",
  "modulation_type": "inhibitor",
  "supporting_evidence": [
    "SERPING1 haploinsufficiency removes the primary brake on plasma kallikrein (KLKB1)",
    "Uncontrolled KLKB1 generates excess bradykinin via high-molecular-weight kininogen cleavage",
    "Bradykinin binds BDKRB2 on endothelium → increased vascular permeability → angioedema",
    "Genetic and pharmacological KLKB1 inhibition abolishes attacks in HAE models"
  ],
  "precedent_drugs": ["sebetralstat (Ekterly, KalVista) — oral KLKB1 inhibitor, FDA approved July 2025", "berotralstat (Orladeyo) — oral KLKB1 inhibitor, FDA approved 2020", "lanadelumab (Takhzyro) — anti-KLKB1 mAb, FDA approved 2018"],
  "confidence_score": 0.95,
  "citations": [
    "Webb DJ et al. KONFIDENT trial (KalVista) sebetralstat Phase 3, 2025",
    "Riedl MA et al. ZENITH-1 trial berotralstat, NEJM 2020",
    "Bhatt DL et al. lanadelumab subcutaneous, NEJM 2017",
    "Kaplan AP & Ghebrehiwet B. The plasma bradykinin-forming pathways, J Allergy Clin Immunol 2010"
  ],
  "rationale": "SERPING1 loss-of-function removes inhibitory control over plasma kallikrein (KLKB1). Restoring this brake by directly inhibiting KLKB1 prevents bradykinin excess that drives angioedema attacks. Three approved drugs (lanadelumab, berotralstat, sebetralstat) validate this target."
}

Example 2 — UMOD/ADTKD:
Input: gene=UMOD, mechanism=misfolding, pathway includes TMED9 TMED2 TMED10 HSP90B1, candidate targets include TMED9
Output JSON:
{
  "target_protein": "TMED9 (transmembrane emp24 domain-containing protein 9)",
  "target_pathway": "COPI vesicle / ER-to-Golgi cargo receptor / ER quality control",
  "modulation_type": "inhibitor",
  "supporting_evidence": [
    "UMOD and MUC1 frameshifts cause mutant protein misfolding and ER retention",
    "TMED9 is a cargo receptor that retains misfolded uromodulin in the ER",
    "BRD4780 binds TMED9, releasing trapped mutant UMOD and MUC1fs for lysosomal degradation",
    "BRD4780 rescued kidney function in Umod(fs/+) mice without affecting WT UMOD secretion",
    "Genetic knockdown of TMED2, TMED9, or TMED10 phenocopies BRD4780 rescue"
  ],
  "precedent_drugs": ["BRD4780 — TMED9 modulator (preclinical, Dvela-Levitt et al. Cell 2019)", "No approved drugs yet; validates TMED pathway as druggable"],
  "confidence_score": 0.88,
  "citations": [
    "Dvela-Levitt M et al. Small molecule targets TMED9 and promotes lysosomal degradation to reverse proteinopathy. Cell. 2019;178(3):521-535.e23.",
    "Rampoldi L et al. Allelism of MCKD, FJHN and GCKD caused by impairment of uromodulin export dynamics. Hum Mol Genet. 2003",
    "Kirby A et al. Mutations causing medullary cystic kidney disease type 1 lie in a large VNTR in MUC1 missed by massively parallel sequencing. Nat Genet. 2013"
  ],
  "rationale": "UMOD misfolding mutations cause ER retention of mutant uromodulin, activating UPR and damaging kidney tubules. TMED9 acts as a cargo receptor trapping the misfolded protein. BRD4780 binds TMED9 to redirect mutant protein to lysosomes for degradation, clearing ER stress. This approach is protein-specific and does not require gene correction."
}

Always return a single valid JSON object. If evidence is weak, lower confidence_score and say so in rationale."""


async def strategy_synthesis_node(state: AgentState) -> dict:
    client = _get_client()

    gene = state.get("gene_symbol") or state["gene"]
    mechanism = state.get("molecular_mechanism", "unknown")
    mechanism_reasoning = state.get("mechanism_reasoning", "")
    pathway_genes = state.get("pathway_genes") or []
    candidate_targets = state.get("candidate_targets") or []
    approved_drugs = state.get("approved_drugs") or []
    phenotype = state["disease_phenotype"]
    mutation = state["mutation"]
    retry = state.get("retry_count", 0)

    targets_text = json.dumps(candidate_targets[:8], indent=2) if candidate_targets else "No druggable targets found from database search"
    pathway_text = ", ".join(pathway_genes[:20]) if pathway_genes else "Not found"
    approved_text = json.dumps(approved_drugs[:5], indent=2) if approved_drugs else "None found"

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

Pathway context (Reactome):
{pathway_text}

Druggable targets found:
{targets_text}

Approved drugs found:
{approved_text}
{critique_ctx}

Apply the mechanism-to-strategy rules. Return ONLY valid JSON matching the strategy schema."""

    try:
        response = client.messages.create(
            model=_get_model(),
            max_tokens=1500,
            system=SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        # Strip markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)

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
        }
    except Exception as e:
        return {
            "strategy": None,
            "errors": [f"strategy_synthesis error: {e}"],
            "reasoning_trace": [f"strategy_synthesis failed: {e}"],
            "retry_count": retry + 1,
        }
