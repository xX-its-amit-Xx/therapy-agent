# End-to-End Demo: BRD4780 / ADTKD Case

**RESEARCH PROTOTYPE. NOT FOR CLINICAL USE.**

This walkthrough runs the therapy-agent on the BRD4780 retrospective case
(UMOD frameshift → TMED9 cargo receptor → lysosomal degradation) and shows
exactly what to look at in the reasoning trace.

---

## Background

**Gene:** UMOD (uromodulin / Tamm-Horsfall protein)  
**Disease:** Autosomal dominant tubulointerstitial kidney disease (ADTKD)  
**Mutation class:** Frameshift causing protein misfolding and ER retention  
**Expected therapeutic strategy:** Modulate TMED9 cargo receptor (BRD4780) to
release trapped mutant uromodulin for lysosomal degradation  
**Key reference:** Dvela-Levitt M et al. *Cell.* 2019;178(3):521–535.e23.

---

## Install

```bash
git clone https://github.com/your-org/therapy-agent
cd therapy-agent
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Run (CLI)

```bash
therapy-agent run \
  --gene UMOD \
  --mutation "frameshift mutation causing protein misfolding and ER retention in kidney epithelial cells" \
  --phenotype "autosomal dominant tubulointerstitial kidney disease (ADTKD) with tubular atrophy and progressive renal failure"
```

### Expected streaming output

```
╭─ Starting analysis ───────────────────────────────────────────────────────╮
│ Therapy Agent                                                               │
│ Gene: UMOD  Mutation: frameshift …  Phenotype: ADTKD …                     │
╰─────────────────────────────────────────────────────────────────────────────╯

╭─ parse_input  Parsing gene/mutation/phenotype input ───────────────────────╮
│   → Parsed: gene=UMOD, mutation_type=frameshift, notes=ER retention        │
╰─────────────────────────────────────────────────────────────────────────────╯

╭─ variant_lookup  Querying ClinVar & g2p-rag ───────────────────────────────╮
│   → ClinVar: found 14 variants for UMOD                                    │
│   → g2p-rag: retrieved 5 chunk(s)                                          │
╰─────────────────────────────────────────────────────────────────────────────╯

╭─ mechanism_classifier  Classifying molecular mechanism ────────────────────╮
│   Mechanism: misfolding  (confidence: 0.92)                                │
╰─────────────────────────────────────────────────────────────────────────────╯

╭─ pathway_expansion  Expanding pathway via Reactome ────────────────────────╮
│   → Pathway genes: TMED9, TMED2, TMED10, HSPA5, CANX, CALR, VCP, DNAJB11  │
╰─────────────────────────────────────────────────────────────────────────────╯

╭─ druggable_target_search  Searching ChEMBL & DrugBank ─────────────────────╮
│   → Searched 16 genes; found 4 druggable, 0 with approved drugs           │
│   → Top targets: TMED9, TMED2, TMED10, VCP                                │
╰─────────────────────────────────────────────────────────────────────────────╯

╭─ strategy_synthesis  Synthesizing therapeutic strategy ────────────────────╮
│   Strategy drafted: TMED9 (transmembrane emp24 domain-containing …) (modulator)│
╰─────────────────────────────────────────────────────────────────────────────╯

╭─ self_critique  Self-critique: reviewing evidence & confidence ─────────────╮
│   Critique: accept (confidence adj: 0.00 → 0.88)                          │
│   Critique: Target well-supported by Dvela-Levitt 2019 Cell paper          │
╰─────────────────────────────────────────────────────────────────────────────╯

╭ Therapeutic Strategy ───────────────────────────────────────────────────────╮
│ Target Protein    TMED9 (transmembrane emp24 domain-containing protein 9)   │
│ Target Pathway    COPI vesicle / ER-to-Golgi cargo receptor                 │
│ Modulation        modulator                                                  │
│ Confidence        0.88                                                       │
│ Rationale         UMOD misfolding mutations cause ER retention of mutant    │
│                   uromodulin. TMED9 acts as a cargo receptor trapping the   │
│                   protein. BRD4780 binds TMED9 to redirect mutant protein   │
│                   to lysosomes for degradation, clearing ER stress.          │
│ Precedent Drug 1  BRD4780 — TMED9 modulator (preclinical, Dvela-Levitt …)  │
│ Citation 1        Dvela-Levitt M et al. Cell. 2019;178(3):521–535.e23.      │
╰─────────────────────────────────────────────────────────────────────────────╯
```

---

## What to look at in the reasoning trace

| Step | Node | What it tells you |
|---|---|---|
| `Parsed: gene=UMOD, mutation_type=frameshift` | `parse_input` | Free-text was correctly interpreted; `frameshift` is the key classification that flows downstream |
| `g2p-rag: retrieved 5 chunk(s)` | `variant_lookup` | g2p-rag is live and returning context — check `g2p_chunks` in the state for the actual documents used |
| `Mechanism: misfolding (confidence=0.92)` | `mechanism_classifier` | The few-shot classifier correctly fired on `frameshift + ER retention`; this single label drives the entire downstream strategy |
| `Pathway genes: TMED9, TMED2, TMED10, …` | `pathway_expansion` | Reactome curated fallback for UMOD includes TMED cargo receptors; this is where the non-obvious hypothesis originates |
| `Searched 16 genes; 4 druggable` | `druggable_target_search` | For `misfolding` mechanism the node injects ER quality-control genes (TMED9/TMED2/TMED10) even when Reactome doesn't return them — check `src/therapy_agent/nodes/druggable_target_search.py` line 18 |
| `Strategy drafted: TMED9 (modulator)` | `strategy_synthesis` | Claude applied mechanism rule 4 ("misfolding → TMED cargo receptor") from the few-shot system prompt |
| `Critique: accept` | `self_critique` | The Dvela-Levitt citation passed the sanity check; if it had flagged `revise`, the agent would have looped back to `strategy_synthesis` |

---

## Run as benchmark case

```bash
python benchmarks/run_benchmarks.py --case brd4780_umod
```

Output in `benchmark_runs/<timestamp>/results.jsonl`:

```jsonc
{
  "schema_version": "1.0",
  "case_id": "brd4780_umod",
  "gene": "UMOD",
  "target_protein": {"name": "TMED9 …", "gene_symbol": "TMED9"},
  "modulation_type": "modulator",
  "confidence_score": 0.88,
  "precedent_drugs": [{"name": "BRD4780 …", "approved": false, "year": 2019}],
  "supporting_evidence": [
    {"claim": "TMED9 is a cargo receptor that retains misfolded UMOD in ER",
     "doi": null},
    {"claim": "Dvela-Levitt M et al. Cell. 2019;178(3):521-535.e23.",
     "doi": "10.1016/j.cell.2019.07.002"}
  ],
  "model_used": "claude-sonnet-4-20250514",
  "input_tokens": 5103,
  "output_tokens": 689,
  "wall_clock_seconds": 16.8
}
```

Feed into bio-rag-eval:
```bash
bio-rag-eval run \
  --input  benchmark_runs/<timestamp>/results.jsonl \
  --schema therapy_strategy_v1
```

---

## Where the agent can go wrong on this case

| Failure mode | How to detect |
|---|---|
| Calls mechanism `lof` instead of `misfolding` | `mechanism_classifier` confidence will be low (< 0.7); check reasoning trace |
| Targets VCP (p97 ATPase) instead of TMED9 | Both are valid ER QC proteins; VCP answer scores as PARTIAL (target aliases list in `brd4780_umod.yaml` includes `TMED`) |
| Confabulates a BRD4780 clinical trial | `self_critique` should flag it; check `critique_notes` for UNSUPPORTED markers |
| Missing Dvela-Levitt citation | Citation match dimension drops to 0/30; still PARTIAL if target is right |
