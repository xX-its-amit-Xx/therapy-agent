# Benchmark Suite

**RESEARCH PROTOTYPE. NOT FOR CLINICAL USE.**

This directory contains benchmark cases for the therapy-agent pipeline and the
runner that produces structured JSONL output consumed by **bio-rag-eval**.

---

## Output Schema (`TherapyStrategyOutput`)

Every case produces one JSON object written to `results.jsonl`.
Schema version is stored in the `schema_version` field and pinned in
`src/therapy_agent/config.py` (`BENCHMARK_SCHEMA_VERSION`).

```jsonc
{
  "schema_version": "1.0",        // bump when fields change
  "case_id": "ekterly_serping1",
  "gene": "SERPING1",
  "mutation": "frameshift causing haploinsufficiency …",
  "disease_phenotype": "hereditary angioedema …",

  // ── strategy ────────────────────────────────────────────────────────
  "target_protein": {
    "name": "KLKB1 (plasma kallikrein)",
    "uniprot_id": null,            // populated if UniProt lookup succeeds
    "gene_symbol": "KLKB1"
  },
  "target_pathway": "Kallikrein-kinin system / contact activation pathway",
  "modulation_type": "inhibitor", // canonical enum — see below
  "supporting_evidence": [
    {"claim": "SERPING1 LoF removes inhibitory control over KLKB1", "doi": null},
    {"claim": "Sebetralstat (Ekterly) inhibits KLKB1 orally; FDA approved July 2025",
     "doi": "10.xxxx/xxxx"}
  ],
  "precedent_drugs": [
    {"name": "sebetralstat (Ekterly)", "approved": true, "year": 2025},
    {"name": "berotralstat (Orladeyo)", "approved": true, "year": 2020}
  ],
  "confidence_score": 0.95,       // 0.0 – 1.0

  // ── transparency ────────────────────────────────────────────────────
  "reasoning_trace": [
    {"node": "pipeline", "content": "Parsed: gene=SERPING1, mutation_type=frameshift"},
    {"node": "pipeline", "content": "Mechanism: lof (confidence=0.95): …"},
    {"node": "pipeline", "content": "Strategy: target=KLKB1 (plasma kallikrein), …"}
  ],

  // ── run metadata ────────────────────────────────────────────────────
  "model_used": "claude-sonnet-4-20250514",
  "input_tokens": 4812,
  "output_tokens": 623,
  "wall_clock_seconds": 14.3,
  "timestamp": "2025-05-22T18:00:00+00:00"
}
```

### `modulation_type` enum

| Value | Meaning |
|---|---|
| `inhibitor` | Small-molecule or biologic that reduces target activity |
| `agonist` | Activates / mimics the target (e.g. MC4R agonist) |
| `chaperone` | Pharmacological chaperone stabilising misfolded protein |
| `modulator` | Indirect modulation (e.g. cargo receptor re-routing) |
| `ASO` | Antisense oligonucleotide — exon skipping, splice switching, or siRNA |
| `gene_therapy` | Viral / non-viral gene delivery |
| `enzyme_replacement` | Recombinant enzyme infusion |
| `other` | None of the above |

---

## Case Sources

| Source | How loaded | Always runs? |
|---|---|---|
| `fda-strategy-triples` package | `load_benchmark_cases()` | Only if `validation_status == "validated"` |
| `benchmarks/cases/*.yaml` | Supplementary YAML | Yes (unless `--no-fda-triples` omits them) |
| `benchmarks/*.yaml` (Ekterly, BRD4780) | Primary gold-standard | Always |

---

## Running

```bash
# Full run — fda-strategy-triples + YAML
python benchmarks/run_benchmarks.py

# Gold-standard cases only
python benchmarks/run_benchmarks.py --primary-only

# Single case
python benchmarks/run_benchmarks.py --case ekterly_serping1

# Pin a specific model
python benchmarks/run_benchmarks.py --model claude-sonnet-4-20250514

# Skip fda-strategy-triples (YAML only)
python benchmarks/run_benchmarks.py --no-fda-triples
```

Output is written to `benchmark_runs/<ISO-timestamp>/`:
```
benchmark_runs/20250522T180000Z/
  results.jsonl    # one TherapyStrategyOutput per line
  summary.json     # pass/partial/fail counts + run metadata
  run.log          # human-readable console transcript
```

---

## Feeding into bio-rag-eval

```bash
bio-rag-eval run \
  --input  benchmark_runs/20250522T180000Z/results.jsonl \
  --schema therapy_strategy_v1 \
  --out    eval_results/
```

Each line in `results.jsonl` is a self-contained evaluation unit.
bio-rag-eval reads `schema_version`, validates the structure, and scores
`supporting_evidence[*].doi` for retrieval recall, `target_protein.gene_symbol`
for target accuracy, and `confidence_score` calibration.

---

## Grading rubric (YAML cases)

| Dimension | Points | Pass criterion |
|---|---|---|
| Target match | 30 | `target_protein.name` contains expected gene symbol or alias |
| Citation match | 30 | Any `precedent_drugs[*].name` or `supporting_evidence[*].claim` matches a key citation |
| Mechanism match | 20 | `modulation_type` is consistent with expected mechanism class |
| Confidence | 10 | `confidence_score >= min_confidence` |
| Rationale quality | 10 | `supporting_evidence` total text > 50 chars |
| **Total** | **100** | ≥ 80 → PASS, 50–79 → PARTIAL, < 50 → FAIL |
