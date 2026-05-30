# Adversarial benchmark cases

Cases designed to probe specific failure modes the v0.7–v0.8 evaluation
surfaced. Unlike `cases/` (canonical biology) and `heldout_2024_2025/`
(post-cutoff generalization), these are hand-crafted to fail an agent
that has the WRONG inductive bias on a documented failure pattern.

Each case is labeled with the failure mode it probes:

| File | Probes |
|---|---|
| `paralog_confusion_dmd_utrn.yaml` | does the agent confuse the disease gene's paralog (UTRN) for the right exon-skipping target (DMD)? |
| `cascade_disambiguation_hae.yaml` | given the SAME disease as Ekterly/Garadacimab, does the agent pick the named drug's specific cascade member? |
| `paralog_vs_repressor_scd.yaml` | does the agent confuse the fetal-globin paralog (HBG1) with its erythroid-enhancer repressor (BCL11A)? |
| `wrong_axis_cah.yaml` | does the agent pick the wrong receptor (MR vs CRHR1) on a multi-receptor endocrine case? |
| `pathway_branch_serping1_complement.yaml` | does the agent pick the C1R/C1S complement branch vs the KLKB1/F12 kinin branch for SERPING1, given a non-edema (complement-mediated) phenotype? |

Run with:

    python run_blinded.py --set adversarial --out adv_results.json

The pass criterion is per-case `expected_target` + `valid_targets`
matching, same as the main benchmarks. Adversarial scores are tracked
SEPARATELY from dev/val (different curation philosophy) and shouldn't
be aggregated.
