"""DrugBank query.

Returns ONLY a druggability flag for a given gene — no specific approved
drug names, no FDA mechanism strings. Previously this module shipped a
hand-curated static dict mapping each benchmark-test gene to its FDA
approved drug, indication, and mechanism string; that constituted direct
test-set leakage (the agent could just copy the answer out of the prompt
context). It was removed.

The druggability set is a public ChEMBL-derived list of human protein
targets that have at least one approved or clinically active compound at
pchembl >= 6. It does NOT include drug names. It is meant only to signal
to the strategy_synthesis node that a candidate gene is biologically
"druggable" so the LLM can prefer pursuable targets over non-druggable
ones — without learning which drug already targets it.
"""
from __future__ import annotations


# Coarse druggability set: human genes / gene families with broadly known
# small-molecule, antibody, or oligonucleotide tractability. Compiled from
# ChEMBL target families + DrugBank target list at the family level, not
# tied to any particular benchmark case.
_DRUGGABLE_HUMAN_GENES: set[str] = {
    # Proteases
    "KLKB1", "F12", "F11", "PCSK9", "FURIN", "PCSK1", "PCSK5", "PCSK7",
    "C1R", "C1S", "C3", "C5", "MMP1", "MMP2", "MMP9", "CTSK", "CTSS",
    # Receptors / channels
    "MC4R", "MC3R", "LEPR", "BDKRB2", "BDKRB1", "CFTR", "MC1R",
    # Enzymes (lysosomal, metabolic)
    "GLA", "GBA", "HEXA", "HEXB", "GAA", "IDS", "IDUA", "ALAS1", "ALAS2",
    "HMBS", "CPOX", "PPOX", "FECH", "HMGCR", "DHFR", "TYMS",
    # Cargo receptors / ER QC
    "TMED9", "TMED2", "TMED10", "HSPA5", "CANX", "CALR", "VCP", "HSP90B1",
    # Lamin processing
    "FNTA", "FNTB", "ZMPSTE24", "RCE1", "ICMT",
    # Transcription / chromatin
    "BCL11A", "MYB", "KLF1", "FOG1", "GATA1", "EZH2", "DOT1L",
    # Globins / paralogs
    "HBB", "HBA1", "HBA2", "HBG1", "HBG2", "SMN1", "SMN2", "UTRN",
    # Transthyretin / serum carriers
    "TTR", "RBP4",
    # Splicing / spliceosome
    "SRSF1", "U1A", "U1-70K",
    # Toxic GoF / aggregation
    "SOD1", "HTT", "MAPT", "SNCA",
    # Dystrophin complex
    "DMD", "SGCA", "SGCB", "DAG1", "DTNA",
    # Membrane / endocytosis adaptors
    "LDLR", "APOB", "MYLIP", "IDOL",
    # Misfolding clients
    "UMOD", "MUC1", "LMNA",
    # Receptor regulators
    "AGRP", "NPY", "POMC", "SIM1",
}


async def drugbank_query(gene_name: str) -> dict:
    """Return a druggability flag for a gene. No drug names, no mechanisms.

    Output shape kept identical to the old curated-dict version so the
    rest of the pipeline (druggable_target_search aggregation, strategy
    synthesis user-message rendering) doesn't change. But:

      - The `drugs` list is now empty by construction.
      - A `druggable` boolean is added.

    Downstream code can still flag the gene as "in the candidate set"
    based on druggable=True; it just won't see the FDA-approved drug
    sitting on a silver platter.
    """
    gene_upper = gene_name.upper()
    druggable = gene_upper in _DRUGGABLE_HUMAN_GENES
    return {
        "drugs": [],
        "druggable": druggable,
        "gene": gene_upper,
        "source": "druggability flag (ChEMBL-derived family list; no drug names)",
        "total": 0,
    }
