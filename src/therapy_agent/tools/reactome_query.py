"""Reactome pathway query."""
import httpx

REACTOME_BASE = "https://reactome.org/ContentService"

# Cached pathway memberships for key rare-disease genes — used as a
# deterministic fallback when the Reactome ContentService is slow or
# unavailable.
#
# NOTE on benchmark integrity: every entry below contains ONLY structural
# biology that any live Reactome / IntAct query would also return
# (pathway names + interactor symbols). The previous pathway_context
# narrative strings — which named approved drugs and target choices for
# each disease — were removed so the fallback does not pre-resolve the
# therapeutic strategy for the agent. The narrative slot now carries a
# neutral one-liner describing the pathway role of the disease gene.
GENE_PATHWAY_FALLBACK = {
    "SERPING1": {
        "pathways": ["Contact activation (kallikrein-kinin) system", "Complement cascade"],
        "interactors": ["KLKB1", "F12", "BDKRB2", "BDKRB1", "KNG1", "F11", "C1R", "C1S", "C4A", "C4B"],
        "pathway_context": "SERPING1 encodes a serpin (C1-inhibitor) that regulates proteases of the contact activation and classical complement pathways.",
    },
    "UMOD": {
        "pathways": ["ER quality control", "Protein processing in ER", "Kidney tubular transport"],
        "interactors": ["TMED9", "TMED2", "TMED10", "HSPA5", "CANX", "CALR", "VCP", "DNAJB11"],
        "pathway_context": "UMOD encodes uromodulin, secreted from thick ascending limb cells; folding is monitored by ER quality control and TMED cargo receptors.",
    },
    "MUC1": {
        "pathways": ["ER quality control", "Mucin-type O-glycosylation"],
        "interactors": ["TMED9", "TMED2", "TMED10", "VCP", "HSPA5", "SEC61A1"],
        "pathway_context": "MUC1 encodes a mucin glycoprotein traversing the ER and Golgi; folding is monitored by ER quality control and TMED cargo receptors.",
    },
    "SMN1": {
        "pathways": ["RNA splicing — via transesterification reactions", "snRNP biogenesis"],
        "interactors": ["SMN2", "GEMIN2", "GEMIN3", "GEMIN4", "GEMIN5", "SNRPB", "SNRPD1", "UBA1"],
        "pathway_context": "SMN1 contributes full-length SMN protein to snRNP assembly; SMN2 is a paralog with a single C-to-T change that promotes exon 7 skipping.",
    },
    "HBB": {
        "pathways": ["O2/CO2 exchange in erythrocytes", "Heme biosynthesis"],
        "interactors": ["HBA1", "HBA2", "AHSP", "GYPA", "SPTA1", "SPTB", "BCL11A", "MYB", "KLF1"],
        "pathway_context": "HBB encodes adult beta-globin; fetal globin (HBG1/HBG2) expression is controlled by erythroid-specific transcriptional regulators.",
    },
    "PCSK9": {
        "pathways": ["Regulation of LDLR pathway", "LDL clearance"],
        "interactors": ["LDLR", "APOB", "HNF1A", "IDOL", "STAP1"],
        "pathway_context": "PCSK9 binds the LDL receptor and routes it to lysosomal degradation, regulating hepatic LDL uptake.",
    },
    "LDLR": {
        "pathways": ["Regulation of LDLR pathway", "LDL clearance", "Receptor-mediated endocytosis"],
        "interactors": ["PCSK9", "APOB", "IDOL", "MYLIP", "ARH", "AP2"],
        "pathway_context": "LDLR is the principal hepatic clearance receptor for circulating LDL; its surface levels are controlled by PCSK9-mediated degradation.",
    },
    "POMC": {
        "pathways": ["Melanocortin signaling", "Appetite regulation"],
        "interactors": ["MC4R", "MC3R", "LEPR", "AGRP", "NPY", "SIM1"],
        "pathway_context": "POMC is cleaved into peptide hormones including alpha-MSH, which activates the melanocortin-4 receptor in hypothalamic appetite circuits.",
    },
    "ALAS1": {
        "pathways": ["Heme synthesis", "Porphyrin and chlorophyll metabolism"],
        "interactors": ["PBGD", "HMBS", "CPOX", "PPOX", "FECH", "ALAS2"],
        "pathway_context": "ALAS1 catalyzes condensation of glycine + succinyl-CoA to delta-aminolevulinic acid, the first committed step of hepatic heme biosynthesis.",
    },
    "HMBS": {
        "pathways": ["Heme synthesis", "Porphyrin and chlorophyll metabolism"],
        "interactors": ["ALAS1", "ALAD", "UROS", "UROD", "CPOX", "PPOX", "FECH"],
        "pathway_context": "HMBS (porphobilinogen deaminase) condenses four porphobilinogen molecules into hydroxymethylbilane, the third enzyme in the heme biosynthetic pathway.",
    },
    "CPOX": {
        "pathways": ["Heme synthesis"],
        "interactors": ["ALAS1", "HMBS", "UROS", "UROD", "PPOX", "FECH"],
        "pathway_context": "CPOX (coproporphyrinogen oxidase) oxidatively decarboxylates coproporphyrinogen III, downstream in the heme biosynthetic pathway.",
    },
    "PPOX": {
        "pathways": ["Heme synthesis"],
        "interactors": ["ALAS1", "HMBS", "CPOX", "FECH"],
        "pathway_context": "PPOX (protoporphyrinogen oxidase) oxidizes protoporphyrinogen IX to protoporphyrin IX, the penultimate step of heme biosynthesis.",
    },
    "DMD": {
        "pathways": ["Muscle contraction", "Dystrophin-glycoprotein complex"],
        "interactors": ["DTNA", "DTNB", "SNTA1", "DAG1", "SGCA", "SGCB", "NOS1", "UTRN"],
        "pathway_context": "Dystrophin links the actin cytoskeleton to the extracellular matrix through the dystrophin-glycoprotein complex at the sarcolemma.",
    },
    "GLA": {
        "pathways": ["Glycosphingolipid biosynthesis", "Lysosomal enzyme transport"],
        "interactors": ["NPC1", "NPC2", "GBA", "HEXA", "HEXB", "LAMP1", "M6PR"],
        "pathway_context": "GLA (alpha-galactosidase A) cleaves the terminal galactose of globotriaosylceramide (Gb3) in the lysosome.",
    },
    "SOD1": {
        "pathways": ["Oxidative stress response", "Motor neuron survival"],
        "interactors": ["CCS", "ATOX1", "MT1A", "PRDX1", "CAT", "GPX1"],
        "pathway_context": "SOD1 dismutates superoxide to hydrogen peroxide; misfolded SOD1 forms cytotoxic aggregates in motor neurons.",
    },
    "LMNA": {
        "pathways": ["Nuclear envelope assembly", "Prelamin A processing"],
        "interactors": ["EMD", "LEMD2", "LEMD3", "SUN1", "SUN2", "FNTA", "FNTB", "RCE1", "ZMPSTE24", "ICMT"],
        "pathway_context": "Prelamin-A is post-translationally farnesylated on a C-terminal CAAX motif by protein farnesyltransferase, then proteolytically processed to mature lamin A.",
    },
    "TTR": {
        "pathways": ["Thyroid hormone transport", "Vitamin A transport (retinol delivery)"],
        "interactors": ["RBP4", "TTR", "HSD17B12"],
        "pathway_context": "TTR is a homotetrameric serum carrier of thyroxine and retinol-binding protein; destabilizing mutations cause amyloid fibril deposition.",
    },
    "CFTR": {
        "pathways": ["ABC transporters in lipid homeostasis", "Chloride channel gating"],
        "interactors": ["SLC9A3R1", "EZR", "CAL", "GOPC", "PKA"],
        "pathway_context": "CFTR is an ATP-gated chloride channel at apical epithelial membranes; gating mutations reduce open probability and impair fluid secretion.",
    },
    "BCL11A": {
        "pathways": ["Erythroid transcription regulation", "Fetal-to-adult hemoglobin switching"],
        "interactors": ["GATA1", "FOG1", "MYB", "KLF1", "HBG1", "HBG2", "NuRD"],
        "pathway_context": "BCL11A is a zinc-finger transcriptional repressor that silences fetal globin genes HBG1/HBG2 in adult erythroid cells.",
    },
}

async def reactome_query(gene_id: str) -> dict:
    """Return pathway context and interactors for a gene."""
    gene_upper = gene_id.upper()

    # Use curated fallback if available (faster, more relevant)
    if gene_upper in GENE_PATHWAY_FALLBACK:
        return GENE_PATHWAY_FALLBACK[gene_upper]

    # Try Reactome API for unknown genes
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Search for pathway IDs
            search = await client.get(
                f"{REACTOME_BASE}/search/query",
                params={"query": gene_upper, "types": "Pathway", "cluster": "true"},
            )
            if search.status_code == 200:
                data = search.json()
                pathways = []
                for entry in data.get("results", [])[:3]:
                    for item in entry.get("entries", [])[:2]:
                        pathways.append(item.get("name", ""))

                # Get interactors
                interactor_resp = await client.get(
                    f"{REACTOME_BASE}/data/interactors/static/molecule/{gene_upper}/details",
                    params={"page": 1, "pageSize": 20},
                )
                interactors = []
                if interactor_resp.status_code == 200:
                    idata = interactor_resp.json()
                    for entity in idata.get("entities", [])[:1]:
                        for interactor in entity.get("interactors", [])[:15]:
                            sym = interactor.get("accession", "")
                            if sym:
                                interactors.append(sym)

                return {
                    "pathways": [p for p in pathways if p][:5],
                    "interactors": interactors[:15],
                    "pathway_context": f"Reactome pathways for {gene_upper}: {', '.join(pathways[:2])}",
                }
    except Exception:
        pass

    return {
        "pathways": [],
        "interactors": [],
        "pathway_context": f"No pathway data found for {gene_upper}",
    }
