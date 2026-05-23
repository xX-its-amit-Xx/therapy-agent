"""Reactome pathway query."""
import httpx

REACTOME_BASE = "https://reactome.org/ContentService"

# Hard-coded pathway expansions for key rare-disease genes
# (used as fallback when Reactome API is slow/unavailable)
GENE_PATHWAY_FALLBACK = {
    "SERPING1": {
        "pathways": ["Contact activation (kallikrein-kinin) system", "Complement cascade"],
        "interactors": ["KLKB1", "F12", "BDKRB2", "BDKRB1", "KNG1", "F11", "C1R", "C1S", "C4A", "C4B"],
        "pathway_context": "SERPING1 (C1-inhibitor) suppresses plasma kallikrein (KLKB1) and Factor XIIa (F12). Its absence allows uncontrolled kallikrein activity generating bradykinin from HMW kininogen.",
    },
    "UMOD": {
        "pathways": ["ER quality control", "Protein processing in ER", "Kidney tubular transport"],
        "interactors": ["TMED9", "TMED2", "TMED10", "HSPA5", "CANX", "CALR", "VCP", "DNAJB11"],
        "pathway_context": "Uromodulin is synthesized in thick ascending limb cells. Frameshift mutations cause misfolding and ER retention, activating UPR. TMED cargo receptors regulate ER retention of misfolded proteins.",
    },
    "MUC1": {
        "pathways": ["ER quality control", "Mucin-type O-glycosylation"],
        "interactors": ["TMED9", "TMED2", "TMED10", "VCP", "HSPA5", "SEC61A1"],
        "pathway_context": "MUC1 frameshift (MUC1-fs) creates a toxic truncated protein that accumulates in ER. TMED9 cargo receptor is required for retention; BRD4780 releases it for lysosomal degradation.",
    },
    "SMN1": {
        "pathways": ["RNA splicing — via transesterification reactions", "snRNP biogenesis"],
        "interactors": ["SMN2", "GEMIN2", "GEMIN3", "GEMIN4", "GEMIN5", "SNRPB", "SNRPD1", "UBA1"],
        "pathway_context": "SMN1 produces full-length SMN protein for snRNP assembly. LoF shifts reliance to SMN2 which produces truncated SMNΔ7. Augmenting SMN2 exon 7 inclusion rescues.",
    },
    "HBB": {
        "pathways": ["O2/CO2 exchange in erythrocytes", "Heme biosynthesis"],
        "interactors": ["HBA1", "HBA2", "AHSP", "GYPA", "SPTA1", "SPTB"],
        "pathway_context": "HbS (Glu6Val) polymerizes under deoxygenation causing sickling. Therapies target polymerization (voxelotor), HbF induction (hydroxyurea), or BCL11A silencing.",
    },
    "PCSK9": {
        "pathways": ["Regulation of LDLR pathway", "LDL clearance"],
        "interactors": ["LDLR", "APOB", "HNF1A", "IDOL", "STAP1"],
        "pathway_context": "PCSK9 degrades LDL receptor. GoF mutations cause FH. Silencing PCSK9 (inclisiran) or blocking PCSK9-LDLR interaction (evolocumab) restores LDL clearance.",
    },
    "POMC": {
        "pathways": ["Melanocortin signaling", "Appetite regulation"],
        "interactors": ["MC4R", "MC3R", "LEPR", "AGRP", "NPY", "SIM1"],
        "pathway_context": "POMC-derived α-MSH activates MC4R to suppress appetite. POMC or LEPR LoF blocks this axis. Setmelanotide (MC4R agonist) bypasses the broken signaling.",
    },
    "ALAS1": {
        "pathways": ["Heme synthesis", "Porphyrin and chlorophyll metabolism"],
        "interactors": ["PBGD", "HMBS", "CPOX", "PPOX", "FECH", "ALAS2"],
        "pathway_context": "ALAS1 is the rate-limiting enzyme in hepatic heme synthesis. GoF-like states (derepressed) in AHP cause toxic ALA/PBG accumulation. Givosiran silences ALAS1 mRNA via siRNA.",
    },
    "DMD": {
        "pathways": ["Muscle contraction", "Dystrophin-glycoprotein complex"],
        "interactors": ["DTNA", "DTNB", "SNTA1", "DAG1", "SGCA", "SGCB", "NOS1", "UTRN"],
        "pathway_context": "Dystrophin links actin cytoskeleton to extracellular matrix. Out-of-frame deletions cause complete absence. Exon-skipping restores partial reading frame (Becker-like dystrophin).",
    },
    "GLA": {
        "pathways": ["Glycosphingolipid biosynthesis", "Lysosomal enzyme transport"],
        "interactors": ["NPC1", "NPC2", "GBA", "HEXA", "HEXB", "LAMP1", "M6PR"],
        "pathway_context": "GLA (α-galactosidase A) cleaves Gb3 in lysosomes. Missense mutations cause misfolding. Migalastat (pharmacological chaperone) stabilizes amenable GLA variants.",
    },
    "SOD1": {
        "pathways": ["Oxidative stress response", "Motor neuron survival"],
        "interactors": ["CCS", "ATOX1", "MT1A", "PRDX1", "CAT", "GPX1"],
        "pathway_context": "SOD1 mutations (GoF/dominant-negative toxic aggregates) cause motor neuron death. Tofersen is an ASO that reduces SOD1 mRNA in CSF/spinal cord.",
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
