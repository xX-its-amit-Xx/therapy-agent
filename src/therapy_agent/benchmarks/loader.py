"""Benchmark case loader.

Primary source: fda-strategy-triples package (structured, versioned).
Supplementary: hand-curated YAML files in benchmarks/cases/ that always run
               regardless of validation_status (used for the two
               retrospective gold-standard cases: Ekterly and BRD4780).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from therapy_agent.config import FDA_TRIPLES_VERSION

logger = logging.getLogger(__name__)

# Repo-root benchmarks directory (sibling of src/)
_BENCHMARKS_DIR = Path(__file__).parent.parent.parent.parent / "benchmarks"


# ── Data model ────────────────────────────────────────────────────────────────

class BenchmarkInput(BaseModel):
    gene: str
    mutation: str
    disease_phenotype: str


class BenchmarkCase(BaseModel):
    id: str
    name: str
    source: str = "yaml"          # "yaml" | "fda_strategy_triples"
    input: BenchmarkInput
    expected_outputs: Optional[dict] = None
    grading: Optional[dict] = None

    # ── factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, data: dict) -> "BenchmarkCase":
        inp = data.get("input", {})
        return cls(
            id=data.get("id", "unknown"),
            name=data.get("name", data.get("id", "unknown")),
            source="yaml",
            input=BenchmarkInput(
                gene=inp.get("gene", ""),
                mutation=inp.get("mutation", ""),
                disease_phenotype=inp.get("disease_phenotype", ""),
            ),
            expected_outputs=data.get("expected_outputs"),
            grading=data.get("grading"),
        )

    @classmethod
    def from_row(cls, row) -> "BenchmarkCase":
        """Build from a fda-strategy-triples DataFrame row."""
        return cls(
            id=str(getattr(row, "case_id", getattr(row, "id", "unknown"))),
            name=str(getattr(row, "name", getattr(row, "case_id", "unknown"))),
            source="fda_strategy_triples",
            input=BenchmarkInput(
                gene=str(row.gene),
                mutation=str(row.mutation),
                disease_phenotype=str(row.disease_phenotype),
            ),
            expected_outputs={
                "target_protein": getattr(row, "target_protein", None),
                "target_aliases": _split_field(getattr(row, "target_aliases", "")),
                "modulation_type": getattr(row, "modulation_type", None),
                "key_citations": _split_field(getattr(row, "key_citations", "")),
                "mechanism_class": getattr(row, "mechanism_class", None),
                "min_confidence": float(getattr(row, "min_confidence", 0.7)),
            },
            grading=None,  # triples dataset uses standard rubric
        )


def _split_field(value) -> list[str]:
    """Split a semicolon- or comma-separated field into a list."""
    if not value or (isinstance(value, float)):
        return []
    sep = ";" if ";" in str(value) else ","
    return [v.strip() for v in str(value).split(sep) if v.strip()]


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_yaml_cases(directory: Path, *, tag: str = "yaml") -> list[BenchmarkCase]:
    """Load all *.yaml files from a directory."""
    cases = []
    for f in sorted(directory.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            if data:
                case = BenchmarkCase.from_yaml(data)
                case = case.model_copy(update={"source": tag})
                cases.append(case)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", f, exc)
    return cases


def _load_fda_triples(
    version: str = FDA_TRIPLES_VERSION,
    min_validation_status: str = "validated",
) -> list[BenchmarkCase]:
    """Load benchmark cases from the fda-strategy-triples package."""
    try:
        from fda_strategy_triples import load_dataset  # type: ignore[import]
    except ImportError:
        logger.warning(
            "fda-strategy-triples not installed — skipping structured dataset. "
            "Install with: pip install 'fda-strategy-triples>=%s'",
            version,
        )
        return []

    try:
        df = load_dataset(version=version)
        if "validation_status" not in df.columns:
            logger.warning(
                "fda-strategy-triples DataFrame missing 'validation_status' column "
                "(got columns: %s). Skipping structured dataset.",
                list(df.columns),
            )
            return []
        df = df[df.validation_status == min_validation_status]
        cases = [BenchmarkCase.from_row(r) for _, r in df.iterrows()]
        logger.info("Loaded %d cases from fda-strategy-triples %s.", len(cases), version)
        return cases
    except Exception as exc:
        logger.warning("fda-strategy-triples load failed: %s", exc)
        return []


def load_benchmark_cases(
    min_validation_status: str = "validated",
    include_supplementary: bool = True,
) -> list[BenchmarkCase]:
    """Return the full benchmark case list.

    Order:
      1. fda-strategy-triples cases (if package installed and cases available)
      2. Supplementary curated YAML cases from benchmarks/cases/ (always included
         when include_supplementary=True)
      3. Primary YAML cases from benchmarks/ root (Ekterly + BRD4780) — always run

    Cases are de-duplicated by id; fda-strategy-triples wins on conflict.
    """
    seen: dict[str, BenchmarkCase] = {}

    # 1. Structured dataset
    for c in _load_fda_triples(min_validation_status=min_validation_status):
        seen[c.id] = c

    # 2. Supplementary curated cases (benchmarks/cases/*.yaml)
    if include_supplementary and (_BENCHMARKS_DIR / "cases").exists():
        for c in _load_yaml_cases(_BENCHMARKS_DIR / "cases", tag="yaml_supplementary"):
            seen.setdefault(c.id, c)

    # 3. Primary gold-standard cases (benchmarks/*.yaml) — always run
    for c in _load_yaml_cases(_BENCHMARKS_DIR, tag="yaml_primary"):
        seen.setdefault(c.id, c)

    cases = list(seen.values())
    logger.info("Total benchmark cases: %d", len(cases))
    return cases
