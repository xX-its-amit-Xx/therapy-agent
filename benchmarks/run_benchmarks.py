#!/usr/bin/env python3
"""Benchmark runner — emits structured JSONL consumed by bio-rag-eval.

Output layout:
    benchmark_runs/<ISO-timestamp>/
        results.jsonl   one TherapyStrategyOutput JSON object per line
        summary.json    pass/partial/fail counts + run metadata
        run.log         human-readable console transcript

Usage:
    python benchmarks/run_benchmarks.py            # all cases
    python benchmarks/run_benchmarks.py --primary-only
    python benchmarks/run_benchmarks.py --case ekterly_serping1
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich import box

# ── path bootstrap ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from therapy_agent.benchmarks.loader import BenchmarkCase, load_benchmark_cases
from therapy_agent.config import DEFAULT_MODEL, get_model
from therapy_agent.graph import run_agent
from therapy_agent.schemas.output import TherapyStrategyOutput, build_output

app = typer.Typer(add_completion=False)
console = Console()
logger = logging.getLogger(__name__)


# ── grading ───────────────────────────────────────────────────────────────────

def grade(case: BenchmarkCase, output: TherapyStrategyOutput) -> dict:
    """Score a TherapyStrategyOutput against the case's expected_outputs.

    Returns a dict with keys: scores, total, max_total, pct, verdict.
    """
    expected = case.expected_outputs or {}
    rubric = case.grading or {
        "target_match": 30,
        "citation_match": 30,
        "mechanism_match": 20,
        "confidence_match": 10,
        "rationale_quality": 10,
    }
    max_total = sum(rubric.values())
    scores: dict[str, int] = {}
    total = 0

    # target match
    target_text = output.target_protein.name.lower()
    exp_target = (expected.get("target_protein") or "").lower()
    aliases = [a.lower() for a in expected.get("target_aliases", [])]
    hit = bool(exp_target and exp_target in target_text) or any(a in target_text for a in aliases)
    scores["target_match"] = rubric["target_match"] if hit else 0
    total += scores["target_match"]

    # citation match — search precedent_drugs + supporting_evidence
    haystack = " ".join([
        " ".join(d.name for d in output.precedent_drugs),
        " ".join(e.claim for e in output.supporting_evidence),
    ]).lower()
    key_cits = [c.lower() for c in expected.get("key_citations", [])]
    cit_hit = any(kc in haystack for kc in key_cits)
    scores["citation_match"] = rubric["citation_match"] if cit_hit else 0
    total += scores["citation_match"]

    # mechanism match
    # mechanism lives in final_state, not in TherapyStrategyOutput directly —
    # carry it via the case id lookup against the run dict (passed via closure)
    # We approximate here: check target_pathway or modulation_type against expected
    exp_mech = (expected.get("mechanism_class") or "").lower()
    # Map mechanism_class to modulation hints
    _MECH_MODULATION = {
        "lof": ["inhibitor", "agonist", "gene_therapy", "enzyme_replacement", "aso"],
        "gof": ["inhibitor", "aso"],
        "misfolding": ["chaperone", "modulator", "inhibitor"],
        "mislocalization": ["modulator", "chaperone"],
        "dominant_negative": ["aso", "inhibitor"],
    }
    allowed_mods = _MECH_MODULATION.get(exp_mech, [])
    mech_hit = not exp_mech or output.modulation_type.lower() in allowed_mods
    scores["mechanism_match"] = rubric["mechanism_match"] if mech_hit else 0
    total += scores["mechanism_match"]

    # confidence match
    min_conf = float(expected.get("min_confidence", 0.7))
    scores["confidence_match"] = rubric["confidence_match"] if output.confidence_score >= min_conf else 0
    total += scores["confidence_match"]

    # rationale quality
    ev_text = " ".join(e.claim for e in output.supporting_evidence)
    scores["rationale_quality"] = rubric["rationale_quality"] if len(ev_text) > 50 else 0
    total += scores["rationale_quality"]

    pct = total / max_total * 100 if max_total else 0
    verdict = "PASS" if pct >= 80 else ("PARTIAL" if pct >= 50 else "FAIL")
    return {"scores": scores, "total": total, "max_total": max_total, "pct": pct, "verdict": verdict}


# ── runner ────────────────────────────────────────────────────────────────────

async def run_case(case: BenchmarkCase, model: str) -> tuple[TherapyStrategyOutput, float]:
    """Run the agent on one case. Returns (output, wall_clock_seconds)."""
    t0 = time.perf_counter()
    try:
        final_state = await run_agent(
            gene=case.input.gene,
            mutation=case.input.mutation,
            disease_phenotype=case.input.disease_phenotype,
        )
    except Exception as exc:
        logger.error("Agent error on case %s: %s", case.id, exc)
        # Return a minimal error output
        from therapy_agent.schemas.output import TargetProtein
        final_state = {
            "gene": case.input.gene,
            "mutation": case.input.mutation,
            "disease_phenotype": case.input.disease_phenotype,
            "final_strategy": None,
            "strategy": None,
            "reasoning_trace": [f"ERROR: {exc}"],
            "token_usage": [],
        }
    wall = time.perf_counter() - t0
    output = build_output(
        case_id=case.id,
        final_state=final_state,
        model=model,
        wall_clock=wall,
    )
    return output, wall


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def run(
    model: str = typer.Option(DEFAULT_MODEL, help="Anthropic model ID (pinned default from config.py)"),
    primary_only: bool = typer.Option(False, "--primary-only", help="Run only the two gold-standard YAML cases"),
    case_id: str = typer.Option(None, "--case", help="Run a single case by ID"),
    out_dir: str = typer.Option("benchmark_runs", "--out-dir", help="Parent directory for run output"),
    no_fda_triples: bool = typer.Option(False, "--no-fda-triples", help="Skip fda-strategy-triples; use YAML only"),
) -> None:
    """Run benchmark cases and write JSONL results for bio-rag-eval."""
    os.environ["ANTHROPIC_MODEL"] = model  # explicit override, not setdefault

    # ── output directory ──────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tee console output to log file
    log_path = run_dir / "run.log"
    file_handler = logging.FileHandler(log_path)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, logging.StreamHandler()])

    # ── load cases ────────────────────────────────────────────────────────────
    if no_fda_triples or primary_only:
        # Load only YAML cases
        from therapy_agent.benchmarks.loader import _load_yaml_cases
        from pathlib import Path as _P
        _bd = _P(__file__).parent
        cases = _load_yaml_cases(_bd, tag="yaml_primary")
        if not primary_only:
            _sub = _bd / "cases"
            if _sub.exists():
                cases += _load_yaml_cases(_sub, tag="yaml_supplementary")
    else:
        cases = load_benchmark_cases()

    if case_id:
        cases = [c for c in cases if c.id == case_id]

    if not cases:
        console.print("[red]No cases found.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]therapy-agent benchmark[/bold]  model=[cyan]{model}[/cyan]  cases={len(cases)}")
    console.print(f"Output directory: [dim]{run_dir}[/dim]\n")

    # ── run ───────────────────────────────────────────────────────────────────
    results: list[dict] = []
    jsonl_path = run_dir / "results.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_fh:
        for case in cases:
            console.print(f"[cyan]▶ {case.id}[/cyan]  {case.name[:60]}")
            output, wall = asyncio.run(run_case(case, model))
            grade_result = grade(case, output)

            # Write one line to JSONL immediately (streaming-friendly)
            line = output.model_dump_json()
            jsonl_fh.write(line + "\n")
            jsonl_fh.flush()

            v = grade_result["verdict"]
            color = {"PASS": "green", "PARTIAL": "yellow", "FAIL": "red"}[v]
            console.print(
                f"  [{color}]{v}[/{color}]  "
                f"{grade_result['total']}/{grade_result['max_total']} "
                f"({grade_result['pct']:.0f}%)  "
                f"target=[white]{output.target_protein.name[:30]}[/white]  "
                f"conf={output.confidence_score:.2f}  "
                f"tok_in={output.input_tokens or '?'}  "
                f"wall={wall:.1f}s"
            )

            results.append({
                "case_id": case.id,
                "name": case.name,
                "grade": grade_result,
                "output_summary": {
                    "target_protein": output.target_protein.name,
                    "modulation_type": output.modulation_type,
                    "confidence_score": output.confidence_score,
                    "input_tokens": output.input_tokens,
                    "output_tokens": output.output_tokens,
                    "wall_clock_seconds": output.wall_clock_seconds,
                },
            })

    # ── summary table ─────────────────────────────────────────────────────────
    table = Table(title="Benchmark Summary", box=box.ROUNDED, expand=False)
    table.add_column("Case ID", style="cyan")
    table.add_column("Verdict", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Target")
    table.add_column("Conf", justify="right")
    table.add_column("In-Tok", justify="right")
    table.add_column("Wall(s)", justify="right")

    pass_n = partial_n = fail_n = 0
    for r in results:
        g = r["grade"]
        v = g["verdict"]
        color = {"PASS": "green", "PARTIAL": "yellow", "FAIL": "red"}[v]
        os_ = r["output_summary"]
        table.add_row(
            r["case_id"],
            f"[{color}]{v}[/{color}]",
            f"{g['total']}/{g['max_total']} ({g['pct']:.0f}%)",
            (os_["target_protein"] or "-")[:28],
            f"{os_['confidence_score']:.2f}",
            str(os_["input_tokens"] or "-"),
            f"{os_['wall_clock_seconds']:.1f}",
        )
        if v == "PASS":
            pass_n += 1
        elif v == "PARTIAL":
            partial_n += 1
        else:
            fail_n += 1

    console.print(table)
    console.print(
        f"\n[bold]Results:[/bold] "
        f"[green]{pass_n} PASS[/green]  "
        f"[yellow]{partial_n} PARTIAL[/yellow]  "
        f"[red]{fail_n} FAIL[/red]  "
        f"/ {len(results)} total"
    )

    # ── summary.json ──────────────────────────────────────────────────────────
    summary = {
        "schema_version": "1.0",
        "run_timestamp": ts,
        "model": model,
        "total_cases": len(results),
        "pass": pass_n,
        "partial": partial_n,
        "fail": fail_n,
        "results_jsonl": str(jsonl_path),
        "cases": results,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    console.print(f"\n[dim]JSONL → {jsonl_path}[/dim]")
    console.print(f"[dim]Summary → {summary_path}[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
