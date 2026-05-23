#!/usr/bin/env python3
"""Run all therapy-agent benchmark cases and report results."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich import box

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from therapy_agent.graph import run_agent
from therapy_agent.state import make_initial_state

app = typer.Typer()
console = Console()

BENCHMARK_DIR = Path(__file__).parent


def load_cases(primary_only: bool = False) -> list[dict]:
    """Load all benchmark YAML files."""
    cases = []

    # Primary cases
    for f in sorted(BENCHMARK_DIR.glob("*.yaml")):
        try:
            case = yaml.safe_load(f.read_text())
            if case:
                case["_file"] = str(f)
                cases.append(case)
        except Exception as e:
            console.print(f"[red]Failed to load {f}: {e}[/red]")

    if primary_only:
        return cases

    # Curated cases
    for f in sorted((BENCHMARK_DIR / "cases").glob("*.yaml")):
        try:
            case = yaml.safe_load(f.read_text())
            if case:
                case["_file"] = str(f)
                cases.append(case)
        except Exception as e:
            console.print(f"[red]Failed to load {f}: {e}[/red]")

    return cases


def grade_result(case: dict, final_state: dict) -> dict:
    """Score the agent's output against expected outputs."""
    expected = case.get("expected_outputs", {})
    grading = case.get("grading", {})

    strategy = final_state.get("final_strategy") or final_state.get("strategy") or {}
    mechanism = final_state.get("molecular_mechanism", "")
    citations = final_state.get("citations", [])

    scores = {}
    total = 0
    max_total = sum(grading.values()) if grading else 100

    # Target match (30 pts)
    target_protein = (strategy.get("target_protein") or "").lower()
    expected_target = (expected.get("target_protein") or "").lower()
    target_aliases = [a.lower() for a in expected.get("target_aliases", [])]
    target_hit = expected_target in target_protein or any(a in target_protein for a in target_aliases)
    scores["target_match"] = grading.get("target_match", 30) if target_hit else 0
    total += scores["target_match"]

    # Citation match (30 pts)
    all_text = " ".join([
        str(strategy.get("citations", [])),
        str(strategy.get("precedent_drugs", [])),
        str(strategy.get("rationale", "")),
        str(citations),
    ]).lower()
    key_citations = [c.lower() for c in expected.get("key_citations", [])]
    cit_hit = any(kc in all_text for kc in key_citations)
    scores["citation_match"] = grading.get("citation_match", 30) if cit_hit else 0
    total += scores["citation_match"]

    # Mechanism match (20 pts)
    mech_hit = mechanism.lower() == (expected.get("mechanism_class") or "").lower()
    scores["mechanism_match"] = grading.get("mechanism_match", 20) if mech_hit else 0
    total += scores["mechanism_match"]

    # Confidence match (10 pts)
    conf = strategy.get("confidence_score", 0.0)
    min_conf = expected.get("min_confidence", 0.7)
    scores["confidence_match"] = grading.get("confidence_match", 10) if conf >= min_conf else 0
    total += scores["confidence_match"]

    # Rationale quality (10 pts) — basic check
    rationale = strategy.get("rationale", "")
    has_rationale = len(rationale) > 50
    scores["rationale_quality"] = grading.get("rationale_quality", 10) if has_rationale else 0
    total += scores["rationale_quality"]

    pct = total / max_total * 100 if max_total > 0 else 0
    if pct >= 80:
        verdict = "PASS"
    elif pct >= 50:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "scores": scores,
        "total": total,
        "max_total": max_total,
        "pct": pct,
        "verdict": verdict,
        "target_found": strategy.get("target_protein", ""),
        "mechanism_found": mechanism,
        "confidence_found": conf,
    }


async def run_case(case: dict, model: str) -> dict:
    """Run a single benchmark case."""
    inp = case["input"]
    try:
        final = await run_agent(
            gene=inp["gene"],
            mutation=inp["mutation"],
            disease_phenotype=inp["disease_phenotype"],
        )
        return final
    except Exception as e:
        console.print(f"[red]Error running {case.get('id', '?')}: {e}[/red]")
        return {}


@app.command()
def run(
    model: str = typer.Option("claude-sonnet-4-6", help="Anthropic model ID"),
    primary_only: bool = typer.Option(False, "--primary-only", help="Run only Case 1 & 2"),
    output_json: str = typer.Option(None, "--output-json", help="Write results JSON to file"),
    case_id: str = typer.Option(None, "--case", help="Run a single case by ID"),
):
    """Run all benchmark cases and print a results table."""
    os.environ.setdefault("ANTHROPIC_MODEL", model)

    cases = load_cases(primary_only)
    if case_id:
        cases = [c for c in cases if c.get("id") == case_id]

    if not cases:
        console.print("[red]No cases found.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Running {len(cases)} benchmark case(s) with model {model}[/bold]\n")

    results = []
    for case in cases:
        console.print(f"[cyan]Running: {case.get('name', case.get('id', '?'))}[/cyan]")
        final = asyncio.run(run_case(case, model))
        grade = grade_result(case, final)
        results.append({
            "id": case.get("id", "?"),
            "name": case.get("name", "?"),
            "grade": grade,
            "final_state": final,
        })
        verdict_color = {"PASS": "green", "PARTIAL": "yellow", "FAIL": "red"}[grade["verdict"]]
        console.print(f"  [{verdict_color}]{grade['verdict']}[/{verdict_color}] "
                     f"({grade['total']}/{grade['max_total']} = {grade['pct']:.0f}%)  "
                     f"target={grade['target_found']!r}  conf={grade['confidence_found']:.2f}")

    # Summary table
    table = Table(title="Benchmark Results", box=box.ROUNDED, expand=False)
    table.add_column("Case ID", style="cyan")
    table.add_column("Name", style="white", max_width=40)
    table.add_column("Verdict", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Target Found")
    table.add_column("Conf", justify="right")

    pass_count = partial_count = fail_count = 0
    for r in results:
        g = r["grade"]
        v = g["verdict"]
        color = {"PASS": "green", "PARTIAL": "yellow", "FAIL": "red"}[v]
        table.add_row(
            r["id"],
            r["name"][:40],
            f"[{color}]{v}[/{color}]",
            f"{g['total']}/{g['max_total']} ({g['pct']:.0f}%)",
            g["target_found"][:25] if g["target_found"] else "-",
            f"{g['confidence_found']:.2f}",
        )
        if v == "PASS":
            pass_count += 1
        elif v == "PARTIAL":
            partial_count += 1
        else:
            fail_count += 1

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] [green]{pass_count} PASS[/green]  [yellow]{partial_count} PARTIAL[/yellow]  [red]{fail_count} FAIL[/red]  "
                 f"out of {len(results)} cases")

    if output_json:
        with open(output_json, "w") as f:
            json.dump([{"id": r["id"], "grade": r["grade"]} for r in results], f, indent=2)
        console.print(f"[dim]JSON results written to {output_json}[/dim]")


def main():
    app()


if __name__ == "__main__":
    main()
