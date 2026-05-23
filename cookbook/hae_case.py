#!/usr/bin/env python3
"""
Cookbook Example 1: Hereditary Angioedema (HAE) — SERPING1 LoF

Run this script to see the therapy-agent reason through the HAE case
and arrive at plasma kallikrein (KLKB1) as the therapeutic target,
citing sebetralstat (Ekterly), FDA approved July 2025.

Usage:
    python cookbook/hae_case.py
"""
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from therapy_agent.graph import run_agent

console = Console()

CASE = {
    "gene": "SERPING1",
    "mutation": "frameshift or large deletion causing haploinsufficiency of C1 esterase inhibitor",
    "disease_phenotype": (
        "hereditary angioedema (HAE) with recurrent attacks of subcutaneous and "
        "mucosal edema, triggered by trauma, stress, or estrogen"
    ),
}

BACKGROUND = """
HAE Background:
  SERPING1 encodes C1-esterase inhibitor (C1-INH), the primary brake on the
  contact activation pathway. Haploinsufficiency leads to uncontrolled plasma
  kallikrein (KLKB1) activity, generating excess bradykinin from high-molecular-
  weight kininogen. Bradykinin binds BDKRB2 on endothelial cells, increasing
  vascular permeability and causing angioedema attacks.

  Three therapeutic approaches are approved:
    1. Replacement: C1-INH concentrate (Berinert, Cinryze)
    2. Kallikrein inhibition: ecallantide, berotralstat, sebetralstat (Ekterly)
    3. Downstream: bradykinin B2 receptor antagonist (icatibant)

  Sebetralstat (Ekterly, KalVista) is the first oral on-demand kallikrein
  inhibitor, approved by FDA in July 2025 based on the KONFIDENT Phase 3 trial.
"""


async def main():
    console.print(Panel(BACKGROUND.strip(), title="HAE Background", expand=False))

    nodes_seen = []

    async def stream_cb(node_name: str, node_output: dict):
        nodes_seen.append(node_name)
        trace = node_output.get("reasoning_trace", [])
        mech = node_output.get("molecular_mechanism")
        pathway = node_output.get("pathway_genes", [])
        strategy = node_output.get("strategy")

        console.print(f"\n[bold cyan][{node_name}][/bold cyan]")
        for t in trace:
            console.print(f"  → {t}")
        if mech:
            conf = node_output.get("mechanism_confidence", 0)
            console.print(f"  Mechanism: [yellow]{mech}[/yellow] (confidence={conf:.2f})")
        if pathway:
            console.print(f"  Pathway: {', '.join(pathway[:8])}")
        if strategy:
            console.print(f"  Strategy: [green]{strategy.get('target_protein')}[/green] / {strategy.get('modulation_type')}")

    console.print(Panel(
        f"Gene: [cyan]{CASE['gene']}[/cyan]\n"
        f"Mutation: [cyan]{CASE['mutation']}[/cyan]\n"
        f"Phenotype: [cyan]{CASE['disease_phenotype']}[/cyan]",
        title="Input",
        expand=False,
    ))

    final = await run_agent(
        gene=CASE["gene"],
        mutation=CASE["mutation"],
        disease_phenotype=CASE["disease_phenotype"],
        stream_callback=stream_cb,
    )

    strat = final.get("final_strategy") or final.get("strategy")
    if strat:
        t = Table(title="Final Therapeutic Strategy", box=box.ROUNDED)
        t.add_column("Field", style="bold cyan")
        t.add_column("Value")
        t.add_row("Target", strat.get("target_protein", ""))
        t.add_row("Pathway", strat.get("target_pathway", ""))
        t.add_row("Modulation", strat.get("modulation_type", ""))
        t.add_row("Confidence", f"{strat.get('confidence_score', 0):.2f}")
        for d in strat.get("precedent_drugs", [])[:3]:
            t.add_row("Precedent Drug", d)
        for c in strat.get("citations", [])[:3]:
            t.add_row("Citation", c)
        t.add_row("Rationale", strat.get("rationale", "")[:300])
        console.print(t)

    # Validation
    console.print("\n[bold]Validation checks:[/bold]")
    target = (strat or {}).get("target_protein", "").lower()
    cit_text = str((strat or {}).get("citations", [])).lower() + str((strat or {}).get("precedent_drugs", [])).lower()

    checks = [
        ("Target is KLKB1/plasma kallikrein", "klkb1" in target or "kallikrein" in target),
        ("Cites sebetralstat or Ekterly", "sebetralstat" in cit_text or "ekterly" in cit_text),
        ("Modulation type = inhibitor", (strat or {}).get("modulation_type") == "inhibitor"),
        ("Confidence >= 0.80", (strat or {}).get("confidence_score", 0) >= 0.80),
    ]
    for desc, passed in checks:
        icon = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  {icon}  {desc}")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]Set ANTHROPIC_API_KEY environment variable first.[/red]")
        sys.exit(1)
    asyncio.run(main())
