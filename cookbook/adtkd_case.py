#!/usr/bin/env python3
"""
Cookbook Example 2: ADTKD — UMOD/MUC1 Misfolding → BRD4780/TMED9

Run this script to see the therapy-agent reason through the ADTKD case
and arrive at TMED cargo receptors as the therapeutic target, citing
BRD4780 (Dvela-Levitt et al. Cell 2019).

Usage:
    python cookbook/adtkd_case.py
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
    "gene": "UMOD",
    "mutation": "frameshift mutation causing protein misfolding and ER retention in kidney epithelial cells",
    "disease_phenotype": (
        "autosomal dominant tubulointerstitial kidney disease (ADTKD) with tubular "
        "atrophy, interstitial fibrosis, and progressive renal failure by the 5th decade"
    ),
}

BACKGROUND = """
ADTKD Background:
  ADTKD-UMOD is caused by dominant mutations (typically missense, occasionally
  frameshift) in UMOD, encoding uromodulin (Tamm-Horsfall protein), the most
  abundant urinary protein. Mutant uromodulin misfolds and is retained in the
  ER of thick ascending limb (TAL) cells, triggering UPR and progressive tubular
  damage.

  The same ER retention mechanism operates in ADTKD-MUC1, caused by a frameshift
  in the VNTR region of MUC1, creating a truncated mucin protein (MUC1-fs).

  BRD4780, identified by Dvela-Levitt et al. (Cell 2019), is a small molecule that
  binds TMED9 — a COPI vesicle cargo receptor — and redirects trapped misfolded
  UMOD and MUC1-fs from ER retention to lysosomal degradation, rescuing kidney
  function in mouse models without affecting wild-type protein secretion.
"""


async def main():
    console.print(Panel(BACKGROUND.strip(), title="ADTKD Background", expand=False))

    async def stream_cb(node_name: str, node_output: dict):
        trace = node_output.get("reasoning_trace", [])
        console.print(f"\n[bold cyan][{node_name}][/bold cyan]")
        for t in trace:
            console.print(f"  → {t}")
        mech = node_output.get("molecular_mechanism")
        if mech:
            conf = node_output.get("mechanism_confidence", 0)
            console.print(f"  Mechanism: [yellow]{mech}[/yellow] (confidence={conf:.2f})")
        pathway = node_output.get("pathway_genes", [])
        if pathway:
            console.print(f"  Pathway genes: {', '.join(pathway[:8])}")
        strategy = node_output.get("strategy")
        if strategy:
            console.print(f"  Strategy: [green]{strategy.get('target_protein')}[/green]")

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
    mech = final.get("molecular_mechanism", "")

    checks = [
        ("Mechanism is misfolding", mech == "misfolding"),
        ("Target is TMED9/TMED2/TMED10 or cargo receptor", any(t in target for t in ["tmed9", "tmed2", "tmed10", "cargo receptor", "tmed"])),
        ("Cites BRD4780 or Dvela-Levitt", "brd4780" in cit_text or "dvela-levitt" in cit_text or "dvela" in cit_text),
        ("Confidence >= 0.75", (strat or {}).get("confidence_score", 0) >= 0.75),
    ]
    for desc, passed in checks:
        icon = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  {icon}  {desc}")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]Set ANTHROPIC_API_KEY environment variable first.[/red]")
        sys.exit(1)
    asyncio.run(main())
