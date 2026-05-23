import asyncio
import json
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from therapy_agent.graph import run_agent

app = typer.Typer(help="Therapy Agent — hypothesize therapeutic strategies from genetic variants.")
console = Console()

NODE_DESCRIPTIONS = {
    "parse_input": "Parsing gene/mutation/phenotype input",
    "variant_lookup": "Querying ClinVar & g2p-rag for variant data",
    "mechanism_classifier": "Classifying molecular mechanism (LoF/GoF/misfolding/...)",
    "pathway_expansion": "Expanding pathway via Reactome",
    "druggable_target_search": "Searching ChEMBL & DrugBank for druggable targets",
    "strategy_synthesis": "Synthesizing therapeutic strategy with Claude",
    "self_critique": "Self-critique: reviewing evidence & confidence",
}


async def _run_with_streaming(gene, mutation, phenotype, model):
    os.environ.setdefault("ANTHROPIC_MODEL", model)

    node_order = []

    async def stream_cb(node_name: str, node_output: dict):
        desc = NODE_DESCRIPTIONS.get(node_name, node_name)
        console.print(Panel(f"[bold cyan]{node_name}[/bold cyan]  [dim]{desc}[/dim]", expand=False))

        trace = node_output.get("reasoning_trace", [])
        for line in trace:
            console.print(f"  [dim]→ {line}[/dim]")

        errs = node_output.get("errors", [])
        for e in errs:
            console.print(f"  [red]⚠ {e}[/red]")

        mech = node_output.get("molecular_mechanism")
        if mech:
            conf = node_output.get("mechanism_confidence")
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
            console.print(f"  [yellow]Mechanism: {mech}  (confidence: {conf_str})[/yellow]")

        pathway = node_output.get("pathway_genes")
        if pathway:
            console.print(f"  [green]Pathway genes: {', '.join(pathway[:8])}{'...' if len(pathway) > 8 else ''}[/green]")

        targets = node_output.get("candidate_targets", [])
        if targets:
            names = [t.get("gene_name", t.get("name", "?")) for t in targets[:5]]
            console.print(f"  [green]Candidate targets: {', '.join(names)}[/green]")

        strat = node_output.get("strategy")
        if strat:
            console.print(f"  [bold green]Strategy drafted: {strat.get('target_protein')} ({strat.get('modulation_type')})[/bold green]")

        notes = node_output.get("critique_notes", [])
        for n in notes:
            console.print(f"  [magenta]Critique: {n}[/magenta]")

        node_order.append(node_name)

    console.print(Panel(
        f"[bold]Therapy Agent[/bold]\n"
        f"Gene: [cyan]{gene}[/cyan]  Mutation: [cyan]{mutation}[/cyan]  Phenotype: [cyan]{phenotype}[/cyan]",
        title="[bold yellow]Starting analysis[/bold yellow]",
        expand=False,
    ))

    final = await run_agent(gene, mutation, phenotype, stream_callback=stream_cb)

    # Print final strategy
    strat = final.get("final_strategy") or final.get("strategy")
    if strat:
        t = Table(title="Therapeutic Strategy", box=box.ROUNDED, expand=False)
        t.add_column("Field", style="bold cyan", no_wrap=True)
        t.add_column("Value", style="white")
        t.add_row("Target Protein", strat.get("target_protein", ""))
        t.add_row("Target Pathway", strat.get("target_pathway", ""))
        t.add_row("Modulation", strat.get("modulation_type", ""))
        t.add_row("Confidence", f"{strat.get('confidence_score', 0):.2f}")
        t.add_row("Rationale", strat.get("rationale", "")[:200])
        for i, d in enumerate(strat.get("precedent_drugs", [])[:3], 1):
            t.add_row(f"Precedent Drug {i}", d)
        for i, c in enumerate(strat.get("citations", [])[:4], 1):
            t.add_row(f"Citation {i}", c)
        console.print(t)
    else:
        console.print("[red]No strategy generated.[/red]")
        if final.get("errors"):
            for e in final["errors"]:
                console.print(f"[red]Error: {e}[/red]")

    return final


@app.command()
def run(
    gene: str = typer.Option(..., help="Gene symbol (e.g. SERPING1)"),
    mutation: str = typer.Option(..., help="Mutation description (e.g. 'frameshift causing haploinsufficiency')"),
    phenotype: str = typer.Option(..., help="Disease phenotype (e.g. 'hereditary angioedema')"),
    model: str = typer.Option("claude-sonnet-4-6", help="Anthropic model ID"),
    output: Optional[str] = typer.Option(None, help="Write JSON output to file"),
):
    """Run the therapy-agent pipeline for a gene/mutation/phenotype."""
    final = asyncio.run(_run_with_streaming(gene, mutation, phenotype, model))

    if output:
        strat = final.get("final_strategy") or final.get("strategy")
        with open(output, "w") as f:
            json.dump({"strategy": strat, "citations": final.get("citations", [])}, f, indent=2)
        console.print(f"[dim]JSON written to {output}[/dim]")


@app.command()
def benchmark(
    case_file: str = typer.Argument(..., help="Path to benchmark YAML file"),
    model: str = typer.Option("claude-sonnet-4-6", help="Model ID"),
):
    """Run a single benchmark case from YAML."""
    import yaml
    with open(case_file) as f:
        case = yaml.safe_load(f)
    inp = case["input"]
    asyncio.run(_run_with_streaming(inp["gene"], inp["mutation"], inp["disease_phenotype"], model))


def main():
    app()


if __name__ == "__main__":
    main()
