"""LangGraph state machine for the therapy-agent."""
import asyncio
import os
from langgraph.graph import StateGraph, END
from therapy_agent.state import AgentState
from therapy_agent.nodes.parse_input import parse_input_node
from therapy_agent.nodes.variant_lookup import variant_lookup_node
from therapy_agent.nodes.mechanism_classifier import mechanism_classifier_node
from therapy_agent.nodes.pathway_expansion import pathway_expansion_node
from therapy_agent.nodes.druggable_target_search import druggable_target_search_node
from therapy_agent.nodes.interactor_biology_lookup import interactor_biology_lookup_node
from therapy_agent.nodes.strategy_synthesis import strategy_synthesis_node
from therapy_agent.nodes.self_critique import self_critique_node


def should_revise(state: AgentState) -> str:
    """Trigger one revise pass on every case (not just low-confidence ones).

    v0.4 found that the 3B Llama reports 0.8-0.9 confidence for almost every
    output -- including ones where the rationale describes a target the
    `target_protein` field doesn't reflect (e.g. rationale names TMED9 as
    the cargo receptor, target_protein says UMOD). Always firing one
    critique pass gives the model a chance to align the target with its own
    rationale before we score it. retry_count gating still caps total
    iterations at 2.
    """
    strategy = state.get("strategy")
    retry_count = state.get("retry_count", 0)
    critique_pass_done = state.get("critique_pass_done", False)
    if strategy and retry_count < 2 and not critique_pass_done:
        return "revise"
    return "done"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("parse_input", parse_input_node)
    builder.add_node("variant_lookup", variant_lookup_node)
    builder.add_node("mechanism_classifier", mechanism_classifier_node)
    builder.add_node("pathway_expansion", pathway_expansion_node)
    builder.add_node("druggable_target_search", druggable_target_search_node)
    builder.add_node("interactor_biology_lookup", interactor_biology_lookup_node)
    builder.add_node("strategy_synthesis", strategy_synthesis_node)
    builder.add_node("self_critique", self_critique_node)

    builder.set_entry_point("parse_input")
    builder.add_edge("parse_input", "variant_lookup")
    builder.add_edge("variant_lookup", "mechanism_classifier")
    builder.add_edge("mechanism_classifier", "pathway_expansion")
    builder.add_edge("pathway_expansion", "druggable_target_search")
    builder.add_edge("druggable_target_search", "interactor_biology_lookup")
    builder.add_edge("interactor_biology_lookup", "strategy_synthesis")
    builder.add_edge("strategy_synthesis", "self_critique")
    builder.add_conditional_edges(
        "self_critique",
        should_revise,
        {"revise": "strategy_synthesis", "done": END},
    )
    return builder.compile()


graph = build_graph()


async def run_agent(gene: str, mutation: str, disease_phenotype: str, stream_callback=None) -> AgentState:
    """Run the full agent pipeline and return final state."""
    from therapy_agent.state import make_initial_state
    initial = make_initial_state(gene, mutation, disease_phenotype)

    final_state = initial
    async for step in graph.astream(initial):
        node_name = list(step.keys())[0]
        node_output = step[node_name]
        # merge into final state
        for k, v in node_output.items():
            final_state = dict(final_state)
            if isinstance(v, list) and isinstance(final_state.get(k), list):
                # for Annotated[list, add] fields
                if k in ("reasoning_trace", "citations", "errors"):
                    final_state[k] = final_state.get(k, []) + v
                else:
                    final_state[k] = v
            else:
                final_state[k] = v
        if stream_callback:
            await stream_callback(node_name, node_output)
    return final_state
