"""Pinned runtime configuration.

All LLM model strings are pinned here so benchmark runs are reproducible.
Never import a model ID as a string literal anywhere else in this package —
always import DEFAULT_MODEL (or a role-specific override) from this module.
"""

# ── LLM ───────────────────────────────────────────────────────────────────────
# Pinned to a specific snapshot so benchmark results don't silently shift when
# Anthropic aliases (e.g. "claude-sonnet-4") are updated to new checkpoints.
DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Override at runtime with ANTHROPIC_MODEL env var; the pin is the fallback.
def get_model() -> str:
    import os
    return os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


# ── Retrieval ─────────────────────────────────────────────────────────────────
G2P_RETRIEVAL_K = 5           # top-k chunks from g2p-rag per query

# ── Benchmarks ────────────────────────────────────────────────────────────────
BENCHMARK_SCHEMA_VERSION = "1.0"
FDA_TRIPLES_VERSION = "v0.1.0"
