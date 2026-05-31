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

# Path to the prebuilt ChromaDB. Override at runtime via G2P_INDEX_DIR env var.
# Default points to the sibling g2p-rag checkout's data/chroma directory; the
# CI workflow downloads the snapshot to the same path before running the bench.
def get_g2p_index_dir() -> str:
    import os
    from pathlib import Path
    if (v := os.environ.get("G2P_INDEX_DIR")):
        return v
    # Sibling-repo default for local dev: ../g2p-rag/data/chroma
    # config.py lives at therapy-agent/src/therapy_agent/config.py
    # parents[0] = therapy_agent/, [1] = src/, [2] = therapy-agent/, [3] = .windsurf/
    sibling_root = Path(__file__).resolve().parents[3]
    return str(sibling_root / "g2p-rag" / "data" / "chroma")

# ── Benchmarks ────────────────────────────────────────────────────────────────
BENCHMARK_SCHEMA_VERSION = "1.0"
FDA_TRIPLES_VERSION = "v0.1.0"
