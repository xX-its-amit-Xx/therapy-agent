"""Pluggable LLM backend.

Three backends, one shape:

    from therapy_agent.llm import get_backend
    client = get_backend()
    resp = client.messages.create(
        model="...",
        max_tokens=1500,
        system="You are...",
        messages=[{"role": "user", "content": "..."}],
    )
    text = resp.content[0].text
    n_in, n_out = resp.usage.input_tokens, resp.usage.output_tokens

- "anthropic"  -> forwards directly to the Anthropic SDK (Claude).
- "llama"      -> local GGUF via llama-cpp-python, response shaped Anthropic-like.
- "openai"     -> calls OpenAI Chat Completions API (GPT-4o / GPT-4.1 / o-series),
                  response shaped Anthropic-like.

Select with env var ``THERAPY_AGENT_LLM_BACKEND`` (default: anthropic).
For the llama backend, ``LLAMA_MODEL_PATH`` points to the GGUF file.
For the openai backend, ``OPENAI_API_KEY`` and optional ``OPENAI_MODEL``
(default ``gpt-4o-mini``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


# ── Anthropic-shape response shims ────────────────────────────────────────────

@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Response:
    content: list[_TextBlock]
    usage: _Usage
    stop_reason: str = "end_turn"
    model: str = ""


# ── Anthropic backend (pass-through) ──────────────────────────────────────────

class _AnthropicBackend:
    """Thin wrapper that just delegates to the real Anthropic SDK."""

    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    @property
    def messages(self):
        return self._client.messages


# ── Llama (llama-cpp-python) backend ──────────────────────────────────────────

class _LlamaMessages:
    """Mimics Anthropic's `client.messages.create(...)` against a local Llama."""

    def __init__(self, backend: "_LlamaBackend") -> None:
        self._b = backend

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str = "",
        temperature: float = 0.2,
        **_: Any,
    ) -> _Response:
        # Build the chat-completion payload llama-cpp wants.
        chat_msgs: list[dict[str, str]] = []
        if system:
            chat_msgs.append({"role": "system", "content": system})
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                # Anthropic-style content blocks → flatten to text.
                parts = [c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") == "text"]
                content = "\n".join(p for p in parts if p)
            chat_msgs.append({"role": m["role"], "content": content})

        resp = self._b.llm.create_chat_completion(
            messages=chat_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {}) or {}
        return _Response(
            content=[_TextBlock(text=text)],
            usage=_Usage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            ),
            stop_reason="end_turn",
            model=model,
        )


class _LlamaBackend:
    """Loads a GGUF Llama model once and exposes Anthropic-shaped `.messages`."""

    def __init__(self) -> None:
        from llama_cpp import Llama
        model_path = os.environ.get(
            "LLAMA_MODEL_PATH",
            "C:/llama-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Llama model file not found at {model_path!r}. "
                "Set LLAMA_MODEL_PATH or download with huggingface_hub."
            )
        self.llm = Llama(
            model_path=model_path,
            n_ctx=int(os.environ.get("LLAMA_N_CTX", "8192")),
            n_threads=int(os.environ.get("LLAMA_N_THREADS",
                                         str(os.cpu_count() or 8))),
            seed=int(os.environ.get("LLAMA_SEED", "7")),
            verbose=False,
        )
        self._messages = _LlamaMessages(self)

    @property
    def messages(self) -> _LlamaMessages:
        return self._messages


# ── OpenAI backend ────────────────────────────────────────────────────────────

class _OpenAIMessages:
    """Mimics Anthropic's `client.messages.create(...)` against OpenAI's
    Chat Completions API. Lets the rest of the pipeline (which targets
    Anthropic's shape) run unchanged with GPT-4o / GPT-4.1 / etc."""

    def __init__(self, backend: "_OpenAIBackend") -> None:
        self._b = backend

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str = "",
        temperature: float = 0.2,
        **_: Any,
    ) -> _Response:
        chat_msgs: list[dict[str, str]] = []
        if system:
            chat_msgs.append({"role": "system", "content": system})
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                parts = [c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") == "text"]
                content = "\n".join(p for p in parts if p)
            chat_msgs.append({"role": m["role"], "content": content})

        # OPENAI_MODEL takes precedence over whatever model string the
        # caller hands us (the caller's `model` typically comes from
        # therapy_agent.config.get_model() which is the Anthropic alias).
        openai_model = os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"

        resp = self._b.client.chat.completions.create(
            model=openai_model,
            messages=chat_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return _Response(
            content=[_TextBlock(text=text)],
            usage=_Usage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            ),
            stop_reason=str(resp.choices[0].finish_reason or "stop"),
            model=openai_model,
        )


class _OpenAIBackend:
    def __init__(self) -> None:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI backend selected but OPENAI_API_KEY env var is not set."
            )
        self.client = openai.OpenAI(api_key=api_key)
        self._messages = _OpenAIMessages(self)

    @property
    def messages(self) -> _OpenAIMessages:
        return self._messages


# ── Singleton accessor ────────────────────────────────────────────────────────

_BACKEND_CACHE: dict[str, Any] = {}


def get_backend(name: str | None = None):
    """Return a singleton client of the chosen backend.

    Backends are cached by name so Llama only loads once per process and
    OpenAI / Anthropic API clients are reused (connection pooling).
    """
    if name is None:
        name = os.environ.get("THERAPY_AGENT_LLM_BACKEND", "anthropic").lower()
    if name not in _BACKEND_CACHE:
        if name == "anthropic":
            _BACKEND_CACHE[name] = _AnthropicBackend()
        elif name == "llama":
            _BACKEND_CACHE[name] = _LlamaBackend()
        elif name == "openai":
            _BACKEND_CACHE[name] = _OpenAIBackend()
        else:
            raise ValueError(
                f"Unknown LLM backend {name!r}. "
                "Choose from: anthropic, llama, openai."
            )
    return _BACKEND_CACHE[name]


def current_backend_name() -> str:
    return os.environ.get("THERAPY_AGENT_LLM_BACKEND", "anthropic").lower()
