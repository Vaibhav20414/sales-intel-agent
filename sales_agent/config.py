"""Runtime configuration (env vars / .env, overridable from the UI)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

DEFAULT_MODEL = "ollama_chat/qwen2.5:7b"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"

# Examples shown in the UI. Any LiteLLM model string works.
MODEL_PRESETS = [
    "ollama_chat/qwen2.5:7b",
    "ollama_chat/llama3.1:8b",
    "ollama_chat/mistral-nemo",
    "anthropic/claude-opus-5-5",
    "openai/gpt-4o-mini",
    "gemini/gemini-2.0-flash",
    "groq/llama-3.3-70b-versatile",
]


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    api_base: Optional[str] = None
    temperature: float = 0.1
    max_iterations: int = 12
    timeout: float = 180.0

    @property
    def is_ollama(self) -> bool:
        return self.model.startswith(("ollama/", "ollama_chat/"))

    @property
    def effective_api_base(self) -> Optional[str]:
        if self.api_base:
            return self.api_base
        if self.is_ollama:
            return os.getenv("OLLAMA_API_BASE", DEFAULT_OLLAMA_BASE)
        return None

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
            api_base=os.getenv("LLM_API_BASE") or None,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_iterations=int(os.getenv("LLM_MAX_ITERATIONS", "12")),
        )
