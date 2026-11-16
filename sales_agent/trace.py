"""Execution trace: one TraceEvent per agent step, exportable as JSON."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

EventKind = Literal["llm_call", "tool_call", "tool_result", "error", "final"]

TRACE_DIR = Path(__file__).resolve().parent.parent / "traces"


@dataclass
class TraceEvent:
    step: int
    kind: EventKind
    tool_name: Optional[str] = None
    input: Any = None
    output: Any = None  # full output (kept for JSON export)
    output_summary: str = ""
    duration_ms: Optional[float] = None
    tokens: Optional[dict] = None  # {"prompt": int, "completion": int}
    model: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="milliseconds"))

    @property
    def title(self) -> str:
        labels = {
            "llm_call": "🧠 LLM decision",
            "tool_call": f"🔧 Call {self.tool_name}",
            "tool_result": f"✅ {self.tool_name}",
            "error": f"⚠️ Error{f' in {self.tool_name}' if self.tool_name else ''}",
            "final": "🏁 Final report",
        }
        return f"Step {self.step} · {labels[self.kind]}"


class Tracer:
    """Collects events and hands out step numbers + timers."""

    def __init__(self, mode: str, question: str, model: Optional[str] = None):
        self.mode, self.question, self.model = mode, question, model
        self.events: list[TraceEvent] = []
        self.started = datetime.now()
        self._step = 0

    def next_step(self) -> int:
        self._step += 1
        return self._step

    def record(self, event: TraceEvent) -> TraceEvent:
        self.events.append(event)
        return event

    @staticmethod
    def timer() -> float:
        return time.perf_counter()

    @staticmethod
    def elapsed_ms(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000, 1)

    def totals(self) -> dict:
        prompt = sum((e.tokens or {}).get("prompt", 0) or 0 for e in self.events)
        completion = sum((e.tokens or {}).get("completion", 0) or 0 for e in self.events)
        return {
            "steps": len(self.events),
            "tool_calls": sum(e.kind == "tool_call" for e in self.events),
            "errors": sum(e.kind == "error" for e in self.events),
            "duration_ms": sum(e.duration_ms or 0 for e in self.events),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        }

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "model": self.model,
            "question": self.question,
            "started": self.started.isoformat(timespec="seconds"),
            "totals": self.totals(),
            "events": [asdict(e) for e in self.events],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, directory: Path = TRACE_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.started:%Y%m%d-%H%M%S}-{self.mode}.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def summarise(obj: Any, limit: int = 220) -> str:
    """Short one-line summary of a tool output for the trace panel."""
    if isinstance(obj, dict):
        if "error" in obj:
            return f"error: {obj['error']}"
        keys = ["total_revenue", "rows", "orders", "product_count", "status"]
        parts = [f"{k}={obj[k]}" for k in keys if k in obj]
        if "overall" in obj:
            parts.append(f"growth={obj['overall'].get('growth_pct')}%")
        if "underperformers" in obj:
            parts.append("underperformers=" + (", ".join(u["product"] for u in obj["underperformers"]) or "none"))
        if parts:
            return "; ".join(parts)[:limit]
    text = json.dumps(obj, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"
