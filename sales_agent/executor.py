"""Shared tool execution: runs one tool and yields its trace events.

Usage inside an agent generator:
    result, ok = yield from execute_tool(ctx, tracer, "calculate_revenue", {...})
"""
from __future__ import annotations

from typing import Any, Generator, Optional

from .tools import TOOL_REGISTRY, SalesContext
from .trace import TraceEvent, Tracer, summarise


def execute_tool(
    ctx: SalesContext,
    tracer: Tracer,
    name: str,
    args: dict[str, Any],
    model: Optional[str] = None,
    note: Optional[str] = None,
) -> Generator[TraceEvent, None, tuple[dict, bool]]:
    shown_input = {**args, "_note": note} if note else args
    yield tracer.record(
        TraceEvent(step=tracer.next_step(), kind="tool_call", tool_name=name, input=shown_input, model=model,
                   output_summary=note or "")
    )
    fn = TOOL_REGISTRY.get(name)
    t0 = tracer.timer()
    if fn is None:
        result = {"error": f"Unknown tool {name!r}. Available: {', '.join(TOOL_REGISTRY)}"}
        ok = False
    else:
        try:
            result = fn(ctx, **args)
            ok = True
        except TypeError as e:  # wrong / missing arguments
            result = {"error": f"Invalid arguments for {name}: {e}"}
            ok = False
        except (ValueError, KeyError) as e:
            result = {"error": f"{name} failed: {e}"}
            ok = False
    ms = tracer.elapsed_ms(t0)
    yield tracer.record(
        TraceEvent(
            step=tracer.next_step(),
            kind="tool_result" if ok else "error",
            tool_name=name,
            input=args,
            output=result,
            output_summary=summarise(result),
            duration_ms=ms,
            model=model,
        )
    )
    return result, ok
