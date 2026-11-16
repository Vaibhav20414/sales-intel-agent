"""Provider-agnostic LLM agent: a manual tool-use loop over litellm.completion.

Works with any LiteLLM model string (ollama_chat/..., anthropic/..., openai/...,
gemini/..., groq/...). Includes safeguards for small local models: argument
cleaning, text-embedded tool-call recovery, error feedback, and an automatic
final report if the model never calls generate_report.
"""
from __future__ import annotations

import inspect
import json
import re
import uuid
from typing import Any, Iterator, Optional

import litellm

from .config import AgentConfig
from .executor import execute_tool
from .periods import parse_period
from .report import rule_based_findings, rule_based_recommendations
from .schemas import TOOLS
from .tools import TOOL_REGISTRY, SalesContext
from .trace import TraceEvent, Tracer, summarise

litellm.drop_params = True  # silently drop params a provider doesn't support
litellm.suppress_debug_info = True
MAX_TOOL_RESULT_CHARS = 6000

SYSTEM_PROMPT = """You are a sales intelligence analyst. You answer questions about the user's sales data \
by calling tools. Never invent numbers; only use numbers returned by tools.

Today's date (latest date in the data) is {today}. The dataset covers {start} to {end}.
Unless the user says otherwise, "this month" means {cur_start} to {cur_end} and should be \
compared with the same days of the previous month: {prev_start} to {prev_end}.

Recommended steps:
1. get_sales_data for the current period
2. calculate_revenue for the current period (group_by "product")
3. get_product_performance comparing current vs previous period
4. calculate_growth comparing current vs previous period (group_by "category")
5. generate_report with a summary, findings and specific recommendations

Always finish by calling generate_report exactly once. Dates must be YYYY-MM-DD."""


# ---- helpers ----------------------------------------------------------------
def build_system_prompt(ctx: SalesContext, question: str) -> str:
    start, end = ctx.coverage
    p = parse_period(question, ctx.today)
    return SYSTEM_PROMPT.format(
        today=ctx.today, start=start, end=end,
        cur_start=p["current_start"], cur_end=p["current_end"],
        prev_start=p["previous_start"], prev_end=p["previous_end"],
    )


def check_tool_support(model: str) -> Optional[bool]:
    """True/False if LiteLLM knows whether the model supports tools, None if unknown."""
    try:
        return bool(litellm.supports_function_calling(model=model))
    except Exception:
        return None


def friendly_error(e: Exception, cfg: AgentConfig) -> str:
    name, msg = type(e).__name__, str(e)
    low = msg.lower()
    if cfg.is_ollama:
        if "connect" in name.lower() or "connection refused" in low or "connecterror" in low or "10061" in low:
            return (f"Ollama not reachable at {cfg.effective_api_base}. Start it with `ollama serve` "
                    "(or open the Ollama app) and try again.")
        if "not found" in low or "notfound" in name.lower():
            tag = cfg.model.split("/", 1)[-1]
            return f"Model '{tag}' is not installed in Ollama. Run `ollama pull {tag}`."
    if "does not support tools" in low or "tool" in low and "not support" in low:
        return (f"Model '{cfg.model}' does not support tool calling. Try a tool-capable model "
                "(e.g. qwen2.5, llama3.1, mistral-nemo) or use Offline mode.")
    if "authentication" in name.lower() or "api key" in low or "api_key" in low:
        return f"Authentication failed for '{cfg.model}'. Set the provider's API key (e.g. OPENAI_API_KEY) in .env."
    if "ratelimit" in name.lower():
        return f"Rate limit hit for '{cfg.model}'. Wait a moment and retry."
    if "timeout" in name.lower():
        return f"The model timed out after {cfg.timeout:.0f}s. Local models can be slow on first load; retry."
    return f"{name}: {msg[:400]}"


def _parse_args(raw: Any) -> tuple[Optional[dict], Optional[str]]:
    if isinstance(raw, dict):
        return raw, None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    try:
        val = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"arguments are not valid JSON ({e}). Send a JSON object."
    if not isinstance(val, dict):
        return None, "arguments must be a JSON object."
    return val, None


def _clean_args(name: str, args: dict) -> tuple[dict, list[str]]:
    """Drop unknown keys and null-ish values that small models often send."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return args, []
    allowed = set(inspect.signature(fn).parameters) - {"ctx"}
    dropped = [k for k in args if k not in allowed]
    cleaned = {k: v for k, v in args.items() if k in allowed and v not in (None, "", "null", "None")}
    return cleaned, dropped


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _text_tool_calls(content: Optional[str]) -> list[dict]:
    """Recover tool calls that a local model wrote as JSON text instead of a real tool call."""
    if not content:
        return []
    m = _JSON_OBJ.search(content)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    items = obj if isinstance(obj, list) else [obj]
    calls = []
    for it in items:
        if isinstance(it, dict) and it.get("name") in TOOL_REGISTRY:
            args = it.get("arguments", it.get("parameters", {}))
            calls.append({"id": f"text_{uuid.uuid4().hex[:8]}", "name": it["name"],
                          "arguments": args if isinstance(args, str) else json.dumps(args)})
    return calls


def _for_llm(name: str, result: dict) -> str:
    if name == "generate_report" and "error" not in result:
        return json.dumps({"status": "report generated. You are done."})
    text = json.dumps(result, separators=(",", ":"), default=str)
    return text if len(text) <= MAX_TOOL_RESULT_CHARS else text[:MAX_TOOL_RESULT_CHARS] + "...(truncated)"


def _usage(resp: Any) -> Optional[dict]:
    u = getattr(resp, "usage", None)
    if not u:
        return None
    return {"prompt": getattr(u, "prompt_tokens", 0) or 0, "completion": getattr(u, "completion_tokens", 0) or 0}


# ---- agent ------------------------------------------------------------------
def run_llm_agent(question: str, ctx: SalesContext, tracer: Tracer, cfg: AgentConfig) -> Iterator[TraceEvent]:
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(ctx, question)},
        {"role": "user", "content": question},
    ]
    extra: dict[str, Any] = {}
    if cfg.effective_api_base:
        extra["api_base"] = cfg.effective_api_base
    if cfg.is_ollama:
        extra["num_ctx"] = 16384  # Ollama's default context is too small for tool results

    report_done = False
    last_text = ""
    fatal = False

    for _ in range(cfg.max_iterations):
        t0 = tracer.timer()
        try:
            resp = litellm.completion(
                model=cfg.model, messages=messages, tools=TOOLS, tool_choice="auto",
                temperature=cfg.temperature, timeout=cfg.timeout, **extra,
            )
        except Exception as e:  # provider errors vary; map to a friendly message
            yield tracer.record(TraceEvent(step=tracer.next_step(), kind="error", model=cfg.model,
                                           output={"error": str(e)}, output_summary=friendly_error(e, cfg),
                                           duration_ms=tracer.elapsed_ms(t0)))
            fatal = True
            break

        msg = resp.choices[0].message
        content = getattr(msg, "content", None) or ""
        native = [
            {"id": tc.id or f"call_{uuid.uuid4().hex[:8]}", "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (getattr(msg, "tool_calls", None) or [])
        ]
        calls = native or _text_tool_calls(content)
        if content:
            last_text = content

        yield tracer.record(TraceEvent(
            step=tracer.next_step(), kind="llm_call", model=cfg.model, duration_ms=tracer.elapsed_ms(t0),
            tokens=_usage(resp),
            output={"content": content, "tool_calls": calls, "finish_reason": resp.choices[0].finish_reason},
            output_summary=(f"→ {', '.join(c['name'] for c in calls)}" if calls else "")
            + (f"  “{content[:140]}”" if content and not native else ""),
        ))

        if not calls:
            break

        messages.append({
            "role": "assistant",
            "content": content if not native else (content or None),
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"] if isinstance(c["arguments"], str)
                              else json.dumps(c["arguments"])}}
                for c in calls
            ],
        })

        for c in calls:
            args, err = _parse_args(c["arguments"])
            if err:
                result = {"error": f"{c['name']}: {err}"}
                yield tracer.record(TraceEvent(step=tracer.next_step(), kind="error", tool_name=c["name"],
                                               input=c["arguments"], output=result, output_summary=result["error"]))
            else:
                args, dropped = _clean_args(c["name"], args)
                note = f"ignored unknown args: {', '.join(dropped)}" if dropped else None
                result, ok = yield from execute_tool(ctx, tracer, c["name"], args, model=cfg.model, note=note)
                if ok and c["name"] == "generate_report":
                    report_done = True
            messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                             "content": _for_llm(c["name"], result)})
        if report_done:
            break

    if not report_done and (not fatal or ctx.results):
        yield from _auto_report(question, ctx, tracer, last_text)
    if ctx.report_markdown:
        yield tracer.record(TraceEvent(step=tracer.next_step(), kind="final", model=cfg.model,
                                       output=ctx.report_markdown, output_summary="report ready"))


def _auto_report(question: str, ctx: SalesContext, tracer: Tracer, last_text: str) -> Iterator[TraceEvent]:
    """The model stopped without calling generate_report: finish the job deterministically."""
    note = "auto: model ended without calling generate_report"
    p = parse_period(question, ctx.today)
    both = {k: p[k] for k in ("current_start", "current_end", "previous_start", "previous_end")}
    if "get_product_performance" not in ctx.results:
        yield from execute_tool(ctx, tracer, "get_product_performance", both, note=note)
    if "calculate_growth" not in ctx.results:
        yield from execute_tool(ctx, tracer, "calculate_growth", {**both, "group_by": "category"}, note=note)
    perf = ctx.results.get("get_product_performance", {})
    growth = ctx.results.get("calculate_growth")
    summary = last_text.strip() or "Automated summary: see key metrics and underperforming products below."
    yield from execute_tool(ctx, tracer, "generate_report", {
        "summary": summary[:1500],
        "findings": rule_based_findings(perf, growth) if perf else [],
        "recommendations": rule_based_recommendations(perf) if perf else [],
    }, note=note)


__all__ = ["run_llm_agent", "check_tool_support", "friendly_error", "build_system_prompt", "summarise"]
