import json
from types import SimpleNamespace as NS

import pytest

from sales_agent import agent_llm
from sales_agent.agent_llm import run_llm_agent
from sales_agent.agent_offline import run_offline_agent
from sales_agent.config import AgentConfig
from sales_agent.trace import Tracer

Q = "Analyse my sales performance this month and tell me which products are underperforming."
P = dict(current_start="2026-11-01", current_end="2026-11-16",
         previous_start="2026-10-01", previous_end="2026-10-16")


# ---- offline ----------------------------------------------------------------
def test_offline_pipeline(ctx):
    tracer = Tracer("offline", Q)
    events = list(run_offline_agent(Q, ctx, tracer))
    calls = [e.tool_name for e in events if e.kind == "tool_call"]
    assert calls == ["get_sales_data", "calculate_revenue", "get_product_performance",
                     "calculate_growth", "generate_report"]
    assert events[-1].kind == "final"
    assert "Desk Lamp" in events[-1].output
    assert not [e for e in events if e.kind == "error"]
    assert json.loads(tracer.to_json())["totals"]["tool_calls"] == 5


# ---- LLM loop with mocked litellm.completion -------------------------------
def _call(name, args, raw=False):
    return NS(id=f"id_{name}", function=NS(name=name, arguments=args if raw else json.dumps(args)))


def _resp(tool_calls=None, content=""):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls),
                          finish_reason="tool_calls" if tool_calls else "stop")],
              usage=NS(prompt_tokens=100, completion_tokens=20))


class Script:
    """Replays a fixed list of responses and records the messages it was sent."""

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.fixture
def cfg():
    return AgentConfig(model="ollama_chat/qwen2.5:7b")


def test_llm_normal_run(ctx, cfg, monkeypatch):
    script = Script([
        _resp([_call("get_sales_data", {"start_date": "2026-11-01", "end_date": "2026-11-16"})]),
        _resp([_call("get_product_performance", P), _call("calculate_growth", {**P, "group_by": "category"})]),
        _resp([_call("generate_report", {"summary": "LLM summary", "findings": ["x"], "recommendations": ["y"]})]),
    ])
    monkeypatch.setattr(agent_llm.litellm, "completion", script)
    events = list(run_llm_agent(Q, ctx, Tracer("llm", Q), cfg))
    tools = [e.tool_name for e in events if e.kind == "tool_call"]
    assert tools == ["get_sales_data", "get_product_performance", "calculate_growth", "generate_report"]
    assert events[-1].kind == "final" and "LLM summary" in events[-1].output
    assert len(script.calls) == 3  # stops right after generate_report
    # Ollama-specific settings were passed through
    assert script.calls[0]["api_base"] == "http://localhost:11434" and script.calls[0]["num_ctx"] == 16384
    # tool results were sent back with matching ids
    tool_msgs = [m for m in script.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "id_get_sales_data"


def test_llm_recovers_from_bad_json_and_bad_args(ctx, cfg, monkeypatch):
    script = Script([
        _resp([_call("get_sales_data", "{not json", raw=True)]),
        _resp([_call("calculate_revenue", {"start_date": "Nov 1", "end_date": "2026-11-16", "colour": "red"})]),
        _resp([_call("get_product_performance", P)]),
        _resp([_call("generate_report", {"summary": "ok", "findings": [], "recommendations": []})]),
    ])
    monkeypatch.setattr(agent_llm.litellm, "completion", script)
    events = list(run_llm_agent(Q, ctx, Tracer("llm", Q), cfg))
    errors = [e for e in events if e.kind == "error"]
    assert len(errors) == 2
    assert "not valid JSON" in errors[0].output_summary
    assert "ISO date" in errors[1].output_summary
    # the model was told about the errors
    fed_back = [m["content"] for m in script.calls[2]["messages"] if m["role"] == "tool"]
    assert any("not valid JSON" in c for c in fed_back) and any("ISO date" in c for c in fed_back)
    # unknown arg was dropped and noted
    call = next(e for e in events if e.kind == "tool_call" and e.tool_name == "calculate_revenue")
    assert "colour" in call.input["_note"]
    assert events[-1].kind == "final"


def test_llm_never_calls_report_gets_auto_report(ctx, cfg, monkeypatch):
    script = Script([
        _resp([_call("get_sales_data", {"start_date": "2026-11-01", "end_date": "2026-11-16"})]),
        _resp(None, content="Earbuds and Yoga Mat are down a lot."),
    ])
    monkeypatch.setattr(agent_llm.litellm, "completion", script)
    events = list(run_llm_agent(Q, ctx, Tracer("llm", Q), cfg))
    auto = [e for e in events if e.kind == "tool_call" and "_note" in (e.input or {})]
    assert [e.tool_name for e in auto] == ["get_product_performance", "calculate_growth", "generate_report"]
    assert events[-1].kind == "final"
    assert "Earbuds and Yoga Mat are down a lot." in events[-1].output
    assert "Coffee Grinder" in events[-1].output


def test_llm_text_embedded_tool_call(ctx, cfg, monkeypatch):
    text_call = json.dumps({"name": "get_product_performance", "arguments": P})
    script = Script([
        _resp(None, content=f"I'll call: {text_call}"),
        _resp([_call("generate_report", {"summary": "s", "findings": [], "recommendations": []})]),
    ])
    monkeypatch.setattr(agent_llm.litellm, "completion", script)
    events = list(run_llm_agent(Q, ctx, Tracer("llm", Q), cfg))
    assert [e.tool_name for e in events if e.kind == "tool_result"] == ["get_product_performance", "generate_report"]


def test_llm_connection_error_is_friendly(ctx, cfg, monkeypatch):
    class APIConnectionError(Exception):
        pass

    def boom(**_):
        raise APIConnectionError("Connection refused")

    monkeypatch.setattr(agent_llm.litellm, "completion", boom)
    events = list(run_llm_agent(Q, ctx, Tracer("llm", Q), cfg))
    assert len(events) == 1 and events[0].kind == "error"
    assert "Ollama not reachable" in events[0].output_summary
