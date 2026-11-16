"""Offline agent: fixed tool pipeline, no LLM required."""
from __future__ import annotations

from typing import Iterator

from .executor import execute_tool
from .periods import parse_period
from .report import rule_based_findings, rule_based_recommendations
from .tools import SalesContext
from .trace import TraceEvent, Tracer, summarise


def run_offline_agent(question: str, ctx: SalesContext, tracer: Tracer) -> Iterator[TraceEvent]:
    p = parse_period(question, ctx.today)
    cur = {"start_date": p["current_start"], "end_date": p["current_end"]}
    both = {k: p[k] for k in ("current_start", "current_end", "previous_start", "previous_end")}

    _, ok = yield from execute_tool(ctx, tracer, "get_sales_data", cur, note=f"Period: {p['label']}")
    if not ok:
        return
    yield from execute_tool(ctx, tracer, "calculate_revenue", {**cur, "group_by": "product"})
    perf, _ = yield from execute_tool(ctx, tracer, "get_product_performance", both)
    growth, _ = yield from execute_tool(ctx, tracer, "calculate_growth", {**both, "group_by": "category"})

    perf_full = ctx.results.get("get_product_performance", perf)
    o = growth.get("overall", {})
    n_under = len(perf_full.get("underperformers", []))
    gp = o.get("growth_pct")
    direction = "up" if (gp or 0) >= 0 else "down"
    summary = (
        f"Revenue for {p['label']} is {direction} {abs(gp or 0):.1f}% "
        f"(${o.get('current_revenue', 0):,.2f} vs ${o.get('previous_revenue', 0):,.2f}). "
        f"{n_under} product(s) are underperforming and need attention."
    )
    rep, ok = yield from execute_tool(
        ctx,
        tracer,
        "generate_report",
        {
            "summary": summary,
            "findings": rule_based_findings(perf_full, growth),
            "recommendations": rule_based_recommendations(perf_full),
        },
    )
    if ok:
        yield tracer.record(
            TraceEvent(step=tracer.next_step(), kind="final", output=rep["report_markdown"],
                       output_summary=summarise({"status": "done", "underperformers": perf_full["underperformers"]}))
        )
