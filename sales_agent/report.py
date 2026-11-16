"""Markdown report builder + rule-based findings/recommendations.

Key numbers always come from the cached tool results on the context, never
from LLM text, so the figures in the report are correct even if a model
misquotes them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .tools import SalesContext


def _fmt_money(x: Optional[float]) -> str:
    return "n/a" if x is None else f"${x:,.2f}"


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:+.1f}%"


def rule_based_findings(perf: dict, growth: Optional[dict] = None) -> list[str]:
    findings = []
    if growth:
        o = growth["overall"]
        findings.append(
            f"Overall revenue {_fmt_money(o['current_revenue'])} vs {_fmt_money(o['previous_revenue'])} "
            f"({_fmt_pct(o['growth_pct'])})."
        )
    if perf.get("top_growers"):
        tops = ", ".join(f"{g['product']} ({_fmt_pct(g['growth_pct'])})" for g in perf["top_growers"])
        findings.append(f"Strongest growth: {tops}.")
    under = perf.get("underperformers", [])
    if under:
        findings.append(f"{len(under)} of {perf['product_count']} products are underperforming.")
        for u in under:
            extra = f" ({'; '.join(u['notes'])})" if u.get("notes") else ""
            findings.append(f"{u['product']}: {'; '.join(u['reasons'])}{extra}.")
    else:
        findings.append("No products met the underperformance criteria.")
    if growth and growth.get("breakdown"):
        worst = growth["breakdown"][0]
        if worst["growth_pct"] is not None and worst["growth_pct"] < 0:
            findings.append(f"Weakest {growth['group_by']}: {worst['group']} ({_fmt_pct(worst['growth_pct'])}).")
    return findings


def rule_based_recommendations(perf: dict) -> list[str]:
    recs = []
    for p in perf.get("products", []):
        if not p["underperforming"]:
            continue
        name = p["product"]
        reasons = " ".join(p["reasons"])
        notes = " ".join(p.get("notes", []))
        if "discount" in notes:
            recs.append(
                f"Review the discount on {name}: deeper discounting has not lifted volume. "
                "Test reverting to the previous discount level."
            )
        elif "erosion" in reasons:
            recs.append(
                f"Re-evaluate pricing for {name}: volume is up but revenue is down. "
                "The price cut is not paying for itself."
            )
        elif "revenue down" in reasons:
            recs.append(
                f"Investigate demand drop for {name} (stock-outs, competitor pricing, listing changes) "
                "and consider a targeted promotion or bundle."
            )
        else:
            recs.append(f"Consider bundling or repositioning {name}, or reducing inventory exposure.")
    for g in perf.get("top_growers", [])[:2]:
        recs.append(f"Protect momentum on {g['product']} ({_fmt_pct(g['growth_pct'])}): ensure stock and visibility.")
    return recs or ["Performance is stable. Keep monitoring weekly."]


def build_report(
    ctx: "SalesContext",
    summary: str,
    findings: list[str],
    recommendations: list[str],
    title: Optional[str] = None,
) -> str:
    perf = ctx.results.get("get_product_performance")
    growth = ctx.results.get("calculate_growth")
    revenue = ctx.results.get("calculate_revenue")

    lines = [f"# {title or 'Sales Performance Report'}", ""]
    if perf:
        cp, pp = perf["current_period"], perf["previous_period"]
        lines.append(f"*Current period:* {cp['start']} → {cp['end']}  ·  *Compared with:* {pp['start']} → {pp['end']}")
        lines.append("")

    lines += ["## Summary", "", summary.strip() or "_No summary provided._", ""]

    # Key metrics from verified tool output
    metrics = []
    if growth:
        o = growth["overall"]
        metrics += [
            ("Revenue (current)", _fmt_money(o["current_revenue"])),
            ("Revenue (previous)", _fmt_money(o["previous_revenue"])),
            ("Revenue growth", _fmt_pct(o["growth_pct"])),
            ("Units growth", _fmt_pct(o["units_growth_pct"])),
        ]
    elif revenue:
        metrics.append(("Revenue", _fmt_money(revenue["total_revenue"])))
    if revenue:
        metrics += [("Orders", f"{revenue['orders']:,}"), ("Avg order value", _fmt_money(revenue["average_order_value"]))]
    if metrics:
        lines += ["## Key metrics", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in metrics]
        lines.append("")

    if perf and perf.get("underperformers"):
        lines += [
            "## Underperforming products",
            "",
            "| Product | Revenue | Previous | Growth | Why |",
            "|---|---:|---:|---:|---|",
        ]
        by_name = {p["product"]: p for p in perf["products"]}
        for u in perf["underperformers"]:
            p = by_name[u["product"]]
            why = "; ".join(u["reasons"] + u.get("notes", []))
            lines.append(
                f"| {p['product']} | {_fmt_money(p['current_revenue'])} | {_fmt_money(p['previous_revenue'])} "
                f"| {_fmt_pct(p['growth_pct'])} | {why} |"
            )
        lines.append("")

    if perf:
        lines += ["## Top products by revenue", "", "| # | Product | Revenue | Growth | Share |", "|---:|---|---:|---:|---:|"]
        for p in perf["products"][:5]:
            lines.append(
                f"| {p['rank']} | {p['product']} | {_fmt_money(p['current_revenue'])} "
                f"| {_fmt_pct(p['growth_pct'])} | {p['share_pct']:.1f}% |"
            )
        lines.append("")

    if findings:
        lines += ["## Findings", ""] + [f"- {f}" for f in findings] + [""]
    if recommendations:
        lines += ["## Recommendations", ""] + [f"{i}. {r}" for i, r in enumerate(recommendations, 1)] + [""]
    return "\n".join(lines).rstrip() + "\n"
