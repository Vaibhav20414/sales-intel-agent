"""The five agent tools. Pure pandas, each returns a compact JSON-safe dict.

Every tool takes a SalesContext as its first argument; the LLM only supplies the
keyword arguments. Results are also cached on the context (ctx.results) so the
report builder and UI charts can reuse the exact numbers.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Optional

import pandas as pd

# ---- underperformance thresholds -------------------------------------------
DECLINE_THRESHOLD_PCT = -15.0  # revenue growth below this => underperforming
BOTTOM_QUANTILE = 0.25  # bottom quartile of revenue share ...
# ... AND growth <= 0 => underperforming
DISCOUNT_JUMP = 0.10  # avg discount rose by >= 10 pts => noted as a cause

GROUP_COLUMNS = {
    "product": "product_name",
    "category": "category",
    "region": "region",
}


@dataclass
class SalesContext:
    df: pd.DataFrame
    results: dict[str, Any] = field(default_factory=dict)
    report_markdown: Optional[str] = None

    @property
    def today(self) -> date:
        """Reference 'today' = latest date in the dataset (works for any CSV)."""
        return self.df["date"].max().date()

    @property
    def coverage(self) -> tuple[date, date]:
        return self.df["date"].min().date(), self.df["date"].max().date()


# ---- helpers ----------------------------------------------------------------
def _d(value: Any, name: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(date.fromisoformat(str(value)[:10]))
    except Exception as e:
        raise ValueError(f"{name} must be an ISO date like 2026-11-01, got {value!r}") from e


def _slice(ctx: SalesContext, start: Any, end: Any, **filters: Optional[str]) -> pd.DataFrame:
    s, e = _d(start, "start_date"), _d(end, "end_date")
    if s > e:
        raise ValueError(f"start date {s.date()} is after end date {e.date()}")
    df = ctx.df[(ctx.df["date"] >= s) & (ctx.df["date"] <= e)]
    for col, val in filters.items():
        if val:
            df = df[df[col].astype(str).str.lower() == str(val).lower()]
    return df


def _pct(cur: float, prev: float) -> Optional[float]:
    if prev == 0:
        return None
    return round((cur - prev) / prev * 100, 1)


def _money(x: float) -> float:
    return round(float(x), 2)


def default_periods(today: date) -> dict[str, str]:
    """Month-to-date vs the same days of the previous month."""
    cur_start = today.replace(day=1)
    prev_month_last = cur_start - timedelta(days=1)
    prev_start = prev_month_last.replace(day=1)
    prev_end_day = min(today.day, calendar.monthrange(prev_start.year, prev_start.month)[1])
    return {
        "current_start": cur_start.isoformat(),
        "current_end": today.isoformat(),
        "previous_start": prev_start.isoformat(),
        "previous_end": prev_start.replace(day=prev_end_day).isoformat(),
    }


# ---- tools ------------------------------------------------------------------
def get_sales_data(
    ctx: SalesContext,
    start_date: str,
    end_date: str,
    category: Optional[str] = None,
    region: Optional[str] = None,
    product: Optional[str] = None,
) -> dict:
    """Retrieve sales records for a period (summary + small sample)."""
    df = _slice(ctx, start_date, end_date, category=category, region=region, product_name=product)
    sample = df.head(5).assign(date=lambda x: x["date"].dt.date.astype(str))
    result = {
        "period": {"start": str(_d(start_date, "start_date").date()), "end": str(_d(end_date, "end_date").date())},
        "filters": {k: v for k, v in {"category": category, "region": region, "product": product}.items() if v},
        "rows": int(len(df)),
        "orders": int(df["order_id"].nunique()),
        "products": int(df["product_id"].nunique()),
        "categories": sorted(df["category"].astype(str).unique().tolist()),
        "regions": sorted(df["region"].astype(str).unique().tolist()),
        "total_revenue": _money(df["revenue"].sum()),
        "total_units": int(df["units"].sum()),
        "dataset_coverage": {"start": str(ctx.coverage[0]), "end": str(ctx.coverage[1])},
        "sample": sample[["date", "product_name", "region", "units", "revenue"]].to_dict("records"),
    }
    if result["rows"] == 0:
        result["note"] = "No sales in this period. Check the dates against dataset_coverage."
    ctx.results["get_sales_data"] = result
    return result


def calculate_revenue(ctx: SalesContext, start_date: str, end_date: str, group_by: str = "none") -> dict:
    """Total revenue, units, orders and AOV, optionally broken down."""
    df = _slice(ctx, start_date, end_date)
    orders = int(df["order_id"].nunique())
    revenue = float(df["revenue"].sum())
    result: dict[str, Any] = {
        "period": {"start": str(_d(start_date, "start_date").date()), "end": str(_d(end_date, "end_date").date())},
        "total_revenue": _money(revenue),
        "total_units": int(df["units"].sum()),
        "orders": orders,
        "average_order_value": _money(revenue / orders) if orders else 0.0,
        "group_by": group_by,
    }
    if group_by and group_by != "none":
        if group_by == "week":
            key = df["date"].dt.to_period("W-SUN").apply(lambda p: str(p.start_time.date()))
            grouped = df.groupby(key)
        elif group_by in GROUP_COLUMNS:
            grouped = df.groupby(GROUP_COLUMNS[group_by])
        else:
            raise ValueError(f"group_by must be one of none, product, category, region, week; got {group_by!r}")
        agg = grouped.agg(revenue=("revenue", "sum"), units=("units", "sum")).reset_index()
        agg.columns = ["group", "revenue", "units"]
        agg["share_pct"] = (agg["revenue"] / revenue * 100).round(1) if revenue else 0.0
        if group_by != "week":
            agg = agg.sort_values("revenue", ascending=False)
        agg["revenue"] = agg["revenue"].round(2)
        agg["units"] = agg["units"].astype(int)
        result["breakdown"] = agg.head(30).to_dict("records")
    ctx.results["calculate_revenue"] = result
    return result


def calculate_growth(
    ctx: SalesContext,
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
    group_by: str = "none",
) -> dict:
    """Growth % between two periods, overall and optionally per group."""
    cur = _slice(ctx, current_start, current_end)
    prev = _slice(ctx, previous_start, previous_end)
    c_rev, p_rev = float(cur["revenue"].sum()), float(prev["revenue"].sum())
    c_units, p_units = int(cur["units"].sum()), int(prev["units"].sum())
    result: dict[str, Any] = {
        "current_period": {"start": current_start, "end": current_end},
        "previous_period": {"start": previous_start, "end": previous_end},
        "overall": {
            "current_revenue": _money(c_rev),
            "previous_revenue": _money(p_rev),
            "change": _money(c_rev - p_rev),
            "growth_pct": _pct(c_rev, p_rev),
            "current_units": c_units,
            "previous_units": p_units,
            "units_growth_pct": _pct(c_units, p_units),
        },
        "group_by": group_by,
    }
    if group_by and group_by != "none":
        if group_by not in GROUP_COLUMNS:
            raise ValueError(f"group_by must be one of none, product, category, region; got {group_by!r}")
        col = GROUP_COLUMNS[group_by]
        c = cur.groupby(col)["revenue"].sum()
        p = prev.groupby(col)["revenue"].sum()
        both = pd.concat([c.rename("current"), p.rename("previous")], axis=1).fillna(0.0)
        rows = [
            {
                "group": str(g),
                "current_revenue": _money(r.current),
                "previous_revenue": _money(r.previous),
                "change": _money(r.current - r.previous),
                "growth_pct": _pct(r.current, r.previous),
            }
            for g, r in both.iterrows()
        ]
        rows.sort(key=lambda r: (r["growth_pct"] is None, r["growth_pct"] if r["growth_pct"] is not None else 0))
        result["breakdown"] = rows
    ctx.results["calculate_growth"] = result
    return result


def get_product_performance(
    ctx: SalesContext,
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
    top_n: Optional[int] = None,
) -> dict:
    """Rank products by revenue and flag underperformers with a reason."""
    cur = _slice(ctx, current_start, current_end)
    prev = _slice(ctx, previous_start, previous_end)

    def agg(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(["product_id", "product_name", "category"]).agg(
            revenue=("revenue", "sum"), units=("units", "sum"), discount=("discount", "mean")
        )

    both = (
        agg(cur)
        .merge(agg(prev), left_index=True, right_index=True, how="outer", suffixes=("_cur", "_prev"))
        .fillna(0.0)
        .reset_index()
    )
    total = float(both["revenue_cur"].sum())
    both["share_pct"] = (both["revenue_cur"] / total * 100) if total else 0.0
    share_cutoff = float(both["share_pct"].quantile(BOTTOM_QUANTILE)) if len(both) else 0.0
    both = both.sort_values("revenue_cur", ascending=False).reset_index(drop=True)

    products = []
    for rank, r in enumerate(both.itertuples(index=False), start=1):
        growth = _pct(r.revenue_cur, r.revenue_prev)
        units_growth = _pct(r.units_cur, r.units_prev)
        reasons = []
        if growth is not None and growth < DECLINE_THRESHOLD_PCT:
            reasons.append(f"revenue down {abs(growth):.1f}% vs previous period")
        if r.share_pct <= share_cutoff and (growth is None or growth <= 0) and r.revenue_prev > 0:
            reasons.append(f"bottom-quartile revenue share ({r.share_pct:.1f}%) with no growth")
        if units_growth is not None and growth is not None and units_growth > 0 and growth < 0:
            reasons.append(
                f"units up {units_growth:.1f}% but revenue down {abs(growth):.1f}% (price/discount erosion)"
            )
        notes = []
        if r.discount_cur - r.discount_prev >= DISCOUNT_JUMP:
            notes.append(f"average discount rose from {r.discount_prev:.0%} to {r.discount_cur:.0%}")
        if r.revenue_prev == 0 and r.revenue_cur > 0:
            notes.append("new product (no sales in previous period)")
        products.append(
            {
                "rank": rank,
                "product_id": r.product_id,
                "product": r.product_name,
                "category": r.category,
                "current_revenue": _money(r.revenue_cur),
                "previous_revenue": _money(r.revenue_prev),
                "growth_pct": growth,
                "current_units": int(r.units_cur),
                "previous_units": int(r.units_prev),
                "units_growth_pct": units_growth,
                "share_pct": round(float(r.share_pct), 1),
                "avg_discount_current": round(float(r.discount_cur), 3),
                "avg_discount_previous": round(float(r.discount_prev), 3),
                "underperforming": bool(reasons),
                "reasons": reasons,
                "notes": notes,
            }
        )

    ranked = products[: int(top_n)] if top_n else products
    growers = [p for p in products if p["growth_pct"] is not None]
    result = {
        "current_period": {"start": current_start, "end": current_end},
        "previous_period": {"start": previous_start, "end": previous_end},
        "product_count": len(products),
        "products": ranked,
        "underperformers": [
            {"product": p["product"], "growth_pct": p["growth_pct"], "reasons": p["reasons"], "notes": p["notes"]}
            for p in products
            if p["underperforming"]
        ],
        "top_growers": [
            {"product": p["product"], "growth_pct": p["growth_pct"]}
            for p in sorted(growers, key=lambda p: p["growth_pct"], reverse=True)[:3]
            if p["growth_pct"] > 0
        ],
        "thresholds": {
            "decline_pct": DECLINE_THRESHOLD_PCT,
            "bottom_share_quantile": BOTTOM_QUANTILE,
        },
    }
    ctx.results["get_product_performance"] = {**result, "products": products}  # keep full list for charts
    return result


def generate_report(
    ctx: SalesContext,
    summary: str,
    findings: Optional[list[str]] = None,
    recommendations: Optional[list[str]] = None,
    title: Optional[str] = None,
) -> dict:
    """Produce the final markdown report from the analysis results."""
    from .report import build_report  # local import avoids a cycle

    md = build_report(ctx, summary=summary, findings=findings or [], recommendations=recommendations or [], title=title)
    ctx.report_markdown = md
    ctx.results["generate_report"] = {"report_markdown": md}
    return {"status": "report generated", "characters": len(md), "report_markdown": md}


ToolFn = Callable[..., dict]
TOOL_REGISTRY: dict[str, ToolFn] = {
    "get_sales_data": get_sales_data,
    "calculate_revenue": calculate_revenue,
    "calculate_growth": calculate_growth,
    "get_product_performance": get_product_performance,
    "generate_report": generate_report,
}
