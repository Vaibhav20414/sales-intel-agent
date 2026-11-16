from datetime import date

import pandas as pd
import pytest

from sales_agent.data import DataError, normalise
from sales_agent.periods import parse_period
from sales_agent.tools import (
    calculate_growth,
    calculate_revenue,
    default_periods,
    generate_report,
    get_product_performance,
    get_sales_data,
)

TINY = dict(current_start="2026-11-01", current_end="2026-11-02",
            previous_start="2026-10-01", previous_end="2026-10-02")


# ---- data -------------------------------------------------------------------
def test_sample_data_shape(sample_df):
    assert len(sample_df) > 5000
    assert sample_df["product_id"].nunique() == 20
    assert sample_df["date"].max().date() == date(2026, 11, 16)


def test_missing_columns_rejected():
    with pytest.raises(DataError, match="revenue"):
        normalise(pd.DataFrame({"date": ["2026-01-01"], "product_id": ["A"], "product_name": ["a"], "units": [1]}))


def test_optional_columns_defaulted():
    df = normalise(pd.DataFrame({"date": ["2026-01-01"], "product_id": [1], "product_name": ["a"],
                                 "units": [2], "revenue": [10.0]}))
    assert df.loc[0, "unit_price"] == 5.0
    assert df.loc[0, "category"] == "Uncategorised"


# ---- periods ----------------------------------------------------------------
def test_default_periods_mtd():
    assert default_periods(date(2026, 11, 16)) == {
        "current_start": "2026-11-01", "current_end": "2026-11-16",
        "previous_start": "2026-10-01", "previous_end": "2026-10-16",
    }


def test_default_periods_clips_short_month():
    p = default_periods(date(2026, 3, 31))
    assert p["previous_end"] == "2026-02-28"


def test_parse_period_variants():
    today = date(2026, 11, 16)
    assert parse_period("how was last month?", today)["current_start"] == "2026-10-01"
    assert parse_period("last month", today)["previous_end"] == "2026-09-30"
    p = parse_period("sales in the last 30 days", today)
    assert (p["current_start"], p["previous_end"]) == ("2026-10-18", "2026-10-17")
    assert parse_period("anything", today)["current_end"] == "2026-11-16"


# ---- tool arithmetic (hand-checked) -----------------------------------------
def test_revenue_totals(tiny_ctx):
    r = calculate_revenue(tiny_ctx, "2026-11-01", "2026-11-02", group_by="product")
    assert r["total_revenue"] == 220.0
    assert r["orders"] == 3
    assert r["average_order_value"] == pytest.approx(73.33)
    assert r["breakdown"][0] == {"group": "Alpha", "revenue": 150.0, "units": 15, "share_pct": 68.2}


def test_growth_math(tiny_ctx):
    g = calculate_growth(tiny_ctx, **TINY, group_by="product")
    assert g["overall"]["previous_revenue"] == 200.0
    assert g["overall"]["growth_pct"] == 10.0  # 220 vs 200
    by = {r["group"]: r for r in g["breakdown"]}
    assert by["Alpha"]["growth_pct"] == 50.0
    assert by["Beta"]["growth_pct"] == -60.0
    assert by["Gamma"]["growth_pct"] is None  # division by zero -> null


def test_performance_flags_tiny(tiny_ctx):
    perf = get_product_performance(tiny_ctx, **TINY)
    flagged = {u["product"] for u in perf["underperformers"]}
    assert flagged == {"Beta"}
    gamma = next(p for p in perf["products"] if p["product"] == "Gamma")
    assert "new product" in gamma["notes"][0]


def test_bad_dates_raise(tiny_ctx):
    with pytest.raises(ValueError, match="ISO date"):
        calculate_revenue(tiny_ctx, "November", "2026-11-02")
    with pytest.raises(ValueError, match="after end"):
        calculate_revenue(tiny_ctx, "2026-11-05", "2026-11-01")


def test_bad_group_by(tiny_ctx):
    with pytest.raises(ValueError, match="group_by"):
        calculate_revenue(tiny_ctx, "2026-11-01", "2026-11-02", group_by="colour")


def test_empty_period_note(tiny_ctx):
    r = get_sales_data(tiny_ctx, "2025-01-01", "2025-01-31")
    assert r["rows"] == 0 and "note" in r


# ---- the planted story in the sample data -----------------------------------
def test_story_underperformers(ctx):
    perf = get_product_performance(ctx, **default_periods(ctx.today))
    flagged = {u["product"]: u for u in perf["underperformers"]}
    for name in ["Wireless Earbuds", "Yoga Mat", "Coffee Grinder", "Desk Lamp"]:
        assert name in flagged, f"{name} should be flagged; got {list(flagged)}"
    assert any("erosion" in r for r in flagged["Desk Lamp"]["reasons"])
    assert any("discount" in n for n in flagged["Coffee Grinder"]["notes"])
    by_name = {p["product"]: p for p in perf["products"]}
    for name in ["Smart Watch", "Protein Powder"]:
        assert name not in flagged
        assert by_name[name]["growth_pct"] > 20
    assert "Smart Watch" in {g["product"] for g in perf["top_growers"]}


def test_report_uses_tool_numbers(ctx):
    p = default_periods(ctx.today)
    calculate_revenue(ctx, p["current_start"], p["current_end"])
    get_product_performance(ctx, **p)
    calculate_growth(ctx, **p)
    out = generate_report(ctx, summary="Test summary", findings=["f1"], recommendations=["r1"])
    md = out["report_markdown"]
    assert "Test summary" in md and "## Underperforming products" in md and "Desk Lamp" in md
    assert f"${ctx.results['calculate_growth']['overall']['current_revenue']:,.2f}" in md
