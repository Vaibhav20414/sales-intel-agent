"""Tool definitions in OpenAI function-calling format (LiteLLM converts per provider).

Parameters are deliberately flat (no nested objects) because small local models
are much more reliable with flat schemas.
"""
from __future__ import annotations

_DATE = {"type": "string", "description": "ISO date YYYY-MM-DD"}
_PERIODS = {
    "current_start": {**_DATE, "description": "Start of the period being analysed (YYYY-MM-DD)"},
    "current_end": {**_DATE, "description": "End of the period being analysed (YYYY-MM-DD)"},
    "previous_start": {**_DATE, "description": "Start of the comparison period (YYYY-MM-DD)"},
    "previous_end": {**_DATE, "description": "End of the comparison period (YYYY-MM-DD)"},
}
_PERIOD_KEYS = list(_PERIODS)


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [
    _fn(
        "get_sales_data",
        "Retrieve sales records for a date range. Returns row/order counts, totals, the products/"
        "categories/regions present, dataset coverage and a small sample. Call this first.",
        {
            "start_date": _DATE,
            "end_date": _DATE,
            "category": {"type": "string", "description": "Optional category filter"},
            "region": {"type": "string", "description": "Optional region filter"},
            "product": {"type": "string", "description": "Optional product name filter"},
        },
        ["start_date", "end_date"],
    ),
    _fn(
        "calculate_revenue",
        "Calculate total revenue, units, orders and average order value for a date range, "
        "optionally broken down by product, category, region or week.",
        {
            "start_date": _DATE,
            "end_date": _DATE,
            "group_by": {"type": "string", "enum": ["none", "product", "category", "region", "week"]},
        },
        ["start_date", "end_date"],
    ),
    _fn(
        "get_product_performance",
        "Rank all products by revenue for the current period, compare with the previous period, "
        "and flag underperforming products with reasons. Use this to find underperformers.",
        {**_PERIODS, "top_n": {"type": "integer", "description": "Optional: only return the top N in the ranking"}},
        _PERIOD_KEYS,
    ),
    _fn(
        "calculate_growth",
        "Calculate revenue and unit growth % between the current and previous periods, overall "
        "and optionally per product, category or region.",
        {**_PERIODS, "group_by": {"type": "string", "enum": ["none", "product", "category", "region"]}},
        _PERIOD_KEYS,
    ),
    _fn(
        "generate_report",
        "Produce the final report. Call this LAST, once, with your conclusions. Key metrics and "
        "tables are filled in automatically from the earlier tool results.",
        {
            "summary": {"type": "string", "description": "2-4 sentence executive summary"},
            "findings": {"type": "array", "items": {"type": "string"}, "description": "Key findings"},
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific, actionable recommendations",
            },
            "title": {"type": "string", "description": "Optional report title"},
        },
        ["summary", "findings", "recommendations"],
    ),
]
