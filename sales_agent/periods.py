"""Turn natural-language period phrases into concrete current/previous date ranges."""
from __future__ import annotations

import re
from datetime import date, timedelta

from .tools import default_periods


def _rng(cs: date, ce: date, ps: date, pe: date, label: str) -> dict[str, str]:
    return {
        "current_start": cs.isoformat(),
        "current_end": ce.isoformat(),
        "previous_start": ps.isoformat(),
        "previous_end": pe.isoformat(),
        "label": label,
    }


def parse_period(question: str, today: date) -> dict[str, str]:
    q = question.lower()

    if "last month" in q or "previous month" in q:
        cur_end = today.replace(day=1) - timedelta(days=1)
        cur_start = cur_end.replace(day=1)
        prev_end = cur_start - timedelta(days=1)
        return _rng(cur_start, cur_end, prev_end.replace(day=1), prev_end, "last full month vs the month before")

    if "this week" in q or "week to date" in q:
        cur_start = today - timedelta(days=today.weekday())
        span = today - cur_start
        prev_start = cur_start - timedelta(days=7)
        return _rng(cur_start, today, prev_start, prev_start + span, "week-to-date vs same days last week")

    m = re.search(r"last (\d{1,3}) days", q)
    if m or "last week" in q:
        n = int(m.group(1)) if m else 7
        cur_start = today - timedelta(days=n - 1)
        prev_end = cur_start - timedelta(days=1)
        return _rng(cur_start, today, prev_end - timedelta(days=n - 1), prev_end, f"last {n} days vs prior {n} days")

    p = default_periods(today)
    return {**p, "label": "month-to-date vs same days last month"}
