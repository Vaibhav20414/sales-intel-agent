import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from generate_sample import generate  # noqa: E402
from sales_agent.data import normalise  # noqa: E402
from sales_agent.tools import SalesContext  # noqa: E402


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return normalise(generate())


@pytest.fixture
def ctx(sample_df) -> SalesContext:
    return SalesContext(df=sample_df)


@pytest.fixture
def tiny_ctx() -> SalesContext:
    """Hand-built data for exact arithmetic checks."""
    rows = [
        # previous period: 2026-10-01..02
        ("O1", "2026-10-01", "A", "Alpha", "Cat1", "North", 10, 10.0, 0.0, 100.0),
        ("O2", "2026-10-02", "B", "Beta", "Cat1", "North", 5, 20.0, 0.0, 100.0),
        # current period: 2026-11-01..02
        ("O3", "2026-11-01", "A", "Alpha", "Cat1", "North", 15, 10.0, 0.0, 150.0),
        ("O4", "2026-11-02", "B", "Beta", "Cat1", "South", 2, 20.0, 0.0, 40.0),
        ("O5", "2026-11-02", "C", "Gamma", "Cat2", "South", 1, 30.0, 0.0, 30.0),  # new product
    ]
    cols = ["order_id", "date", "product_id", "product_name", "category", "region",
            "units", "unit_price", "discount", "revenue"]
    return SalesContext(df=normalise(pd.DataFrame(rows, columns=cols)))
