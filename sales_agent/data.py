"""Loading and validating sales data."""
from __future__ import annotations

from pathlib import Path
from typing import IO, Union

import pandas as pd

REQUIRED_COLUMNS = ["date", "product_id", "product_name", "units", "revenue"]
OPTIONAL_DEFAULTS = {
    "order_id": None,  # filled with a row index
    "category": "Uncategorised",
    "region": "All",
    "unit_price": None,  # derived from revenue / units
    "discount": 0.0,
}

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales.csv"


class DataError(ValueError):
    """Raised when a sales file is missing required columns or is unreadable."""


def load_sales(source: Union[str, Path, IO, None] = None) -> pd.DataFrame:
    """Load a sales CSV (path or file-like) and normalise it.

    Required columns: date, product_id, product_name, units, revenue.
    Optional columns get sensible defaults.
    """
    source = source if source is not None else DEFAULT_DATA_PATH
    try:
        df = pd.read_csv(source)
    except FileNotFoundError as e:
        raise DataError(f"Sales file not found: {source}. Run `python data/generate_sample.py` first.") from e
    except Exception as e:  # pandas raises many parser error types
        raise DataError(f"Could not read sales file: {e}") from e
    return normalise(df)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Missing required column(s): {', '.join(missing)}. Required: {', '.join(REQUIRED_COLUMNS)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates == len(df):
        raise DataError("Could not parse any values in the 'date' column.")
    df = df.dropna(subset=["date"])

    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    df["product_id"] = df["product_id"].astype(str)
    df["product_name"] = df["product_name"].astype(str)

    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            if col == "order_id":
                df[col] = [f"R{i:07d}" for i in range(len(df))]
            elif col == "unit_price":
                df[col] = (df["revenue"] / df["units"].where(df["units"] != 0)).fillna(0.0)
            else:
                df[col] = default
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce").fillna(0.0)
    return df.sort_values("date").reset_index(drop=True)
