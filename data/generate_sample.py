"""Generate a deterministic sample sales dataset with a built-in "story".

Story (applies from 2026-11-01 onward, compared with October):
  * Wireless Earbuds, Yoga Mat         -> sharp decline in demand
  * Coffee Grinder                     -> heavy discount + lower demand (revenue collapses)
  * Desk Lamp                          -> price cut: units UP, revenue DOWN
  * Smart Watch, Protein Powder        -> strong growth
  * everything else                    -> steady with noise and a weekly pattern

Usage:  python data/generate_sample.py [--out data/sales.csv] [--end 2026-11-16]
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = date(2026, 8, 1)
DEFAULT_END = date(2026, 11, 16)
STORY_START = pd.Timestamp("2026-11-01")

# product_id, name, category, unit_price, base daily orders
PRODUCTS = [
    ("P01", "Wireless Earbuds", "Electronics", 79.0, 4.0),
    ("P02", "Smart Watch", "Electronics", 199.0, 2.0),
    ("P03", "Bluetooth Speaker", "Electronics", 59.0, 3.0),
    ("P04", "USB-C Hub", "Electronics", 39.0, 3.5),
    ("P05", "Phone Case", "Electronics", 19.0, 6.0),
    ("P06", "Coffee Grinder", "Home & Kitchen", 49.0, 3.0),
    ("P07", "Desk Lamp", "Home & Kitchen", 45.0, 3.0),
    ("P08", "Blender", "Home & Kitchen", 89.0, 2.0),
    ("P09", "Chef Knife", "Home & Kitchen", 65.0, 2.5),
    ("P10", "Water Bottle", "Home & Kitchen", 24.0, 5.0),
    ("P11", "Yoga Mat", "Sports & Fitness", 35.0, 4.0),
    ("P12", "Protein Powder", "Sports & Fitness", 54.0, 3.0),
    ("P13", "Resistance Bands", "Sports & Fitness", 22.0, 3.5),
    ("P14", "Running Shoes", "Sports & Fitness", 120.0, 2.0),
    ("P15", "Dumbbell Set", "Sports & Fitness", 99.0, 1.5),
    ("P16", "Notebook Pack", "Office", 12.0, 6.0),
    ("P17", "Ergonomic Mouse", "Office", 49.0, 3.0),
    ("P18", "Monitor Stand", "Office", 69.0, 2.0),
    ("P19", "Gel Pens", "Office", 9.0, 7.0),
    ("P20", "Backpack", "Office", 75.0, 2.5),
]

REGIONS = ["North", "South", "East", "West"]
REGION_WEIGHTS = [0.3, 0.25, 0.25, 0.2]

# November story: demand multiplier, price multiplier, discount override
STORY = {
    "P01": dict(demand=0.55),
    "P11": dict(demand=0.60),
    "P06": dict(demand=0.80, discount=0.35),
    "P07": dict(demand=1.25, price=0.55),
    "P02": dict(demand=1.80),
    "P12": dict(demand=1.80),
}
STEADY_UPLIFT = 1.04  # mild seasonal lift for everything else
VOLUME = 3.0  # scales order counts; higher = less noise
WEEKDAY_FACTOR = [0.9, 0.95, 1.0, 1.0, 1.1, 1.3, 1.2]  # Mon..Sun


def generate(end: date = DEFAULT_END, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, end, freq="D")
    rows = []
    order_no = 1
    for day in days:
        in_story = day >= STORY_START
        wk = WEEKDAY_FACTOR[day.weekday()]
        for pid, name, cat, price, base in PRODUCTS:
            story = STORY.get(pid, {}) if in_story else {}
            demand = story.get("demand", STEADY_UPLIFT if in_story else 1.0)
            unit_price = round(price * story.get("price", 1.0), 2)
            n_orders = rng.poisson(base * VOLUME * wk * demand)
            for _ in range(n_orders):
                units = 1 + int(rng.poisson(0.6))
                if "discount" in story:
                    discount = story["discount"]
                else:
                    discount = float(rng.choice([0.0, 0.0, 0.0, 0.05, 0.10]))
                region = rng.choice(REGIONS, p=REGION_WEIGHTS)
                rows.append(
                    {
                        "order_id": f"O{order_no:06d}",
                        "date": day.date().isoformat(),
                        "product_id": pid,
                        "product_name": name,
                        "category": cat,
                        "region": region,
                        "units": units,
                        "unit_price": unit_price,
                        "discount": discount,
                        "revenue": round(units * unit_price * (1 - discount), 2),
                    }
                )
                order_no += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(Path(__file__).with_name("sales.csv")))
    parser.add_argument("--end", default=DEFAULT_END.isoformat(), help="last date (YYYY-MM-DD)")
    args = parser.parse_args()
    df = generate(end=date.fromisoformat(args.end))
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows ({df['date'].min()} -> {df['date'].max()}) to {args.out}")


if __name__ == "__main__":
    main()
