from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OPTIONS_RAW = ROOT / "data" / "SPY_options_raw.csv"
HISTORY = ROOT / "data" / "SPY_history.csv"
OPTIONS_CLEAN = ROOT / "data" / "spy_options_clean.csv"

MAX_SPREAD_MID_RATIO = 0.30
DIVIDEND_LOOKBACK_DAYS = 365


def parse_float(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_spot_and_dividend_yield(history_path: Path) -> tuple[float, float, datetime, float]:
    with history_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {history_path}")

    rows.sort(key=lambda row: parse_date(row["Date"]))
    latest = rows[-1]
    latest_date = parse_date(latest["Date"])
    s0 = parse_float(latest["Close"])

    lookback_start = latest_date - timedelta(days=DIVIDEND_LOOKBACK_DAYS)
    trailing_dividends = sum(
        parse_float(row["Dividends"])
        for row in rows
        if parse_date(row["Date"]) > lookback_start and parse_float(row["Dividends"]) > 0
    )
    dividend_yield = trailing_dividends / s0 if s0 > 0 else 0.0
    return s0, dividend_yield, latest_date, trailing_dividends


def clean_options() -> dict[str, int | float | str]:
    s0, dividend_yield, latest_date, trailing_dividends = load_spot_and_dividend_yield(HISTORY)

    with OPTIONS_RAW.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    extra_fields = ["spread", "spread_mid_ratio", "S0", "dividend_yield"]
    output_fieldnames = fieldnames + [name for name in extra_fields if name not in fieldnames]

    counts = {
        "raw_rows": len(rows),
        "dropped_zero_bid": 0,
        "dropped_low_open_interest": 0,
        "dropped_wide_spread": 0,
        "clean_rows": 0,
        "S0": s0,
        "dividend_yield": dividend_yield,
        "trailing_dividends": trailing_dividends,
        "history_latest_date": latest_date.date().isoformat(),
    }

    cleaned: list[dict[str, str]] = []
    for row in rows:
        bid = parse_float(row["bid"])
        ask = parse_float(row["ask"])
        mid = parse_float(row.get("mid", ""))
        open_interest = parse_float(row["openInterest"])

        if bid == 0:
            counts["dropped_zero_bid"] += 1
            continue
        if open_interest <= 1:
            counts["dropped_low_open_interest"] += 1
            continue

        spread = ask - bid
        spread_mid_ratio = spread / mid if mid > 0 else float("inf")
        if spread_mid_ratio > MAX_SPREAD_MID_RATIO:
            counts["dropped_wide_spread"] += 1
            continue

        row["spread"] = f"{spread:.10g}"
        row["spread_mid_ratio"] = f"{spread_mid_ratio:.10g}"
        row["S0"] = f"{s0:.10g}"
        row["dividend_yield"] = f"{dividend_yield:.10g}"
        cleaned.append(row)

    counts["clean_rows"] = len(cleaned)

    with OPTIONS_CLEAN.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(cleaned)

    return counts


if __name__ == "__main__":
    summary = clean_options()
    for key, value in summary.items():
        print(f"{key}: {value}")
