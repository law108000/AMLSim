#!/usr/bin/env python3
"""Summarize AMLSim outputs (tx_count.csv and tx_log.csv)."""

import argparse
import csv
from pathlib import Path


def parse_tx_count(path: Path):
    stats = {
        "steps": 0,
        "normal_total": 0,
        "sar_total": 0,
        "max_normal": (None, -1),
        "max_sar": (None, -1),
        "sar_steps": 0,
    }

    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            step = int(row["step"])
            normal = int(row["normal"])
            sar = int(row["SAR"])
            stats["steps"] += 1
            stats["normal_total"] += normal
            stats["sar_total"] += sar
            if normal > stats["max_normal"][1]:
                stats["max_normal"] = (step, normal)
            if sar > stats["max_sar"][1]:
                stats["max_sar"] = (step, sar)
            if sar > 0:
                stats["sar_steps"] += 1

    return stats


def parse_tx_log(path: Path):
    stats = {
        "rows": 0,
        "sar_rows": 0,
        "min_amount": None,
        "max_amount": None,
        "total_amount": 0.0,
    }

    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            amount = float(row["amount"])
            stats["rows"] += 1
            stats["sar_rows"] += int(row["isSAR"])
            stats["total_amount"] += amount
            if stats["min_amount"] is None or amount < stats["min_amount"]:
                stats["min_amount"] = amount
            if stats["max_amount"] is None or amount > stats["max_amount"]:
                stats["max_amount"] = amount

    return stats


def main():
    parser = argparse.ArgumentParser(description="Summarize AMLSim run outputs.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to outputs/<scenario> directory containing tx_count.csv and tx_log.csv",
    )
    args = parser.parse_args()
    tx_count = args.output_dir / "tx_count.csv"
    tx_log = args.output_dir / "tx_log.csv"

    if not tx_count.exists() or not tx_log.exists():
        raise SystemExit(f"Expected files not found in {args.output_dir}")

    count_stats = parse_tx_count(tx_count)
    log_stats = parse_tx_log(tx_log)

    print(f"Scenario directory: {args.output_dir}")
    print("--- Transaction counts ---")
    print(f"Steps: {count_stats['steps']}")
    print(f"Total normal tx: {count_stats['normal_total']:,}")
    print(f"Total SAR tx: {count_stats['sar_total']:,}")
    print(
        f"Peak normal step: {count_stats['max_normal'][0]} ({count_stats['max_normal'][1]})"
    )
    print(f"Peak SAR step: {count_stats['max_sar'][0]} ({count_stats['max_sar'][1]})")
    print(
        f"Steps with SAR activity: {count_stats['sar_steps']} (~{count_stats['sar_steps']/count_stats['steps']*100:.1f}% of steps)"
    )

    print("\n--- Transaction log ---")
    print(f"Rows: {log_stats['rows']:,}")
    print(f"SAR rows: {log_stats['sar_rows']:,}")
    print(
        f"Min/Max amount: {log_stats['min_amount']:.2f} / {log_stats['max_amount']:.2f}"
    )
    avg_amount = log_stats['total_amount'] / log_stats['rows'] if log_stats['rows'] else 0
    print(f"Average amount: {avg_amount:.2f}")


if __name__ == "__main__":
    main()
