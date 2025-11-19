#!/usr/bin/env python3
"""Evaluate Falcon-TST on AMLSim by forecasting per-account activity and scoring anomalies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from transformers import AutoModel

DEFAULT_FREQ = "1h"
CHANNEL_NAMES = ["out_amt", "in_amt", "out_cnt", "in_cnt"]
EPS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--simulation-name", type=str, default="sample")
    parser.add_argument(
        "--falcon-path",
        type=Path,
        default=Path("training") / "falcon-tst",
        help="Path to the locally cloned Falcon-TST_Large repository.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=96,
        help="Number of future time bins to predict for evaluation.",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.2,
        help="Fraction of the timeline reserved for evaluation (rest used as history).",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default=DEFAULT_FREQ,
        help="Resampling frequency for aggregating transactions (pandas offset alias).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run Falcon-TST on.",
    )
    parser.add_argument(
        "--max-accounts",
        type=int,
        default=50,
        help="Keep only the top-N most active origin accounts (<=0 keeps all).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("training") / "models",
        help="Directory where Falcon-TST metrics will be stored.",
    )
    return parser.parse_args()


def load_transactions(outputs_dir: Path, simulation_name: str) -> pd.DataFrame:
    tx_path = outputs_dir / simulation_name / "transactions.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"Missing transactions file: {tx_path}")
    tx = pd.read_csv(tx_path)
    tx["base_amt"] = pd.to_numeric(tx["base_amt"], errors="coerce")
    tx["tran_timestamp"] = pd.to_datetime(tx["tran_timestamp"], errors="coerce")
    tx = tx.dropna(subset=["base_amt", "tran_timestamp", "orig_acct", "is_sar"])
    tx["is_sar"] = tx["is_sar"].astype(int)
    return tx


def select_accounts(tx: pd.DataFrame, max_accounts: int) -> List[str]:
    counts = tx.groupby("orig_acct").size().sort_values(ascending=False)
    if max_accounts <= 0 or len(counts) <= max_accounts:
        return counts.index.tolist()
    return counts.head(max_accounts).index.tolist()


def build_time_index(tx: pd.DataFrame, freq: str) -> pd.DatetimeIndex:
    start = tx["tran_timestamp"].min().floor(freq)
    end = tx["tran_timestamp"].max().ceil(freq)
    return pd.date_range(start=start, end=end, freq=freq)


def build_account_series(
    tx: pd.DataFrame,
    accounts: List[str],
    time_index: pd.DatetimeIndex,
    freq: str,
) -> Tuple[np.ndarray, Dict[str, int], Dict[pd.Timestamp, int], pd.DataFrame]:
    account_to_idx = {acct: idx for idx, acct in enumerate(accounts)}
    time_to_idx = {ts: idx for idx, ts in enumerate(time_index)}
    num_accounts = len(accounts)
    num_steps = len(time_index)
    series = np.zeros((num_accounts, num_steps, len(CHANNEL_NAMES)), dtype=np.float32)

    tx = tx.copy()
    tx["time_bin"] = tx["tran_timestamp"].dt.floor(freq)
    tx = tx[tx["time_bin"].isin(time_to_idx)]

    out_groups = tx.groupby(["orig_acct", "time_bin"]).agg(
        out_amt=("base_amt", "sum"),
        out_cnt=("base_amt", "size"),
    )
    for (acct, ts), row in out_groups.iterrows():
        idx = account_to_idx.get(acct)
        if idx is None:
            continue
        t_idx = time_to_idx[ts]
        series[idx, t_idx, 0] = row["out_amt"]
        series[idx, t_idx, 2] = row["out_cnt"]

    in_groups = tx.groupby(["bene_acct", "time_bin"]).agg(
        in_amt=("base_amt", "sum"),
        in_cnt=("base_amt", "size"),
    )
    for (acct, ts), row in in_groups.iterrows():
        idx = account_to_idx.get(acct)
        if idx is None:
            continue
        t_idx = time_to_idx[ts]
        series[idx, t_idx, 1] = row["in_amt"]
        series[idx, t_idx, 3] = row["in_cnt"]

    return series, account_to_idx, time_to_idx, tx


def compute_windows(num_steps: int, eval_fraction: float, fallback_horizon: int) -> Tuple[int, int]:
    eval_steps = max(1, min(fallback_horizon, math.floor(num_steps * eval_fraction)))
    train_steps = max(1, num_steps - eval_steps)
    eval_steps = min(eval_steps, num_steps - train_steps)
    return train_steps, eval_steps


def forecast_accounts(
    model,
    account_series: np.ndarray,
    train_steps: int,
    horizon: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    num_accounts, num_steps, _ = account_series.shape
    scores = np.zeros((num_accounts, num_steps), dtype=np.float32)
    mask = np.zeros((num_accounts, num_steps), dtype=bool)

    model.to(device)
    model.eval()

    seq_limit = getattr(model.config, "seq_length", train_steps)

    for idx in range(num_accounts):
        series = account_series[idx]
        history = series[:train_steps]
        future = series[train_steps : train_steps + horizon]
        if not np.any(history[:, 0]) or not np.any(future[:, 0]):
            continue
        if history.shape[0] > seq_limit:
            history = history[-seq_limit:]
        input_tensor = torch.from_numpy(history).unsqueeze(0).to(device).float()
        with torch.no_grad():
            predictions = model.predict(input_tensor, forecast_horizon=horizon)
        pred = predictions.squeeze(0).cpu().numpy()
        pred = pred[: future.shape[0]]
        future = future[: pred.shape[0]]
        diff = np.abs(future - pred)
        err = diff[:, 0] / (np.abs(future[:, 0]) + EPS)
        scores[idx, train_steps : train_steps + len(err)] = err
        mask[idx, train_steps : train_steps + len(err)] = True
    return scores, mask


def evaluate_transactions(
    tx: pd.DataFrame,
    account_to_idx: Dict[str, int],
    time_to_idx: Dict[pd.Timestamp, int],
    scores: np.ndarray,
    score_mask: np.ndarray,
    train_steps: int,
    horizon: int,
) -> Dict[str, float]:
    eval_start_idx = train_steps
    eval_end_idx = min(scores.shape[1] - 1, train_steps + horizon - 1)

    eval_times = {ts for ts, idx in time_to_idx.items() if eval_start_idx <= idx <= eval_end_idx}
    tx_eval = tx[tx["time_bin"].isin(eval_times) & tx["orig_acct"].isin(account_to_idx)].copy()
    if tx_eval.empty:
        return {}

    tx_eval["acct_idx"] = tx_eval["orig_acct"].map(account_to_idx)
    tx_eval["time_idx"] = tx_eval["time_bin"].map(time_to_idx)
    tx_eval = tx_eval.dropna(subset=["acct_idx", "time_idx"])
    if tx_eval.empty:
        return {}

    tx_eval["has_score"] = tx_eval.apply(
        lambda row: score_mask[int(row["acct_idx"]), int(row["time_idx"])]
        if 0 <= int(row["time_idx"]) < score_mask.shape[1]
        else False,
        axis=1,
    )
    tx_eval = tx_eval[tx_eval["has_score"]]
    if tx_eval.empty:
        return {}

    tx_eval["score"] = tx_eval.apply(
        lambda row: scores[int(row["acct_idx"]), int(row["time_idx"])]
        if 0 <= int(row["time_idx"]) < scores.shape[1]
        else 0.0,
        axis=1,
    )
    valid = tx_eval
    if valid.empty or valid["is_sar"].nunique() < 2:
        return {}

    y_true = valid["is_sar"].astype(int)
    y_score = valid["score"].astype(float)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "classification_report": classification_report(y_true, y_score > y_score.mean(), output_dict=True),
        "num_samples": int(len(valid)),
        "positive_ratio": float(y_true.mean()),
    }
    return metrics


def load_baseline_metrics(model_dir: Path, simulation_name: str) -> Dict[str, float] | None:
    baseline_path = model_dir / f"{simulation_name}_metrics.json"
    if not baseline_path.exists():
        return None
    return json.loads(baseline_path.read_text())


def main():
    args = parse_args()
    tx = load_transactions(args.outputs_dir, args.simulation_name)
    candidate_accounts = select_accounts(tx, args.max_accounts)
    tx = tx[tx["orig_acct"].isin(candidate_accounts) | tx["bene_acct"].isin(candidate_accounts)].copy()
    time_index = build_time_index(tx, args.freq)
    series, account_to_idx, time_to_idx, tx = build_account_series(tx, candidate_accounts, time_index, args.freq)
    train_steps, eval_steps = compute_windows(len(time_index), args.eval_fraction, args.forecast_horizon)

    model = AutoModel.from_pretrained(args.falcon_path, trust_remote_code=True)
    scores, score_mask = forecast_accounts(model, series, train_steps, eval_steps, args.device)
    metrics = evaluate_transactions(tx, account_to_idx, time_to_idx, scores, score_mask, train_steps, eval_steps)

    if not metrics:
        raise RuntimeError("Falcon-TST evaluation did not produce valid metrics (insufficient data).")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.model_dir / f"{args.simulation_name}_falcon_tst_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))

    baseline = load_baseline_metrics(args.model_dir, args.simulation_name)
    comparison = {}
    if baseline:
        for key in ("roc_auc", "average_precision"):
            if key in baseline and key in metrics:
                comparison[key] = metrics[key] - baseline[key]

    print("Falcon-TST metrics saved to", out_path)
    print(json.dumps(metrics, indent=2))
    if comparison:
        print("Comparison vs logistic baseline:")
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
