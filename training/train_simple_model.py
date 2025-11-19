#!/usr/bin/env python3
"""Simple transaction monitoring model training on AMLSim outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory that contains simulation subfolders (defaults to ./outputs).",
    )
    parser.add_argument(
        "--simulation-name",
        type=str,
        default="sample",
        help="Name of the simulation subdirectory under outputs/.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("training") / "models",
        help="Directory where the trained model and metrics will be stored.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for the test split (default 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_transactions(outputs_dir: Path, simulation_name: str) -> pd.DataFrame:
    tx_path = outputs_dir / simulation_name / "transactions.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"Missing transactions file: {tx_path}")
    tx = pd.read_csv(tx_path)
    if "base_amt" in tx.columns:
        tx["base_amt"] = pd.to_numeric(tx["base_amt"], errors="coerce")
    if "tran_timestamp" in tx.columns:
        tx["tran_timestamp"] = pd.to_datetime(tx["tran_timestamp"], errors="coerce")
    return tx


def enrich_with_accounts(outputs_dir: Path, simulation_name: str, tx: pd.DataFrame) -> pd.DataFrame:
    acct_path = outputs_dir / simulation_name / "accounts.csv"
    if not acct_path.exists():
        return tx

    acct_cols = [
        "acct_id",
        "prior_sar_count",
        "tx_behavior_id",
        "bank_id",
        "initial_deposit",
    ]
    header = pd.read_csv(acct_path, nrows=0)
    usecols = [c for c in acct_cols if c in header.columns]
    accounts = pd.read_csv(acct_path, usecols=usecols)

    def merge_side(df: pd.DataFrame, suffix: str, key: str) -> pd.DataFrame:
        subset = accounts.rename(columns={col: f"{col}_{suffix}" for col in accounts.columns if col != "acct_id"})
        subset = subset.rename(columns={"acct_id": key})
        return df.merge(subset, on=key, how="left")

    tx = merge_side(tx, "orig", "orig_acct")
    tx = merge_side(tx, "bene", "bene_acct")
    return tx


def engineer_features(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["amount"] = tx["base_amt"].fillna(0.0)
    tx["amount_log"] = np.log1p(tx["amount"].clip(lower=0))

    tx["hour"] = tx["tran_timestamp"].dt.hour.fillna(-1)
    tx["dayofweek"] = tx["tran_timestamp"].dt.dayofweek.fillna(-1)
    tx["is_alert"] = (
        tx.get("alert_id", pd.Series([None] * len(tx)))
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
        .astype(int)
    )

    tx["orig_freq"] = tx["orig_acct"].map(tx["orig_acct"].value_counts())
    tx["bene_freq"] = tx["bene_acct"].map(tx["bene_acct"].value_counts())

    bool_map = {True: 1, False: 0, "true": 1, "false": 0, "True": 1, "False": 0}
    for col in tx.columns:
        if col.endswith("prior_sar_count"):
            tx[col] = tx[col].map(bool_map)
        if col.endswith("tx_behavior_id"):
            tx[col] = pd.to_numeric(tx[col], errors="coerce")
        if col.endswith("initial_deposit"):
            tx[col] = pd.to_numeric(tx[col], errors="coerce")

    return tx


def build_dataset(tx: pd.DataFrame):
    if "is_sar" not in tx.columns:
        raise ValueError("transactions.csv does not include is_sar column.")

    feature_columns_numeric = [
        "amount",
        "amount_log",
        "hour",
        "dayofweek",
        "is_alert",
        "orig_freq",
        "bene_freq",
        "prior_sar_count_orig",
        "prior_sar_count_bene",
        "tx_behavior_id_orig",
        "tx_behavior_id_bene",
        "initial_deposit_orig",
        "initial_deposit_bene",
    ]
    present_numeric = [col for col in feature_columns_numeric if col in tx.columns and tx[col].notna().any()]

    categorical_candidates = [
        "tx_type",
        "bank_id_orig",
        "bank_id_bene",
    ]
    categorical_columns = [col for col in categorical_candidates if col in tx.columns and tx[col].notna().any()]

    feature_columns = present_numeric + categorical_columns
    if not feature_columns:
        raise ValueError("No usable feature columns were found in transactions.csv.")

    df = tx[feature_columns + ["is_sar"]].replace([np.inf, -np.inf], np.nan).dropna(subset=["is_sar"])
    return df, present_numeric, categorical_columns


def train_model(df: pd.DataFrame, numeric_cols, categorical_cols, test_size, random_state, model_dir: Path, simulation_name: str):
    X = df.drop(columns=["is_sar"])
    y = df["is_sar"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=None,
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_scores)),
        "average_precision": float(average_precision_score(y_test, y_scores)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "test_size": int(y_test.shape[0]),
        "positive_ratio_test": float(y_test.mean()),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{simulation_name}_logreg.joblib"
    metrics_path = model_dir / f"{simulation_name}_metrics.json"

    joblib.dump(clf, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return model_path, metrics_path, metrics


def main():
    args = parse_args()
    tx = load_transactions(args.outputs_dir, args.simulation_name)
    tx = enrich_with_accounts(args.outputs_dir, args.simulation_name, tx)
    tx = engineer_features(tx)
    df, numeric_cols, categorical_cols = build_dataset(tx)

    model_path, metrics_path, metrics = train_model(
        df,
        numeric_cols,
        categorical_cols,
        test_size=args.test_size,
        random_state=args.random_state,
        model_dir=args.model_dir,
        simulation_name=args.simulation_name,
    )

    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
