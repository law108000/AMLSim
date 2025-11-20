#!/usr/bin/env python3
"""Temporal GNN (EvolveGCN style) transaction monitoring model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


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
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Number of hidden channels.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to run the model on.",
    )
    return parser.parse_args()


def load_and_process_data(outputs_dir: Path, simulation_name: str) -> Tuple[List[Data], int, int]:
    tx_path = outputs_dir / simulation_name / "transactions.csv"
    acct_path = outputs_dir / simulation_name / "accounts.csv"
    
    if not tx_path.exists():
        raise FileNotFoundError(f"Missing transactions file: {tx_path}")
    
    tx = pd.read_csv(tx_path)
    tx["base_amt"] = pd.to_numeric(tx["base_amt"], errors="coerce").fillna(0.0)
    tx["tran_timestamp"] = pd.to_datetime(tx["tran_timestamp"], errors="coerce")
    tx["day"] = tx["tran_timestamp"].dt.floor("d")
    
    accounts = pd.DataFrame()
    if acct_path.exists():
        accounts = pd.read_csv(acct_path)
        
    # Map accounts
    all_accts = set(tx["orig_acct"]).union(set(tx["bene_acct"]))
    if not accounts.empty:
        all_accts = all_accts.union(set(accounts["acct_id"]))
    acct_to_idx = {acct: idx for idx, acct in enumerate(sorted(list(all_accts)))}
    num_nodes = len(acct_to_idx)
    
    # Split by day
    daily_groups = tx.groupby("day")
    snapshots = []
    
    # Global scaler for node features
    # We'll compute node features per snapshot but scale them globally if possible, 
    # or just scale per snapshot. Let's scale per snapshot for simplicity.
    
    for day, group in daily_groups:
        # Edge Index
        src = group["orig_acct"].map(acct_to_idx).values
        dst = group["bene_acct"].map(acct_to_idx).values
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        # Edge Features
        group["amount_log"] = np.log1p(group["base_amt"])
        group["hour"] = group["tran_timestamp"].dt.hour
        edge_attr = torch.tensor(group[["amount_log", "hour"]].values, dtype=torch.float)
        
        # Labels
        y = torch.tensor(group["is_sar"].astype(int).values, dtype=torch.float)
        
        # Node Features (Degree & Volume for this day)
        node_feats = np.zeros((num_nodes, 4), dtype=np.float32)
        out_degree = group["orig_acct"].map(acct_to_idx).value_counts()
        in_degree = group["bene_acct"].map(acct_to_idx).value_counts()
        out_amt = group.groupby(group["orig_acct"].map(acct_to_idx))["base_amt"].sum()
        in_amt = group.groupby(group["bene_acct"].map(acct_to_idx))["base_amt"].sum()
        
        for idx in range(num_nodes):
            node_feats[idx, 0] = in_degree.get(idx, 0)
            node_feats[idx, 1] = out_degree.get(idx, 0)
            node_feats[idx, 2] = in_amt.get(idx, 0)
            node_feats[idx, 3] = out_amt.get(idx, 0)
            
        node_feats = np.log1p(node_feats)
        scaler = StandardScaler()
        node_feats = scaler.fit_transform(node_feats)
        x = torch.tensor(node_feats, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        snapshots.append(data)
        
    return snapshots, num_nodes, 4


class RecurrentGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.gru = torch.nn.GRUCell(hidden_channels, hidden_channels)
        
        # Edge classifier
        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels + 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
        self.hidden_channels = hidden_channels

    def forward(self, x, edge_index, edge_attr, h_prev):
        # GCN Step
        h_curr = self.conv(x, edge_index)
        h_curr = F.relu(h_curr)
        
        # GRU Step (Evolve node states)
        if h_prev is None:
            h_prev = torch.zeros_like(h_curr)
        h_next = self.gru(h_curr, h_prev)
        
        # Edge Classification
        src, dst = edge_index
        x_src = h_next[src]
        x_dst = h_next[dst]
        
        edge_input = torch.cat([x_src, x_dst, edge_attr], dim=1)
        out = self.edge_classifier(edge_input)
        
        return out.squeeze(), h_next


def train_model(snapshots, num_nodes, args):
    device = torch.device(args.device)
    model = RecurrentGCN(
        in_channels=snapshots[0].num_node_features,
        hidden_channels=args.hidden_channels
    ).to(device)
    
    # Split snapshots into train/test (temporal split)
    # Adjust split to ensure test set has anomalies (SARs end in July 2018, data ends Dec 2018)
    # We split at 50% to ensure the test set covers the latter half of SARs
    split_idx = int(len(snapshots) * 0.5)
    train_snapshots = snapshots[:split_idx]
    test_snapshots = snapshots[split_idx:]
    
    # Check distribution
    train_pos = sum(d.y.sum().item() for d in train_snapshots)
    test_pos = sum(d.y.sum().item() for d in test_snapshots)
    print(f"Train snapshots: {len(train_snapshots)} (SARs: {int(train_pos)})")
    print(f"Test snapshots: {len(test_snapshots)} (SARs: {int(test_pos)})")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Calculate global pos_weight for training
    total_pos = sum(d.y.sum().item() for d in train_snapshots)
    total_count = sum(len(d.y) for d in train_snapshots)
    pos_weight = torch.tensor([(total_count - total_pos) / max(total_pos, 1.0)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print(f"Training RecurrentGCN on {args.device} with {len(snapshots)} snapshots...")
    
    for epoch in range(args.epochs):
        model.train()
        h = None
        total_loss = 0
        
        # Train on sequence
        for data in train_snapshots:
            data = data.to(device)
            optimizer.zero_grad()
            
            out, h = model(data.x, data.edge_index, data.edge_attr, h)
            # Detach h to prevent backprop through entire history (Truncated BPTT)
            h = h.detach()
            
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d}: Avg Loss: {total_loss / len(train_snapshots):.4f}")

    # Evaluation
    model.eval()
    h = None
    # Warm up hidden state on train data
    with torch.no_grad():
        for data in train_snapshots:
            data = data.to(device)
            _, h = model(data.x, data.edge_index, data.edge_attr, h)
            
    # Predict on test data
    all_y_true = []
    all_y_scores = []
    
    with torch.no_grad():
        for data in test_snapshots:
            data = data.to(device)
            logits, h = model(data.x, data.edge_index, data.edge_attr, h)
            scores = torch.sigmoid(logits)
            all_y_true.extend(data.y.cpu().numpy())
            all_y_scores.extend(scores.cpu().numpy())
            
    y_true = np.array(all_y_true)
    y_scores = np.array(all_y_scores)
    y_pred = (y_scores > 0.5).astype(int)
    
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "average_precision": float(average_precision_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "classification_report": classification_report(y_true, y_pred, output_dict=True) if len(np.unique(y_true)) > 1 else {},
        "test_size": int(len(y_true)),
        "positive_ratio_test": float(y_true.mean()),
    }
    
    return model, metrics


def load_baseline_metrics(model_dir: Path, simulation_name: str) -> dict | None:
    baseline_path = model_dir / f"{simulation_name}_metrics.json"
    if not baseline_path.exists():
        return None
    return json.loads(baseline_path.read_text())


def main():
    args = parse_args()
    snapshots, num_nodes, num_features = load_and_process_data(args.outputs_dir, args.simulation_name)
    
    if len(snapshots) < 2:
        print("Not enough snapshots for temporal training.")
        return

    model, metrics = train_model(snapshots, num_nodes, args)
    
    args.model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.model_dir / f"{args.simulation_name}_evolvegcn_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    
    print(f"EvolveGCN (Temporal) Metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))
    
    baseline = load_baseline_metrics(args.model_dir, args.simulation_name)
    if baseline:
        print("\nComparison vs Logistic Regression Baseline:")
        comparison = {}
        for key in ["roc_auc", "average_precision"]:
            if key in baseline and key in metrics:
                diff = metrics[key] - baseline[key]
                comparison[key] = {
                    "evolvegcn": metrics[key],
                    "baseline": baseline[key],
                    "diff": diff
                }
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
