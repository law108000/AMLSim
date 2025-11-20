#!/usr/bin/env python3
"""GNN transaction monitoring model training on AMLSim outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv


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
        "--model-type",
        type=str,
        default="graphsage",
        choices=["gcn", "graphsage"],
        help="Type of GNN model to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Number of hidden channels in GNN layers.",
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to run the model on.",
    )
    return parser.parse_args()


def load_data(outputs_dir: Path, simulation_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tx_path = outputs_dir / simulation_name / "transactions.csv"
    acct_path = outputs_dir / simulation_name / "accounts.csv"
    
    if not tx_path.exists():
        raise FileNotFoundError(f"Missing transactions file: {tx_path}")
    
    tx = pd.read_csv(tx_path)
    if "base_amt" in tx.columns:
        tx["base_amt"] = pd.to_numeric(tx["base_amt"], errors="coerce")
    if "tran_timestamp" in tx.columns:
        tx["tran_timestamp"] = pd.to_datetime(tx["tran_timestamp"], errors="coerce")
        
    accounts = pd.DataFrame()
    if acct_path.exists():
        accounts = pd.read_csv(acct_path)
        
    return tx, accounts


def build_graph(tx: pd.DataFrame, accounts: pd.DataFrame) -> Data:
    # Map account IDs to continuous indices
    all_accts = set(tx["orig_acct"]).union(set(tx["bene_acct"]))
    if not accounts.empty:
        all_accts = all_accts.union(set(accounts["acct_id"]))
    
    acct_to_idx = {acct: idx for idx, acct in enumerate(sorted(list(all_accts)))}
    num_nodes = len(acct_to_idx)
    
    # Edge Index
    src = tx["orig_acct"].map(acct_to_idx).values
    dst = tx["bene_acct"].map(acct_to_idx).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # Edge Features (Transaction Features)
    # We want to classify edges, so we need edge features and labels
    tx["amount"] = tx["base_amt"].fillna(0.0)
    tx["amount_log"] = np.log1p(tx["amount"].clip(lower=0))
    tx["hour"] = tx["tran_timestamp"].dt.hour.fillna(-1)
    tx["dayofweek"] = tx["tran_timestamp"].dt.dayofweek.fillna(-1)
    
    edge_features = tx[["amount_log", "hour", "dayofweek"]].values
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Edge Labels
    y = torch.tensor(tx["is_sar"].astype(int).values, dtype=torch.float)
    
    # Node Features
    # If we have account features, use them. Otherwise, use simple stats or identity.
    # For simplicity, let's use degree and some aggregated stats if available, or just random/ones if not.
    # Here we'll construct simple node features based on transaction history (in/out degree, total amount)
    
    # Initialize node features
    node_features = np.zeros((num_nodes, 4), dtype=np.float32) # [in_degree, out_degree, in_amt, out_amt]
    
    # Compute stats
    out_degree = tx["orig_acct"].map(acct_to_idx).value_counts()
    in_degree = tx["bene_acct"].map(acct_to_idx).value_counts()
    out_amt = tx.groupby(tx["orig_acct"].map(acct_to_idx))["amount"].sum()
    in_amt = tx.groupby(tx["bene_acct"].map(acct_to_idx))["amount"].sum()
    
    for idx in range(num_nodes):
        node_features[idx, 0] = in_degree.get(idx, 0)
        node_features[idx, 1] = out_degree.get(idx, 0)
        node_features[idx, 2] = in_amt.get(idx, 0)
        node_features[idx, 3] = out_amt.get(idx, 0)
        
    # Log transform amounts/degrees
    node_features = np.log1p(node_features)
    
    # Normalize node features
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type="graphsage"):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "gcn":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == "graphsage":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            
        # Edge classifier
        # Concatenate source node emb, target node emb, and edge features
        # Edge features dim is 3 (amount_log, hour, dayofweek)
        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels + 3, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        # Node embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Edge classification
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]
        
        # Concatenate
        edge_input = torch.cat([x_src, x_dst, edge_attr], dim=1)
        out = self.edge_classifier(edge_input)
        return out.squeeze()


def train_model(data, args):
    device = torch.device(args.device)
    data = data.to(device)
    
    # Split edges into train/test
    num_edges = data.num_edges
    indices = np.arange(num_edges)
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=args.test_size, 
        stratify=data.y.cpu().numpy(), 
        random_state=args.random_state
    )
    
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    model = GNN(
        in_channels=data.num_node_features,
        hidden_channels=args.hidden_channels,
        out_channels=1,
        model_type=args.model_type
    ).to(device)
    
    # Calculate positive weight for imbalance handling
    num_pos = data.y[train_mask].sum().item()
    num_neg = len(data.y[train_mask]) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print(f"Training {args.model_type.upper()} on {args.device} with pos_weight={pos_weight.item():.2f}...")
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_attr)
        # out is sigmoid output in previous code, but BCEWithLogitsLoss takes logits.
        # We need to change the model to return logits.
        loss = criterion(out[train_mask], data.y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index, data.edge_attr)
                pred = torch.sigmoid(logits)
                train_auc = roc_auc_score(data.y[train_mask].cpu(), pred[train_mask].cpu())
                test_auc = roc_auc_score(data.y[test_mask].cpu(), pred[test_mask].cpu())
                print(f"Epoch {epoch+1:03d}: Loss: {loss.item():.4f}, Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")

    # Final Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        y_score = torch.sigmoid(logits)[test_mask].cpu().numpy()
        y_true = data.y[test_mask].cpu().numpy()
        y_pred = (y_score > 0.5).astype(int)
        
        metrics = {
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "average_precision": float(average_precision_score(y_true, y_score)),
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
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
    tx, accounts = load_data(args.outputs_dir, args.simulation_name)
    data = build_graph(tx, accounts)
    
    model, metrics = train_model(data, args)
    
    args.model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.model_dir / f"{args.simulation_name}_{args.model_type}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    
    print(f"{args.model_type.upper()} Metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))
    
    baseline = load_baseline_metrics(args.model_dir, args.simulation_name)
    if baseline:
        print("\nComparison vs Logistic Regression Baseline:")
        comparison = {}
        for key in ["roc_auc", "average_precision"]:
            if key in baseline and key in metrics:
                diff = metrics[key] - baseline[key]
                comparison[key] = {
                    args.model_type: metrics[key],
                    "baseline": baseline[key],
                    "diff": diff
                }
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
