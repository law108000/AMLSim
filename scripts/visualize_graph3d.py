#!/usr/bin/env python3
"""Render AMLSim transaction graphs in 3D using Matplotlib."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple

import math
import fractions

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
import numpy as np

if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

import networkx as nx


def load_accounts(path: Path) -> Dict[str, Dict[str, str]]:
    accounts: Dict[str, Dict[str, str]] = {}
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            accounts[row["ACCOUNT_ID"]] = row
    return accounts


def build_graph(account_rows: Dict[str, Dict[str, str]], tx_path: Path, max_nodes: int | None) -> nx.Graph:
    g = nx.Graph()
    for account_id, attrs in account_rows.items():
        if max_nodes is not None and len(g) >= max_nodes:
            break
        g.add_node(
            account_id,
            is_sar=attrs["IS_SAR"].lower() == "true",
            bank_id=attrs.get("BANK_ID", ""),
        )

    with tx_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            src, dst = row["src"], row["dst"]
            if src in g and dst in g:
                g.add_edge(src, dst, ttype=row.get("ttype", ""))
    return g


def plot_graph(
    graph: nx.Graph,
    output: Path,
    seed: int,
    max_edges: int,
    figsize: Tuple[int, int],
    show: bool,
):
    if graph.number_of_nodes() == 0:
        raise SystemExit("Graph is empty; check input filters")

    old_state = np.random.get_state()
    np.random.seed(seed)
    pos = nx.spring_layout(graph, dim=3)
    np.random.set_state(old_state)

    xs_sar, ys_sar, zs_sar = [], [], []
    xs_norm, ys_norm, zs_norm = [], [], []
    for node, (x, y, z) in pos.items():
        node_attrs = graph.node[node]
        if node_attrs.get("is_sar"):
            xs_sar.append(x)
            ys_sar.append(y)
            zs_sar.append(z)
        else:
            xs_norm.append(x)
            ys_norm.append(y)
            zs_norm.append(z)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(xs_norm, ys_norm, zs_norm, s=10, c="#1f77b4", alpha=0.7, label="Normal")
    if xs_sar:
        ax.scatter(xs_sar, ys_sar, zs_sar, s=25, c="#d62728", alpha=0.9, label="SAR")

    for idx, (src, dst) in enumerate(graph.edges()):
        if idx >= max_edges:
            break
        x_vals = [pos[src][0], pos[dst][0]]
        y_vals = [pos[src][1], pos[dst][1]]
        z_vals = [pos[src][2], pos[dst][2]]
        ax.plot(x_vals, y_vals, z_vals, c="gray", alpha=0.1)

    ax.set_axis_off()
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize AMLSim graph in 3D using Matplotlib")
    parser.add_argument("accounts", type=Path, help="Path to accounts.csv")
    parser.add_argument("transactions", type=Path, help="Path to transactions.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("graph3d.png"),
        help="Output image file (PNG)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for layout")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Limit number of nodes plotted (useful for huge graphs)",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=5000,
        help="Limit number of edges drawn to keep plot readable",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(10, 8),
        metavar=("WIDTH", "HEIGHT"),
        help="Matplotlib figure size",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the Matplotlib window in addition to saving the image",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    accounts = load_accounts(args.accounts)
    graph = build_graph(accounts, args.transactions, args.max_nodes)
    plot_graph(graph, args.output, args.seed, args.max_edges, tuple(args.figsize), args.show)
    print(f"Saved 3D visualization to {args.output}")


if __name__ == "__main__":
    main()
