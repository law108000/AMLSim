#!/usr/bin/env python3
"""NetworkX MultiDiGraph demo using AMLSim outputs (inspired by chess example)."""

from __future__ import annotations

import argparse
import csv
import math
import fractions
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

if not hasattr(cbook, "iterable"):
    def _iterable(obj):
        try:
            iter(obj)
            return True
        except TypeError:
            return False

    cbook.iterable = _iterable

if not hasattr(cbook, "is_string_like"):
    def _is_string_like(obj):
        try:
            obj + ""
            return True
        except Exception:  # pylint: disable=broad-except
            return False

    cbook.is_string_like = _is_string_like

if not hasattr(cbook, "is_numlike"):
    def _is_numlike(obj):
        try:
            float(obj)
            return True
        except Exception:  # pylint: disable=broad-except
            return False

    cbook.is_numlike = _is_numlike

if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

import networkx as nx  # noqa: E402


def load_accounts(accounts_path: Path) -> dict[str, dict[str, str]]:
    accounts = {}
    with accounts_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            accounts[row["ACCOUNT_ID"]] = row
    return accounts


def build_graph(accounts: dict[str, dict[str, str]], tx_log_path: Path) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for acc_id, row in accounts.items():
        graph.add_node(
            acc_id,
            is_sar=row.get("IS_SAR", "false").lower() == "true",
            bank=row.get("BANK_ID", ""),
        )

    with tx_log_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            src = row["nameOrig"]
            dst = row["nameDest"]
            if src not in graph:
                graph.add_node(src, is_sar=False, bank="generated")
            if dst not in graph:
                graph.add_node(dst, is_sar=False, bank="generated")
            graph.add_edge(
                src,
                dst,
                amount=float(row["amount"]),
                result=row.get("Result", row.get("ttype", row.get("type", ""))),
                is_sar=row.get("isSAR", "0") == "1",
                step=int(row["step"]),
            )
    return graph


def summarize_graph(graph: nx.MultiDiGraph):
    print(
        "Loaded {edges} transactions between {nodes} accounts".format(
            edges=graph.number_of_edges(), nodes=graph.number_of_nodes()
        )
    )
    undirected = graph.to_undirected()
    components = [undirected.subgraph(c) for c in nx.connected_components(undirected)]
    if len(components) > 1:
        print("Disconnected components detected (showing small ones):")
        for comp in components[1:]:
            print(sorted(comp.nodes())[:10])
    sar_edges = sum(1 for _, _, d in graph.edges(data=True) if d.get("is_sar"))
    print(f"SAR-flagged edges: {sar_edges}")

    openings = {d.get("result", d.get("Result")) for _, _, d in graph.edges(data=True)}
    print(f"Unique transaction tags stored: {len(openings)}")


def compute_sizes(graph: nx.MultiDiGraph, simple: nx.Graph):
    edgewidth = []
    for u, v in simple.edges():
        data = graph.get_edge_data(u, v)
        width = len(data) if data else 1
        edgewidth.append(width)
    wins = defaultdict(float)
    for src, dst, d in graph.edges(data=True):
        amt = d.get("amount", 0.0)
        wins[src] += amt
    nodesize = [wins[n] * 0.02 for n in simple]
    return edgewidth, nodesize


def draw_graph(graph: nx.MultiDiGraph, output: Path, title: str):
    simple = nx.Graph(graph)
    edgewidth, nodesize = compute_sizes(graph, simple)
    pos = nx.spring_layout(simple)
    node_colors = ["#d62728" if graph.node[n].get("is_sar") else "#1f77b4" for n in simple]

    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_edges(simple, pos, alpha=0.2, width=edgewidth, edge_color="gray", ax=ax)
    nx.draw_networkx_nodes(
        simple,
        pos,
        node_size=nodesize,
        node_color=node_colors,
        alpha=0.9,
        ax=ax,
    )
    label_options = {"ec": "k", "fc": "white", "alpha": 0.6}
    nx.draw_networkx_labels(simple, pos, bbox=label_options, font_size=9, ax=ax)

    ax.set_title(title)
    ax.text(
        0.75,
        0.1,
        "edge width = # transactions",
        transform=ax.transAxes,
        horizontalalignment="center",
    )
    ax.text(
        0.75,
        0.06,
        "node size = total outgoing amount",
        transform=ax.transAxes,
        horizontalalignment="center",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved visualization to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="NetworkX AMLSim demo")
    parser.add_argument("accounts", type=Path, help="Path to accounts.csv")
    parser.add_argument("tx_log", type=Path, help="Path to tx_log.csv")
    parser.add_argument("--output", type=Path, default=Path("amlsim_networkx.png"))
    parser.add_argument("--title", type=str, default="AMLSim MultiDiGraph demo")
    return parser.parse_args()


def main():
    args = parse_args()
    accounts = load_accounts(args.accounts)
    graph = build_graph(accounts, args.tx_log)
    summarize_graph(graph)
    draw_graph(graph, args.output, args.title)


if __name__ == "__main__":
    main()
