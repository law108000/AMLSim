import os
import sys
import csv
import json
import math
import fractions
import warnings
from collections import defaultdict

if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

if not hasattr(matplotlib, "cbook") or not hasattr(matplotlib.cbook, "deprecation"):
    class _DummyDeprecationWarning(Warning):
        MatplotlibDeprecationWarning = Warning

    if not hasattr(matplotlib, "cbook"):
        class _DummyCbook:
            deprecation = _DummyDeprecationWarning

        matplotlib.cbook = _DummyCbook()
    else:
        matplotlib.cbook.deprecation = _DummyDeprecationWarning

if not hasattr(matplotlib.cbook, "is_string_like"):
    def _is_string_like(obj):
        return isinstance(obj, str)

    matplotlib.cbook.is_string_like = _is_string_like

if not hasattr(matplotlib.cbook, "iterable"):
    def _iterable(obj):
        try:
            iter(obj)
            return True
        except TypeError:
            return False

    matplotlib.cbook.iterable = _iterable

if not hasattr(matplotlib.cbook, "is_numlike"):
    def _is_numlike(obj):
        try:
            float(obj)
            return True
        except (TypeError, ValueError):
            return False

    matplotlib.cbook.is_numlike = _is_numlike

warnings.filterwarnings('ignore', category=matplotlib.cbook.deprecation.MatplotlibDeprecationWarning)


def _resolve_path(primary_dir, filename, fallback_dir=None, fallback_name=None, required=True):
    candidates = []
    if filename:
        candidates.append(os.path.join(primary_dir, filename))
    if fallback_dir:
        candidates.append(os.path.join(fallback_dir, fallback_name or filename))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    if required:
        raise FileNotFoundError(f"None of the candidate files exist: {candidates}")
    return None


def _field_lookup(fieldnames, *candidates):
    if not fieldnames:
        return None
    lowered = {name.lower(): name for name in fieldnames if name}
    for cand in candidates:
        key = cand.lower()
        if key in lowered:
            return lowered[key]
    return None


def _add_edges(reader, graph, filter_fn=None):
    orig_field = _field_lookup(
        reader.fieldnames,
        "orig_acct",
        "orig_id",
        "orig",
        "nameorig",
        "src",
    )
    bene_field = _field_lookup(
        reader.fieldnames,
        "bene_acct",
        "dest_id",
        "dest",
        "namedest",
        "dst",
    )
    amt_field = _field_lookup(reader.fieldnames, "amount", "base_amt", "amt")
    date_field = _field_lookup(reader.fieldnames, "tran_timestamp", "timestamp", "date", "step")

    if not orig_field or not bene_field or not amt_field:
        raise ValueError("Could not locate required transaction columns")

    for row in reader:
        if filter_fn and not filter_fn(row):
            continue
        orig_id = row[orig_field]
        bene_id = row[bene_field]
        if orig_id not in graph or bene_id not in graph:
            continue
        amount = row.get(amt_field, "")
        date_val = row.get(date_field, "") if date_field else ""
        if isinstance(date_val, str) and "T" in date_val:
            date_val = date_val.split("T")[0]
        label = amount
        if date_val:
            label += "\n" + str(date_val)
        graph.add_edge(orig_id, bene_id, amount=amount, date=date_val, label=label)


def load_alerts(conf_json):
    graph = nx.DiGraph()
    bank_accts = defaultdict(list)

    with open(conf_json, "r") as rf:
        conf = json.load(rf)

    sim_name = conf["general"]["simulation_name"]
    output_dir = os.path.join(conf["output"]["directory"], sim_name)
    temporal_dir = os.path.join(conf["temporal"]["directory"], sim_name)

    alert_members_file = _resolve_path(
        output_dir,
        conf["output"].get("alert_members"),
        fallback_dir=temporal_dir,
        fallback_name=conf["temporal"].get("alert_members"),
    )

    alert_tx_file = _resolve_path(
        output_dir,
        conf["output"].get("alert_transactions"),
        fallback_dir=temporal_dir,
        fallback_name=conf["temporal"].get("alert_transactions"),
        required=False,
    )

    tx_log_file = os.path.join(output_dir, conf["output"].get("transaction_log", "tx_log.csv"))

    with open(alert_members_file, "r") as rf:
        reader = csv.DictReader(rf)
        acct_field = _field_lookup(reader.fieldnames, "account_id", "acct_id", "accountid")
        bank_field = _field_lookup(reader.fieldnames, "bank_id", "bankid")
        if not acct_field or not bank_field:
            raise ValueError("Could not locate account or bank columns in alert member file")

        for row in reader:
            acct_id = row[acct_field]
            bank_id = row[bank_field]
            graph.add_node(acct_id, bank_id=bank_id)
            bank_accts[bank_id].append(acct_id)

    if alert_tx_file and os.path.exists(alert_tx_file):
        with open(alert_tx_file, "r") as rf:
            reader = csv.DictReader(rf)
            _add_edges(reader, graph)
    elif os.path.exists(tx_log_file):
        with open(tx_log_file, "r") as rf:
            reader = csv.DictReader(rf)
            alert_field = _field_lookup(reader.fieldnames, "alert_id", "alertid")

            def _filter(row):
                if not alert_field:
                    return True
                val = row.get(alert_field, "")
                return val not in ("", "-1", -1, None)

            _add_edges(reader, graph, filter_fn=_filter)
    else:
        raise FileNotFoundError("No alert transaction or transaction log file found")

    return graph, bank_accts


def plot_alerts(graph, bank_accts, output_png):
    cmap = plt.get_cmap("tab10")
    pos = nx.nx_agraph.graphviz_layout(graph)

    plt.figure(figsize=(12.0, 8.0))
    plt.axis('off')

    for i, bank_id in enumerate(bank_accts.keys()):
        color = cmap(i % cmap.N)
        members = bank_accts[bank_id]
        nx.draw_networkx_nodes(graph, pos, members, node_size=300, node_color=color, label=bank_id)
        nx.draw_networkx_labels(graph, pos, {n: n for n in members}, font_size=10)

    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=6)

    plt.legend(numpoints=1)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(output_png, dpi=120)


if __name__ == "__main__":
    argv = sys.argv

    if len(argv) < 3:
        print("Usage: python3 %s [ConfJSON] [OutputPNG]" % argv[0])
        sys.exit(1)

    conf_json = argv[1]
    output_png = argv[2]
    graph, bank_accts = load_alerts(conf_json)
    plot_alerts(graph, bank_accts, output_png)
