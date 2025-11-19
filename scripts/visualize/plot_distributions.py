"""
Plot statistical distributions from the transaction graph.
"""

import os
import sys
import csv
import json
import math
import fractions
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta

if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

import networkx as nx
import powerlaw
import numpy as np

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

category = matplotlib.cbook.deprecation.MatplotlibDeprecationWarning
warnings.filterwarnings('ignore', category=category)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def _field_lookup(fieldnames, *candidates):
    if not fieldnames:
        return None
    lowered = {name.lower(): name for name in fieldnames if name}
    for cand in candidates:
        if not cand:
            continue
        cand_lower = cand.lower()
        if cand_lower in lowered:
            return lowered[cand_lower]
    return None


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    return bool(value)


def _parse_date_value(raw_value, base_date):
    if raw_value is None or raw_value == "":
        return base_date
    value = str(raw_value).strip()
    if not value:
        return base_date
    if value.isdigit():
        return base_date + timedelta(days=int(value))
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    if "T" in value:
        try:
            return datetime.strptime(value.split("T")[0], "%Y-%m-%d")
        except ValueError:
            pass
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return base_date


def _resolve_data_path(conf, sim_name, key, *, fallback_key=None, required=True):
    output_dir = conf.get("output", {}).get("directory", "outputs")
    temporal_dir = conf.get("temporal", {}).get("directory")
    primary_name = conf.get("output", {}).get(key)
    fallback_name = conf.get("temporal", {}).get(fallback_key or key)
    candidates = []
    if primary_name:
        candidates.append(os.path.join(output_dir, sim_name, primary_name))
    if temporal_dir and fallback_name:
        candidates.append(os.path.join(temporal_dir, sim_name, fallback_name))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    if required:
        raise FileNotFoundError("Could not locate file for '%s'. Tried: %s" % (key, candidates))
    return None


def get_date_list(_g):
    all_dates = list(nx.get_edge_attributes(_g, "date").values())
    start_date = min(all_dates)
    end_date = max(all_dates)
    days = (end_date - start_date).days + 1
    date_list = [start_date + timedelta(days=n) for n in range(days)]
    return date_list


def construct_graph(_acct_csv, _tx_csv, _schema, _base_date):
    """Load transaction data (accounts + transactions) and construct Graph."""

    if isinstance(_base_date, str):
        base_date = datetime.strptime(_base_date, "%Y-%m-%d")
    else:
        base_date = _base_date

    _g = nx.MultiDiGraph()

    with open(_acct_csv, "r") as _rf:
        reader = csv.DictReader(_rf)
        if not reader.fieldnames:
            raise ValueError("Account CSV %s has no header" % _acct_csv)
        acct_field = _field_lookup(reader.fieldnames, "account_id", "acct_id", "accountid")
        bank_field = _field_lookup(reader.fieldnames, "bank_id", "bankid", "bank")
        sar_field = _field_lookup(reader.fieldnames, "is_sar", "sar_flag", "sar", "issar")

        if not acct_field:
            raise ValueError("Could not find account id column in %s" % _acct_csv)
        if not bank_field:
            raise ValueError("Could not find bank id column in %s" % _acct_csv)

        for row in reader:
            acct_id = row.get(acct_field)
            if acct_id is None:
                continue
            bank_id = row.get(bank_field, "UNKNOWN")
            is_sar = _as_bool(row.get(sar_field, False)) if sar_field else False
            _g.add_node(acct_id, bank_id=bank_id, is_sar=is_sar)

    with open(_tx_csv, "r") as _rf:
        reader = csv.DictReader(_rf)
        if not reader.fieldnames:
            raise ValueError("Transaction CSV %s has no header" % _tx_csv)
        orig_field = _field_lookup(reader.fieldnames, "orig_acct", "orig_id", "orig", "nameorig", "src", "from", "payer", "sender")
        bene_field = _field_lookup(reader.fieldnames, "bene_acct", "dest_id", "dest", "namedest", "dst", "to", "receiver")
        amt_field = _field_lookup(reader.fieldnames, "amount", "base_amt", "amt", "baseamount", "transactionamount")
        date_field = _field_lookup(reader.fieldnames, "tran_timestamp", "timestamp", "date", "step", "trans_date", "txdate")
        type_field = _field_lookup(reader.fieldnames, "tx_type", "transaction_type", "type", "ttype")
        sar_field = _field_lookup(reader.fieldnames, "is_sar", "sar_flag", "issar")
        alert_field = _field_lookup(reader.fieldnames, "alert_id", "alertid")

        required_fields = [
            ("origin", orig_field),
            ("destination", bene_field),
            ("amount", amt_field),
            ("date", date_field),
        ]
        missing = [name for name, field in required_fields if not field]
        if missing:
            raise ValueError("Transaction CSV %s missing required columns: %s" % (_tx_csv, ", ".join(missing)))

        for row in reader:
            orig = row.get(orig_field)
            bene = row.get(bene_field)
            if orig is None or bene is None:
                continue
            try:
                amount = float(row.get(amt_field, 0.0) or 0.0)
            except ValueError:
                continue
            date = _parse_date_value(row.get(date_field), base_date)
            tx_type = row.get(type_field, "") if type_field else ""
            is_sar = _as_bool(row.get(sar_field, False)) if sar_field else False
            alert_id = row.get(alert_field) if alert_field else None

            if orig not in _g:
                _g.add_node(orig, bank_id="UNKNOWN", is_sar=False)
            if bene not in _g:
                _g.add_node(bene, bank_id="UNKNOWN", is_sar=False)

            _g.add_edge(orig, bene, amount=amount, date=date, type=tx_type, is_sar=is_sar, alert_id=alert_id)

    return _g


def plot_degree_distribution(_g, _conf, _plot_img):
    """Plot degree distribution for accounts (vertices)
    :param _g: Transaction graph
    :param _conf: Configuration object
    :param _plot_img: Degree distribution image (log-log plot)
    :return:
    """
    # Load parameter files
    _input_conf = _conf["input"]
    _input_dir = _input_conf["directory"]
    _input_acct = _input_conf["accounts"]
    _input_deg = _input_conf["degree"]
    input_acct_path = os.path.join(_input_dir, _input_acct)
    input_deg_path = os.path.join(_input_dir, _input_deg)

    if not os.path.isfile(input_acct_path):
        print("Account parameter file %s is not found." % input_acct_path)
        return

    total_num_accts = 0
    with open(input_acct_path, "r") as _rf:
        reader = csv.reader(_rf)
        header = next(reader)
        count_idx = None
        for i, col in enumerate(header):
            if col == "count":
                count_idx = i
                break
        for row in reader:
            total_num_accts += int(row[count_idx])

    if not os.path.isfile(input_deg_path):
        print("Degree parameter file %s is not found." % input_deg_path)
        return

    deg_num_accts = 0
    in_degrees = list()
    in_deg_seq = list()
    in_deg_hist = list()
    out_degrees = list()
    out_deg_seq = list()
    out_deg_hist = list()
    with open(input_deg_path, "r") as _rf:
        reader = csv.reader(_rf)
        next(reader)
        for row in reader:
            deg = int(row[0])
            in_num = int(row[1])
            out_num = int(row[2])
            if in_num > 0:
                in_degrees.extend([deg] * in_num)
                in_deg_seq.append(deg)
                in_deg_hist.append(in_num)
                deg_num_accts += in_num
            if out_num > 0:
                out_degrees.extend([deg] * out_num)
                out_deg_seq.append(deg)
                out_deg_hist.append(out_num)

    multiplier = total_num_accts // deg_num_accts
    # print(total_num_accts, deg_num_accts, multiplier)
    in_degrees = [d * multiplier for d in in_degrees]
    in_deg_hist = [d * multiplier for d in in_deg_hist]
    out_degrees = [d * multiplier for d in out_degrees]
    out_deg_hist = [d * multiplier for d in out_deg_hist]

    # ax1, ax2: Expected in/out-degree distributions from parameter files
    # ax3, ax4: Output in/out-degree distributions from the output transaction list
    plt.clf()
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    pw_result = powerlaw.Fit(in_degrees, verbose=False)
    alpha = pw_result.power_law.alpha
    alpha_text = "alpha = %.2f" % alpha
    ax1.loglog(in_deg_seq, in_deg_hist, "bo-")
    ax1.set_title("Expected in-degree distribution")
    plt.text(0.75, 0.9, alpha_text, transform=ax1.transAxes)
    ax1.set_xlabel("In-degree")
    ax1.set_ylabel("Number of account vertices")

    pw_result = powerlaw.Fit(out_degrees, verbose=False)
    alpha = pw_result.power_law.alpha
    alpha_text = "alpha = %.2f" % alpha
    ax2.loglog(out_deg_seq, out_deg_hist, "ro-")
    ax2.set_title("Expected out-degree distribution")
    plt.text(0.75, 0.9, alpha_text, transform=ax2.transAxes)
    ax2.set_xlabel("Out-degree")
    ax2.set_ylabel("Number of account vertices")

    # Get degree from the output transaction list
    in_degrees = [len(_g.pred[n].keys()) for n in _g.nodes()]  # list(_g.in_degree().values())
    in_deg_seq = sorted(set(in_degrees))
    in_deg_hist = [in_degrees.count(x) for x in in_deg_seq]
    pw_result = powerlaw.Fit(in_degrees, verbose=False)
    alpha = pw_result.power_law.alpha
    alpha_text = "alpha = %.2f" % alpha
    ax3.loglog(in_deg_seq, in_deg_hist, "bo-")
    ax3.set_title("Output in-degree distribution")
    plt.text(0.75, 0.9, alpha_text, transform=ax3.transAxes)
    ax3.set_xlabel("In-degree")
    ax3.set_ylabel("Number of account vertices")

    out_degrees = [len(_g.succ[n].keys()) for n in _g.nodes()]  # list(_g.out_degree().values())
    # print("max out-degree", max(out_degrees))
    out_deg_seq = sorted(set(out_degrees))
    out_deg_hist = [out_degrees.count(x) for x in out_deg_seq]
    pw_result = powerlaw.Fit(out_degrees, verbose=False)
    alpha = pw_result.power_law.alpha
    alpha_text = "alpha = %.2f" % alpha
    ax4.loglog(out_deg_seq, out_deg_hist, "ro-")
    ax4.set_title("Output out-degree distribution")
    plt.text(0.75, 0.9, alpha_text, transform=ax4.transAxes)
    ax4.set_xlabel("Out-degree")
    ax4.set_ylabel("Number of account vertices")

    plt.savefig(_plot_img)


def plot_wcc_distribution(_g, _plot_img):
    """Plot weakly connected components size distributions
    :param _g: Transaction graph
    :param _plot_img: WCC size distribution image (log-log plot)
    :return:
    """
    all_wcc = nx.weakly_connected_components(_g)
    wcc_sizes = Counter([len(wcc) for wcc in all_wcc])
    size_seq = sorted(wcc_sizes.keys())
    size_hist = [wcc_sizes[x] for x in size_seq]

    plt.figure(figsize=(16, 12))
    plt.clf()
    plt.loglog(size_seq, size_hist, 'ro-')
    plt.title("WCC Size Distribution")
    plt.xlabel("Size")
    plt.ylabel("Number of WCCs")
    plt.savefig(_plot_img)


def plot_alert_stat(_alert_acct_csv, _alert_tx_csv, _schema, _plot_img, _base_date, tx_log_csv=None):

    if isinstance(_base_date, str):
        base_date = datetime.strptime(_base_date, "%Y-%m-%d")
    else:
        base_date = _base_date

    if not os.path.exists(_alert_acct_csv):
        print("Alert member file %s not found. Skipping alert plots." % _alert_acct_csv)
        return

    alert_member_count = Counter()
    alert_tx_count = Counter()
    alert_init_amount = dict()  # Initial amount
    alert_amount_list = defaultdict(list)  # All amount list
    alert_dates = defaultdict(list)
    label_alerts = defaultdict(list)  # label -> alert IDs

    with open(_alert_acct_csv, "r") as _rf:
        reader = csv.DictReader(_rf)
        if not reader.fieldnames:
            print("Alert member file %s has no header. Skipping" % _alert_acct_csv)
            return
        alert_field = _field_lookup(reader.fieldnames, "alert_id", "alertid")
        type_field = _field_lookup(reader.fieldnames, "alert_type", "reason", "type")
        sar_field = _field_lookup(reader.fieldnames, "is_sar", "sar_flag", "issar")

        if not alert_field:
            print("Alert member file %s missing alert ID column. Skipping" % _alert_acct_csv)
            return

        for row in reader:
            alert_id = row.get(alert_field)
            if alert_id is None:
                continue
            alert_type = row.get(type_field) if type_field else "Unknown"
            is_sar = _as_bool(row.get(sar_field, False)) if sar_field else False

            alert_member_count[alert_id] += 1
            label = ("SAR" if is_sar else "Normal") + ":" + (alert_type or "Unknown")
            label_alerts[label].append(alert_id)

    tx_source = None
    if _alert_tx_csv and os.path.exists(_alert_tx_csv):
        tx_source = _alert_tx_csv
    elif tx_log_csv and os.path.exists(tx_log_csv):
        tx_source = tx_log_csv
    else:
        print("Alert transaction data not found. Skipping alert plots.")
        return

    with open(tx_source, "r") as _rf:
        reader = csv.DictReader(_rf)
        if not reader.fieldnames:
            print("Alert transaction file %s has no header. Skipping" % tx_source)
            return
        alert_field = _field_lookup(reader.fieldnames, "alert_id", "alertid")
        amt_field = _field_lookup(reader.fieldnames, "amount", "base_amt", "amt", "baseamount")
        date_field = _field_lookup(reader.fieldnames, "tran_timestamp", "timestamp", "date", "step")

        if not alert_field or not amt_field or not date_field:
            print("Alert transaction columns missing in %s. Skipping" % tx_source)
            return

        for row in reader:
            alert_id = row.get(alert_field)
            if alert_id is None or alert_id in ("", "-1", -1):
                continue
            if alert_id not in alert_member_count:
                # Skip alerts we don't know about to avoid noise
                continue
            try:
                amount = float(row.get(amt_field, 0.0) or 0.0)
            except ValueError:
                continue
            date = _parse_date_value(row.get(date_field), base_date)

            alert_tx_count[alert_id] += 1
            alert_amount_list[alert_id].append(amount)
            alert_dates[alert_id].append(date)
            if alert_id not in alert_init_amount:
                alert_init_amount[alert_id] = amount

    if not label_alerts:
        print("No alert membership data to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    cmap = plt.get_cmap("tab10")
    for i, (label, alerts) in enumerate(label_alerts.items()):
        color = cmap(i % cmap.N)
        x_accounts = [alert_member_count.get(a, 0) for a in alerts]
        y_init = np.array([alert_init_amount.get(a, 0.0) for a in alerts])

        ax1.scatter(x_accounts, y_init, s=50, color=color, label=label, edgecolors="none")
        for j, alert_id in enumerate(alerts):
            ax1.annotate(alert_id, (x_accounts[j], y_init[j]))

        x_tx = [alert_tx_count.get(a, 0) for a in alerts]
        y_period = []
        for a in alerts:
            dates = alert_dates.get(a, [])
            if dates:
                period = (max(dates) - min(dates)).days + 1
            else:
                period = 0
            y_period.append(period)

        ax2.scatter(x_tx, y_period, s=100, color=color, label=label, edgecolors="none")
        for j, alert_id in enumerate(alerts):
            ax2.annotate(alert_id, (x_tx[j], y_period[j]))

    ax1.set_xlabel("Number of accounts per alert")
    ax1.set_ylabel("Initial transaction amount")
    ax1.legend()
    ax2.set_xlabel("Number of transactions per alert")
    ax2.set_ylabel("Transaction period")
    ax2.legend()
    plt.savefig(_plot_img)


def plot_aml_rule(aml_csv, _plot_img):
    """Plot the number of AML typologies
    :param aml_csv: AML typology pattern parameter CSV file
    :param _plot_img: Output image file (bar plot)
    """
    aml_types = Counter()
    num_idx = None
    type_idx = None

    if not os.path.isfile(aml_csv):
        print("AML typology file %s is not found." % aml_csv)
        return

    with open(aml_csv, "r") as _rf:
        reader = csv.reader(_rf)
        header = next(reader)
        for i, k in enumerate(header):
            if k == "count":
                num_idx = i
            elif k == "type":
                type_idx = i

        for row in reader:
            if "#" in row[0]:
                continue
            num = int(row[num_idx])
            aml_type = row[type_idx]
            aml_types[aml_type] += num

    x = list()
    y = list()
    for aml_type, num in aml_types.items():
        x.append(aml_type)
        y.append(num)

    plt.figure(figsize=(16, 12))
    plt.clf()
    plt.bar(range(len(x)), y, tick_label=x)
    plt.title("AML typologies")
    plt.xlabel("Typology name")
    plt.ylabel("Number of patterns")
    plt.savefig(_plot_img)


def plot_tx_count(_g, _plot_img):
    """Plot the number of normal and SAR transactions
    :param _g: Transaction graph
    :param _plot_img: Output image file path
    """
    date_list = get_date_list(_g)
    normal_tx_count = Counter()
    sar_tx_count = Counter()

    for _, _, attr in _g.edges(data=True):
        is_sar = attr["is_sar"]
        date = attr["date"]
        if is_sar:
            sar_tx_count[date] += 1
        else:
            normal_tx_count[date] += 1

    normal_tx_list = [normal_tx_count[d] for d in date_list]
    sar_tx_list = [sar_tx_count[d] for d in date_list]

    plt.figure(figsize=(16, 12))
    plt.clf()
    p_n = plt.plot(date_list, normal_tx_list, "b")
    p_f = plt.plot(date_list, sar_tx_list, "r")
    plt.yscale('log')
    plt.legend((p_n[0], p_f[0]), ("Normal", "SAR"))
    plt.title("Number of transactions per step")
    plt.xlabel("Simulation step")
    plt.ylabel("Number of transactions")
    plt.savefig(_plot_img)


def plot_clustering_coefficient(_g, _plot_img, interval=30):
    """Plot the clustering coefficient transition
    :param _g: Transaction graph
    :param _plot_img: Output image file
    :param interval: Simulation step interval for plotting
    (it takes too much time to compute clustering coefficient)
    :return:
    """
    date_list = get_date_list(_g)

    gg = nx.Graph()
    edges = defaultdict(list)
    for k, v in nx.get_edge_attributes(_g, "date").items():
        e = (k[0], k[1])
        edges[v].append(e)

    sample_dates = list()
    values = list()
    for i, t in enumerate(date_list):
        gg.add_edges_from(edges[t])
        if i % interval == 0:
            v = nx.average_clustering(gg) if gg.number_of_nodes() else 0.0
            sample_dates.append(t)
            values.append(v)
            print("Clustering coefficient at %s: %f" % (str(t), v))

    plt.figure(figsize=(16, 12))
    plt.clf()
    plt.plot(sample_dates, values, 'bo-')
    plt.title("Clustering Coefficient Transition")
    plt.xlabel("date")
    plt.ylabel("Clustering Coefficient")
    plt.savefig(_plot_img)


def plot_diameter(dia_csv, _plot_img):
    """Plot the diameter and the average of largest distance transitions
    :param dia_csv: Diameter transition CSV file
    :param _plot_img: Output image file
    :return:
    """
    x = list()
    dia = list()
    aver = list()

    with open(dia_csv, "r") as _rf:
        reader = csv.reader(_rf)
        next(reader)
        for row in reader:
            step = int(row[0])
            d = float(row[1])
            a = float(row[2])
            x.append(step)
            dia.append(d)
            aver.append(a)

    plt.figure(figsize=(16, 12))
    plt.clf()
    plt.ylim(0, max(dia) + 1)
    p_d = plt.plot(x, dia, "r")
    p_a = plt.plot(x, aver, "b")
    plt.legend((p_d[0], p_a[0]), ("Diameter", "Average"))
    plt.title("Diameter and Average Distance")
    plt.xlabel("Simulation step")
    plt.ylabel("Distance")
    plt.savefig(_plot_img)


def plot_bank2bank_count(_g: nx.MultiDiGraph, _plot_img: str):
    acct_bank = nx.get_node_attributes(_g, "bank_id")
    bank_list = sorted(set(acct_bank.values()))
    bank2bank_all = Counter()
    bank2bank_sar = Counter()

    for orig, bene, attr in _g.edges(data=True):
        orig_bank = acct_bank[orig]
        bene_bank = acct_bank[bene]
        is_sar = attr["is_sar"]
        bank_pair = (orig_bank, bene_bank)
        bank2bank_all[bank_pair] += 1
        if is_sar:
            bank2bank_sar[bank_pair] += 1

    total_num = _g.number_of_edges()
    internal_num = sum([num for pair, num in bank2bank_all.items() if pair[0] == pair[1]])
    external_num = total_num - internal_num
    internal_ratio = internal_num / total_num * 100
    external_ratio = external_num / total_num * 100
    internal_sar_num = sum([num for pair, num in bank2bank_sar.items() if pair[0] == pair[1]])
    external_sar_num = sum([num for pair, num in bank2bank_sar.items() if pair[0] != pair[1]])

    all_count_data = list()
    sar_count_data = list()
    for orig_bank in bank_list:
        all_count_row = [bank2bank_all[(orig_bank, bene_bank)] for bene_bank in bank_list]
        all_count_total = sum(all_count_row)
        all_count_data.append(all_count_row + [all_count_total])
        sar_count_row = [bank2bank_sar[(orig_bank, bene_bank)] for bene_bank in bank_list]
        sar_count_total = sum(sar_count_row)
        sar_count_data.append(sar_count_row + [sar_count_total])

    all_count_total = list()
    sar_count_total = list()
    for bene_bank in bank_list:
        all_count_total.append(sum([bank2bank_all[(orig_bank, bene_bank)] for orig_bank in bank_list]))
        sar_count_total.append(sum([bank2bank_sar[(orig_bank, bene_bank)] for orig_bank in bank_list]))
    all_count_total.append(sum(all_count_total))
    sar_count_total.append(sum(sar_count_total))

    all_count_data.append(all_count_total)
    sar_count_data.append(sar_count_total)

    all_count_csv = list()
    sar_count_csv = list()
    for row in all_count_data:
        all_count_csv.append(["{:,}".format(num) for num in row])
    for row in sar_count_data:
        sar_count_csv.append(["{:,}".format(num) for num in row])

    cols = ["To: %s" % bank for bank in bank_list] + ["Total"]
    rows = ["From: %s" % bank for bank in bank_list] + ["Total"]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(9, 6))
    table_attr = {"rowLabels": rows, "colLabels": cols, "colWidths": [0.15 for _ in cols],
                  "loc": "center", "bbox": [0.15, 0.3, 0.75, 0.6]}
    ax1.axis("off")
    ax1.table(cellText=all_count_csv, **table_attr)
    ax1.set_title("Number of all bank-to-bank transactions")

    ax2.axis("off")
    ax2.table(cellText=sar_count_csv, **table_attr)
    ax2.set_title("Number of SAR bank-to-bank transactions")

    fig.suptitle("Internal bank transactions: Total = {:,} ({:.2f}%), SAR = {:,}".
                 format(internal_num, internal_ratio, internal_sar_num) + "\n" +
                 "External bank transactions: Total = {:,} ({:.2f}%), SAR = {:,}"
                 .format(external_num, external_ratio, external_sar_num),
                 y=0.1)
    plt.tight_layout()
    fig.savefig(_plot_img)


if __name__ == "__main__":
    argv = sys.argv

    if len(argv) < 2:
        print("Usage: python3 %s [ConfJSON]" % argv[0])
        exit(1)

    conf_json = argv[1]
    with open(conf_json, "r") as rf:
        conf = json.load(rf)

    input_dir = conf["input"]["directory"]
    schema_json = conf["input"]["schema"]
    schema_path = os.path.join(input_dir, schema_json)

    with open(schema_path, "r") as rf:
        schema = json.load(rf)

    sim_name = argv[2] if len(argv) >= 3 else conf["general"]["simulation_name"]
    work_dir = os.path.join(conf["output"]["directory"], sim_name)
    os.makedirs(work_dir, exist_ok=True)

    try:
        acct_path = _resolve_data_path(conf, sim_name, "accounts")
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)

    tx_log_path = _resolve_data_path(conf, sim_name, "transaction_log", required=False)
    tx_path = tx_log_path or _resolve_data_path(conf, sim_name, "transactions")
    if not os.path.exists(tx_path):
        print("Transaction list CSV file %s not found." % tx_path)
        exit(1)

    print("Constructing transaction graph")
    g = construct_graph(acct_path, tx_path, schema, conf["general"].get("base_date", "2017-01-01"))

    v_conf = conf["visualizer"]
    deg_plot = v_conf["degree"]
    wcc_plot = v_conf["wcc"]
    alert_plot = v_conf["alert"]
    count_plot = v_conf["count"]
    cc_plot = v_conf["clustering"]
    dia_plot = v_conf["diameter"]
    b2b_plot = "bank2bank.png"

    print("Plot degree distributions")
    plot_degree_distribution(g, conf, os.path.join(work_dir, deg_plot))

    print("Plot weakly connected component size distribution")
    plot_wcc_distribution(g, os.path.join(work_dir, wcc_plot))

    param_dir = conf["input"]["directory"]
    alert_param_file = conf["input"]["alert_patterns"]
    param_path = os.path.join(param_dir, alert_param_file)
    plot_path = os.path.join(work_dir, alert_plot)
    print("Plot AML typology count")
    plot_aml_rule(param_path, plot_path)

    alert_acct_path = _resolve_data_path(
        conf,
        sim_name,
        "alert_members",
        fallback_key="alert_members",
        required=False,
    )
    alert_tx_path = _resolve_data_path(
        conf,
        sim_name,
        "alert_transactions",
        fallback_key="alert_transactions",
        required=False,
    )

    if alert_acct_path:
        print("Plot alert attribute distributions")
        plot_alert_stat(
            alert_acct_path,
            alert_tx_path,
            schema,
            os.path.join(work_dir, "alert_dist.png"),
            conf["general"].get("base_date", "2017-01-01"),
            tx_log_csv=tx_log_path,
        )
    else:
        print("Alert member file not found. Skipping alert distribution plots.")

    print("Plot transaction count per date")
    plot_tx_count(g, os.path.join(work_dir, count_plot))

    print("Plot clustering coefficient of the transaction graph")
    plot_clustering_coefficient(g, os.path.join(work_dir, cc_plot))

    dia_path = _resolve_data_path(conf, sim_name, "diameter_log", required=False)
    if dia_path and os.path.exists(dia_path):
        plot_img = os.path.join(work_dir, dia_plot)
        print("Plot diameter of the transaction graph")
        plot_diameter(dia_path, plot_img)
    else:
        missing_label = dia_path if dia_path else "<not provided>"
        print("Diameter log file %s not found." % missing_label)

    print("Plot bank-to-bank transaction counts")
    plot_bank2bank_count(g, os.path.join(work_dir, b2b_plot))
