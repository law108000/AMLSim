## AMLSim Quick HOWTO

### Prerequisites
- Install Java 8 and download required jars into `jars/`. If using Maven, only `jars/mason.20.jar` needs to be copied and registered via `mvn install:install-file`.
- Ensure Python 3.7 is available and install dependencies:

```bash
pip3 install -r requirements.txt
```

- Duplicate and edit `conf.json` (or a template under `paramFiles/`) so the `input`, `general`, and `output` sections point to the desired parameter set, simulation settings, and output directory.

### Step 1 – Generate transactions (Python)
```bash
cd /workspaces/AMLSim
python3 scripts/transaction_graph_generator.py conf.json
```
This reads parameter CSVs (e.g., `paramFiles/1K/*`) and writes intermediate graph data.

### Step 2 – Build and run simulator (Java)
```bash
bash scripts/build_AMLSim.sh
bash scripts/run_AMLSim.sh conf.json
```
The first command compiles all Java sources (using jars or Maven). The second runs the simulator using the `general` section of `conf.json` and drops raw logs under `outputs/<simulation_name>/`.

### Step 3 – Convert logs to CSVs
```bash
python3 scripts/convert_logs.py conf.json
```
This produces `transactions.csv`, `alert_transactions.csv`, `tx_log.csv`, etc. based on the `output` paths in `conf.json`.

### Optional workflows
- Plot distributions:

```bash
python3 scripts/visualize/plot_distributions.py conf.json
```

- Validate alert subgraphs:

```bash
python3 scripts/validation/validate_alerts.py conf.json
```

- Clean generated artifacts:

```bash
sh scripts/clean_logs.sh
```

Repeat steps 1–3 for each configuration you want to simulate (e.g., different folders in `paramFiles/`).
