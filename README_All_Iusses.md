# Graph-Based Cyber-Physical Risk — Issues 1–7

This guide describes the project framework, Issues 1–7, and the procedure for running the experiments on the configured datasets using 10 random seeds.

The project evaluates whether graph-based intrusion detection generalizes beyond familiar hosts, network topology, and traffic distributions. It compares content-only and graph-based models, audits data splits, evaluates topology and feature interventions, tests strict temporal protocols, and measures uncertainty across multiple random seeds.

The dataset paths are configured in the main notebook and should remain unchanged. This README should be placed alongside the notebook, `src/`, and `experiments/` directories. The README does not modify the notebook or source code.

## Contents

1. [Quick Start](#quick-start)
2. [Framework and Project Structure](#framework-and-project-structure)
3. [Configuration and Seeds](#configuration-and-seeds)
4. [Issues 1–7](#issues-17)
5. [Protocol and Leakage Rules](#protocol-and-leakage-rules)
6. [Resume, Rerun, and Terminal Execution](#resume-rerun-and-terminal-execution)

---

## Quick Start

### 1. Open the Project

Extract the complete project into a folder on the machine used for running the experiments.

Open the full-data `notebook.ipynb` **inside the project folder**. The notebook imports the accompanying Python modules and is not intended to run as a standalone file.

Select the Python kernel associated with the experiment environment. The configured dataset paths must exist on the machine running the notebook.

### 2. Select the Environment

Use an environment in which the compatible CUDA-enabled PyTorch and PyTorch Geometric installations are already available.

The first notebook cell reports:

* Python executable
* Project directory
* PyTorch version
* CUDA availability
* Source commit

If the required project dependencies are not already installed, activate the experiment environment and run the following command from the project root:

```bash
python -m pip install -r requirements.txt
```

The existing compatible PyTorch installation should be preserved. PyTorch should not be force-reinstalled or upgraded solely to run this project.

The `requirements.txt` file is a dependency specification and is not a fully version-pinned environment lockfile.

### 3. Run the Setup and Configuration Cells

Run the environment setup cell followed by the dataset and configuration cell.

The configured dataset paths should remain unchanged. Confirm that the displayed paths point to the intended CSV files.

The configuration cell:

* verifies that both CSV files exist and are non-empty;
* defines the random seed list and training settings;
* creates or verifies the results configuration;
* writes `workflow_config.json` to the project root.

### 4. Run Issues 1–7 in Order

Run the notebook from top to bottom in the following order:

**Issue 1 → Issue 2 → Issue 3 → Issue 4 → Issue 5 → Issue 6 → Issue 7**

Each issue uses the shared workflow and generates its corresponding reports and saved artifacts. The display cells present selected tables and figures.

Issue 7 depends on the saved results from Issues 3–6. Running Issue 7 in an empty results directory will not generate all of its prerequisite comparisons.

---

# Framework and Project Structure

The notebook orchestrates the experiments through:

```text
src/experiments/workflow.py
```

Data loaders and split functions prepare the datasets. Experiment runners train and evaluate the models, while reporting scripts generate tables and figures from the saved artifacts.

| Location                       | Purpose                                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------------------- |
| `notebook.ipynb`               | Main full-data entry point containing dataset paths and configuration.                            |
| `src/data/`                    | SCADANet flow features, temporal windows, BATADAL sensor/window graphs, and preprocessing.        |
| `src/models/`                  | Graph-based models and non-GNN baselines.                                                         |
| `src/graph/`                   | Graph construction and rewiring controls.                                                         |
| `src/evaluation/`              | Splits, audits, metrics, aggregation, and prediction diagnostics.                                 |
| `src/utils/`                   | Seeds, result writing, source provenance, split persistence, and configuration helpers.           |
| `src/experiments/`             | Reference, baseline, ablation, temporal, host-normalization, robustness, and explanation runners. |
| `experiments/`                 | Audit and report builders and the final artifact validator.                                       |
| `experiments/results/q1/`      | Generated experiment outputs.                                                                     |
| `requirements.txt`             | Python dependency specification.                                                                  |
| `workflow_config.example.json` | Example configuration settings.                                                                   |
| `workflow_config.json`         | Configuration generated by the notebook and used for terminal runs.                               |
| `SOURCE_PROVENANCE.json`       | Source-snapshot verification when Git metadata is unavailable.                                    |

## Dataset Representations

### SCADANet

IP addresses form graph nodes, while observed source/destination pairs form message-passing edges. Individual flows are treated as query edges carrying traffic-content features.

The loader uses:

* `Source`
* `Destination`
* `Attack_Type`

Normal labels are converted to the binary normal/attack target.

The feature schema is defined in:

```text
src/data/scadanet.py
```

It includes packet/flow quantities and categorical protocol fields. The expected column names and temporal fields of the dataset must be preserved.

### BATADAL

Sensors form graph nodes, correlation relationships form graph edges, and sliding windows form classification samples.

The default configuration uses:

* 24 rows per window
* 6-row stride

These correspond to 24-hour windows and 6-hour strides only when the input data is sampled hourly.

The loader uses `ATT_FLAG` and maps `-999` to the normal class (`0`).

The strict evaluation path derives graph statistics and normalization parameters from training rows.

Historical reference functions intentionally retain their original behavior. Their results should therefore not be described as strict leakage-free results.

---

# Configuration and Seeds

The following configuration is used for the 10-seed experimental run.

The notebook configuration must contain the complete seed list before starting the experiments.

| Setting               | Run Value                                      | Meaning                                                                              |
| --------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------ |
| `scadanet_path`       | Configured notebook path                       | SCADANet CSV location.                                                               |
| `batadal_path`        | Configured notebook path                       | BATADAL CSV location.                                                                |
| `seeds`               | `[42, 7, 123, 1, 2024, 13, 21, 99, 2025, 314]` | Shared set of 10 random seeds.                                                       |
| `epochs`              | `300`                                          | Main SCADANet neural-training budget.                                                |
| `batadal_epochs`      | `100`                                          | BATADAL training budget.                                                             |
| `host_epochs`         | `400`                                          | Track-B host-normalization training budget.                                          |
| `threshold`           | `"validation"`                                 | Validation-selected threshold for configurable baseline paths.                       |
| `cpu_threads`         | `4`                                            | PyTorch CPU thread setting.                                                          |
| `explanation_samples` | `150`                                          | Requested samples per prediction category for explanations, subject to availability. |
| `topology_examples`   | `5`                                            | Requested topology explanation examples per prediction category.                     |
| `resume`              | `True`                                         | Skip stages with completed-stage markers.                                            |
| `profile`             | `"full_data"`                                  | Descriptive run label.                                                               |

The shared threshold configuration does not override protocol-specific thresholds.

Issue 5 uses a fixed threshold of `0.5` for matched temporal comparisons.

Issue 6 explicitly compares fixed and validation-selected thresholds.

## Running 10 Seeds

The seed configuration should be:

```python
"seeds": [42, 7, 123, 1, 2024, 13, 21, 99, 2025, 314],
```

After setting the seed list, rerun the configuration cell and use the same seed list for Issues 3–7.

Issue 7 requires matching prerequisite results for all 10 seeds.

Issue 1 uses the first configured seed for its four historical reference experiments. Issue 2 audits split structure and does not train every model for every seed.

If an existing run was generated with a different seed list, the workflow treats it as a different configuration. The existing results should be archived before starting the new configuration.

## Hardware and Runtime

Seeds and experiment stages are executed sequentially.

Neural baseline paths detect CUDA when available. However, historical/frozen temporal and host-normalization routines may retain CPU execution paths. Therefore, `CUDA: True` in the notebook does not imply that every experiment runs on the GPU.

Full runs include multiple model families, validation searches, and explanation stages. Runtime depends on dataset size, hardware, and the selected experiment configuration.

No verified full-data runtime estimate is provided.

Many training stages suppress detailed epoch output. A quiet notebook cell may still be actively training.

---

# Issues 1–7

| Local Issue | Tracker Reference                                         | Main Objective                                                      |
| ----------- | --------------------------------------------------------- | ------------------------------------------------------------------- |
| 1           | #22                                                       | Reusable and reproducible experiment pipeline.                      |
| 2           | #23                                                       | Split, coverage, shortcut, and leakage audits.                      |
| 3           | #24; also referred to as #3 in the supplied specification | Fair non-GNN/GNN baselines.                                         |
| 4           | #25                                                       | Topology-versus-content counterfactuals.                            |
| 5           | #26                                                       | Strict temporal evaluation.                                         |
| 6           | #27                                                       | Host normalization and threshold calibration.                       |
| 7           | #28                                                       | Multi-seed statistics, operational diagnostics, and explainability. |

---

## Issue 1 — Reusable Pipeline and Historical References

### Code

```text
src/experiments/runner.py
src/utils/io.py
src/utils/reproducibility.py
```

Issue 1 runs four inherited experiments:

1. SCADANet Track B
2. SCADANet historical temporal
3. BATADAL historical temporal
4. BATADAL static cross-validation

These experiments establish reference behavior and provenance. They are not the primary source of strict temporal evidence.

Split functions save exact split outputs together with input fingerprints.

JSON result files record:

* source commit;
* source hash;
* split-manifest references;
* environment metadata.

---

## Issue 2 — Split and Shortcut Audit

### Code

```text
src/evaluation/audit.py
experiments/run_split_audits.py
experiments/build_table_a2_and_figures.py
```

Issue 2 audits:

* attack coverage;
* host overlap and novelty;
* pair overlap and novelty;
* source and destination diversity;
* numeric-feature drift;
* class balance;
* temporal overlap.

Normal traffic is excluded from attack-type coverage. Coverage counts are computed directly from the dataset rather than hardcoded.

The audit includes:

* SCADANet Track A;
* SCADANet Track B;
* SCADANet temporal analysis;
* strict/fixed topology checks;
* all five BATADAL static-CV folds;
* three purged BATADAL chronological boundaries.

Audit warnings are diagnostic outputs requiring interpretation. They do not automatically prove that a model succeeds or fails.

---

## Issue 3 — Fair Baselines

### Code

```text
src/experiments/baseline_runner.py
src/models/baselines.py
experiments/build_table_b1_and_figures.py
```

| Experiment | Dataset / Protocol       | Model Families                                         |
| ---------- | ------------------------ | ------------------------------------------------------ |
| B1         | SCADANet Track B         | Logistic Regression, Tree, MLP, GraphSAGE, GCN, GAT    |
| B2         | SCADANet strict temporal | Tree, MLP, GraphSAGE, and prespecified alternative GNN |
| B3         | BATADAL temporal         | MLP, GraphSAGE, and prespecified alternative GNN       |

Models use the relevant split and content-feature inputs.

Enhanced preprocessing is fitted only on the appropriate training/fitting rows.

Hyperparameters and thresholds are selected using validation data rather than test scores.

GCN is the prespecified alternative GNN and should not be selected based on test performance.

The workflow saves:

* per-model/per-seed results;
* per-attack recall;
* discrimination and calibration metrics where supported;
* computational diagnostics;
* `baseline_configs.json`.

Some backends may not provide parameter counts or storage fields. Peak-memory measurement is also not guaranteed for every backend.

---

## Issue 4 — Topology/Content Ablations

### Code

```text
src/experiments/ablation_runner.py
src/graph/rewiring.py
experiments/build_table_c1_and_figures.py
```

| Condition | Intervention                                            |
| --------- | ------------------------------------------------------- |
| A         | Content-only MLP without graph message passing.         |
| B         | Topology-only model without flow content features.      |
| C         | Full model using the real graph and content features.   |
| D         | Randomized topology with the same sample/content setup. |
| E         | Degree-preserving rewiring.                             |
| F         | No-message-passing control.                             |

The paired conditions and feature interventions are evaluated across the configured seeds.

Rewiring diagnostics include:

* node counts;
* edge counts;
* degree statistics;
* connected components.

Reporting includes paired metric changes, uncertainty, and all-seed per-attack summaries.

### Temporal Qualification

Conditions D and E use fixed training-boundary rewiring controls.

Conditions B and C use cumulative causal topology.

Therefore, all six temporal conditions should not be described as identical dynamic-topology interventions.

Track B is the principal A–F topology comparison.

---

## Issue 5 — Strict Temporal Evaluation

### Code

```text
src/experiments/temporal_runner.py
experiments/build_table_t_and_figures.py
```

| Protocol                      | Topology Supplied at Test Time                         | Model             |
| ----------------------------- | ------------------------------------------------------ | ----------------- |
| T1 — Previous Temporal        | Final topology may expose future test edges.           | Frozen            |
| T2 — Strict Causal Cumulative | Only topology observed before the current test window. | Same frozen model |
| T3 — Fixed Training Boundary  | Training-boundary topology throughout testing.         | Same frozen model |

T1, T2, and T3 use the same model and fixed threshold within each seed.

T2 predicts each test window before adding that window's observed topology.

Independent audit artifacts check for unobserved edges.

Table T2 and Figure T4 compare strict temporal recall with a separately trained matched Track-B reference.

The matched reference uses:

* the same architecture;
* the same training epochs;
* the same learning rate;
* the same binary loss-weighting rule;
* a fixed `0.5` threshold.

It is distinct from the validation-tuned B1 model.

Because Track B and the temporal protocols use different splits, this comparison represents a generalization comparison rather than an isolated topology-only intervention.

The reported difference is:

```text
Delta recall = strict temporal recall − matched Track-B recall
```

### BATADAL Temporal Evaluation

BATADAL retains the overlapping historical reference and additionally evaluates purged:

* 60/40 chronological split;
* 70/30 chronological split;
* 80/20 chronological split.

The same purge rule and fixed threshold are used.

Boundary audits record:

* purged-window counts;
* train/test raw-row overlap.

---

## Issue 6 — Host Normalization

### Code

```text
src/experiments/host_norm_runner.py
experiments/build_table_h_and_figures.py
```

The experiment evaluates the following conditions on both Track B and strict temporal data:

| Condition | Features                      | Threshold           |
| --------- | ----------------------------- | ------------------- |
| H1        | Original/global preprocessing | Fixed `0.5`         |
| H2        | Original/global preprocessing | Validation-selected |
| H3        | Host-normalized               | Fixed `0.5`         |
| H4        | Host-normalized               | Validation-selected |

This separates the effect of feature normalization from threshold calibration.

Host statistics are calculated from fitting data, with fallback behavior for hosts that do not have suitable training statistics.

Seed archives support:

* aggregate uncertainty;
* host-specific false positives;
* per-attack recall changes.

Recall losses and temporal FPR increases are retained wherever observed. Host normalization is not assumed to improve every attack or host.

---

## Issue 7 — Statistics, Operational Metrics, and Explanations

### Code

```text
src/experiments/robustness_runner.py
src/experiments/explainability_runner.py
experiments/build_table_s_and_figures.py
```

Issue 7 collects saved principal comparisons from Issues 3–6 and computes multi-seed summaries and paired statistics.

The principal comparisons are not all retrained by the unified Issue 7 workflow. Operational diagnostics and explanation stages perform their own explicit fits where required.

The analysis includes:

* mean;
* sample standard deviation;
* median;
* confidence intervals;
* paired bootstrap comparisons;
* effect sizes.

Operational diagnostics cover calibration and applicable inference/model-size measures.

False alarms per hour, event recall, and time-to-detect are reported only when the available data support the required exposure or event definitions. Otherwise, the metric remains `N/A` with an explanation.

### Explanation Analysis

Explanation sampling is stratified across:

* True Positive (TP);
* False Positive (FP);
* False Negative (FN);
* True Negative (TN).

Sampling also considers host and attack diversity within each category.

The reports include:

* feature attribution rankings;
* cross-seed rank/top-feature stability;
* feature-intervention uncertainty.

Missing prediction categories are not fabricated.

---

# Protocol and Leakage Rules

The following rules apply throughout the experimental workflow.

### 1. Fit Preprocessing on Training/Fitting Data

Enhanced and strict paths restrict:

* imputation;
* scaling;
* categorical vocabulary.

BATADAL strict graph statistics and sensor ranges are derived from training rows.

### 2. Keep the Strict SCADANet Model Frozen

Earlier test-window topology may accumulate, but test labels must never update model weights.

### 3. Predict Before Adding the Current Window's Edges

For the causal temporal protocol, each window must be evaluated before its observed edges are added to the topology.

The causal audit must pass for every tested window.

### 4. Purge Overlapping BATADAL Windows

Training and testing window supports must not share raw rows.

The boundary audit and purge counts must be preserved for every split fraction.

### 5. Keep Comparison Settings Matched

Seeds must be paired across comparisons.

Each experiment must follow its defined model and threshold settings.

A validation-tuned B1 result must not be substituted for the matched Issue 5 reference.

### 6. Report Attack-Level Failures

Attack-level results must retain:

* `vuln_scan`;
* `modbus_fdi`;
* `modbus_abuse`;
* `insider_threat`.

Low or zero recall should be reported where observed.

An attack that is absent from a split is not equivalent to an attack with zero recall.

### SCADANet Causality Scope

SCADANet causality is defined using ordered windows.

Equal timestamps may span multiple windows, so the implementation does not guarantee strict timestamp inequality for every individual row.

Future node IDs may exist as isolated constant-feature placeholders, while future pair edges are excluded from T2.

This scope should be explicitly stated when describing the no-leakage condition.

---

# Resume, Rerun, and Terminal Execution

## Resume After a Disconnect

Reopen the same project and select the same experiment environment.

Run the setup and configuration cells with unchanged values, then rerun the relevant issue cells.

With:

```python
resume = True
```

completed stages are skipped.

A partially completed stage may run again. This is **stage-level resumption**, not restoration of an interrupted training epoch or optimizer state.

Do not delete completed-stage markers while expecting completed stages to be skipped.

Likewise, do not keep completion markers after deleting the result files they represent.

## Change Settings or Datasets

The results directory is bound to the configuration, source, and dataset hashes.

Changes to the following may trigger a configuration mismatch:

* seed list;
* training epochs;
* dataset paths;
* source code;
* commit;
* other stored configuration values.

Before starting a different experiment configuration, archive or rename the entire:

```text
experiments/results/
```

directory to a unique location outside the active results directory.

Then start the new configuration.

Existing results should be preserved for comparison.

Do not edit `run_config.json` to bypass the configuration guard.

## Run from a Terminal

After the notebook configuration cell has generated:

```text
workflow_config.json
```

the same workflow can be executed from the project root using:

```bash
python -m src.experiments.workflow --config workflow_config.json --issue all
```

To run a specific issue:

```bash
python -m src.experiments.workflow --config workflow_config.json --issue 5
```

Valid issue numbers are:

```text
1 2 3 4 5 6 7
```

The terminal working directory must remain at the project root.

The notebook and terminal workflow must not be executed concurrently against the same results directory.
