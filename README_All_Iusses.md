# Graph-based cyber-physical risk — Issues 1–7

This guide explains the project framework, Issues 1–7, and how to run the experiments on your datasets using 10 seeds.

The project evaluates whether graph-based intrusion detection generalizes beyond familiar hosts, topology and traffic distributions. It compares content-only and graph models, audits data splits, tests topology and feature interventions, evaluates strict temporal protocols, and measures uncertainty across random seeds.

**Your dataset paths are already set in the notebook. Keep those values.** Place this README beside that notebook, `src/`, and `experiments/`. This document does not modify your notebook or code.

## Contents

1. [Quick start](#quick-start)
2. [Framework and project structure](#framework-and-project-structure)
3. [Configuration and seeds](#configuration-and-seeds)
4. [Issues 1–7](#issues-17)
5. [Protocol and leakage rules](#protocol-and-leakage-rules)
6. [Resume, rerun and terminal execution](#resume-rerun-and-terminal-execution)

## Quick start

### 1. Open the project

Extract the complete project into a folder on the machine that will run the experiments. Open your full-data `notebook.ipynb` **inside that folder**. The notebook imports the accompanying Python modules; it cannot run as a standalone file.

Select the Python kernel from your experiment environment. The dataset paths must exist on the machine running the notebook.

### 2. Select the environment

Use the environment in which your CUDA-compatible PyTorch and PyTorch Geometric already work. The first notebook cell prints the Python executable, project folder, PyTorch version, CUDA availability and source commit.

For an environment that needs the project dependencies, run this from the project folder with that environment active:

```bash
python -m pip install -r requirements.txt
```

Preserve your existing compatible PyTorch installation. Do not force-reinstall or upgrade PyTorch merely to run this project. If your environment already has the dependencies, no installation step is needed. The requirements file is not a version-pinned environment lockfile.

### 3. Run the setup and configuration cells

Run the first environment cell, then the dataset/configuration cell. Since you have already entered the real CSV paths, leave them unchanged. Confirm that the displayed paths refer to the intended files.

The configuration cell:

- verifies that both CSV files exist and are nonempty;
- defines the seed list and training settings;
- creates or verifies the results configuration;
- writes `workflow_config.json` in the project folder.

### 4. Run the seven issues in order

Run the notebook from top to bottom: Issue 1, Issue 2, Issue 3, Issue 4, Issue 5, Issue 6, then Issue 7. Each issue calls the shared workflow and builds its reports. The following display cells show selected tables and figures.

Issue 7 depends on the saved results of Issues 3–6. Running Issue 7 alone in an empty results folder will not generate all its prerequisite comparisons.

## Framework and project structure

The notebook orchestrates experiments through `src/experiments/workflow.py`. Data loaders and split functions prepare inputs; experiment runners fit and evaluate models; reporting scripts build tables and figures from saved artifacts.

| Location | Purpose |
|---|---|
| `notebook.ipynb` | Main full-data entry point; your paths and configuration live here. |
| `src/data/` | SCADANet flow features, temporal windows, BATADAL sensor/window graphs and preprocessing. |
| `src/models/` | Graph models and non-GNN baselines. |
| `src/graph/` | Graph rewiring controls. |
| `src/evaluation/` | Splits, audits, metrics, aggregation and prediction diagnostics. |
| `src/utils/` | Seeds, result writing, source provenance, split persistence and configuration helpers. |
| `src/experiments/` | Reference, baseline, ablation, temporal, host-normalization, robustness and explanation runners. |
| `experiments/` | Audit/report builders and the final artifact validator. |
| `experiments/results/q1/` | Generated experiment outputs. |
| `requirements.txt` | Python dependencies. |
| `workflow_config.example.json` | Example settings; not a substitute for your configured notebook. |
| `workflow_config.json` | Written by the notebook configuration cell; used for terminal runs. |
| `SOURCE_PROVENANCE.json` | Source-snapshot verification when Git metadata is unavailable. |

### Dataset representations

**SCADANet:** IP addresses form graph nodes; observed source/destination pairs form message-passing edges. Individual flows are query edges carrying traffic content features. The loader uses `Source`, `Destination`, and `Attack_Type`; normal labels are converted to the binary normal/attack target. The feature schema is defined in `src/data/scadanet.py` and includes packet/flow quantities and categorical protocol fields. Preserve the dataset's expected column names and temporal fields.

**BATADAL:** sensors form graph nodes, correlation relationships form graph edges, and sliding windows form classification samples. Defaults are 24 rows per window and a stride of 6 rows, corresponding to hours only when the input is sampled hourly. The loader uses `ATT_FLAG` and maps `-999` to normal (`0`). The strict path derives graph statistics and normalization from training rows.

Historical reference functions intentionally retain their original behavior. Their scores should not be described as strict leakage-free results.

## Configuration and seeds

Use the following configuration for your 10-seed run. Confirm that the notebook’s `CONFIG["seeds"]` contains the full list below before starting; editing this README does not change the notebook settings.

| Setting | Run value | Meaning |
|---|---|---|
| `scadanet_path` | Your notebook path | SCADANet CSV location. |
| `batadal_path` | Your notebook path | BATADAL CSV location. |
| `seeds` | `[42, 7, 123, 1, 2024, 13, 21, 99, 2025, 314]` | Shared set of 10 seeds for repeated experiments. |
| `epochs` | `300` | Main SCADANet neural-training budget. |
| `batadal_epochs` | `100` | BATADAL training budget. |
| `host_epochs` | `400` | Track-B host-normalization budget; temporal host runs use `epochs`. |
| `threshold` | `"validation"` | Validation-selected threshold in baseline/configurable paths. |
| `cpu_threads` | `4` | PyTorch CPU thread setting; not parallel seed execution. |
| `explanation_samples` | `150` | Requested samples per prediction category for explanations, limited by availability. |
| `topology_examples` | `5` | Requested topology explanation examples per prediction category. |
| `resume` | `True` | Skip stages with completed-stage markers. |
| `profile` | `"full_data"` | Descriptive run label. |

The shared threshold setting does **not** override protocol-defined thresholds everywhere. Issue 5 deliberately uses fixed `0.5` for matched temporal comparisons. Issue 6 deliberately compares fixed and validation-selected thresholds.

### Running 10 seeds

Set the notebook seed configuration to:

```python
"seeds": [42, 7, 123, 1, 2024, 13, 21, 99, 2025, 314],
```

Run the configuration cell again and use the same seed list for Issues 3–7. Issue 7 requires matching prerequisite results for all 10 seeds. Issue 1 uses the first configured seed for its four historical references; Issue 2 audits split structure rather than training every model for every seed.

If an existing run uses a different seed list, the workflow treats this as a new configuration. Archive the existing results and start a fresh run. It does not automatically extend an existing configuration while retaining its completion markers.

### Hardware and runtime

Seeds and stages run sequentially. Neural baseline paths detect CUDA, but historical/frozen temporal and host routines retain CPU execution paths. A notebook printout of `CUDA: True` does not mean every experiment runs on GPU.

Full runs include multiple model families, validation searches and explanation fits, so runtime depends heavily on dataset size and hardware. There is no verified full-data runtime estimate for your machine. The workflow suppresses many epoch logs; a quiet cell can still be training.

## Issues 1–7

| Local issue | Tracker reference | Main objective |
|---|---|---|
| 1 | #22 | Reusable, reproducible experiment pipeline. |
| 2 | #23 | Split, coverage, shortcut and leakage audits. |
| 3 | #24; also called #3 in the supplied specification | Fair non-GNN/GNN baselines. |
| 4 | #25 | Topology-versus-content counterfactuals. |
| 5 | #26 | Strict temporal evaluation. |
| 6 | #27 | Host normalization and threshold calibration. |
| 7 | #28 | Multi-seed statistics, operational diagnostics and explainability. |

### Issue 1 — reusable pipeline and historical references

**Code:** `src/experiments/runner.py`, `src/utils/io.py`, `src/utils/reproducibility.py`, and shared data/model/evaluation modules.

Runs four inherited experiments: SCADANet Track B, SCADANet historical temporal, BATADAL historical temporal, and BATADAL static cross-validation. These establish reference behavior and provenance; they are not the primary strict temporal evidence.

Split functions save exact split outputs with input fingerprints. JSON results record the source commit, source hash, split-manifest references and environment metadata. 

### Issue 2 — split and shortcut audit

**Code:** `src/evaluation/audit.py`, `experiments/run_split_audits.py`, `experiments/build_table_a2_and_figures.py`.

Audits attack coverage, host and pair overlap/novelty, source and destination diversity, numeric-feature drift, class balance and temporal overlap. Normal traffic is excluded from attack-type coverage; coverage counts are computed from the dataset rather than hardcoded.

Includes SCADANet Track A, Track B and temporal analyses, strict/fixed topology checks, all five BATADAL static-CV folds, and the three purged BATADAL boundaries. Audit warnings are diagnostics requiring interpretation, not automatic proof that a model succeeds or fails.

### Issue 3 — fair baselines

**Code:** `src/experiments/baseline_runner.py`, `src/models/baselines.py`, `experiments/build_table_b1_and_figures.py`.

| Experiment | Dataset/protocol | Families |
|---|---|---|
| B1 | SCADANet Track B | Logistic regression, tree, MLP, GraphSAGE, GCN, GAT. |
| B2 | SCADANet strict temporal | Tree, MLP, GraphSAGE and the prespecified alternative GNN, GCN by default. |
| B3 | BATADAL temporal | MLP, GraphSAGE and the prespecified alternative GNN. |

Models share the relevant split and content-feature inputs. Enhanced preprocessing fits on training/fitting rows. Hyperparameters and thresholds use validation data rather than test scores. GCN is a prespecified alternative; do not select the alternative after inspecting test performance.

Saves per-model/per-seed results, per-attack recall, discrimination/calibration metrics where supported, computational diagnostics and `baseline_configs.json`. Parameter or storage fields may be unavailable for some backends; peak-memory measurement is not a guaranteed output.

### Issue 4 — topology/content ablations

**Code:** `src/experiments/ablation_runner.py`, `src/graph/rewiring.py`, `experiments/build_table_c1_and_figures.py`.

| Condition | Intervention |
|---|---|
| A | Content-only MLP, without graph message passing. |
| B | Topology-only model, without flow content features. |
| C | Full model with real graph and content; comparison reference. |
| D | Randomized topology with the same sample/content setup. |
| E | Degree-preserving rewiring. |
| F | No-message-passing control. |

Runs paired conditions and feature interventions across seeds. Rewiring diagnostics include node/edge counts, degree statistics and connected components. Reporting includes paired metric changes, uncertainty and all-seed per-attack summaries.

**Temporal qualification:** D/E use fixed training-boundary rewiring controls; B/C use cumulative causal topology. Do not describe all six temporal conditions as identical dynamic-topology interventions. Track B is the principal A–F topology comparison.

### Issue 5 — strict temporal evaluation

**Code:** `src/experiments/temporal_runner.py`, `experiments/build_table_t_and_figures.py`.

| Protocol | Topology supplied at test time | Model |
|---|---|---|
| T1 — previous temporal | Final topology may expose future test edges. | Frozen. |
| T2 — strict causal cumulative | Only topology observed before the current test window. | Same frozen model. |
| T3 — fixed training boundary | Training-boundary topology throughout testing. | Same frozen model. |

T1/T2/T3 use the same model and fixed threshold within a seed. T2 predicts each window before adding that window's observed topology. Independent audit artifacts check for unobserved edges.

Table T2 and Figure T4 compare strict temporal recall with a **separately trained matched Track-B reference**. This reference matches architecture, epochs, learning rate, binary loss-weighting rule and fixed `0.5` threshold. It is distinct from the validation-tuned B1 model. Because Track B and temporal splits differ, this is a generalization comparison, not an isolated topology-only intervention.

`Delta recall = strict temporal recall − matched Track-B recall`.

BATADAL retains the overlapping historical reference and evaluates purged 60/40, 70/30 and 80/20 chronological boundaries using the same purge rule and fixed threshold. The boundary audits record purged-window counts and zero train/test raw-row overlap.

### Issue 6 — host normalization

**Code:** `src/experiments/host_norm_runner.py`, `experiments/build_table_h_and_figures.py`.

Runs the following conditions on both Track B and strict temporal data:

| Condition | Features | Threshold |
|---|---|---|
| H1 | Original/global preprocessing | Fixed `0.5`. |
| H2 | Original/global preprocessing | Validation-selected. |
| H3 | Host-normalized | Fixed `0.5`. |
| H4 | Host-normalized | Validation-selected. |

This separates the effect of feature normalization from threshold calibration. Host statistics use fitting data with a fallback for hosts lacking suitable training statistics. Seed archives support aggregate uncertainty, host-specific false positives and per-attack recall changes.

Preserve recall losses and temporal FPR increases wherever observed; normalization is not assumed to improve every attack or host.

### Issue 7 — statistics, operational metrics and explanations

**Code:** `src/experiments/robustness_runner.py`, `src/experiments/explainability_runner.py`, `experiments/build_table_s_and_figures.py`.

Collects saved principal comparisons from Issues 3–6, then computes multi-seed summaries and paired statistics. These principal comparisons are not all retrained by the unified Issue 7 workflow. Operational diagnostics and explanation stages do perform their own explicit fits.

Includes mean, sample standard deviation, median, confidence intervals, paired bootstrap comparisons and effect sizes. Operational diagnostics cover calibration and applicable inference/model-size measures. False alarms per hour, event recall and time-to-detect are reported only when the data support the required exposure or event definitions; otherwise they remain N/A with reasons.

Explanation sampling is stratified over TP, FP, FN and TN, with host/attack diversity within categories. Reports feature attribution rankings, cross-seed rank/top-feature stability and feature-intervention uncertainty. Missing prediction categories are not fabricated.

## Protocol and leakage rules

1. **Fit preprocessing on fitting/training data.** Enhanced/strict paths restrict imputation/scaling and categorical vocabulary; BATADAL strict graph statistics and sensor ranges use training rows.
2. **Keep the strict SCADANet model frozen.** Earlier test-window topology may accumulate, but test labels do not update model weights.
3. **Predict before adding the current window's edges.** The causal audit must pass for every tested window.
4. **Purge overlapping BATADAL windows.** Train and test window supports must not share raw rows. Preserve the boundary audit and purge counts for every split fraction.
5. **Keep comparison settings matched.** Pair seeds and respect each experiment's model/threshold definition. Do not substitute a tuned B1 row for the matched Issue 5 reference.
6. **Report attack-level failures.** Retain `vuln_scan`, `modbus_fdi`, `modbus_abuse` and `insider_threat` results when present, including low/zero recall. An absent attack is not equivalent to zero recall.

SCADANet causality is defined by ordered windows. Equal timestamps can span windows; the implementation does not guarantee strict timestamp inequality for every row. Future node IDs can exist as isolated constant-feature placeholders, while future pair edges are excluded from T2. State this scope when describing the no-leakage condition.

## Resume, rerun and terminal execution

### Resume after a disconnect

Reopen the same project, select the same environment, run setup/configuration with unchanged values, then rerun the issue cells. With `resume=True`, completed stages are skipped. A partially completed stage can rerun; this is stage-level resumption, **not restoration of an interrupted epoch or optimizer state**.

Do not delete completed markers while expecting already completed stages to be skipped. Do not keep markers while deleting the result files they represent.

### Change settings or datasets

The results directory is bound to the configuration, source and dataset hashes. A change to seeds, epochs, paths, source/commit or other stored configuration values can trigger a mismatch, even if the change seems minor.

Before starting a different experiment, archive or rename the entire `experiments/results/` folder to a unique name outside the active results location. Then run the new configuration. Preserve the old results for comparison; do not edit `run_config.json` to bypass the guard.

### Run from a terminal

After the notebook configuration cell has written your paths into `workflow_config.json`, the same workflow can run from a terminal:

```bash
python -m src.experiments.workflow --config workflow_config.json --issue all
```

Or run one issue:

```bash
python -m src.experiments.workflow --config workflow_config.json --issue 5
```

Use issue numbers 1–7. Keep the terminal working directory at the project root. Do not run the notebook and terminal workflow concurrently against the same results directory.
