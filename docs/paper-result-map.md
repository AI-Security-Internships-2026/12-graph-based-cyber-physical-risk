# Paper Result Map

This document maps the tables, figures, and supporting numerical claims in the final manuscript to the committed result artifacts and generation scripts.

Result paths are relative to:

`experiments/results/q1/`

Experiment runners are under:

`src/experiments/`

Table and figure generation scripts are under:

`experiments/`

Unless otherwise stated, reported aggregate results use the ten seeds:

`42, 7, 123, 1, 2024, 13, 21, 99, 2025, 314`

---

## Tables

| Manuscript item | Content | Final source artifact(s) | Generation / source script |
|---|---|---|---|
| **Table 1 (`tab:datasets`)** | Dataset and graph summary | `split_audits/table_a2_scadanet_coverage.csv`, `temporal/scadanet_graph_growth.csv`, `ablations/rewiring_metadata.json`, `run_config.json` | `experiments/build_table_a2_and_figures.py`, `src/experiments/temporal_runner.py`, `src/experiments/ablation_runner.py` |
| **Table 2 (`tab:configs`)** | GraphSAGE strict-temporal configurations | `temporal/scadanet_config_seed*.json`, `ablations/c1_temporal/*C_full_model*_seed*.json`, `baselines/scadanet_temporal_baselines.csv`, `host_normalization/host_normalization_config.json` | `src/experiments/temporal_runner.py`, `src/experiments/ablation_runner.py`, `src/experiments/baseline_runner.py`, `src/experiments/host_norm_runner.py` |
| **Table 3 (`tab:audit`)** | Split and leakage audit | `split_audits/table_a1_protocol_audit.csv`, `split_audits/*_audit.json` | `experiments/run_split_audits.py` |
| **Table 4 (`tab:strict`)** | Track B vs strict temporal; BATADAL static vs purged | `baselines/table_b1_main_comparison.csv`, `baselines/table_b2_temporal_comparison.csv`, `temporal/table_t1_old_vs_strict.csv`, `temporal/table_t3_batadal_split_sensitivity.csv`, `references/batadal_static_cv_seed42.json` | `experiments/build_table_b1_and_figures.py`, `experiments/build_table_t_and_figures.py` |
| **Table 5 (`tab:baselines`)** | Model baseline comparison | `baselines/scadanet_trackb_baselines.csv`, `baselines/scadanet_temporal_baselines.csv`, `baselines/batadal_temporal_baselines.csv`, `baselines/baseline_configs.json` | `src/experiments/baseline_runner.py`, `experiments/build_table_b1_and_figures.py` |
| **Table 6 (`tab:ablation`)** | Topology and content counterfactuals | `ablations/table_c1_trackb.csv`, `ablations/table_c1_temporal.csv`, `ablations/c1_trackb/*.json`, `ablations/c1_temporal/*.json` | `src/experiments/ablation_runner.py`, `experiments/build_table_c1_and_figures.py` |
| **Table 7 (`tab:interventions`)** | Track B feature interventions | `ablations/table_c2_feature_interventions.csv` | `src/experiments/ablation_runner.py`, `experiments/build_table_c1_and_figures.py` |
| **Table 8 (`tab:pertype`)** | Strict-temporal recall by attack class | `temporal/table_t2_scadanet_per_attack.csv`, `baselines/b2/*_seed*.json`, `split_audits/table_a2_scadanet_coverage.csv` | `src/experiments/temporal_runner.py`, `src/experiments/baseline_runner.py`, `experiments/build_table_t_and_figures.py` |
| **Table 9 (`tab:mitigation`)** | Host normalization and threshold conditions H1-H4 | `host_normalization/table_h1_mitigation_comparison.csv`, `host_normalization/table_h1_temporal_mitigation_comparison.csv`, `host_normalization/table_h2_fp_concentration_by_host.csv` | `src/experiments/host_norm_runner.py`, `experiments/build_table_h_and_figures.py` |
| **Table 10 (`tab:cost`)** | Model parameters, training time and inference time | `baselines/table_b3_computational_cost.csv` | `src/experiments/baseline_runner.py`, `experiments/build_table_b1_and_figures.py` |

---

## Figures

| Manuscript item | Content | Final source artifact | Generation script |
|---|---|---|---|
| **Figure 1 — left** | Track B topology/content counterfactuals | `ablations/figure_c1_trackb.png` | `experiments/build_table_c1_and_figures.py` |
| **Figure 1 — right** | Strict-temporal topology/content counterfactuals | `ablations/figure_c1_temporal.png` | `experiments/build_table_c1_and_figures.py` |
| **Figure 2** | Evaluation-audit workflow | Drawn directly in the manuscript | Manuscript source |
| **Figure 3 — left** | SCADANet attack-class coverage by split | `split_audits/figure_a1_coverage_heatmap.png` | `experiments/build_table_a2_and_figures.py` |
| **Figure 3 — right** | Static vs strict temporal per-attack recall | `temporal/figure_t4_static_vs_strict_heatmap.png` | `experiments/build_table_t_and_figures.py` |
| **Figure 4 — left** | False positives by source host under H1-H4 | `host_normalization/figure_h1_fp_by_host.png` | `experiments/build_table_h_and_figures.py` |
| **Figure 4 — right** | Track B F1 distribution across ten seeds | `robustness/figure_s1_seed_distribution.png` | `experiments/build_table_s_and_figures.py` |

---

## Supporting Numerical Claims

| Manuscript value / claim | Final source artifact(s) | Generation / analysis script |
|---|---|---|
| Paired statistics and effect sizes | `robustness/table_s2_statistical_comparison.csv` | `src/experiments/robustness_runner.py`, `experiments/build_table_s_and_figures.py` |
| ECE and Brier scores | `robustness/table_s3_operational_metrics.csv` | `src/experiments/robustness_runner.py`, `experiments/build_table_s_and_figures.py` |
| Integrated Gradients rankings and cross-seed stability | `robustness/table_s4_explainability_stability.csv`, `robustness/explainability_summary.csv` | `src/experiments/explainability_runner.py`, `experiments/build_table_s_and_figures.py` |
| F1 variation across seeds | `robustness/table_s1_robustness_summary.csv`, `baselines/scadanet_trackb_baselines.csv` | `src/experiments/robustness_runner.py`, `experiments/build_table_s_and_figures.py` |
| `vuln_scan` test count and 45.3% share of test attack flows | `temporal/table_t2_scadanet_per_attack.csv` | `src/experiments/temporal_runner.py`, `experiments/build_table_t_and_figures.py` |
| GCN strict-temporal per-seed results | `baselines/b2/*gcn*_seed*.json` | `src/experiments/baseline_runner.py` |
| Topology-only strict-temporal per-seed results | `ablations/c1_temporal/*B_topology_only*_seed*.json` | `src/experiments/ablation_runner.py` |
| `modbus_abuse` per-seed recall | `baselines/b2/*_seed*.json`, `ablations/c1_temporal/*.json` | `src/experiments/baseline_runner.py`, `src/experiments/ablation_runner.py` |
| `os_scan` recall under H1-H4 | `host_normalization/trackb_per_attack_recall.csv`, `host_normalization/trackb_host_normalization_summary.csv` | `src/experiments/host_norm_runner.py`, `experiments/build_table_h_and_figures.py` |
| False positives for hosts `.140` and `.141` | `host_normalization/trackb_false_positives_by_host.csv` | `src/experiments/host_norm_runner.py`, `experiments/build_table_h_and_figures.py` |
| H1-H4 top-two-host FP share | Derived from `host_normalization/trackb_false_positives_by_host.csv` and total FP values | `experiments/build_table_h_and_figures.py` |
| BATADAL purged 60/40, 70/30 and 80/20 results | `temporal/table_t3_batadal_split_sensitivity.csv` | `src/experiments/temporal_runner.py`, `experiments/build_table_t_and_figures.py` |
| BATADAL temporal aggregate results | `temporal/batadal_purged_temporal_metrics.csv` | `src/experiments/temporal_runner.py` |
| Track B and strict-temporal edge-attribute dimensions | `baselines/b1/*.json`, `baselines/b2/*.json` (`notes.edge_attr_dim`) | `src/experiments/baseline_runner.py` |
| Tree-model backend | `baselines/b1/*tree_xgboost*_seed*.json`, `baselines/b2/*tree_xgboost*_seed*.json` | `src/models/baselines.py`, `src/experiments/baseline_runner.py` |
| Software/library versions | Per-seed JSON `library_versions`; `run_config.json` | `src/experiments/workflow.py` |
| Git commit, source fingerprint, dataset hashes and seed configuration | `run_config.json` and per-seed JSON metadata | `src/experiments/workflow.py` |
| Result-validation checks | `validation_report.json` | `experiments/validate_results.py` |

---

## Dataset and Preprocessing Evidence

Some descriptive dataset facts used in the manuscript originate from the experimental notebooks rather than aggregated result tables.

| Manuscript claim | Source |
|---|---|
| SCADANet contains 534,841 flows and 60 raw columns | `experiments/notebooks/Week6-8.ipynb` |
| SCADANet attack-flow count and feature/column audit | `experiments/notebooks/Week8.ipynb` |
| BATADAL contains 4,177 rows and 43 sensor/actuator variables | `experiments/notebooks/Week5-8.ipynb` |
| ICS-Flow exploratory knowledge-graph counts | Week 5, 7, 8, 10 and 12 experimental notebooks |
| Track A reference results and attack coverage | `experiments/notebooks/Week8.ipynb` together with `split_audits/scadanet_track_a_audit.json` |
| Temporal-bin construction observations | `experiments/notebooks/Week10.ipynb` |

These notebook-derived values are descriptive only; the final controlled experimental results are mapped to committed CSV/JSON artifacts above.

---

## Split Reproducibility

The exact train/validation/test and purged split manifests used by the final experiments are stored under:

`experiments/results/q1/splits/`

The manifests include seed-specific Track B assignments, strict-temporal positions, BATADAL split definitions, and validation/purge assignments.

Because the complete manifest set is large, the full archive is also made available through the external link documented in:

`docs/split-manifests.md`

Split summary and leakage diagnostics are stored separately under:

`experiments/results/q1/split_audits/`

---

## Final Reproducibility Files

The final result set should be interpreted together with:

- `run_config.json`
- `validation_report.json`
- `splits/`
- `split_audits/`
- final baseline, temporal, ablation, robustness and host-normalization artifacts
- the final manuscript source
- this `docs/paper-result-map.md`

`validation_report.json` provides the final automated consistency checks for the committed result package.
