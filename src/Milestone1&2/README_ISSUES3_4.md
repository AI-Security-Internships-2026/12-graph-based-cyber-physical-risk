# Milestone 2 — Baselines and topology/content ablations

This README covers **Issue #3** (fair non-GNN and GNN baselines) and **Issue #4** (topology versus content ablations). It records the implementation, the real result artifacts currently available, and the remaining experiments. Additional seed runs and the final aggregated results **will be run and reported next**; they are not claimed as completed here.

The status below was checked against the uploaded `experiments.zip` and the supplied source code. Model training was not rerun for this README.

---

## 1. What is implemented

| Issue | Implementation | Result location |
|---|---|---|
| #3 | Logistic Regression, MLP, XGBoost tree, GCN, GraphSAGE, and GAT under SCADANet Track B; principal models under strict SCADANet temporal and purged BATADAL temporal evaluation. A validation carve-out selects hyperparameters and thresholds. | `experiments/results/q1/baselines/` |
| #4 | Content-only, topology-only, full GraphSAGE, random topology, degree-preserving rewiring, and endpoint-only conditions; selected content-feature permutations. | `experiments/results/q1/ablations/` |

Relevant code is in `src/models/baselines.py`, `src/models/gnn.py`, `src/experiments/baseline_runner.py`, `src/experiments/ablation_runner.py`, and `src/graph/rewiring.py`. Table and figure builders are `experiments/build_table_b1_and_figures.py` and `experiments/build_table_c1_and_figures.py`.

**Dependency:** B2, B3, and the strict temporal ablations call the Issue #5 temporal protocol implementation. The separately prepared Issues 1–4 code-only ZIP does not contain that helper and cannot rerun those temporal conditions by itself. Use the complete source project when reproducing temporal results. The legacy Issue #1 chronological results are historical references, not substitutes for the strict temporal conditions.

---

## 2. Issue #3 — Fair baseline comparison

### Protocols and available runs

| Experiment | Models | Per-run JSONs found | Seeds found |
|---|---|---:|---|
| B1 — SCADANet Track B | Six families | 18 | 42, 7, 123 |
| B2 — strict SCADANet temporal | MLP, tree, GraphSAGE, GCN | 12 | 42, 7, 123 |
| B3 — purged BATADAL temporal | MLP, GraphSAGE, GCN | 9 | 42, 7, 123 |

The JSONs include precision, recall, F1, AUPRC, AUROC, FPR, thresholds, timing, parameter counts where applicable, and SCADANet per-attack recall. B1 and B2 use the same split within each seed across their respective model families. B3 JSONs record a 24-row window, 6-row stride, and a purge of four boundary windows for the 70/30 split.

The saved **seed 42** B1 table reports F1 of approximately **0.9857 for the tree, 0.9856 for GCN, and 0.9845 for GraphSAGE**. These close values do not establish a GraphSAGE advantage. In B2, the seed 42 GraphSAGE F1 is approximately **0.5666**, with FPR **0.4143**; the temporal gap is material and should be discussed alongside per-attack recall, particularly `vuln_scan` and `modbus_abuse`.

**Artifact correction needed:** The archived B1/B2/B3 summary CSVs contain seeds **7 and 123**, while the saved tables and figures were generated when seed **42** was available. Regenerate the summary CSVs and the B tables/figures from all three per-run JSON sets before treating them as the final submitted summaries. The requested `baseline_configs.json` is not present in the uploaded results archive; include the actual search-space configuration in the final submission.

### Issue #3 status

- [x] All six B1 model families have real per-run results.
- [x] B2 and B3 principal model results are saved for three seeds.
- [x] Per-attack recall and computational-cost fields are in the saved JSONs.
- [x] Tables B1–B3 and Figures B1–B3 have been generated.
- [ ] Rebuild summary CSVs, tables, and figures from the complete set of saved runs.
- [ ] Include the baseline configuration artifact and the final interpretation of whether the graph models add value.

---

## 3. Issue #4 — Topology and feature counterfactuals

### Conditions and available runs

| Experiment | Available evidence | Current coverage |
|---|---|---|
| C1 — SCADANet Track B | JSONs for Conditions A–F at seed 42; A–E at seed 1 | One complete seed, one incomplete seed |
| C1 — strict SCADANet temporal | JSONs and summary CSV for Conditions A–F | Seed 42 only |
| C2 — feature permutation | No permutation, `Protocol_TCP`, `Tcp_flags_reset_Set`, `frame_len`, top three jointly, all content shuffled | Seed 42 only |

The saved seed 42 Track B Table C1 gives F1 **0.98454** for full GraphSAGE, **0.98516** for random topology, and **0.98526** for degree-preserving rewiring. This single-seed comparison does not show a benefit from the true communication topology. Table C2 shows that permuting `frame_len` lowers F1 from about **0.98451** to **0.95958**. These are preliminary, condition-specific observations; their stability must be checked with paired seeds, AUPRC, FPR, and per-attack recall.

The uploaded results include Tables C1/C2 and Figures C1–C4. Rewiring metadata is present, but the archived metadata file currently describes **seed 1**, whereas the existing table/figure set was generated for **seed 42**. Keep rewiring metadata per seed when rebuilding Figure C4. The required `scadanet_trackb_topology_ablation.csv` is missing from this archive even though the seed 42 Table C1 and underlying JSONs exist.

**Temporal comparison limit:** In the implemented strict temporal ablation, Conditions B/C use evolving causal topology. The rewired D/E conditions are evaluated against a fixed rewired training-boundary graph. Report that difference when comparing them; it is not an identical topology-update intervention.

### Issue #4 status

- [x] Six Track B conditions have seed 42 JSON results.
- [x] Strict temporal and feature-permutation results have seed 42 JSONs.
- [x] Per-attack recall is saved in condition JSONs; Tables C1/C2 and Figures C1–C4 exist.
- [ ] Finish Condition F for seed 1 and reach at least **five complete paired seeds** for the required comparisons.
- [ ] Run and report the additional temporal and feature-counterfactual seeds as appropriate.
- [ ] Restore the Track B ablation summary CSV and regenerate C tables/figures from complete paired runs.
- [ ] Save/identify rewiring metadata for the same seeds represented by the figures and report paired uncertainty intervals.

---

## 4. Reproducing and completing the experiments

Run these commands from the repository root, with the required SCADANet/BATADAL data and Python dependencies available. Use the **complete source project** for the strict temporal commands.

```bash
python -m src.experiments.baseline_runner --experiment b1 --seed <seed>
python -m src.experiments.baseline_runner --experiment b2 --seed <seed> --best-alt-gnn gcn
python -m src.experiments.baseline_runner --experiment b3 --seed <seed> --data-path BATADAL_dataset04.csv

python -m src.experiments.ablation_runner --experiment c1_trackb --seed <seed> --conditions A,B,C,D,E,F
python -m src.experiments.ablation_runner --experiment c1_temporal --seed <seed> --conditions A,B,C,D,E,F
python -m src.experiments.ablation_runner --experiment c2_features --seed <seed>

python -m experiments.build_table_b1_and_figures
python -m experiments.build_table_c1_and_figures
```

For B2, choose `--best-alt-gnn` from the B1 validation policy and document the choice; `gcn` above reflects the currently reported B2 runs. Keep the same seed set across paired Issue #4 conditions. The next report will include the additional seed results, refreshed aggregate summaries, mean and standard deviation, and confidence intervals for the principal topology comparisons. Do not fill missing rows with estimates or interpret the current single-seed C1 table as the final multi-seed result.

---

## 5. Submission status

This is a **progress submission** for Issues #3 and #4. The code and seed 42 experiments support an initial comparison, and Issues #3's three-seed per-run results are archived. Issue #4 does not yet meet its five-seed minimum, and some generated summaries need rebuilding. Final paper claims about GraphSAGE, content, and topology should be made only after the paired results and temporal dependency are packaged reproducibly.
