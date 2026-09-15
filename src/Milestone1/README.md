# Milestone 1 — Reproducible Evaluation Foundation

Covers **Issue #1** (reusable/reproducible experiment pipeline) and
**Issue #2** (split & shortcut audit). This README replaces
`README_ISSUE1.md` / `README_ISSUE2.md` / `PR_DESCRIPTION.md` /
`WEEKLY_PROGRESS_ENTRY.md` as the single source of truth for what's done,
what's verified, and what's left.

Every claim below was checked against `Week12_fix.ipynb` cell-by-cell and
against the real result artifacts (not just described in a doc) before
being written down here.

---

## 1. What this milestone delivers

| | |
|---|---|
| **Issue #1** | A `src/` package + `src.experiments.runner` CLI that reproduces 4 inherited experiments (SCADANet Track B, SCADANet temporal, BATADAL temporal, BATADAL static-CV) with controlled seeds and a fixed result schema. |
| **Issue #2** | A reusable `audit_split()` function run against all 5 currently-defined protocols (Track A, Track B, SCADANet temporal, BATADAL static-CV, BATADAL temporal), producing the audit JSONs, `split_audit_summary.csv` (Table A1), Table A2, and Figures A1–A3. |

Both issues share the same underlying data/model code in `src/`, which is
why they're delivered together.

---

## 2. Repository layout

```
src/
├── data/
│   ├── scadanet.py        SCADANet loading, topology + edge-attr construction,
│   │                      equal-count temporal windowing (Cells 106/107/109/
│   │                      116/117/156/157/159)
│   └── batadal.py         BATADAL loading, ATT_FLAG fix, correlation-graph
│                          topology, sliding-window graphs (Cells 42/43/45/91)
├── evaluation/
│   ├── metrics.py         Single compute_metrics(), replaces 5 duplicated
│   │                      inline metric blocks across the notebook
│   ├── splits.py          scadanet_track_a_split, scadanet_track_b_split,
│   │                      temporal_window_split, batadal_static_cv_folds
│   │                      (Cells 110/121/122/159/92)
│   └── audit.py           audit_split() — Issue #2's reusable audit
├── models/
│   └── gnn.py             SAGEEmbedder, NodeClassifier, EdgeClassifier,
│                          EdgeClassifierWithAttr, WindowGraphClassifier
│                          (Cells 77/92/118), + class-weighting helpers
├── experiments/
│   └── runner.py          CLI entrypoint, one run_* function per protocol
└── utils/
    ├── seed.py            set_seed() — Python/NumPy/PyTorch/CUDA + deterministic cuDNN
    └── io.py               write_result() — enforces the Issue #1 result schema

experiments/
├── run_split_audits.py             Runs audit_split() on all 5 protocols
├── build_table_a2_and_figures.py   Builds Table A2 + Figures A1-A3 from the audit JSONs
└── results/q1/
    ├── *.json                                   one file per reference experiment
    ├── reproducibility_check.csv                Issue #1 Table R1 data
    ├── compare.py                                fills reproducibility_check.csv
    └── split_audits/
        ├── *_audit.json                          one per protocol (Issue #2)
        ├── split_audit_summary.csv                Table A1 data
        ├── table_a2_scadanet_coverage.{csv,md}
        └── figure_a{1,2,3}_*.png
```

`src/data/ics_flow.py` and `src/graph/` are **not started** — not needed
for the 4 reference experiments in this milestone; add them when ICS-Flow
work starts.

---

## 3. Issue #1 — Reproducible pipeline

### What was moved into `src/`

Every function below was moved from the notebook, not re-derived from
memory, and hand-diffed line-by-line against its source cell:

- **Models** (Cell 77, 92, 118) — `SAGEEmbedder`, `NodeClassifier`,
  `EdgeClassifier`, `EdgeClassifierWithAttr`, `WindowGraphClassifier`.
- **Splits** (Cell 110, 121/122, 92, 171) — Track A (IP-held-out), Track B
  (single-fixed-IP flow-level), BATADAL `StratifiedKFold` static-CV,
  BATADAL chronological windows.
- **Loss weighting** (Cell 123) — `compute_subtype_balanced_weights()`,
  used **only** for SCADANet Track B (per-subtype inverse-frequency
  weighting on the loss, not just binary attack/normal — Track B mixes
  scan- and flood-type subtypes at very different volumes, and a plain
  binary weight would let the dominant subtype dominate the gradient).
  All other protocols use plain binary `build_class_weights()`.
- **Data loading** (Cell 42/43/45/91, 106/107/109/116/117) — SCADANet
  (kagglehub) and BATADAL (local CSV) loading, topology construction,
  edge-attribute feature matrix, sliding-window graph construction.
- **Seed control** — `set_seed()` sets Python/NumPy/PyTorch/CUDA seeds
  and forces deterministic cuDNN kernels; called first line of every
  `run_*` function in `runner.py`.
- **Result schema** — `write_result()` refuses to write a result missing
  any of: `dataset, model, split_protocol, seed, train_size,
  validation_size, test_size, feature_set, graph_mode, threshold,
  accuracy, precision, recall, f1, auprc, fpr, confusion_matrix, runtime,
  library_versions, git_commit`. Not-applicable fields are stored as
  explicit `null`, never omitted.

### Running it

```bash
python -m src.experiments.runner \
  --dataset scadanet --split track_b \
  --seed 42 --output experiments/results/q1/scadanet_graphsage_trackb_seed42.json

python -m src.experiments.runner \
  --dataset scadanet --split temporal \
  --seed 42 --output experiments/results/q1/scadanet_graphsage_temporal_seed42.json

python -m src.experiments.runner \
  --dataset batadal --split temporal --data-path BATADAL_dataset04.csv \
  --seed 42 --output experiments/results/q1/batadal_graphsage_temporal_seed42.json

python -m src.experiments.runner \
  --dataset batadal --split static_cv --data-path BATADAL_dataset04.csv \
  --seed 42 --output experiments/results/q1/batadal_graphsage_staticcv_seed42.json
```

SCADANet downloads via `kagglehub` by default; pass `--data-path` to use
a local copy instead. BATADAL has no kagglehub source — always pass
`--data-path` or place `BATADAL_dataset04.csv` in the working directory.

### Reproducibility results (Table R1)

All 4 reference experiments were run end-to-end and compared against the
notebook's previously reported values:

| Experiment | Old F1 | Reproduced F1 | Abs. difference | Status |
|---|---:|---:|---:|---|
| SCADANet Track B | 0.974661 | 0.974661 | 0.0 | PASS — exact to 16 sig. figs |
| SCADANet Temporal | 0.6614 | 0.661348 | 0.0000518 | PASS — near-exact, see note below |
| BATADAL Temporal | 0.7778 | 0.777778 | 0.0000222 | PASS |
| BATADAL Static CV | 0.8621 | 0.862081 | 0.0000194 | PASS — mean of 5 folds, each fold exact |

**SCADANet Temporal note:** the near-exact result traces to ~3 flows out
of 160,452 landing on different sides of the temporal window boundary,
caused by `pandas.sort_values("Time")` in the original notebook not
specifying a stable sort — tied timestamps can break differently across
pandas versions/runs. `add_temporal_windows()` in `src/data/scadanet.py`
fixes this going forward with a **content-derived tiebreak** (a hash over
several row columns, not just `kind="stable"`, since sort *stability*
alone still depends on the row order the file arrived in) so future runs
no longer depend on file/download order at all. This does not, and isn't
meant to, reproduce the *old* environment's specific tie-breaking — it
makes new runs deterministic, which is what Issue #1 asks for.

### Bugs found and fixed during the port

1. **Track B loss weighting.** First draft used plain binary class
   weighting; the notebook (Cell 123) actually uses per-sample
   subtype-balanced weighting. Fixed before any result was reported.
2. **`EdgeClassifier` head.** First draft used a single `Linear`; Cell 77
   uses `Linear → ReLU → Linear`. Fixed. (Not exercised by the 4
   reference experiments — only `EdgeClassifierWithAttr` is — but would
   have affected future ICS-Flow work.)
3. **SCADANet-temporal sort-order sensitivity** — see note above.

### Issue #1 acceptance criteria

- [x] Core logic moved from notebooks into reusable modules
- [x] All random seeds explicitly controlled
- [x] Split definitions deterministic and reusable given a seed
- [x] ≥4 inherited experiments reproduced — **4/4, all PASS**
- [x] Result JSON contains model/data/split/seed/metrics/threshold/environment metadata
- [x] Single command runs an experiment, no notebook cells required
- [x] Existing results not silently changed — one near-exact difference, root-caused and documented
- [ ] Notebooks import reusable code instead of duplicating it — **deferred to a follow-up PR**, deliberately, so this milestone carries zero risk of silently changing a result before reproduction was confirmed

---

## 4. Issue #2 — Split & shortcut audit

### What `audit_split()` reports

One call covers every section the issue asks for, returning
`applicable: False` with a reason instead of silently omitting a section
that doesn't apply to a given split (e.g. BATADAL has no per-record host
identity, since its topology is one fixed sensor-correlation graph
shared by every window):

- **Identity/topology overlap** — host overlap, pair overlap, % of test
  nodes/edges unseen in training.
- **Attack coverage** — per-subtype train/test counts, unique source
  IPs, whether each attack subtype is evaluable under the split, %
  of the taxonomy covered in test.
- **Temporal leakage** — raw-timestamp range overlap (flow-level) and
  sliding-window raw-row overlap (window-level).
- **Distribution shift** — overall and per-label base-rate change
  between train and test.
- **Descriptive warnings** — e.g. "N/M attack classes absent from test
  set," "X% of test hosts seen in training," "attack subtype Y has one
  fixed source IP." These are facts about the split, not automatic
  proof of leakage — interpretation is left to a human/later issue, per
  the issue's own instruction not to treat overlap alone as causation.

### Protocols audited (all 5 currently defined; item 6 — Issue #5's
strict temporal protocols — is explicitly deferred, as the issue allows)

1. SCADANet Track A
2. SCADANet Track B
3. SCADANet chronological (temporal)
4. BATADAL static cross-validation
5. BATADAL current chronological split

### Running it

```bash
python -m experiments.run_split_audits \
  --scadanet-path /path/to/scadanet.csv \
  --batadal-path BATADAL_dataset04.csv

python -m experiments.build_table_a2_and_figures
```

The first command writes the 5 `*_audit.json` files and
`split_audit_summary.csv` (Table A1). The second builds Table A2 and
Figures A1–A3 from those JSONs.

### Key empirical result — Table A1 (from `split_audit_summary.csv`)

| Protocol | Train | Test | Host overlap | Attack types covered | New test nodes | New test edges |
|---|---:|---:|---:|---:|---:|---:|
| SCADANet Track A | 523,901 | 10,940 | 0.3% | **3/14** | 99.7% | 100% |
| SCADANet Track B | 422,844 | 105,711 | **100%** | 13/13 | 0% | 0.4% |
| SCADANet Temporal | 374,389 | 160,452 | 0.2% | 12/14 | 99.8% | 99.5% |
| BATADAL Static CV | 554 | 139 | N/A (fixed topology) | 2/2 | N/A | N/A |
| BATADAL Temporal | 485 | 208 | N/A (fixed topology) | 2/2 | N/A | N/A |

This is the headline finding the issue asked Issue #2 to surface:
**Track A tests only 3 of 14 attack types** (Figure A1 shows 11 attack
types have literally zero test samples under Track A), while Track B —
the split with the strongest reported numbers in earlier work — has
**~100% host overlap** between train and test. Neither number alone
proves the reported model performance is invalid; both are facts about
what each protocol is actually able to measure, which is exactly what
this issue exists to make visible before performance numbers are
interpreted.

### Figures

- **Figure A1** (`figure_a1_coverage_heatmap.png`) — log-scale heatmap of
  test-sample count per attack type × protocol, ✗ marking zero-coverage
  cells. Makes the Track-A shortcut visible without relying on aggregate
  metrics, as the issue requested.
- **Figure A2** (`figure_a2_overlap_novelty.png`) — % seen vs. unseen
  test hosts per protocol (BATADAL excluded — no per-record host
  identity at that granularity).
- **Figure A3** (`figure_a3_base_rate_shift.png`) — train vs. test
  prevalence per attack type under the SCADANet temporal split,
  `vuln_scan` highlighted per the issue's explicit spec (train ≈90%,
  test ≈24% — the large shift the issue flagged as already observed).

### Issue #2 acceptance criteria

- [x] Reusable split-audit function exists (`audit_split()`)
- [x] Audit results generated automatically from actual train/test data
- [x] All 5 currently-defined protocols audited
- [x] Per-attack-type test coverage reported
- [x] Host/node/pair overlap reported where meaningful
- [x] Temporal/window overlap reported where meaningful
- [x] Distribution/base-rate changes quantified
- [x] Warnings saved in machine-readable form (`warnings` list in each JSON)
- [x] Table A1 data exportable directly from committed results
- [x] Table A2 and Figures A1–A3 generated from saved artifacts, not typed by hand

### Known limitation

The BATADAL static-CV audit currently audits **fold 0 of 5** as
representative, not all 5 folds. Looping `audit_split()` over each
fold is straightforward to add; not done yet because the 5 folds are
expected to tell a very similar story and it wasn't clear whether the
paper needs "typical fold" or "every fold" reported. Decide before
treating the current single-fold audit as final.

---

## 5. What's explicitly *not* done in this milestone (by design)

Per each issue's "Do Not Do" section, this milestone does **not**:
- change model architecture or tune thresholds,
- delete any attack category, however inconvenient,
- use model test performance to define or justify a split,
- interpret overlap numbers alone as proof of leakage,
- add new datasets, a dashboard, or unrelated features.

Also outstanding, tracked as follow-up work rather than gaps in this
milestone:
- Replacing the notebook's duplicated cells with `from src... import ...`
  (Issue #1, deferred until reproduction was confirmed — now that it is,
  this is unblocked).
- All 5 folds of the BATADAL static-CV audit (Issue #2, see above).
- Issue #5's strict temporal protocols, to be added to the audit once
  that issue lands (explicitly allowed by Issue #2's own scope).

---

## 6. Verification performed

- Every model class, split function, and data-loading function in `src/`
  was diffed line-by-line against its exact source cell in
  `Week12_fix.ipynb` (cell numbers above refer to notebook execution
  count, matching how the notebook itself is commented). No architecture,
  hyperparameter, or logic changes were found beyond the 3 bugs listed
  in Section 3 and the sort-stability fix — all caught and fixed *before*
  being reported as a result, not after.
- All 4 Issue #1 reference experiments and all 5 Issue #2 audits were run
  against the real datasets (Colab, CPU) — every number in this document
  comes from a generated file in `experiments/results/q1/`, not from a
  plan or a synthetic-data dry run.
