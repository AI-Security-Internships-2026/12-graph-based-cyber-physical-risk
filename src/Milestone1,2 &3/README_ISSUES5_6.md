# Milestone 3 — Strict temporal evaluation and host normalization

This README covers **Issue #5** (strict leakage-free temporal evaluation) and **Issue #6** (host-relative feature normalization). It distinguishes the implemented protocols, the results currently displayed in `Milestone2_5&6 (1).ipynb`, and the remaining reporting work.

The figures and numbers below are transcribed from the notebook's **later seed 7 tables**, unless a different seed is stated. The notebook also displays earlier seed 42 outputs. The experiment runners currently write each protocol's summary to a fixed filename, so a later seed can replace an earlier seed's summary. Training was not rerun for this README, and the separate full Issue 5/6 results directory was not provided for an independent file-by-file audit.

---

## 1. What this milestone implements

| Issue | Main change | Output directory |
|---|---|---|
| #5 | SCADANet frozen-model evaluation with causally evolving topology, plus a fixed-training-topology comparison; BATADAL purged chronological splits with 60/40, 70/30, and 80/20 sensitivity. | `experiments/results/q1/temporal/` |
| #6 | Training-derived per-source-host normalization for selected continuous SCADANet features, with a global training-statistics fallback; original and normalized models at default and validation-selected thresholds. | `experiments/results/q1/host_normalization/` |

Relevant code includes `src/evaluation/temporal_protocol.py`, `src/experiments/temporal_runner.py`, `src/evaluation/host_normalization.py`, and `src/experiments/host_norm_runner.py`. The table and figure scripts are `experiments/build_table_t_and_figures.py` and `experiments/build_table_h_and_figures.py`.

---

## 2. Issue #5 — Strict temporal evaluation

### Protocol

For SCADANet, T1 is the historical chronological reference that can expose future test topology. T2 evaluates each test window using only topology observed before that window; the trained model stays frozen. T3 uses the topology available at the training boundary for every test window. BATADAL T4 is the historical overlapping-window reference. T5/T6 remove windows crossing the raw-row train/test boundary before evaluation.

### Results displayed for seed 7

| Dataset | Protocol | F1 | AUPRC | FPR | Interpretation |
|---|---|---:|---:|---:|---|
| SCADANet | T1 — previous temporal | 0.630686 | 0.771219 | 0.080519 | Historical reference; future topology possible |
| SCADANet | T2 — strict causal topology | 0.630671 | 0.685439 | 0.080532 | Primary causal-topology result |
| SCADANet | T3 — fixed training topology | 0.630907 | 0.674420 | 0.079897 | No test-period topology update |
| BATADAL | T4 — overlapping temporal | 0.792453 | 0.921624 | 0.058824 | Historical reference; raw-row overlap |
| BATADAL | T5 — purged 70/30 | 0.792453 | 0.917480 | 0.059783 | Four boundary windows purged |

The SCADANet F1 values are nearly unchanged across T1–T3 for this seed, while AUPRC differs. BATADAL F1 is unchanged between T4 and T5 for this seed, even though the purged protocol removes the boundary overlap. These observations support reporting protocol effects rather than assuming that leakage removal must always lower F1.

For BATADAL sensitivity after purging, the displayed F1 values are **0.746988** (60/40), **0.792453** (70/30), and **0.777778** (80/20). Each split reports **four purged windows**. The notebook logs show that per-window SCADANet metrics, graph growth, per-attack results, the BATADAL boundary audit, Tables T1–T3, and Figures T1–T5 were written.

**Reporting gap:** Table T2 shows `NaN` in every Static/Track-B recall and Δ recall cell, and Figure T4 therefore plots only strict temporal recall. The Track B per-attack values should be loaded from the Issue #3 per-run GraphSAGE JSON for a matched seed, then Table T2 and Figure T4 regenerated. In the displayed seed 7 strict run, `vuln_scan` recall is about **0.0085**, while `insider_threat`, `modbus_fdi`, and `modbus_abuse` are **0**; these attack-specific failures should not be hidden behind aggregate F1.

### Issue #5 status

- [x] SCADANet T1, T2, and T3 were executed and displayed in the notebook.
- [x] BATADAL T4, T5, and 60/40–80/20 purged sensitivity were executed and displayed.
- [x] Four purged BATADAL boundary windows were reported for each displayed purged split.
- [x] Table T1/T3 and Figures T1–T3/T5 were generated in the notebook session.
- [ ] Populate matched Track B recall and Δ recall in Table T2 and regenerate Figure T4 as a true comparison.
- [ ] Check and package the saved per-window CSV, graph-state diagnostics, and BATADAL boundary-audit JSON with the submitted results.
- [ ] Preserve seed-specific outputs before aggregating additional seeds.

---

## 3. Issue #6 — Host-normalized features

### Method and experimental conditions

The code fits per-host center and scale from training rows for `frame_len`, `ip_len`, and `Tcp_window_size`. Hosts without enough training observations use global **training-period** statistics. The experiment compares H1 (original, threshold 0.50), H2 (original, validation-calibrated threshold), H3 (host-normalized, threshold 0.50), and H4 (host-normalized, validation-calibrated threshold) on both Track B and strict SCADANet temporal evaluation. The source-host diagnostics explicitly track `192.168.119.140` and `192.168.119.141`.

### Track B results displayed for seed 7

| Condition | F1 | Normal FPR | Total FP | Recall |
|---|---:|---:|---:|---:|
| H1 — original, 0.50 | 0.974640 | 0.274129 | 4,604 | 0.999753 |
| H2 — original, calibrated | 0.984393 | 0.139863 | 2,349 | 0.994872 |
| H3 — host-normalized, 0.50 | 0.976166 | 0.257219 | 4,320 | 0.999764 |
| H4 — host-normalized, calibrated | 0.985556 | 0.127538 | 2,142 | 0.994928 |

H3 alone has a modest FP reduction relative to H1. H4 has fewer FPs than H2, but **`os_scan` recall falls from 0.994937 (H1) to 0.513924 (H4)**, while `vuln_scan` recall falls from 0.997678 to 0.969575. The host-level improvement is uneven: the displayed false positives for `.141` fall from **2,058 to 9**, while those for `.140` fall from **2,299 to 2,130**. This is a trade-off, not evidence that the host shortcut has been fully removed.

### Strict temporal results displayed for seed 7

| Condition | F1 | Normal FPR | Total FP | AUPRC |
|---|---:|---:|---:|---:|
| H1 — original, 0.50 | 0.630937 | 0.080014 | 6,170 | 0.676098 |
| H2 — original, calibrated | 0.575333 | 0.376012 | 28,995 | 0.676098 |
| H3 — host-normalized, 0.50 | 0.630525 | 0.080052 | 6,173 | 0.819340 |
| H4 — host-normalized, calibrated | 0.663786 | 0.106235 | 8,192 | 0.819340 |

Temporal H3 improves AUPRC, but its default-threshold F1 and FPR are approximately unchanged. Temporal H4 raises F1 relative to H1 while increasing FPR and FP count. Validation-selected thresholds should therefore be assessed against both detection and false-alarm goals; calibration does not guarantee a lower FPR in a later time period.

The notebook generated Tables H1–H3 and Figures H1–H2 through the figure script. The separate Figure H3 cell produced `figure_h3_distribution_frame_len.png`. Its raw-feature panel is compressed by extreme `frame_len` values, so a log or clearly labeled quantile-limited view is needed before using it as visual evidence for the hypothesis.

**Figure correction:** Figure H2 plots **FPR versus recall**, despite its “Precision/FPR” title. The caption saying “lower-left … strictly better” is wrong for recall. Relabel it as an FPR/recall trade-off and describe the desirable direction as **lower FPR and higher recall**.

### Issue #6 status

- [x] H1–H4 were executed for Track B and strict temporal evaluation in the notebook.
- [x] The notebook displays total FPs, FPR, precision/recall/F1/AUPRC, host counts, and per-attack recall tables.
- [x] The two previously problematic hosts are shown explicitly in Table H2.
- [x] Figure H3 was generated by a separate cell using row-level feature values.
- [ ] Correct Figure H2's labels and improve Figure H3's raw-feature scale.
- [ ] Package and check the saved source/destination host FP CSVs, configuration JSON, and per-attack CSVs.
- [ ] Retain outputs by seed and report paired multi-seed results before making a final mitigation claim.

---

## 4. Reproducing the runs

Run from the complete source project with the same datasets and dependencies used for the reported experiments:

```bash
python -m src.experiments.temporal_runner --experiment scadanet --seed 42
python -m src.experiments.temporal_runner --experiment batadal --data-path BATADAL_dataset04.csv --seed 42
python -m experiments.build_table_t_and_figures

python -m src.experiments.host_norm_runner --protocol both --seed 42
python -m experiments.build_table_h_and_figures
```

Figure H3 also requires the notebook's separate row-level plotting cell, or an equivalent script supplied with the result package. When running further seeds, save each seed's generated files separately **before** rerunning a command that writes the same summary filename. Build final tables from the preserved per-seed artifacts and label the seed set used for each figure.

## 5. Current research interpretation

The strict temporal protocol changes SCADANet ranking performance in the displayed run even where thresholded F1 changes little. Host normalization provides limited Track B false-positive relief on its own and gives mixed results when combined with calibration. Some attack recall and temporal FPR trade-offs are substantial. The final paper should keep threshold adjustment distinct from feature-level mitigation and avoid claiming that host normalization universally solves the shortcut.
