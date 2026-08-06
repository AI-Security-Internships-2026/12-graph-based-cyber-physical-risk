
# Weekly Progress Log: Graph-Based Risk Assessment for Cyber-Physical Systems

**Student:** Misbah Shaheen  
**GitHub username:** Misbah-shaheen  

---

## How to Use This File

Add a new section every Friday before opening your weekly Pull Request.  
Be honest — problems and blockers are normal and help your supervisor support you.

---

## Week 1

**Branch:** `misbahshaheen-week-01`  
**PR link:** https://github.com/AI-Security-Internships-2026/12-graph-based-cyber-physical-risk/pull/1

### Completed this week
- [✔] Read README and project proposal  
- [✔] Set up local development environment (Python venv, dependencies)  
- [✔] Successfully ran `src/main.py`  
- [✔] Created and switched to working branch (`misbahshaheen-week-01`)  
- [✔] Identified 5 relevant research papers, tools, and datasets related to GNN-based cyber-physical risk assessment  

### Personal Introduction
I am a 3rd-year Data Science student at NUST with hands-on experience in Graph Neural Networks using PyTorch. I have worked on a T5 + GNN-based system for structured reasoning over 16K+ samples. I also have experience in NLP pipelines and deep learning workflows. I am joining this internship to apply graph-based reasoning to cyber-physical risk problems and to strengthen my skills in knowledge graph construction and GNN-based modeling using tools like NetworkX and Neo4j.

### Problems / Blockers
- Initial confusion in Git branching workflow, later understood after practice  

### Next week plan
- Deeply study the 5 selected research papers  
- Start drafting `docs/proposal.md` (system architecture design)  
- Explore datasets (SWaT, CIC-IDS-2017, MITRE ATT&CK ICS)  
- Begin graph modeling design using NetworkX and Neo4j  

---

## Week 2
**Branch:** `misbahshaheen-week-02`      
**PR link:** https://github.com/AI-Security-Internships-2026/12-graph-based-cyber-physical-risk/pull/3

### Completed this week
- [✔] Revised literature review per supervisor feedback — restricted all sources to 2023–2026
- [✔] Searched and added papers using supervisor's suggested terms: "GNN cyber-physical system attack detection," "knowledge graph ICS SCADA risk assessment," "graph neural network OT security"
- [✔] Replaced outdated datasets (SWaT 2015, CIC-IDS-2017) with three 2023–2026 ICS/OT datasets: ICS-Flow (2023), Cyber4OT (2025), ICS-ADD (2024)
- [✔] Identified and added 4 active GitHub repositories (sepses/ics-sec-kg, zhenlus/GNN-IDS, mbdlrocks/PhD_Replication_Package, lorenzo9uerra/GraphIDS) relevant to GNN-based ICS security
- [✔] Fact-checked all paper citations, DOIs, and dataset statistics against original sources to confirm accuracy before submission


### Problems / Blockers
- No problem

### Next week plan
- Begin designing the knowledge graph schema (node/edge types: Asset, Vulnerability, Technique, Process Variable) in NetworkX before migrating to Neo4j
- Start mapping ICS-Flow's network flow + process variable logs onto the planned dual-layer graph structure
- Draft the system architecture section of `docs/proposal.md` (Week 3 deliverable)
- Begin exploratory data loading/preprocessing of ICS-Flow and Cyber4OT

---

## Week 3
**Branch:** `misbahshaheen-week-03`     
**PR link:** https://github.com/AI-Security-Internships-2026/12-graph-based-cyber-physical-risk/pull/4

### Completed this week
- [✔] Drafted and finalised `docs/proposal.md` — research questions, methodology,
      dataset roles, evaluation metrics, risk table, and weekly deliverable plan
- [✔] Identified flow paths in ICS-Flow attack graph using `nx.all_simple_paths`
      on a directed MultiDiGraph built from attack-labelled flows; 13 unique attack
      paths enumerated from source devices to the primary target node (.41)
- [✔] Constructed attack flow graph (ICS-Flow) — directed, colour-coded by attack
      type (port-scan → MITM → replay → DDoS); saved as `week3_attack_flow_graph.png`
- [✔] Built attack timeline showing temporal sequencing of attack types:
      port-scan (12:15) → MITM (12:17) → replay (12:22) → DDoS (12:34)
- [✔] Ran graph construction on BATADAL (real ICS physical-layer dataset,
      dataset04.csv, 4177 timesteps, 43 sensors/actuators) — built sensor
      correlation graphs for normal and attack periods using Pearson correlation
      thresholding; saved visualisation as `week3_batadal_graph.png`
- [✔] Identified 4 anomalous sensors in BATADAL deviating >2σ from normal during
      attack periods — flagged as candidate high-risk nodes for GNN training (Week 6)
- [✔] Compared normal graph (43 nodes, 37 edges) vs attack graph (43 nodes,
      30 edges) in BATADAL — 11 edges lost and 4 new edges appeared during attacks,
      indicating structural disruption of sensor correlations under attack conditions

### Problems / Blockers
- No problem

### Next week plan
- Integrate ICS-Flow PLC snapshot files (`snapshots_PLC1.csv`, `snapshots_PLC2.csv`)
  with the cyber communication graph to form the dual-layer knowledge graph
- Assign node roles (PLC, HMI, Gateway, Sensor, Actuator) and build `controls`
  and `monitors` edges for the physical process layer
- Engineer the node feature matrix: per-device flow statistics (cyber layer) and
  per-sensor mean/std/deviation ratio (physical layer)
- Export the completed dual-layer graph to NetworkX and begin Neo4j schema design

---

## Week 4
**Branch:** `misbahshaheen-week-04`  
**PR link:** https://github.com/AI-Security-Internships-2026/12-graph-based-cyber-physical-risk/pull/8

### Completed this week
- [✔] Built a cybersecurity Knowledge Graph (KG) from the ICS-Flow dataset using NetworkX
- [✔] Added Asset, Attacker, MITRE ATT&CK Technique, CVE, and Alert nodes
- [✔] Implemented relationships: `COMMUNICATES_WITH`, `USES_TECHNIQUE`, `TARGETS`, `VULNERABLE_TO`, `ENABLES`, and `TRIGGERED_ON`
- [✔] Integrated MITRE ATT&CK ICS techniques and CVE information into the graph
- [✔] Computed node risk scores based on attack activity, communication volume, and CVSS severity
- [✔] Created knowledge graph visualizations and exported graph data for Neo4j (`week4_kg_nodes.csv`, `week4_kg_edges.csv`)
- [✔] Converted the graph into a PyTorch Geometric (PyG) format
- [✔] Implemented GraphSAGE-based node embeddings, node classification, and flow-level classification
- [✔] Evaluated the flow-level classifier using Accuracy, Precision, Recall, and F1-score
- [✔] Added BATADAL evaluation hook for Week 5
- [✔] Updated literature review and added 3 recent papers (2025–2026)

### Problems / Blockers
- Limited number of devices in ICS-Flow makes node-level evaluation difficult.
- Used flow-level classification to obtain a more meaningful evaluation.

### Next week plan
- Import the knowledge graph into Neo4j
- Design Cypher queries for graph analysis
- Preprocess BATADAL using the same graph feature schema
- Evaluate GraphSAGE on BATADAL
- Begin cross-dataset comparison and risk analysis

---

## Week 5
**Branch:** `misbahshaheen-week-05`     
**PR link:** https://github.com/AI-Security-Internships-2026/12-graph-based-cyber-physical-risk/pull/9

### Completed this week
- [✔] Ran GraphSAGE edge-level classifier on ICS-Flow (44,156 flows, 80/20 split, inverse-frequency class weighting) — reported test accuracy, precision, recall, F1 as the held-out generalization metric             
- [✔] Identified and corrected a methodology issue in the initial BATADAL cross-dataset validation: a single static correlation graph produced only 41 total labelled edges (4 anomalous), giving a degenerate 8-edge test split
- [✔] Rebuilt BATADAL validation using sliding 24-hour windows (6-hour stride) → 693 graph instances (56 positive, ~8%), fixed topology from the Week 3 normal-period correlation graph, per-window node features (mean/std/z-score deviation from baseline)
- [✔] Trained a graph-level GraphSAGE classifier (SAGEConv + global mean pooling) with stratified 5-fold cross-validation on BATADAL — accuracy 0.974 ± 0.017, precision 0.790 ± 0.116, recall 0.965 ± 0.043, F1 0.862 ± 0.054
- [✔] Loaded the Week 4 knowledge graph (`week4_kg_nodes.csv`, `week4_kg_edges.csv`) into Neo4j via the Python driver; validated with the ranked attack-path Cypher query from Issue #7's acceptance criteria

### Problems / Blockers
- ICS-Flow and BATADAL report at different units of prediction (edge-level vs. window/graph-level) — flagging this explicitly rather than presenting them as directly comparable

### Next week plan
- Extend cross-dataset validation further using the SCADANet dataset (Kaggle: `ealgul/scada-dataset-v01`) — a virtual Modbus/TCP SCADA testbed with 14 classes (1 normal + 13 attack types), which gives a third, flow-level dataset alongside ICS-Flow and BATADAL for validating the GraphSAGE pipeline
- Begin RQ3 scoping: physical process impact integration

---

## Week 6

**Branch:** `misbahshaheen-week-06`
**PR link:** https://github.com/AI-Security-Internships-2026/12-graph-based-cyber-physical-risk/pull/10

### Completed this week
- [✔] Downloaded and profiled the SCADANet dataset (`ealgul/scada-dataset-v01`) — 534,841 packets, 60 columns; confirmed traffic split (451,099 attack / 83,742 normal) and top attack types (`udp_flood`, `vuln_scan`, `http_flood`, `icmp_flood`, plus 9 smaller classes)
- [✔] Confirmed complete IP-level class separability in SCADANet (every source/destination IP is 100% attack or 100% normal) — designed an IP-held-out train/test split so no IP appears in both sets, avoiding trivial identity memorization
- [✔] Trained a baseline GraphSAGE edge classifier (topology + 2-dim in/out-degree node features only) — test accuracy 0.9985, precision 0.9876, recall 1.0000, F1 0.9938
- [✔] Performed a full column-by-column audit of all 60 raw SCADANet columns, classifying each as dead/constant, label-leak, identifier/free-text/fingerprint, high-cardinality (deliberately held out), or genuine per-flow signal — 10 numeric + 13 categorical columns (224 dims after one-hot encoding) retained as edge features
- [✔] Built an enriched GraphSAGE variant (`EdgeClassifierWithAttr`) that projects the 224-dim edge feature vector and concatenates it with the node-embedding pair before the classification head
- [✔] Ran a set of diagnostics (weight-norm inspection, edge-feature ablation via shuffling/zeroing, logit-margin analysis) to determine whether the enriched model's edge features were actually influencing predictions — confirmed they are correctly wired and do move the logits, but never by enough to flip a prediction once trained, because node-identity alone already separates the dominant high-volume traffic (e.g. `udp_flood`, 338,656/534,841 flows) with a large, fixed margin
- [✔] Benchmarked model size and inference latency for all four trained models (ICS-Flow, BATADAL, SCADANet baseline, SCADANet enriched)
- [✔] Exported all Week 6 metrics, the column audit, and benchmark results to `week6_metrics.json`

### Problems / Blockers
- **Found and fixed a reproducibility bug:** the SCADANet training functions seeded only the train/test IP split (`torch.Generator().manual_seed(seed)`), not PyTorch's global RNG. This left model weight initialization and dropout unseeded, so re-running the same code produced different results on different kernel restarts — I observed the enriched model swing from 0.9982 accuracy on one run to 0.6563 on another, with identical code. Added `torch.manual_seed(seed)` at the start of both training functions to fix this.
- After seeding, the enriched model still reproducibly collapses to accuracy 0.6563 / precision 0.2535 / recall 1.0000 / F1 0.4045 under seed=42 — consistently, on repeated reruns. This is not yet resolved: I don't know whether seed=42 specifically triggers a bad initialization for the 224-dim edge-feature architecture, or whether this instability is systematic across seeds. A multi-seed sweep is queued to determine which.
- Because of the above, I'm treating the enriched-model comparison as provisional rather than final pending the seed sweep, even though the baseline model and the diagnostic findings (edge features are real but currently redundant on the high-volume majority of flows) are solid and reproducible.

### Next week plan
- Run the enriched SCADANet model across multiple seeds (e.g. 1–7) to determine whether the accuracy collapse is seed-specific or a systematic instability of the 224-dim one-hot edge-feature architecture
- If systematic: investigate whether dimensionality reduction on the categorical one-hot block (e.g. embedding layers instead of raw one-hot, or dropping near-constant presence-flag columns) stabilizes training
- Add a stratified evaluation slice (low-frequency IP pairs vs. high-volume flood pairs) to check whether the enriched features provide value specifically on traffic the topology-based node embeddings can't already separate by identity alone
- Finalize and write up the SCADANet baseline-vs-enriched comparison with honest variance reporting (mean ± std across seeds, matching the BATADAL 5-fold protocol from Week 5) rather than a single run
- Update `week6_metrics.json` once the seeded, multi-run results are finalized

---

## Week 7

**Branch:** `misbahshaheen-week-07`
**PR link:** https://github.com/AI-Security-Internships-2026/12-graph-based-cyber-physical-risk/pull/11

### Completed this week
- [✔] Traced Week 6's recall = 1.0000 result to an IP-identity artifact, not real detection: 10 of 14 attack types share one fixed attacker IP each, and the other two (`tcp_syn_flood`, `udp_flood`) use spoofed IPs appearing only 1-2 times — trivially separable by raw node degree alone
- [✔] Confirmed this empirically: enriched 224-dim edge features left recall unchanged (`fn=0`), proving the shortcut lives in the shared node-degree encoder, not the content features
- [✔] Split evaluation into two honestly-scoped tracks: **Track A** (IP-held-out, `tcp_syn_flood`/`udp_flood` only) and **Track B** (flow-level, subtype-balanced, covers the 12 single-IP attack types + normal)
- [✔] Results — Track A: acc 0.9983, precision 0.9853, recall 1.0000 (n=10,940). Track B: acc 0.9563, precision 0.9508, recall 0.9997 (n=105,711)
- [✔] Flagged Track B's 27.2% normal-traffic false-positive rate as an open, unresolved limitation
- [✔] Added per-attack-type breakdowns, an interpretation writeup, and `track_a`/`track_b` blocks in `week7_metrics.json`

### Problems / Blockers
- Week 6's queued multi-seed sweep not done — likely tied to the same degree-shortcut issue, so deferred to Week 8 to re-run against Track A/B instead of the old blended split
- Track B's 27.2% FPR not yet root-caused

### Next week plan
- Root-cause Track B's normal-traffic FPR
- Re-run enriched model to full 400 epochs
- Run the deferred multi-seed sweep against Track A/B
- Test dropping raw IP-degree from node features on Track A
- Update `week7_metrics.json`; begin RQ3 scoping if time permits

---

## Week 8

**Branch:** `misbahshaheen-week-08`
**PR link:** https://github.com/AI-Security-Internships-2026/12-graph-based-cyber-physical-risk/pull/12

### Completed this week
- [✔] Root-caused Track B's 27.2% normal-traffic false-positive rate flagged in Week 7
- [✔] Ruled out topology/degree leakage as the cause: ablated the node-embedding pathway directly and confirmed embeddings carry no meaningful signal (std = 0.056 across all nodes, not identical but negligible) — this is not the same shortcut-learning mechanism found in Week 6/7
- [✔] Identified the actual cause: ~95% of false positives concentrate on two high-volume hosts (`192.168.119.140`, `192.168.119.141`) that are also the dataset's dominant attack sources; their genuinely-normal traffic differs systematically from other hosts' normal traffic on content features (`Tcp_flags_reset_Set`: 44.1% in false positives vs. 0.1% in true negatives; `ip_ttl`/`frame_len` both shift substantially) — consistent with congestion artifacts from these hosts' concurrent attack activity bleeding into their normal traffic
- [✔] Applied a decision-threshold recalibration as a mitigation: selected threshold (0.832) on a 15% validation split carved from Track B's training set, never touching the test set, then applied once to test
- [✔] Result: precision 0.9508 → 0.9722, recall 0.9997 → 0.9967, FPR on normal traffic 27.2% → 15.0%
- [✔] Documented this explicitly as a mitigation, not a root-cause fix — the underlying host-level content similarity is unchanged; only the model's decision boundary is more conservative
- [✔] Extended `week7_metrics.json` → `week8_metrics.json` carrying `ics_flow`/`batadal`/`scadanet` forward unchanged

### Problems / Blockers
- The deeper fix for the FPR issue (per-host feature normalization, so a busy host's normal traffic is judged against its own baseline rather than the global distribution) is proposed but not yet implemented — threshold recalibration was the fast, validated mitigation for this week

### Next week plan
- Apply GNNExplainer/PGExplainer to formally confirm which features drive predictions on the two high-volume hosts specifically (per the roadmap's Aug 9 milestone)
- Implement per-host feature normalization as the deeper fix for Track B's remaining 15.0% FPR, and compare against the threshold-calibration mitigation
- Run the deferred multi-seed sweep against Track A/B

---

## Week 9

**Branch:** `misbahshaheen-week-09`  
**PR link:** https://github.com/AI-Security-Internships-2026/12-graph-based-cyber-physical-risk/pull/15

### Completed this week
- [✔] Applied **GNNExplainer**, **PGExplainer**, and **Captum Integrated Gradients** to formally explain Track B false-positive predictions on the two high-volume hosts (`192.168.119.140`, `192.168.119.141`).
- [✔] Implemented **FlowExplainWrapper** to adapt the custom `EdgeClassifierWithAttr` interface for compatibility with the PyG Explainer API.
- [✔] Addressed PyG PGExplainer API limitations by treating each flow as a graph-level prediction and using the model's predicted class as the explanation target.
- [✔] Applied **Integrated Gradients** to attribute predictions to the 23 raw `edge_attr` features, complementing the topology explainers.
- [✔] Evaluated explainability on **40 sampled false-positive flows** (from 4,374 false positives across the two hosts).
- [✔] Confirmed that **Protocol_TCP (0.273)**, **Tcp_flags_reset_Set (0.154)**, and **frame_len (0.098)** are the dominant packet-content features, quantitatively validating the Week 8 qualitative root-cause analysis.
- [✔] GNNExplainer (**edge mask = 0.2922**, std = **0.0023**) and PGExplainer (**edge mask = 0.1250**, std = **0.0426**) consistently assigned low importance to graph topology, while the low saturation fraction (**0.0113**) confirmed successful GNNExplainer convergence.
- [✔] Improved result reporting by replacing the non-interpretable `node_mask.sum()` metric with **node-mask mean** and **saturation fraction**, and separated topology and Integrated Gradients plots to avoid comparing different explanation scales.
- [✔] Added the explainability discussion, generated `week9_explainability.png`, and extended `week8_metrics.json` into `week9_metrics.json` with the Week 9 explainability results.

### Problems / Blockers
- PGExplainer explanations exhibited higher variance than GNNExplainer because a separate explainer network is trained for each flow; therefore, GNNExplainer is reported as the primary topology estimate and PGExplainer as supporting evidence.
- Per-flow explanation is computationally expensive, so the analysis was limited to **40 sampled flows** instead of the full false-positive set.

### Next week plan
- Extend the analysis to **temporal/streaming graph updates**, where nodes and edges arrive over time instead of using a static graph snapshot.

---
