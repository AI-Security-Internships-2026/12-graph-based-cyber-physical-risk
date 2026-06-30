# Research Proposal: Graph-Based Risk Assessment for Cyber-Physical Systems

**Student:** Misbah Shaheen
**Supervisor:** [Eng. Rana Abu Bakar](https://github.com/engranaabubakar)

**Start date:** 10 June 2026
**Expected end date:** 10 August 2026

---

## 1. Background

Model industrial control system (ICS/SCADA) networks as knowledge graphs and apply graph neural networks to predict attack paths, propagation likelihood, and impact on physical processes.

This project is carried out within the AI Security research agenda of CNIT/PNTLab Pisa (TECIP, Scuola Superiore Sant'Anna).

---

## 2. Problem Statement

Most existing ICS security solutions analyse network traffic and physical process data separately. Network-based approaches can detect suspicious traffic but often fail to explain how an attack propagates through the system, while process-based approaches identify abnormal sensor behaviour without providing information about the cyber events that caused it.

This lack of integration limits the ability to understand attack progression and assess potential operational impact. The project addresses this challenge by constructing a unified cyber-physical graph from publicly available ICS datasets and investigating whether Graph Neural Networks can improve attack classification, attack-path analysis, and cyber-physical risk assessment.

---

## 3. Research Questions

1. **RQ1 — Graph Construction:** Can a dual-layer cyber-physical graph be constructed
   from ICS network-flow data and PLC snapshot data, and does the resulting graph reveal
   exploitable attack paths that are not visible from individual flow records alone?

2. **RQ2 — GNN-Based Attack Analysis:** Can Graph Neural Networks (GCN, GraphSAGE,
   or GAT) operating on the dual-layer graph accurately identify attack-related nodes
   and predict the propagation path of an attack across the ICS network topology?

3. **RQ3 — Physical Process Integration:** Does incorporating physical-process information (sensor readings and actuator states) improve attack detection and cyber-physical risk assessment compared to using network-flow information alone?

---

## 4. Proposed Methodology

### 4.1 Data Collection / Dataset

Three publicly available datasets are used in a phased manner:

**Phase 1 — Primary training dataset: ICS-Flow**
- **Full name:** ICS-Flow — Anomaly Detection Dataset for Industrial Control Systems
- **Source:** Dehlaghi-Ghadim et al., IEEE Access 2023;
  available on Kaggle (`alirezadehlaghi/icssim`)
- **Content:** 45,718 network flow records (64 features) collected from a simulated ICS
  environment containing PLCs, HMIs, a gateway, and an attacker node.
  Includes matched PLC snapshots (`snapshots_PLC1.csv`, `snapshots_PLC2.csv`) and raw
  PCAP (`traffic.pcap`).
- **Attack classes:** Normal, DDoS, IP Scan, Port Scan, MITM, Replay
- **Licence:** CC BY 4.0
- **Role in project:** Build the cyber communication graph; enrich with PLC snapshot
  features to form the physical process layer; train the primary GNN classifier.

**Phase 2 — Attack propagation labelling: Cyber4OT**
- **Full name:** Cyber4OT Dataset
- **Content:** Multi-stage ICS attack sequences with labelled propagation steps
  (Reconnaissance → Vulnerability Discovery → Exploitation → Persistence → PLC Takeover).
- **Role in project:** Intended to support directed attack-path graph construction and provide propagation labels for attack-path prediction (RQ2), subject to successful dataset integration.
- **Licence:** Publicly available for research use.

**Phase 3 — External validation: ICS-ADD**
- **Full name:** ICS Attack Detection Dataset (ICS-ADD)
- **Content:** Real-hardware testbed data including False Data Injection (FDI), Malware,
  MITM, and DoS attacks with recorded physical consequences.
- **Role in project:** Held-out evaluation only; tests whether graph patterns learned
  from ICS-Flow generalise to a different hardware environment (RQ3).
- **Licence:** Publicly available for research use.

**Supplementary dataset (Week 3 graph construction): BATADAL**
- **Full name:** Battle of the Attack Detection Algorithms Dataset
- **Source:** http://www.batadal.net/data.html (dataset04)
- **Content:** 4,177 hourly timesteps of a water distribution system; 43 sensors/actuators
  (tank levels, pump flows, pressures, valve states); 5.2 % attack-labelled rows.
- **Role in project:** Validates the physical-layer graph construction pipeline on a
  well-known benchmark before applying the same approach to ICS-Flow PLC snapshots.
- **Licence:** Public benchmark, freely downloadable.

---

### 4.2 Approach

The pipeline follows three sequential phases, described below.

#### Phase 1 — Dual-Layer Knowledge Graph Construction (ICS-Flow)

**Cyber layer (network graph):**
Each unique IP address becomes a node. A directed, weighted edge is added between two
nodes for every observed flow, with the edge weight equal to the flow count and edge
attributes encoding protocol, byte volume, and attack label. A `MultiDiGraph` is built
using NetworkX to preserve multiple attack-type edges between the same node pair.

Node features (per device) are derived by aggregating all flow records associated with
that IP: total flows sent/received, proportion of each attack class in outgoing traffic,
dominant protocol, and mean/max byte counts.

**Physical layer (process graph):**
PLC snapshot data (`snapshots_PLC1.csv`, `snapshots_PLC2.csv`) is used to add sensor and
actuator nodes. A `controls` edge is drawn from each PLC to the sensors/actuators it
manages. Sensor nodes carry time-series summary features (mean, standard deviation, and
deviation ratio between normal and attack periods). Anomalous sensors (those deviating
more than 2 standard deviations from their normal mean during attack periods) are flagged
as high-risk nodes.

**Dual-layer integration:**
The two graphs are merged: PLC nodes appear in both layers. A `monitors` edge connects
each HMI node to the PLCs it supervises. The resulting heterogeneous knowledge graph
captures both the cyber topology and the physical consequence structure.

#### Phase 2 — GNN Training for Attack Classification and Path Prediction

Three GNN architectures will be evaluated and compared:

| Model | Rationale |
|---|---|
| GCN (Kipf & Welling 2017) | Baseline; uniform neighbourhood aggregation |
| GraphSAGE (Hamilton et al. 2017) | Inductive; handles unseen nodes; scalable |
| GAT (Veličković et al. 2018) | Attention weights highlight high-risk edges |

**Node classification task (RQ1/RQ2):** Each node is labelled with the dominant attack
class observed in its outgoing traffic (Normal, DDoS, MITM, Port Scan, Replay, IP Scan).
The GNN is trained to predict this label from node features and neighbourhood structure.

**Edge classification / attack-path prediction (RQ2):** Using Cyber4OT propagation
labels, each directed edge is annotated with the attack stage it participates in. A link-level prediction head will be explored to identify which edges form part of a multi-stage attack path, depending on the availability and quality of propagation labels.

**Physical-impact estimation (RQ3):** An optional regression component will be investigated to predict the deviation ratio of downstream sensors given a detected attack on a cyber node. This is
evaluated with and without physical-layer node features to isolate the contribution of
the physical process graph.

All models are implemented in PyTorch Geometric. The graph is split 70 % train /
15 % validation / 15 % test at the node level, with stratification by attack class to
handle class imbalance. Class weighting is applied during training (ICS-Flow is
approximately 66 % Normal).

#### Phase 3 — External Validation (ICS-ADD)

The trained models are applied without retraining to graphs constructed from ICS-ADD.
Metrics are compared to Phase 1 test-set results to quantify generalisation degradation.
A qualitative analysis examines which attack types transfer well and which do not.

---

### 4.3 Evaluation Metrics

| Task | Primary Metric | Secondary Metrics |
|---|---|---|
| Node classification | Macro F1 | Per-class F1, Accuracy, ROC-AUC |
| Attack-path prediction | Precision@K (attack paths) | Recall@K, Mean Average Precision |
| Physical-impact estimation | Mean Absolute Error (MAE) | R², per-sensor deviation accuracy |
| Generalisation (ICS-ADD) | Macro F1 drop vs. in-distribution | False Negative Rate on attack classes |

Class imbalance in the test set is handled by reporting macro-averaged (not
micro-averaged) F1, which weights each class equally regardless of frequency.

---

### 4.4 Tooling

| Category | Tool / Library | Version (pinned) |
|---|---|---|
| Graph construction | NetworkX | ≥ 3.3 |
| GNN framework | PyTorch Geometric (PyG) | ≥ 2.5 |
| Deep learning backend | PyTorch | ≥ 2.3 |
| Graph database | Neo4j (optional, for knowledge graph storage) | ≥ 5.x |
| Data manipulation | Pandas, NumPy | latest stable |
| Statistical analysis | SciPy | latest stable |
| Visualisation | Matplotlib, Seaborn | latest stable |
| Dashboard / demo | Streamlit | ≥ 1.35 |
| Environment | Python 3.10, Google Colab (T4 GPU) / University HPC | — |
| Version control | Git, GitHub (AI-Security-Internships-2026 org) | — |

Pinned versions are recorded in `requirements.txt` in the repository root.

---
## 5. Expected Outcome

The primary deliverable is a working prototype implementing the core graph-construction and GNN-analysis pipeline. The prototype will focus on dual-layer graph construction from ICS-Flow, node feature generation, GNN-based attack classification, and graph-based visualisation. Advanced components such as attack-path prediction, physical-impact estimation, and external validation using additional datasets will be included where project progress, dataset availability, and integration complexity permit.

The prototype will be evaluated on the ICS-Flow dataset (in-distribution) and, where feasible, on ICS-ADD as an out-of-distribution validation dataset.

Secondary deliverables include:

- `docs/literature-review.md` — a systematic review of ICS intrusion detection, graph-based cybersecurity analysis, knowledge graphs, and Graph Neural Networks.
- `experiments/results/` — trained model checkpoints, evaluation metrics, confusion matrices, and visualisation outputs.
- `docs/final-report.md` — a technical report structured as a short research paper including methodology, experiments, results, discussion, limitations, and future work.
- A final presentation summarising the project's methodology, findings, and lessons learned.

The project aims to achieve competitive classification performance while providing interpretable graph-based analysis of ICS attack behaviour. A key outcome will be the demonstration of how graph representations can reveal relationships between network activity and physical-process components that are difficult to observe using conventional flow-based approaches alone.

---

## 6. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|--------|------------|----------|------------|
| Cyber4OT or ICS-ADD unavailable or difficult to integrate | Medium | High | Prioritise ICS-Flow as the primary dataset; use additional datasets only after the core pipeline is operational. |
| Class imbalance affects model performance | High | Medium | Apply class weighting, balanced sampling techniques, and report macro-averaged metrics. |
| Graph structure too sparse for effective GNN learning | Medium | Medium | Enrich graph construction with protocol, communication-frequency, and physical-process relationships. |
| Computational limitations during model training | Low | Medium | Use Google Colab GPU resources and graph sampling techniques where necessary. |
| Project scope exceeds the available timeline | High | Medium | Prioritise graph construction and node classification as core objectives. Advanced tasks such as attack-path prediction, physical-impact estimation, and external validation will be treated as stretch goals if required. |
| Dataset schema differences complicate graph generation | Medium | Low | Develop dataset-agnostic preprocessing and graph-construction modules. |
| Limited generalisation to unseen environments | Medium | Medium | Report findings transparently and analyse environment-specific graph characteristics as part of the evaluation. |

---

## 7. Weekly Deliverable Plan

| Week | Branch | Key Output |
|--------|--------|------------|
| 1 | `week-01` | Repository setup, environment configuration, literature search initiated |
| 2 | `week-02` | Literature review draft completed; ICS-Flow dataset exploration and preprocessing |
| 3 | `week-03` | Research proposal finalised; attack-flow graph construction and visualisation; BATADAL graph exploration |
| 4 | `week-04` | Dual-layer graph construction from ICS-Flow and PLC snapshot data; node feature engineering |
| 5 | `week-05` | Baseline GNN model (GCN or GraphSAGE) implemented and evaluated |
| 6 | `week-06` | Core graph-analysis pipeline completed; additional GNN architectures evaluated where feasible |
| 7 | `week-07` | Advanced experiments including attack-path analysis and external validation, subject to successful dataset integration |
| 8 | `week-08` | Final report, presentation materials, dashboard prototype, and project submission |


---


*Last updated: 2026-06-26*
