
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
