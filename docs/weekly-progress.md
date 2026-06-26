
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