# Literature Review: Graph-Based Risk Assessment for Cyber-Physical Systems

**Student:** Misbah Shaheen
**Updated:** 2026-07-03

---

## Instructions

For each paper or resource you read, complete one entry below.
Aim for at least **10 papers** by the end of Week 2.
Use Google Scholar, IEEE Xplore, ACM DL, arXiv, or USENIX Security.


## Paper Summary Template

---

### Paper 1 — Cyber-Physical GNN-Based Intrusion Detection in Smart Power Grids

| Field | Content |
|---|---|
| **Full title** | Cyber-Physical GNN-Based Intrusion Detection in Smart Power Grids |
| **Authors** | Sweeten, J., Takiddin, A., Ismail, M., Refaat, S. S., & Atat, R. |
| **Year** | 2023 |
| **Venue** | IEEE International Conference on Communications, Control, and Computing Technologies for Smart Grids (SmartGridComm 2023) |
| **URL / DOI** | https://ieeexplore.ieee.org/document/10333949 |
| **Method** | Models a power grid as a connected, weighted graph with heterogeneous physical nodes (substations) and cyber nodes (routers), connected by intra- and inter-layer edges; applies GNN layers for a multi-modal IDS that fuses cyber and physical data |
| **Dataset** | Practical cyber-physical testbed (OPAL-RT and a cyber range) with synchronized cyber and physical measurements; benign operation plus false data injection, ransomware, brute force, reverse shell, and backdoor attacks |
| **Key result** | GNN-based IDS outperforms all three benchmark models by exploiting spatial and temporal correlations across the coupled cyber-physical graph |
| **Limitation** | No knowledge graph layer — the graph is built directly from sensor/network topology, with no ontology, CVE grounding, or MITRE ATT&CK integration. No attack-path or propagation-likelihood prediction — the model only classifies intrusion vs. normal at a single timestep. Evaluated on simulated grid data only, never on real ICS hardware. No physical process impact modeling beyond the detection signal itself. |
| **Relevance to our project** | Earliest and simplest proof that a GNN can fuse cyber and physical data for cyber-physical intrusion detection. Establishes the heterogeneous node/edge graph construction pattern (physical assets + cyber assets, domain-weighted edges) that our Neo4j-to-GNN pipeline adapts. |

**Notes / Quotes:**
> Inter-edges between cyber and physical layers are weighted by line admittance values — a direct encoding of domain knowledge into the graph. Our project can use a similar domain-weighted edge scheme for ICS process-variable relationships.

---

### Paper 2 — Heterogeneous GNN with Express Edges for Intrusion Detection in Cyber-Physical Systems

| Field | Content |
|---|---|
| **Full title** | Heterogeneous GNN with Express Edges for Intrusion Detection in Cyber-Physical Systems |
| **Authors** | Li, H. & Chasaki, D. |
| **Year** | 2024 |
| **Venue** | IEEE International Conference on Computing, Networking and Communications (ICNC 2024), pp. 523–529 |
| **URL / DOI** | https://ieeexplore.ieee.org/document/10556029 |
| **Method** | Introduces "Express Edges" — direct connections between non-adjacent nodes in a heterogeneous CPS graph — to capture multi-hop attack propagation that standard message-passing misses |
| **Dataset** | ToN-IoT, NF-BoT-IoT, GraSec-IoT |
| **Key result** | Outperforms standard homogeneous GNN baselines on all three datasets by explicitly modeling structurally distant node relationships |
| **Limitation** | Evaluated on general IoT/IT datasets, not ICS/SCADA protocol traffic — no Modbus, DNP3, or PLC-level data. No knowledge graph layer; Express Edges are manually hand-defined rather than derived from a CVE/ATT&CK ontology. Detects intrusions only — does not output a ranked attack path or propagation probability. No physical-process consequence modeling. |
| **Relevance to our project** | Direct architectural improvement on Paper 1's graph design — solves the specific weakness of standard GNNs missing multi-hop attack chains. The Express Edges technique is a candidate component for modeling how a compromise at one ICS asset (e.g., a field sensor) can affect a structurally distant asset (e.g., a PLC) without an intermediate hop in the raw network graph. |

**Notes / Quotes:**
> Standard GNNs miss attack paths that skip intermediate nodes; Express Edges shortcut these paths — directly relevant to multi-hop ICS attack chains our project needs to capture between non-adjacent assets.

---

### Paper 3 — GNN-IDS: Graph Neural Network Based Intrusion Detection System

| Field | Content |
|---|---|
| **Full title** | GNN-IDS: Graph Neural Network Based Intrusion Detection System |
| **Authors** | Sun, Z., Teixeira, A. M. H., & Toor, S. |
| **Year** | 2024 |
| **Venue** | ACM ARES 2024 (19th International Conference on Availability, Reliability and Security), Vienna, Austria |
| **URL / DOI** | https://doi.org/10.1145/3664476.3664515 |
| **Method** | Incorporates a static attack graph (generated with the MulVAL tool) alongside live network measurements into a single GNN; the attack graph supplies structural context while live data supplies dynamic state |
| **Dataset** | A synthetic dataset and a second dataset derived from CICIDS-2017 |
| **Key result** | Detects anomalies and identifies the specific malicious action that caused them — explainability is built directly into the architecture, evaluated via uncertainty, explainability, and robustness analysis |
| **Limitation** | The attack graph is generated for general enterprise IT topology (via MulVAL), not ICS-specific assets or protocols — no PLC, RTU, or HMI node types. No formal knowledge graph (Neo4j/ontology) layer; the attack graph is a flat structural artifact, not a queryable, continuously-updated knowledge base. No physical process impact prediction — explainability stops at "which action caused this," not "what physical consequence follows." |
| **Relevance to our project** | The closest existing architecture to our overall pipeline: a static graph (our knowledge graph) fused with a GNN over live data (our detection layer) to produce explainable output. This is the direct baseline our KG-to-GNN fusion design is benchmarked against. |

**Notes / Quotes:**
> "By incorporating an attack graph, GNN-IDS could not only detect anomalies but also identify the malicious actions causing them" — the explainability property central to our project's value proposition.

---

### Paper 4 — Graph Neural Network-Based Attack Prediction for Communication-Based Train Control Systems (GNN-AP)

| Field | Content |
|---|---|
| **Full title** | Graph Neural Network-Based Attack Prediction for Communication-Based Train Control Systems |
| **Authors** | Zhao, J., Tang, T., Bu, B., et al. |
| **Year** | 2024 |
| **Venue** | CAAI Transactions on Intelligence Technology (Wiley) |
| **URL / DOI** | https://doi.org/10.1049/cit2.12288 |
| **Method** | Builds an Attack Scenario Graph from multi-step APT alert sequences using an encoder-decoder model and time-based DFS, then combines this with CBTC system topology to form an Attack-Target Graph; converts attack *prediction* into a link-prediction task solved with a GNN |
| **Dataset** | DARPA 1999/2000 IDS alert logs for training/evaluation; a CBTC system simulation dataset for validation |
| **Key result** | Successfully predicts the attacker's next target via link prediction on the attack-target graph; reduces false alarms by incorporating causal relationships between alerts |
| **Limitation** | Domain-specific to rail signaling (CBTC) — not generalized to ICS/SCADA sectors such as water treatment, manufacturing, or power. No knowledge graph layer; the attack-target graph is purpose-built per incident, not a persistent, queryable CVE/ATT&CK-grounded KG. Predicts the *next attack target*, not the physical-process consequence of reaching it. Training data (DARPA 1999/2000 alert logs) is decades-old IT data, even though the CBTC validation set is current. |
| **Relevance to our project** | The core technical method for our "predict attack paths and propagation likelihood" objective — link prediction on a graph is directly implementable with PyTorch Geometric and is the most concrete precedent for our propagation-prediction layer. |

**Notes / Quotes:**
> "GNN is used to identify the attack intent from the attack target graph and convert the attack target prediction task into link prediction" — the exact reframing our project applies to ICS asset graphs.

---

### Paper 5 — Graph-Theoretic Approach for Manufacturing Cybersecurity Risk Modeling and Assessment

| Field | Content |
|---|---|
| **Full title** | Graph-Theoretic Approach for Manufacturing Cybersecurity Risk Modeling and Assessment (published as "Taxonomy-Driven Graph-Theoretic Framework for Manufacturing Cybersecurity Risk Modeling and Assessment") |
| **Authors** | Rahman, M. H., Yazdandoost Hamedani, E., Son, Y.-J., & Shafae, M. |
| **Year** | 2023 (arXiv preprint) / 2024 (journal version) |
| **Venue** | ASME Journal of Computing and Information Science in Engineering (JCISE), 24(7); arXiv:2301.07305 |
| **URL / DOI** | https://arxiv.org/abs/2301.07305 |
| **Method** | Taxonomy-driven cyber-physical attack graph framework for Industry 4.0 manufacturing systems; threat attributes are derived from manufacturing cyberattack taxonomies and represented as an attack graph; graphs are analyzed (rule-based, not learned) to trace propagation and quantify risk |
| **Dataset** | Manufacturing cybersecurity threat taxonomies and an illustrative case study (no live network traffic) |
| **Key result** | Generates comprehensive cyber-physical attack graphs, identifies the highest-likelihood attack path, and ranks critical manufacturing assets requiring prioritized security controls |
| **Limitation** | No GNN or any learning component — graph analysis is entirely rule-based, so it cannot generalize to unseen attack patterns the way a trained model can. No real network or sensor data — validated only on an illustrative case study. No real-time monitoring capability. |
| **Relevance to our project** | The earliest-stage foundation in this review: establishes that attack graphs are a valid structure for representing ICS risk propagation, before any learning is introduced. This is the conceptual starting point our knowledge graph design builds on. |

**Notes / Quotes:**
> "The graphical approach helps model the interdependence of threat attributes, and graphs are analyzed to explore how threat events can propagate through the manufacturing value chain."

---

### Paper 6 — Automated Knowledge-Based Cybersecurity Risk Assessment of Cyber-Physical Systems (Spyderisk)

| Field | Content |
|---|---|
| **Full title** | Automated Knowledge-Based Cybersecurity Risk Assessment of Cyber-Physical Systems |
| **Authors** | Phillips, S. C., Taylor, S., Boniface, M., Modafferi, S., & Surridge, M. |
| **Year** | 2024 |
| **Venue** | IEEE Access, 12, 82482–82505 |
| **URL / DOI** | https://eprints.soton.ac.uk/490296/2/Automated_Knowledge-Based_Cybersecurity_Risk_Assessment_of_Cyber-Physical_Systems.pdf |
| **Method** | Ontology/knowledge-graph-driven automated risk assessment aligned with the ISO 27005 standard; models trust relationships and threat propagation across multi-stakeholder ICT/CPS systems using the open-source Spyderisk software |
| **Dataset** | A published real-world case study — the German steel mill cyberattack — modeled and reproduced in Spyderisk |
| **Key result** | Automatically identifies attack paths, threat propagation, and risk levels; the shortest, highest-likelihood attack path it finds coincides with the originally published human analysis of the steel mill attack |
| **Limitation** | No GNN or machine learning component at all — reasoning is purely ontology- and rule-based, so it cannot learn from new data or improve with more examples. Operates on modeled scenarios rather than live, continuously-ingested ICS traffic. No attack-path *prediction* for unseen configurations — it only reproduces and analyzes scenarios that have been explicitly modeled by an analyst. |
| **Relevance to our project** | The knowledge-graph counterpart to Paper 5: proves that a well-designed ontology can automate standards-aligned (ISO 27005) risk reasoning over a cyber-physical system. Our Neo4j knowledge graph schema is modeled on the same structural classes (Assets, Relations, Threats, Controls, Consequences) Spyderisk defines. |

**Notes / Quotes:**
> "This paper describes a simulation-based approach for automated risk assessment of complex cyber-physical systems to support implementers of ISO 27005."

---

### Paper 7 — Physics-Informed Graph Neural Networks for Attack Path Prediction (PIGNN)

| Field | Content |
|---|---|
| **Full title** | Physics-Informed Graph Neural Networks for Attack Path Prediction |
| **Authors** | François, M., Arduin, P.-E., & Merad, M. |
| **Year** | 2025 |
| **Venue** | Journal of Cybersecurity and Privacy, 5(2), 15 |
| **URL / DOI** | https://doi.org/10.3390/jcp5020015 (replication package: https://github.com/mbdlrocks/PhD_Replication_Package) |
| **Method** | Combines physics-informed neural network constraints with a GNN for full attack-path prediction, tested in inductive, deductive, and hybrid settings; introduces a self-supervised module for initial-access and impact prediction |
| **Dataset** | A purpose-built dataset of 1,033 graph-based representations of Active Directory environments, with attack paths stored as binary adjacency matrices |
| **Key result** | F1 = 0.93 for full attack-path prediction; F1 = 0.978 for initial-access prediction; F1 = 0.82 for impact prediction — substantially ahead of standard (non-physics-informed) GNN baselines |
| **Limitation** | Trained and evaluated entirely on Active Directory (enterprise identity/IT) graphs — no ICS assets (PLC, RTU, HMI), no SCADA protocols, no industrial topology of any kind. "Physics-informed" here means generic structural/network constraints, not real physical-process equations (pressure, flow, voltage) — so it does not model physical process impact in the way our project requires. No knowledge graph or CVE/ATT&CK grounding; node types are fixed to AD entities (User, Computer, OU, Group, GPO, Domain). |
| **Relevance to our project** | The most advanced attack-path prediction architecture in this review — moves beyond single-link prediction (Paper 4) to full-path prediction with physics-style constraints, directly foreshadowing the physical-process-aware propagation model our project needs. We adapt this architecture's design, not its dataset, re-mapping node/edge types to ICS assets. |

**Notes / Quotes:**
> "To ensure the reproducibility of our findings, all data and algorithms presented in this paper can be found in the replication package on GitHub" — code is publicly available for direct architectural reference.

---

### Paper 8 — BRIDG-ICS: AI-Grounded Knowledge Graphs for Intelligent Threat Analytics in Industry 5.0 Cyber-Physical Systems

| Field | Content |
|---|---|
| **Full title** | BRIDG-ICS: AI-Grounded Knowledge Graphs for Intelligent Threat Analytics in Industry 5.0 Cyber-Physical Systems |
| **Authors** | Nandiya, P., Mohsin, A., Ibrahim, A., Sarker, I. H., & Janicke, H. |
| **Year** | 2025 (arXiv preprint) → 2026 (Springer Cybersecurity journal) |
| **Venue** | arXiv:2512.12112 [cs.CR] → Springer Cybersecurity (2026) |
| **URL / DOI** | https://arxiv.org/abs/2512.12112 |
| **Method** | Fuses CVE, CWE, CAPEC, MITRE ATT&CK for ICS, and live OT telemetry (OPC UA, Purdue Model data) into a single, continuously-updated Industrial Security Knowledge Graph; simulates multi-stage attack paths over the graph and computes probabilistic risk metrics (exploit likelihood, attack cost) per asset |
| **Dataset** | CVE/CWE/CAPEC/MITRE ATT&CK for ICS feeds plus synthetic OPC UA and sensor telemetry from a modular ICS testbed |
| **Key result** | Cross-domain reasoning over the unified KG surfaces threat insight siloed defenses cannot provide; anchoring detections to asset-vulnerability relationships reduces false positives in risk scoring |
| **Limitation** | No GNN or learning component at all — risk propagation is computed via graph-analytic simulation, not a trained model, so it cannot improve from labeled attack outcomes or generalize the way a GNN can. Not real-time — designed for periodic risk assessment, not live intrusion detection. No public dataset or code repository confirmed, limiting direct reproducibility. |
| **Relevance to our project** | The most complete and most recent prior work in this review, and the closest match to our overall research problem: a full ICS-specific knowledge graph integrating vulnerability, technique, and live telemetry data with attack-path simulation and risk scoring. Our project's central contribution is adding the GNN learning layer this paper does not have — taking BRIDG-ICS's KG completeness and combining it with the GNN-based prediction methods developed in Papers 1–4 and 7. |

**Notes / Quotes:**
> ICS components that were once isolated now routinely interface with enterprise IT systems and cloud-based platforms — this gradual integration expands the attack surface and weakens the traditional OT air-gap security model, motivating the need for unified KG-based threat analytics.

---
---
 
### Paper 9 — ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation
 

 
| Field | Content |
|---|---|
| **Full title** | ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation |
| **Authors** | Wu, Y., Velazco, M., Zhao, A., Meléndez Luján, M. R., Movva, S., Roy, Y. K., Nguyen, Q., Rodriguez, R., Wu, Q., Albada, M., Kiseleva, J., & Mudgerikar, A. (Microsoft) |
| **Year** | 2025 (v1: July 2025) / updated May 2026 (v3) |
| **Venue** | arXiv:2507.14201 [cs.CR] — open access, CC BY 4.0 |
| **URL / DOI** | https://arxiv.org/abs/2507.14201 |
| **Method** | Constructs bipartite alert–entity investigation graphs from 8 real multi-stage attack chains across 59 Azure security log tables; generates multi-hop Q&A pairs from graph paths mapped to MITRE ATT&CK tactics; evaluates LLM agents on traversing these graphs to investigate and classify threats |
| **Dataset** | Microsoft Azure tenant security logs — 8 real attack chains including ransomware, lateral movement, and credential theft; 59 distinct log table types |
| **Key result** | Average reward across tested models ≈ 0.249 (best model o4-mini ≈ 0.368) in the original benchmark; later updates report GPT-5 high-reasoning variants outperforming lower-reasoning configurations, though task performance remains far from saturated — demonstrating that LLM agents traversing multi-hop attack graphs without structured pre-processing perform poorly, even at frontier model scale. |
| **Limitation** | Enterprise IT only (Azure cloud tenant) — no ICS, SCADA, PLC, or OT protocol context. LLM queries a SQL database directly, not a GNN output — the GNN+LLM fusion our project proposes is not evaluated here. Benchmark paper only, not a deployed production system. |
| **Relevance to our project** | The low recall result (3.82%) directly motivates our project's GNN pre-processing layer: a GNN that compresses multi-hop ICS attack paths into structured embeddings before LLM reasoning would address ExCyTIn's core finding. The bipartite alert–entity graph structure maps onto our KG node types (Asset, Alert, Technique). The ICS adaptation — replacing Azure cloud entities with PLC/HMI/SCADA nodes — is the architectural gap our GNN+LLM pipeline is designed to fill. Directly supports the LLM layer added to project scope by supervisor. |
 
**Notes / Quotes:**
> "Real-world security analysts must sift through a large number of heterogeneous alert signals and security logs, follow multi-hop chains of evidence, and compile an incident report" — the exact analyst workflow our LLM layer is intended to automate for ICS operators.
 
---
 
### Paper 10 — Spatio-Temporal Attention GNN: Explaining Causalities with Attention (STA-GNN)
 

 
| Field | Content |
|---|---|
| **Full title** | Spatio-Temporal Attention Graph Neural Network: Explaining Causalities with Attention |
| **Authors** | Koistinen, K., Hellsten, K., Kaski, K. K., & Herttuainen, J. (Aalto University, Finland) |
| **Year** | 2026 |
| **Venue** | arXiv:2603.10676 [cs.CR] — open access |
| **URL / DOI** | https://arxiv.org/abs/2603.10676 |
| **Method** | Unsupervised GNN combining temporal self-attention (Transformer-style) with dynamic spatial graph attention for multi-modal ICS anomaly detection; ICS sensors, PLCs, and actuators are represented as graph nodes; learns inter-device dependency structure dynamically rather than from a fixed topology; produces explainable attention graphs showing which device relationships caused a detected anomaly; tested on physical SCADA data, NetFlow, and CIP payload simultaneously |
| **Dataset** | SWaT 2015, 2017, 2019 datasets (physical-level SCADA sensor data + NetFlow + CIP payload from a real water-treatment ICS testbed) |
| **Key result** | F1 = 0.77 on SWaT 2015 physical-level data; conformal prediction thresholding reduces false-positive rate to FPR ≈ 0.001; attention graphs correctly identify causal propagation paths in 60–75% of detected attacks |
| **Limitation** | Evaluated exclusively on SWaT water-treatment benchmark — not tested on Modbus/DNP3 or manufacturing ICS protocols. No knowledge graph or CVE/ATT&CK integration. Detects and explains anomalies but does not predict where an attack will propagate next. Authors acknowledge significant concept drift between SWaT dataset versions, limiting long-term deployment without periodic retraining. |
| **Relevance to our project** | Directly addresses two of our core objectives: GNN-based detection on real ICS data (PLCs, SCADA, sensors as graph nodes) and explainability of which device relationships caused an alert. The attention graph output — causal pathways between ICS components — is structurally the same as the propagation edges in our knowledge graph. The SWaT F1 = 0.77 result provides a concrete baseline for comparing our own GNN results in Week 7. Crucially, the paper's future work explicitly proposes integrating LLM reasoning over attention graphs for non-expert users — exactly the LLM layer our supervisor has added to project scope. |
 
**Notes / Quotes:**
> "As future work, we aim to integrate the learned attention structures with large language models (LLMs) to further enhance explainability, particularly for non-expert users." — This is precisely the three-layer architecture (KG → GNN → LLM) our project is now implementing for ICS.
 
---
 
### Paper 11 — MITRE ATT&CK for ICS (Living Framework, Current Version v19)
 

 
| Field | Content |
|---|---|
| **Full title** | MITRE ATT&CK for ICS — Tactics, Techniques, and Procedures for Industrial Control Systems |
| **Authors** | MITRE Corporation |
| **Year** | Continuously updated — current version v19 (April 2026); ICS matrix first released 2020 |
| **Venue** | MITRE ATT&CK (official framework, not a peer-reviewed paper) |
| **URL** | https://attack.mitre.org/matrices/ics/ |
| **Method** | Structured taxonomy of adversary TTPs specific to ICS/OT environments organised into 12 tactics. v19 (April 2026) added ICS sub-techniques for the first time (18 sub-techniques across 79 techniques). Machine-readable STIX data available via GitHub (`mitre-attack/attack-stix-data`). Ingested in Python via the `mitreattack-python` library. Evidence base is real-world ICS incidents including TRITON, Industroyer, and the 2015/2016 Ukraine power grid attacks |
| **Dataset** | Real-world ICS incidents used as evidence base; 14 tracked threat groups (e.g., Sandworm, XENOTIME); 8 documented campaigns |
| **Key result** | Authoritative, community-validated ICS attack taxonomy: 12 tactics, 79 techniques, 18 sub-techniques (v19), 52 mitigations, 18 assets, 14 groups. Sub-techniques were added in v19 — a significant structural change from v16 (October 2024) which had 83 techniques and 0 sub-techniques |
| **Limitation** | Descriptive taxonomy only — no risk scoring, no graph structure, no learned model. Coverage is limited to publicly disclosed incidents; novel or undisclosed attack techniques are not represented until after public disclosure. Not a detection system. |
| **Relevance to our project** | Primary source for Technique nodes and tactic-level edge labels in the project's Neo4j knowledge graph. Every ATT&CK for ICS technique (T0800-series) maps directly to a Technique node; tactic sequences (e.g., Reconnaissance → Lateral Movement → Impair Process Control → Impact) become directed edges encoding known attack progressions. The STIX-format machine-readable data is ingested via the `mitreattack-python` library. Also used as the ground-truth label source for mapping detected GNN anomalies to named ICS techniques, which the LLM layer then translates into natural-language analyst reports. |

 ---
 
## Reference Table (Quick Overview)

| # | Title (short) | Authors | Year | Method | Dataset | Relevance |
|---|---|---|---|---|---|---|
| 1 | CPS GNN IDS — Smart Grids | Sweeten et al. | 2023 | Heterogeneous layered GNN fusing cyber + physical graph layers | OPAL-RT hardware-in-the-loop cyber-physical testbed | Earliest proof a GNN can fuse cyber and physical data for CPS detection; source of our heterogeneous node/edge graph design |
| 2 | Heterogeneous GNN + Express Edges | Li & Chasaki | 2024 | Heterogeneous GNN with non-adjacent "Express Edges" for multi-hop propagation | ToN-IoT, NF-BoT-IoT, GraSec-IoT | Candidate technique for modeling propagation between structurally distant ICS assets (e.g., sensor → distant PLC) |
| 3 | GNN-IDS + Attack Graph | Sun, Teixeira & Toor | 2024 | GNN fused with a static MulVAL attack graph for explainable detection | Synthetic dataset + CICIDS-2017-derived dataset | Closest existing architecture to our KG-to-GNN fusion pipeline; direct baseline for our detection layer |
| 4 | GNN-AP (CBTC) | Zhao, Tang, Bu et al. | 2024 | GNN + link prediction on an Attack-Target Graph built from APT alert sequences + topology | DARPA 1999/2000 IDS alert logs + CBTC simulation dataset | Core technical method for our "predict attack paths and propagation likelihood" objective |
| 5 | Graph-Theoretic Manufacturing Risk | Rahman et al. | 2023/2024 | Taxonomy-driven, rule-based cyber-physical attack graph | Manufacturing cybersecurity taxonomies + illustrative case study | Foundational attack-graph construction methodology our knowledge graph design builds on |
| 6 | Automated KG Risk Assessment (Spyderisk) | Phillips et al. | 2024 | Ontology/knowledge-graph-driven risk assessment aligned with ISO 27005 | German steel mill cyberattack case study (Spyderisk) | Structural blueprint for our Neo4j knowledge graph schema (Assets, Relations, Threats, Controls, Consequences) |
| 7 | Physics-Informed GNN (PIGNN) | François et al. | 2025 | Physics-informed constraints + GNN for full attack-path prediction | 1,033 synthetic Active Directory environment graphs | Most advanced attack-path architecture reviewed; open-source code adapted for our physical-process-aware propagation model |
| 8 | BRIDG-ICS Knowledge Graph | Nandiya et al. | 2025/2026 | AI-grounded knowledge graph fusing CVE/CWE/CAPEC/ATT&CK for ICS + OT telemetry | CVE/CWE/CAPEC/ATT&CK for ICS feeds + synthetic OPC UA testbed telemetry | Most complete prior work; our project adds the GNN learning layer this KG-only approach lacks |
| 9 | ExCyTIn-Bench — LLM Agents on Threat Investigation | Wu et al. (Microsoft) | 2025/2026 |LLM agents traversing bipartite alert–entity graphs for multi-hop threat investigation; attack stages are labeled with MITRE ATT&CK techniques | Motivates GNN pre-processing before LLM reasoning; bipartite alert–entity graph maps onto our KG node types; directly supports LLM layer |
| 10 | STA-GNN — ICS Explainable Anomaly Detection | Koistinen et al. (Aalto) | 2026 | Unsupervised spatio-temporal GNN with attention-based explainability on real ICS data (PLCs, SCADA sensors) | SWaT 2015/2017/2019 (physical SCADA + NetFlow + CIP payload) | ICS-specific GNN on real SCADA/PLC data with causal propagation graphs; F1=0.77 baseline; future work explicitly proposes GNN+LLM integration |
| 11 | MITRE ATT&CK for ICS (v19) | MITRE Corporation | 2026 | Living ICS attack taxonomy — 12 tactics, 79 techniques, 18 sub-techniques; machine-readable STIX | Real-world ICS incidents (TRITON, Ukraine grid, Industroyer) | Primary source for Technique nodes and tactic-sequence edge labels in the Neo4j KG; ground-truth label source for LLM-generated analyst reports |
 
---
 
*None of the papers reviewed combine a GNN, a formal ICS-specific knowledge graph, attack-path/propagation prediction, and an LLM reasoning layer in one system. This points to a gap among the sources surveyed, though it is not an exhaustive claim across the literature.*
 
---

## Tools and Datasets Identified

| Name | Type | URL | Notes |
|---|---|---|---|
| ICS-Flow | Dataset | https://www.kaggle.com/datasets/alirezadehlaghi/icssim | Network flows + process variable logs from a simulated bottle-filling plant. Primary dataset for knowledge graph construction and GNN training. IEEE Access, 2023. |
| Cyber4OT | Dataset | https://github.com/krzysztof-cabaj/Cyber4OT | Multi-stage attack traces on real PLCs in a lab testbed. Candidate source for attack-path edge labels. SoftwareX, 2025. |
| ICS-ADD | Dataset | https://ieee-dataport.org/documents/ics-add-smart-industry-testbed-dataset-cyber-physical-security-monitoring-testing | Real SCADA/PLC hardware traffic with SIEM alerts. Used for supplementary, out-of-distribution evaluation. IEEE Access, 2024. |
| `sepses/ics-sec-kg` | Repository | https://github.com/sepses/ics-sec-kg | ICS cybersecurity knowledge graph ontology (MITRE ATT&CK for ICS, CISA advisories, CVE). Used as a schema reference for the Neo4j design; engine is Java, not run directly. ISWC 2024. |
| `zhenlus/GNN-IDS` | Repository | https://github.com/zhenlus/GNN-IDS | GNN fused with a static attack graph for explainable detection. Architectural baseline for the knowledge-graph-to-GNN pipeline. ARES 2024. |
| `mbdlrocks/PhD_Replication_Package` (PIGNN) | Repository | https://github.com/mbdlrocks/PhD_Replication_Package | Physics-informed GNN for attack-path prediction. Architecture reference, re-mapped from Active Directory to ICS node types. J. Cybersecurity & Privacy, 2025. |
| `lorenzo9uerra/GraphIDS` | Repository | https://github.com/lorenzo9uerra/GraphIDS | Self-supervised GNN for network intrusion detection. Reference for handling scarce labelled ICS attack data. NeurIPS 2025. |
| PyTorch Geometric | Library / Tool | https://github.com/pyg-team/pytorch_geometric | Core GNN implementation framework. |
| Neo4j Graph Data Science | Library / Tool | https://github.com/neo4j/graph-data-science | Graph algorithms (centrality, pathfinding) applied to the knowledge graph. |
| NetworkX | Library / Tool | https://networkx.org | Graph construction and preprocessing before Neo4j ingestion. |
| ICSFlowGenerator / ICSSIM | Library / Tool | https://github.com/AlirezaDehlaghi/ICSFlow | Tools used to generate and extract features for ICS-Flow; available for custom data generation if needed. |
| Streamlit | Library / Tool | https://streamlit.io | Prototype dashboard for visualising results in later weeks. |

---

## Related Tools and Competitors

### Commercial platforms
*(context for production-grade OT security; not directly extensible for research, based on public documentation only)*

**Claroty** — passive ICS network monitoring with automatic asset inventory and behavioural anomaly detection. Public documentation does not indicate a knowledge-graph or attack-path-prediction capability. Closed-source.

**Dragos** — OT-specific threat-intelligence platform with detection playbooks and tracked ICS threat-actor groups. Operates primarily as a signature/intelligence platform per public materials; no graph-based propagation modelling is described.

**Microsoft Defender for IoT** — agentless monitoring that builds a device communication map across common ICS protocols. Per public documentation, this map is used mainly for visualisation in Microsoft Sentinel, not for inference or risk scoring.

### Open-source research tools
*(closer architectural comparators for this project)*

**BloodHound** (`SpecterOps/BloodHound`) — models Active Directory relationships in Neo4j and queries attack paths with Cypher. Structural inspiration for this project's knowledge-graph design, applied to ICS assets rather than AD entities; no ICS protocol awareness or learned inference.

**`sepses/ics-sec-kg`** — an ICS cybersecurity knowledge graph ontology integrating MITRE ATT&CK for ICS, CISA advisories, and CVE data. Used here as a schema reference; performs no learning and produces no risk score.

**`zhenlus/GNN-IDS`** — fuses a static attack graph with a GNN over live traffic for explainable detection on general IT data. Identifies the cause of a detected anomaly (backward-looking), which differs from this project's forward-looking propagation-prediction goal.

**`lorenzo9uerra/GraphIDS`** — self-supervised GNN achieving strong results on IT NetFlow data without labelled attacks. Not evaluated on ICS traffic; relevant mainly for handling scarce labelled data.

**GRFICS v3** — open-source ICS simulation testbed generating labelled Modbus traffic with visible physical-process effects. A data-generation environment, not a detection or KG tool; considered as a supplementary source and set aside in favour of ICS-Flow and Cyber4OT, which provide ready-labelled data.

### Gap summary

| Tool | KG | GNN | Attack-Path Prediction | Physical Impact | ICS Protocols | Open Source |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **This Project** | ✅ Neo4j | ✅ PyG | ✅ (planned) | Partial — KG layer | ✅ | ✅ |
| Claroty | No (per public docs) | No | No | Partial | ✅ | ❌ |
| Dragos | Partial (TI) | No | No | Partial | ✅ | ❌ |
| MS Defender for IoT | Partial (viz only) | No | No | No | ✅ | ❌ |
| BloodHound | ✅ Neo4j | No | Static (Cypher) | No | ❌ | ✅ |
| `sepses/ics-sec-kg` | ✅ RDF | No | No | No | ✅ | ✅ |
| `zhenlus/GNN-IDS` | Partial | ✅ | Detection only | No | ❌ | ✅ |
| `lorenzo9uerra/GraphIDS` | No | ✅ | No | No | ❌ | ✅ |

Based on the tools reviewed, none of the commercial platforms appear, from public documentation, to offer a knowledge-graph or GNN reasoning layer, and none of the open-source tools reviewed combine ICS protocol awareness with both a knowledge graph and a learned, forward-looking attack-path prediction model. This points to a gap among the tools reviewed, though a broader search may surface additional comparators.

---

## Dataset Detail

 
### Dataset A — ICS-Flow
 
| Field | Details |
|---|---|
| Full name | Anomaly Detection Dataset for Industrial Control Systems (ICS-Flow) |
| Authors | Dehlaghi-Ghadim, A., Moghadam, M. H., Balador, A., Hansson, H. |
| Source | Mälardalen University, Sweden / RISE Research Institutes of Sweden |
| Publication | IEEE Access, Vol. 11, pp. 107982–107996 (2023) |
| DOI | 10.1109/ACCESS.2023.3320928 · arXiv:2305.09678 |
| Access | https://www.kaggle.com/datasets/alirezadehlaghi/icssim (open, no request needed) |
| Format | CSV (network flows + process variable logs) + PCAP (raw packets) |
| Scale | 25M+ raw packets; 45,719 labelled flow records; 50 flow features |
| Environment | ICSSIM-simulated bottle-filling process (PLCs, sensors, actuators, HMI, SCADA) |
| Attack types | Benign, DDoS, IP scanning, port scanning, MITM, replay |
 
**Strengths**
- Combines network-flow records with PLC process variable logs from a single environment, supporting a dual-layer knowledge graph (network edges + physical process node features).
- Open-source ICSFlowGenerator tool allows regenerating flows with custom feature sets.
- Dual labelling strategies (IT and NST) support both supervised and unsupervised training.
- Freely downloadable; purpose-built for ICS security research.


**Limitations**
- Generated on a simulated testbed, not an operational plant — dynamics may not fully match production ICS.
- Only 45,719 labelled flows, small relative to enterprise IT benchmarks.
- No false data injection or firmware-tampering attacks in the current release — network-layer attacks only.


**Justification**


Among the datasets reviewed, ICS-Flow is the one that most directly pairs network-flow data with physical process variable logs from the same environment, which maps onto the project's two intended graph layers(network communication and physical process state). This makes it the natural choice as the primary dataset for knowledge graph construction and GNN training, though the dual-layer mapping will need to be validated experimentally during prototyping.
 
---
 
### Dataset B — Cyber4OT
 
| Field | Details |
|---|---|
| Full name | Cyber4OT: Network Traces for Cyber-Security Vulnerability Evaluation in Industrial Control Systems |
| Authors | Cabaj, K., Plamowski, S., Chaber, P., Ławryńczuk, M., Marusak, P., Nebeluk, R., Wojtulewicz, A., Zarzycki, K. |
| Source | Warsaw University of Technology, Poland |
| Publication | SoftwareX (2025), DOI 10.1016/j.softx.2025.102147 |
| Access | https://github.com/krzysztof-cabaj/Cyber4OT (open) |
| Format | PCAP Next Generation (.pcapng) |
| Scale | 96 labelled trace files, 4.25M+ packets, 385 MB |
| Environment | Laboratory ICS testbed, real industrial PLCs, SCADA + HMI, Modbus/TCP |
 
**Attack progression (working interpretation, to be cross-checked against full paper):**
 
| Phase | Description |
|---|---|
| 1 — Reconnaissance | Network scanning, ICS device discovery |
| 2 — Vulnerability analysis | Service enumeration on identified PLCs |
| 3 — Exploitation | Credential attack / protocol exploitation on PLC |
| 4 — Persistence | Persistent control channel established |
| 5 — Full control | Attacker control of target PLC |
 
**Strengths**
- Captures a multi-stage IT→OT attack progression on real PLC hardware — a candidate source for directed edge labels in the knowledge graph.
- Small (385 MB), fully loadable in memory; freely available; peer-reviewed (2025).
- Attack pattern (IT→OT lateral movement to PLC takeover) is broadly consistent with real incidents such as TRITON and the 2015 Ukraine grid attack.


**Limitations**
- Modbus/TCP only — no IEC 60870, DNP3, or OPC-UA coverage.
- Small corpus (96 traces); augmentation likely needed for GNN training.
- Additional attack types (measurement injection, firmware tampering, DoS) are mentioned as future work but not yet released.
- PCAP requires preprocessing (scapy/pyshark) before graph construction.

  
**Justification**


Of the datasets reviewed, Cyber4OT is the one that most clearly documents a full multi-stage attack lifecycle, from reconnaissance to PLC takeover, on real industrial hardware.This sequential structure is the project's intended source of directed propagation-edge labels for attack-path prediction, though the granularity of phase-to-edge mapping needs confirmation during implementation rather than being assumed.
 
---
 
### Dataset C — ICS-ADD
 
| Field | Details |
|---|---|
| Full name | Industrial Control System – Anomaly Detection Dataset (ICS-ADD) |
| Authors | Gaggero, G. B., Armellin, A., Portomauro, G., Marchese, M. |
| Source | University of Genoa, Italy — DITEN Department |
| Publication | IEEE Access, Vol. 12, pp. 64140–64149 (2024), DOI 10.1109/ACCESS.2024.3395991 |
| Access | IEEE DataPort (open) |
| Format | PCAP (raw traffic) + CSV (SCADA events) + CSV (SIEM alerts) |
| Environment | Real, physically wired testbed: SCADA (ScadaBr), PLC (OpenPLC), firewall (PfSense), switch with SPAN port |
| Attack types | DoS, MITM, malware infiltration, false data injection (FDI) |
 
**Strengths**
- Real physical SCADA/PLC hardware — complements the simulated and lab-testbed sources used for training.
- Three data modalities in one dataset: raw traffic, SCADA process events, SIEM alerts.
- Documented FDI consequence: pump activated without HMI command at 12:21:18 — directly relevant to the "physical process impact" objective.
- Compact, practical size for a Week 7 validation pass; open access, no institutional login required.
  
**Limitations**
- Narrower attack variety than ICS-Flow or Cyber4OT (no reconnaissance or multi-stage lateral movement).
- Single vendor/hardware configuration; less diversity than ICS-NAD.
- Short, single-interval capture window rather than multi-day data.

**Justification**

ICS-ADD is used for supplementary, exploratory evaluation on hardware the model was not trained on — an early sanity-check of whether model behaviour holds up outside the training distribution, appropriate in scope to an 8-week prototype. It is not used to claim demonstrated generalisation across industrial hardware.
 
---
 
### Dataset Selection Summary
 
The three datasets were chosen to be complementary rather than overlapping:
 
- **ICS-Flow** — dual-layer node features (network + process) for KG construction and training.
- **Cyber4OT** — multi-stage attack sequences on real PLCs, as the basis for propagation-edge labelling.
- **ICS-ADD** — a real-hardware environment, separate from training data, for supplementary evaluation.
This selection reflects the datasets identified during this project's literature and dataset review and is not presented as an exhaustive survey of every ICS dataset that could serve these roles. A fourth dataset, ICS-NAD, was identified but deferred to future work for practical reasons (scale).
 
