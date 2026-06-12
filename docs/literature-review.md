# Literature Review: Graph-Based Risk Assessment for Cyber-Physical Systems

**Student:** Misbah Shaheen
**Updated:** 2026-06-12

---

## Instructions

For each paper or resource you read, complete one entry below.
Aim for at least **10 papers** by the end of Week 2.
Use Google Scholar, IEEE Xplore, ACM DL, arXiv, or USENIX Security.

---

## Paper Summary Template

### Paper 1 — Graph-Theoretic Approach for Manufacturing Cybersecurity Risk Modeling

| Field | Content |
|---|---|
| **Full title** | Graph-Theoretic Approach for Manufacturing Cybersecurity Risk Modeling and Assessment |
| **Authors** | Md Habibor Rahman, Erfan Yazdandoost Hamedani, Young-Jun Son, Mohammed Shafae (University of Arizona & Purdue University) |
| **Year** | 2023 |
| **Venue** | ASME Journal of Computing and Information Science in Engineering (JCISE); also available on arXiv (2301.07305) |
| **URL / DOI** | https://arxiv.org/pdf/2301.07305 |
| **Method** |Taxonomy-driven cyber-physical attack graph framework for modeling attack propagation and cybersecurity risk in Industry 4.0 manufacturing systems|
| **Dataset** | Manufacturing cybersecurity taxonomies and illustrative case study |
| **Key result** | Generates comprehensive cyber-physical attack graphs, identifies attack paths, estimates attack likelihood and risk, and determines critical manufacturing assets requiring prioritized security controls |
| **Limitation** | Does not apply GNNs — graph analysis is rule-based; focused on discrete manufacturing, may not generalise to all ICS sectors; no real-time monitoring capability |
| **Relevance to our project** | Directly addresses Industry 4.0 ICS risk modelling using attack graphs — the exact problem we are solving. Provides the attack graph construction methodology our GNN will operate on |

**Notes / Quotes:**
> "The graphical approach helps model the interdependence of threat attributes, and graphs are analyzed to explore how threat events can propagate through the manufacturing value chain."
>
> Attack graph construction methodology here is the foundation before applying GNN inference.

---

### Paper 2 — GNN-AP: Attack Prediction for Cyber-Physical Systems

| Field | Content |
|---|---|
| **Full title** | Graph Neural Network-Based Attack Prediction for Communication-Based Train Control Systems |
| **Authors** | Zhao et al. |
| **Year** | 2024 |
| **Venue** | CAAI Transactions on Intelligence Technology (Wiley) |
| **URL / DOI** | https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12288 |
| **Method** | Constructs an Attack-Target Graph by combining multi-step APT attack sequences (via encoder-decoder + time-based DFS) with system topology; uses GNN to extract attacker features and converts attack prediction into a link prediction task |
| **Dataset** | DARPA 1999/2000 IDS alert logs (training and evaluation) and a CBTC system simulation dataset for validation |
| **Key result** | Successfully predicts next attack target from attack-target graph using link prediction; reduces impact of false alarms via reinforcement learning for causal relationships |
| **Limitation** | Domain-specific to rail control systems; limited generalisation to other ICS sectors; relies on alert quality |
| **Relevance to our project** | Core technical idea for our prototype — modelling attack propagation as link prediction on a graph. Directly implementable with PyTorch Geometric |

**Notes / Quotes:**
> "GNN is used to identify the attack intent from the attack target graph and convert the attack target prediction task into link prediction."

---

### Paper 3 — Physics-Informed GNN for Attack Path Prediction

| Field | Content |
|---|---|
| **Full title** | Physics-Informed Graph Neural Networks for Attack Path Prediction |
| **Authors** | Marin François, Pierre-Emmanuel Arduin, Myriam Merad (Paris Dauphine University – PSL) |
| **Year** | 2025 |
| **Venue** | MDPI Cybersecurity Journal |
| **URL / DOI** | https://www.mdpi.com/2624-800X/5/2/15 |
| **Method** | Combines Physics-Informed Neural Networks (PINNs) with Graph Neural Networks for attack-path prediction in inductive, deductive, and hybrid settings; replication package publicly available |
| **Dataset** | Synthetic enterprise-network attack-path dataset with identity-originated attack scenarios |
| **Key result** | Physics-informed constraints improved attack-path prediction performance over standard GNN approaches and enabled accurate prediction of attack paths, origins, and destinations |
| **Limitation** | Synthetic data only; restricted attack origin assumptions; no live ICS deployment tested |
| **Relevance to our project** | Open-source code available to build on directly. Physics constraints are critical for cyber-physical systems where physical process behaviour (e.g. water pressure, voltage) must be respected in risk modelling |

**Notes / Quotes:**
> "To ensure the reproducibility of our findings, all data and algorithms presented in this paper can be found in the replication package on GitHub."
>
> Replication package is publicly available — useful for reproducing baseline results.

---

### Paper 4 — Automated Knowledge-Based Cybersecurity Risk Assessment

| Field | Content |
|---|---|
| **Full title** | Automated Knowledge-Based Cybersecurity Risk Assessment of Cyber-Physical Systems |
| **Authors** | Stephen C. Phillips et al. (University of Southampton) |
| **Year** | 2024 |
| **Venue** | IEEE Access |
| **URL / DOI** | https://eprints.soton.ac.uk/490296/2/Automated_Knowledge-Based_Cybersecurity_Risk_Assessment_of_Cyber-Physical_Systems.pdf |
| **Method** | Simulation-based automated risk assessment using knowledge graphs; models trust relationships in multi-stakeholder ICT/CPS systems; aligned with ISO 27005 standard |
| **Dataset** | German steel mill cyberattack case study modeled in Spyderisk; validation performed using a knowledge-base-driven simulation of a cyber-physical industrial system |
| **Key result** | Spyderisk automatically identifies attack paths, threat propagation, and risk levels in cyber-physical systems, successfully reproducing the German steel mill attack scenario while supporting ISO 27005-based risk assessment |
| **Limitation** | Focused on simulation rather than real-time monitoring; does not use GNNs; limited to modelled scenarios |
| **Relevance to our project** | Shows how automated risk assessment can be aligned with ISO 27005 using ontology-based reasoning and attack-path analysis, making it valuable for industrial cybersecurity and compliance-focused tools |

**Notes / Quotes:**
> "This paper describes a simulation-based approach for automated risk assessment of complex cyber-physical systems to support implementers of ISO 27005."
>
> The framework aligns cybersecurity risk assessment with ISO 27005 and demonstrates automated attack-path and risk analysis for cyber-physical systems

---

### Paper 5 — Industry Baseline: OT Security Platform Comparison

| Field | Content |
|---|---|
| **Full title** | Navigating the OT Security Landscape: A Comparison of Claroty, Nozomi Networks, and Dragos |
| **Authors** | IoT Security Institute |
| **Year** | 2025 |
| **Venue** | IoT Security Institute Industry Report |
| **URL / DOI** | https://iotsecurityinstitute.com/iotsec/iot-security-institute-cyber-security-articles/247-navigating-the-ot-security-landscape-a-comparison-of-claroty,-nozomi-networks,-and-dragos |
| **Method** | Comparative analysis of three leading commercial OT security platforms across asset discovery, threat detection, incident response, and integration capabilities |
| **Dataset** | Industry use cases and deployment examples from manufacturing, energy, utilities, and critical infrastructure sectors |
| **Key result** | Claroty provides comprehensive CPS protection and flexible deployment, Dragos emphasizes threat intelligence and incident response, and Nozomi Networks specializes in OT/IoT visibility and monitoring. The article concludes that each platform has strengths depending on organizational requirements |
| **Limitation** | Proprietary commercial platforms with limited transparency regarding internal algorithms, making independent evaluation and research reproducibility difficult |
| **Relevance to our project** | Highlights the need for an open and explainable graph-based cybersecurity risk assessment system, providing a benchmark against current commercial OT security platforms|

**Notes / Quotes:**
> "Claroty, Nozomi Networks, and Dragos each offer robust OT security capabilities with different strengths and approaches."

---

## Reference Table (Quick Overview)

| # | Title (short) | Authors | Year | Method | Dataset | Relevance |
|---|---|---|---|---|---|---|
| 1 | Graph-Theoretic Manufacturing Risk | Rahman et al. | 2023 | Taxonomy-driven attack graph for Industry 4.0 ICS | Manufacturing cybersecurity taxonomies and illustrative case study | Industry 4.0 attack graph construction methodology; foundation for our GNN input |
| 2 | GNN-AP | Zhao et al. | 2024 | GNN + Link Prediction on Attack-Target Graph | DARPA 1999/2000 IDS alert logs+ CBTC system simulation dataset | Core method for our prototype implementation |
| 3 | Physics-Informed GNN | François et al. | 2025 | PIGNN for attack path prediction | Synthetic infrastructure dataset | Open-source code; physics constraints for CPS |
| 4 | Automated KG Risk Assessment | Phillips et al. | 2024 | Simulation-based KG risk assessment (ISO 27005) |German steel mill cyberattack case study modeled in Spyderisk | Industry compliance blueprint aligned with ISO 27005 risk assessment |
| 5 | OT Security Landscape Comparison | IoT Security Institute | 2025 | Comparative commercial platform analysis | Real industrial deployments | Defines the commercial gap our product fills |

---

## Tools and Datasets Identified

| Name | Type | URL | Notes |
|---|---|---|---|
| SWaT Dataset | Dataset | https://itrust.sutd.edu.sg/itrust-labs_datasets/ | Real ICS water treatment testbed from SUTD Singapore; most cited ICS security benchmark dataset |
| CIC-IDS-2017 | Dataset | https://www.unb.ca/cic/datasets/ids-2017.html | Network intrusion dataset with labeled attack flows; good for baseline experiments |
| MITRE ATT&CK for ICS | Framework / Dataset | https://attack.mitre.org/matrices/ics/ |Free taxonomy of ICS attack techniques; used as ground truth in most research papers |
| CISA ICS Advisories | Dataset / Resource | https://www.cisa.gov/ics-advisories | Free, continuously updated ICS vulnerability advisories from US government |
| PyTorch Geometric | Library | https://pyg.org/ | Core GNN implementation framework; required for prototype (already in requirements.txt) |
| Neo4j | Library / Tool | https://neo4j.com/ | Knowledge graph database for storing ICS asset-vulnerability graphs (already in requirements.txt) |
| NetworkX | Library | https://networkx.org/ | Python graph construction and analysis; used for graph preprocessing before GNN input |
| Dragos Platform | Industry Tool | https://www.dragos.com/ | Leading commercial OT security platform; primary commercial competitor to our product |
| Nozomi Networks | Industry Tool | https://www.nozominetworks.com/ | AI-driven OT/IoT visibility and anomaly detection platform; commercial competitor |
| Claroty | Industry Tool | https://claroty.com/ | Unified XIoT cyber-physical security platform; commercial competitor |
