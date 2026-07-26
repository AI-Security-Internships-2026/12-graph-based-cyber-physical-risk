# Graph-Based Risk Assessment for Cyber-Physical Systems

> **CNIT/PNTLab Pisa · TECIP · Scuola Superiore Sant'Anna — AI Security Internship 2026**

---

## Research Problem

Model industrial control system (ICS/SCADA) networks as knowledge graphs and apply graph neural networks to predict attack paths, propagation likelihood, and impact on physical processes.

---

## Objectives

1. Conduct a systematic literature review on the topic.
2. Design and implement a proof-of-concept prototype.
3. Evaluate the prototype on real or benchmark datasets.
4. Document findings in a final technical report.
5. Present results to the research group.

---

## Expected Deliverables

| Deliverable | Due |
|---|---|
| Literature review (`docs/literature-review.md`) | Week 2 |
| Architecture design document (`docs/proposal.md`) | Week 3 |
| Working prototype (`src/`) | Week 6 |
| Evaluation results (`experiments/results/`) | Week 7 |
| Final report (`docs/final-report.md`) | Week 8 |

---

## Recommended Technology Stack

```
Python, PyTorch Geometric, NetworkX, Neo4j, Pandas, Streamlit
```

See `requirements.txt` for pinned dependencies.

---

## Weekly Workflow

```
Monday     – Review weekly tasks in tasks/week-XX.md
Tue–Thu    – Implementation / experiments
Friday     – Document progress in docs/weekly-progress.md
Friday     – Open weekly Pull Request from your branch → dev
```

---

## Branching Policy

| Branch | Purpose |
|---|---|
| `main` | Stable, supervisor-reviewed code only |
| `dev` | Integration branch — merge weekly PRs here |
| `<your-name>-week-XX` | Your working branch for each week |

**Students must never push directly to `main`.**

---

## Pull Request Policy

- One PR per week, targeting the `dev` branch.
- PR title format: `[Week XX] Brief description`
- PR description must reference the weekly task file and summarise what was done.
- A supervisor or co-student must review before merging.

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/AI-Security-Internships-2026/12-graph-based-cyber-physical-risk.git
cd 12-graph-based-cyber-physical-risk

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your weekly branch
git checkout dev
git pull origin dev
git checkout -b your-name-week-01

# 5. Run the starter script
python src/main.py
```

---

## Roadmap to September 8, 2026

**Current state:** furthest ahead in the cohort — GraphSAGE pipeline validated on SCADANet/BATADAL, and a genuinely rigorous catch already made (the IP-topology leakage / shortcut-learning finding in PR #11). No forward schedule existed until now.

**Novel contribution target:** build on the leakage finding rather than move past it — formally explain *why* the model relies on topology features (via GNN explainability), then push into dynamic/streaming graphs, since almost all comparable GNN-based cyber-physical IDS work assumes a static graph snapshot.

| Date | Milestone |
|---|---|
| Aug 2 | Investigate the remaining Track B false-positive source flagged as a known limitation in PR #11 |
| Aug 9 | Apply a GNN explainability method (GNNExplainer/PGExplainer) to formally characterize which topology features drive predictions |
| Aug 16 | Extend to temporal/streaming graph updates (nodes/edges arriving over time) instead of a static snapshot |
| Aug 23 | Benchmark static vs. temporal approach on SCADANet/BATADAL |
| Aug 30 | Full write-up combining the leakage analysis, explainability, and temporal extension |
| Sep 6 | Paper draft |
| **Sep 8** | **Final submission** |

---

## Supervisor Note

This repository is managed by **CNIT/PNTLab Pisa, TECIP, Scuola Superiore Sant'Anna**.
Please contact your supervisor before making architectural changes.
All code must be original or properly attributed.
Do **not** commit API keys, passwords, or large datasets — see `.gitignore`.
