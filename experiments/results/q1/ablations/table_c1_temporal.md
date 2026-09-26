# Table C1 — Topology/content ablation (temporal)

| Condition                 | Content features   | True topology   | Message passing   |     F1 |   AUPRC |    FPR |    ΔF1 |
|:--------------------------|:-------------------|:----------------|:------------------|-------:|--------:|-------:|-------:|
| Content only              | ✓                  | ✗               | ✗                 | 0.6594 |  0.7815 | 0.1147 | 0.0927 |
| Topology only             | ✗                  | ✓               | ✓                 | 0.7252 |  0.8978 | 0.7128 | 0.1585 |
| Full GraphSAGE            | ✓                  | ✓               | ✓                 | 0.5667 |  0.6819 | 0.4135 | 0      |
| Random topology           | ✓                  | ✗ random        | ✓                 | 0.6077 |  0.7785 | 0.2848 | 0.0411 |
| Degree-preserved rewiring | ✓                  | ✗ rewired       | ✓                 | 0.6371 |  0.7753 | 0.1978 | 0.0704 |
| No message passing        | ✓                  | partial         | ✗                 | 0.6022 |  0.7755 | 0.3046 | 0.0355 |
