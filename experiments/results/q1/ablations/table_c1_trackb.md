# Table C1 — Topology/content ablation (trackb)

| Condition                 | Content features   | True topology   | Message passing   |     F1 |   AUPRC |    FPR |     ΔF1 |
|:--------------------------|:-------------------|:----------------|:------------------|-------:|--------:|-------:|--------:|
| Content only              | ✓                  | ✗               | ✗                 | 0.9839 |  0.9975 | 0.1401 | -0.0006 |
| Topology only             | ✗                  | ✓               | ✓                 | 0.98   |  0.9729 | 0.1385 | -0.0046 |
| Full GraphSAGE            | ✓                  | ✓               | ✓                 | 0.9845 |  0.9985 | 0.1398 |  0      |
| Random topology           | ✓                  | ✗ random        | ✓                 | 0.9852 |  0.9984 | 0.129  |  0.0006 |
| Degree-preserved rewiring | ✓                  | ✗ rewired       | ✓                 | 0.9853 |  0.9986 | 0.1271 |  0.0007 |
| No message passing        | ✓                  | partial         | ✗                 | 0.9856 |  0.9986 | 0.1282 |  0.0011 |
