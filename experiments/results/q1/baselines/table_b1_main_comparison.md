# Table B1 — Main baseline comparison (SCADANet Track B)

| Dataset   | Protocol   | Model               |   Precision |   Recall |     F1 |   AUPRC |    FPR |
|:----------|:-----------|:--------------------|------------:|---------:|-------:|--------:|-------:|
| scadanet  | track_b    | Logistic Regression |      0.9762 |   0.9823 | 0.9793 |  0.9972 | 0.1272 |
| scadanet  | track_b    | Tree model          |      0.9772 |   0.995  | 0.986  |  0.9988 | 0.1234 |
| scadanet  | track_b    | MLP                 |      0.9743 |   0.9942 | 0.9841 |  0.9979 | 0.1393 |
| scadanet  | track_b    | GCN                 |      0.975  |   0.9971 | 0.9859 |  0.9985 | 0.1357 |
| scadanet  | track_b    | GraphSAGE           |      0.9729 |   0.997  | 0.9848 |  0.9983 | 0.1477 |
| scadanet  | track_b    | GAT                 |      0.9743 |   0.9941 | 0.9841 |  0.9979 | 0.1396 |
