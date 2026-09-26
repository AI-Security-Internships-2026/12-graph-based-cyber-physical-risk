# Table B1 — Main baseline comparison (SCADANet Track B)

| Dataset   | Protocol   | Model               |   Precision |   Recall |     F1 |   AUPRC |    FPR |
|:----------|:-----------|:--------------------|------------:|---------:|-------:|--------:|-------:|
| scadanet  | track_b    | Logistic Regression |      0.9761 |   0.9819 | 0.979  |  0.9945 | 0.1268 |
| scadanet  | track_b    | Tree model          |      0.9766 |   0.9949 | 0.9857 |  0.9987 | 0.1255 |
| scadanet  | track_b    | MLP                 |      0.9739 |   0.9941 | 0.9839 |  0.9975 | 0.1401 |
| scadanet  | track_b    | GCN                 |      0.9748 |   0.9967 | 0.9856 |  0.9985 | 0.1358 |
| scadanet  | track_b    | GraphSAGE           |      0.974  |   0.9953 | 0.9845 |  0.9985 | 0.1399 |
| scadanet  | track_b    | GAT                 |      0.9739 |   0.9941 | 0.9839 |  0.9978 | 0.1402 |
