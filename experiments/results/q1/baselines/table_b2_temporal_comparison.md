# Table B2 — Temporal baseline comparison

ΔF1 vs static is temporal F1 minus the same model's SCADANet Track B (static) F1; positive means the model held up better under the strict temporal protocol.

| Dataset   | Model      |   Precision |   Recall |     F1 |   AUPRC |    FPR |   ΔF1 vs static |
|:----------|:-----------|------------:|---------:|-------:|--------:|-------:|----------------:|
| scadanet  | MLP        |      0.9216 |   0.5461 | 0.6858 |  0.8314 | 0.0503 |         -0.2983 |
| scadanet  | Tree model |      0.8428 |   0.5798 | 0.6869 |  0.8891 | 0.1172 |         -0.2991 |
| scadanet  | GraphSAGE  |      0.9092 |   0.5473 | 0.6829 |  0.916  | 0.0607 |         -0.3019 |
| scadanet  | GCN        |      0.8919 |   0.9374 | 0.914  |  0.9646 | 0.1234 |         -0.0719 |
