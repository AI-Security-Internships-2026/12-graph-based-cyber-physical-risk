# Table B2 — Temporal baseline comparison

ΔF1 vs static is temporal F1 minus the same model's SCADANet Track B (static) F1; positive means the model held up better under the strict temporal protocol.

| Dataset   | Model      |   Precision |   Recall |     F1 |   AUPRC |    FPR |   ΔF1 vs static |
|:----------|:-----------|------------:|---------:|-------:|--------:|-------:|----------------:|
| scadanet  | GCN        |      0.8242 |   0.9362 | 0.8766 |  0.9325 | 0.2158 |         -0.109  |
| scadanet  | GraphSAGE  |      0.5879 |   0.5469 | 0.5666 |  0.6825 | 0.4143 |         -0.4179 |
| scadanet  | MLP        |      0.8368 |   0.5441 | 0.6594 |  0.7815 | 0.1147 |         -0.3245 |
| scadanet  | Tree model |      0.8344 |   0.5792 | 0.6838 |  0.8337 | 0.1242 |         -0.3019 |
