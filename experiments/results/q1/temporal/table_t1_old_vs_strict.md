| Dataset   | Protocol                |   Precision |   Recall |       F1 |    AUPRC |       FPR | Leakage condition                    |
|:----------|:------------------------|------------:|---------:|---------:|---------:|----------:|:-------------------------------------|
| SCADANet  | Previous temporal       |    0.869157 | 0.4949   | 0.630686 | 0.771219 | 0.0805192 | future test topology possible        |
| SCADANet  | Strict causal topology  |    0.869136 | 0.494888 | 0.630671 | 0.685439 | 0.0805322 | none identified                      |
| SCADANet  | Fixed training topology |    0.870035 | 0.494888 | 0.630907 | 0.67442  | 0.0798968 | none identified (no topology update) |
| BATADAL   | Previous temporal       |    0.65625  | 1        | 0.792453 | 0.921624 | 0.0588235 | overlapping raw rows                 |
| BATADAL   | Purged temporal         |    0.65625  | 1        | 0.792453 | 0.91748  | 0.0597826 | none identified                      |