# Table B3 — Computational cost

Parameters is blank for the tree baseline (not a meaningful concept for a tree ensemble — see src/models/baselines.py TreeBaseline.n_trainable_params). Timings are single-process CPU wall time on whatever machine produced this file; see library_versions in the per-model result JSON for the environment.

| Protocol          | Dataset   | Model                       |   Parameters |   Train time (s) |   Inference time (ms) |
|:------------------|:----------|:----------------------------|-------------:|-----------------:|----------------------:|
| SCADANet Track B  | scadanet  | logistic_regression         |        218.4 |          15.3122 |               11.9883 |
| SCADANet Track B  | scadanet  | mlp                         |      16232.4 |         540.16   |               11.3847 |
| SCADANet Track B  | scadanet  | tree_xgboost                |        nan   |           6.5503 |               10.7483 |
| SCADANet Track B  | scadanet  | graphsage_edge_attr         |      28437.2 |        1775.82   |               63.8385 |
| SCADANet Track B  | scadanet  | gcn_edge_attr               |      19202   |        1789.02   |               67.0534 |
| SCADANet Track B  | scadanet  | gat_edge_attr               |      36133.2 |        1679.31   |               68.8612 |
| SCADANet temporal | scadanet  | mlp                         |      13464.4 |         475.173  |               14.8206 |
| SCADANet temporal | scadanet  | tree_xgboost                |        nan   |           5.5932 |               22.3883 |
| SCADANet temporal | scadanet  | graphsage_edge_attr         |      27678.8 |        1662.52   |               24.5703 |
| SCADANet temporal | scadanet  | gcn_edge_attr               |      18565.2 |        1637      |               22.2978 |
| BATADAL temporal  | batadal   | mlp_flattened_window        |       6658   |           1.7662 |                1.4062 |
| BATADAL temporal  | batadal   | graphsage_window_classifier |       2434   |         269.658  |               69.8495 |
| BATADAL temporal  | batadal   | gcn_window_classifier       |       1282   |         318.148  |               54.064  |
