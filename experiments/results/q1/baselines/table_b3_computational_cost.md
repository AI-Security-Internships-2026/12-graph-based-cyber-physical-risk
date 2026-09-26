# Table B3 — Computational cost

Parameters is blank for the tree baseline (not a meaningful concept for a tree ensemble — see src/models/baselines.py TreeBaseline.n_trainable_params). Timings are single-process CPU wall time on whatever machine produced this file; see library_versions in the per-model result JSON for the environment.

| Protocol          | Dataset   | Model                       |   Parameters |   Train time (s) |   Inference time (ms) |
|:------------------|:----------|:----------------------------|-------------:|-----------------:|----------------------:|
| BATADAL temporal  | batadal   | gcn_window_classifier       |         1282 |         116.688  |               34.5389 |
| BATADAL temporal  | batadal   | graphsage_window_classifier |         2434 |         114.584  |               25.3336 |
| BATADAL temporal  | batadal   | mlp_flattened_window        |         6658 |           2.4109 |                0.351  |
| SCADANet Track B  | scadanet  | gat_edge_attr               |        45442 |         789.62   |              646.739  |
| SCADANet Track B  | scadanet  | gcn_edge_attr               |        31362 |         780.738  |              615.471  |
| SCADANet Track B  | scadanet  | graphsage_edge_attr         |        35714 |         775.885  |              632.712  |
| SCADANet Track B  | scadanet  | logistic_regression         |          225 |          69.8511 |              143.522  |
| SCADANet Track B  | scadanet  | mlp                         |        18690 |        2454.16   |               70.1087 |
| SCADANet Track B  | scadanet  | tree_xgboost                |          nan |          89.2117 |              267.372  |
| SCADANet temporal | scadanet  | gcn_edge_attr               |        11586 |         725.531  |              145.127  |
| SCADANet temporal | scadanet  | graphsage_edge_attr         |        12738 |         719.36   |              136.685  |
| SCADANet temporal | scadanet  | mlp                         |        18690 |        2128.63   |              109.656  |
| SCADANet temporal | scadanet  | tree_xgboost                |          nan |          95.1096 |              831.254  |
