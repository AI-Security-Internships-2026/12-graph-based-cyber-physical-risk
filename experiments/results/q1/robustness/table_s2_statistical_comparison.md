| Comparison                                           | Metric   |   Mean difference | 95% CI             |   Effect size (Cohen's d) |     p-value | Test used     |
|:-----------------------------------------------------|:---------|------------------:|:-------------------|--------------------------:|------------:|:--------------|
| GraphSAGE_trackb vs MLP_trackb                       | f1       |       0.00067252  | [0.0006, 0.0007]   |                  6.3376   | 8.91599e-09 | paired_t_test |
| GraphSAGE_trackb vs Tree_trackb                      | f1       |      -0.00118736  | [-0.0015, -0.0009] |                 -2.62568  | 1.64252e-05 | paired_t_test |
| GraphSAGE_trackb vs RandomTopology_trackb            | f1       |      -0.000609463 | [-0.0009, -0.0003] |                 -1.31938  | 0.00240355  | paired_t_test |
| GraphSAGE_trackb vs DegreePreservedTopology_trackb   | f1       |      -0.000608822 | [-0.0009, -0.0003] |                 -1.32532  | 0.00233757  | paired_t_test |
| Original_H1_trackb vs HostNormalized_H4_trackb       | fpr      |       0.141834    | [0.1375, 0.1461]   |                 23.6221   | 6.98465e-14 | paired_t_test |
| MatchedGraphSAGE_trackb vs GraphSAGE_strict_temporal | recall   |       0.435066    | [0.4022, 0.4679]   |                  9.47674  | 2.5074e-10  | paired_t_test |
| GraphSAGE_trackb vs ContentOnly_trackb               | f1       |       0.00067252  | [0.0006, 0.0007]   |                  6.3376   | 8.91599e-09 | paired_t_test |
| Original_H1_temporal vs HostNormalized_H4_temporal   | fpr      |      -0.0491571   | [-0.1040, 0.0057]  |                 -0.640778 | 0.0733694   | paired_t_test |