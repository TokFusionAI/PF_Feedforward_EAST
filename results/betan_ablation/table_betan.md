# Cross-β_N-mode 划分 (低β_N训→高β_N测, 干净时序+β_N筛选 leave-one-mode-out, ∥时间防泄漏) 消融结果

划分: `meta/split_by_order_betan` (在 igbt 干净时序上叠加 β_N 筛选: train=低β_N(<0.8)早炮 8048 / val=低β_N 1022 / test=高β_N(>0.8)晚炮 931, held-out 模式); 硬时间边界 train max<val min<test min, β_N 区间不相交, 严格无泄漏无交叠。6 配置 × 3 seed (11, 44, 20260424)。
norm_stats 仅由 betan train (低β_N) 重算 (19 维 notime)。模型与随机划分消融一致 (仅划分不同)。

| Model (PE) | n | Test RMSE (kA) | Test R²_med | Test MSE | 随机RMSE | ΔRMSE |
|---|---|---|---|---|---|---|
| Transformer (proposed), time (sinusoidal) | 3 | 1.028±0.036 | 0.8671±0.0094 | 1.05844±0.07364 | 0.307 | +0.721 |
| Transformer, causal, none | 3 | 1.211±0.018 | 0.8135±0.0051 | 1.46674±0.04435 | 0.328 | +0.883 |
| Transformer, causal, time (sinusoidal) | 3 | 1.202±0.005 | 0.8151±0.0012 | 1.44443±0.01238 | 0.340 | +0.862 |
| Transformer (bidir), none | 3 | 1.365±0.100 | 0.7602±0.0333 | 1.87278±0.28125 | 0.358 | +1.007 |
| LSTM, none | 3 | 1.281±0.019 | 0.7500±0.0038 | 1.64058±0.04927 | - | - |
| MLP, none | 3 | 1.773±0.033 | 0.6735±0.0017 | 3.14553±0.11560 | - | - |
