# IGBT-PWM 时序划分 (chronological) 消融结果

划分: `meta/split_by_order_igbt` (shot>=117203, 2022-11 PS11/12 IGBT-PWM 更新后;
train 17038 / val 2130 / test 2129, 严格时序无交叠), 6 配置 × 3 seed (11, 44, 20260424)。
norm_stats 仅由 igbt train 重算 (19 维 notime)。模型与随机划分消融一致 (仅划分不同)。

| Model (PE) | n | Test RMSE (kA) | Test R²_med | Test MSE | 随机RMSE | ΔRMSE |
|---|---|---|---|---|---|---|
| Transformer (proposed), time (sinusoidal) | 3 | 1.150±0.010 | 0.8840±0.0063 | 1.32241±0.02312 | nan | +nan |
| Transformer, causal, none | 3 | 1.269±0.009 | 0.8664±0.0027 | 1.60979±0.02367 | nan | +nan |
| Transformer, causal, time (sinusoidal) | 3 | 1.382±0.006 | 0.8486±0.0046 | 1.91133±0.01660 | nan | +nan |
| Transformer (bidir), none | 3 | 1.481±0.051 | 0.8006±0.0100 | 2.19490±0.15142 | nan | +nan |
| LSTM, none | 3 | 1.465±0.042 | 0.8037±0.0118 | 2.14718±0.12202 | nan | +nan |
| MLP, none | 3 | 1.766±0.028 | 0.7457±0.0082 | 3.12084±0.09872 | nan | +nan |
