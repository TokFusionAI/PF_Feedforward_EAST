# `bc/`：行为克隆（BC）与 freegsnke 前向验证

端到端命令与依赖顺序见仓库根目录 **[`docs/PIPELINE.md`](../docs/PIPELINE.md)**。

## 单炮（`gs_forward`）与批量（`batch_freegsnke`）

| 场景 | 子包 | 说明 |
|------|------|------|
| **单炮** | **`bc/gs_forward/`** | `run_freegsnke_eval`、`infer_one_shot`、`precursor_export`、MDS/EFIT 读取、EAST 几何等**可复用实现**。 |
| **批量** | **`bc/batch_freegsnke/`** | 多炮、每 phase 多时刻、子进程串联 eval；通过 `from bc.gs_forward...` 调用底层，**只做编排**。 |

PIPELINE 中 **§7 freegsnke** 与 **§8 批量** 之间有更细的说明（含典型 `python -m` 入口）。

根目录下的 `bc/train.py`、`bc/run_freegsnke_eval.py` 等为**薄包装**，便于旧命令与 `python -m bc.<name>` 兼容；逻辑在对应子包内。
