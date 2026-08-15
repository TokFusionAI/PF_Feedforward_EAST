"""公用常量, 与 scripts/dataset_io.py 的约定保持一致.

state 21 维顺序 (见 plans/bc_transformer_v1.md §3):
    R8(8) + Z8(8) + lmsr + lmsz + PCRL01 + time + dt
"""

from __future__ import annotations

import json
from pathlib import Path

PCPF_NAMES: list[str] = [f"PCPF{i}" for i in list(range(1, 9)) + list(range(11, 15))]

STATE_LAYOUT: list[str] = [
    *(f"R8_{i}" for i in range(8)),
    *(f"Z8_{i}" for i in range(8)),
    "lmsr",
    "lmsz",
    "PCRL01",
    "time",
    "dt",
]

D_STATE: int = 21
D_ACTION: int = 12

T_MAX: int = 128

DATASET_ROOT: Path = Path("/data/PF_ATIME_dataset")
META_DIR: Path = Path("meta")

# 推理硬过滤 (反归一化后单位)
HARD_LIMIT_I_KA: float = 14.5

# -------------------- dI/dt 数据驱动阈值 ------------------------------------
#
# 历史 plan 假设 `|dI/dt| <= 4 kA/s` 单值, 但数据统计
# (meta/train_shots.txt 上 26638 炮 × 31.97M 时间步, 见 bc.analysis.analyze_didt 脚本产物
#  results/didt_stats/proposed_thresholds.json) 显示:
#   - 老阈值 4 kA/s 全库实际违规率 7.41%
#   - 不同 PCPF 的 P99 差距 2.2x (PCPF7 4.50 vs PCPF6 10.08)
#   - 这些都是 EAST 专家运行员在真实放电里用出来的数值
#
# 用 per-channel P99 作 "soft" 阈值 (监控训练质量; 期望违规率 ~1%),
# P99.9 作 "relaxed" 阈值 (鼓励匹配专家行为).
# 旧的 `HARD_LIMIT_DIDT_KAPS=4.0` 保留供需要"严格工程/电源上限"场合使用.

# per-channel P99 阈值 (kA/s), 来自全量 train-set |dI/dt| 分位数
DIDT_P99_KAPS: dict[str, float] = {
    "PCPF1":  6.20,
    "PCPF2":  6.12,
    "PCPF3":  7.28,
    "PCPF4":  7.88,
    "PCPF5":  9.79,
    "PCPF6": 10.08,
    "PCPF7":  4.50,
    "PCPF8":  4.70,
    "PCPF11": 7.14,
    "PCPF12": 7.25,
    "PCPF13": 6.71,
    "PCPF14": 6.81,
}

# per-channel P99.9 阈值 (kA/s)
DIDT_P999_KAPS: dict[str, float] = {
    "PCPF1":  8.84,
    "PCPF2":  9.07,
    "PCPF3": 12.07,
    "PCPF4": 12.39,
    "PCPF5": 14.41,
    "PCPF6": 14.61,
    "PCPF7":  5.72,
    "PCPF8":  5.91,
    "PCPF11": 8.30,
    "PCPF12": 8.45,
    "PCPF13": 11.82,
    "PCPF14": 11.80,
}

# 全局兜底阈值 (kA/s), 仅在 per-channel 不可用时用
DIDT_GLOBAL_P99_KAPS: float = 7.41
DIDT_GLOBAL_P999_KAPS: float = 12.00

# 旧的一刀切阈值 (保留作"严格工程上限"参考)
HARD_LIMIT_DIDT_KAPS: float = 4.0


def didt_limit_array(percentile: str = "P99") -> list[float]:
    """返回 12 路 PCPF 阈值的 list[float], 顺序与 PCPF_NAMES 对齐."""
    src = {"P99": DIDT_P99_KAPS, "P99.9": DIDT_P999_KAPS}[percentile]
    return [src[n] for n in PCPF_NAMES]


# -------------------- phase-aware |dI/dt| thresholds -----------------------
#
# 运行 `python -m bc.analyze_didt` 产出 results/didt_stats/proposed_phase_thresholds.json.
# 默认约束用每 phase 的 max (= 运行员真实用过的物理上限, 数据驱动);
# 也可选 P99.99 / P99.9 / P99 得到越来越宽松的上限.
#
# phase 检测见 bc.data.phases (IP_PLATEAU_FRAC=0.9, MIN_PLATEAU_S=0.3s).

PHASE_NAMES: list[str] = ["ramp_up", "flat_top", "ramp_down"]
# bc/common/constants.py → 仓库根为 parent.parent.parent
DIDT_PHASE_THRESHOLDS_JSON = (
    Path(__file__).resolve().parent.parent.parent / "results" / "didt_stats" / "proposed_phase_thresholds.json"
)


def load_phase_limits(
    which: str = "max",
    json_path: Path | str | None = None,
) -> dict[str, dict[str, float]]:
    """返回 {phase: {PCPF_name: threshold_kAps}}.

    which: 'max' | 'P99.99' | 'P99.9' | 'P99'  -- 选哪个分位作阈值.
    json_path: 默认读 results/didt_stats/proposed_phase_thresholds.json.
    若 JSON 不存在, 返回 empty dict (调用方应 fallback 到 DIDT_P99_KAPS).
    """
    path = Path(json_path) if json_path is not None else DIDT_PHASE_THRESHOLDS_JSON
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    key_map = {
        "max": "per_channel_max",
        "P99.99": "per_channel_P99_99",
        "P99.9": "per_channel_P99_9",
        "P99": "per_channel_P99",
    }
    if which not in key_map:
        raise ValueError(f"which must be one of {list(key_map.keys())}, got {which!r}")
    section = key_map[which]
    out: dict[str, dict[str, float]] = {}
    for phase in PHASE_NAMES:
        if phase in payload.get("phases", {}):
            out[phase] = dict(payload["phases"][phase].get(section, {}))
    return out


def phase_limits_array(
    which: str = "max",
    json_path: Path | str | None = None,
) -> dict[str, list[float]]:
    """返回 {phase: list[float] of shape (12,)}; 缺失 channel 用 inf (不触发违规)."""
    limits = load_phase_limits(which=which, json_path=json_path)
    arrs: dict[str, list[float]] = {}
    for phase in PHASE_NAMES:
        d = limits.get(phase, {})
        arrs[phase] = [float(d.get(n, float("inf"))) for n in PCPF_NAMES]
    return arrs


DT_BUCKETS: list[tuple[float, float]] = [
    (0.05, 0.08),
    (0.08, 0.13),
    (0.13, 0.30),
]
