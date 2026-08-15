"""推理后处理: 基于 phase-aware per-channel |dI/dt| 阈值的硬过滤器.

用法:
    from bc.inference_filter import apply_didt_filter

    pred_kA_filt = apply_didt_filter(
        pred_kA,      # (T, 12)
        time_s,       # (T,) 秒
        Ip_A,         # (T,) 安培, 用于 phase 判定
        which='P99.9',
    )

设计:
    - step-by-step, causal: pred_filt[t] 基于 pred_filt[t-1] (不是原始 pred[t-1]).
      这是 "一次过滤, 终生有效" 的硬约束.
    - 每个 step t 属于哪个 phase → 取对应阈值 thr[c] (kA/s)
    - 允许偏移 delta_max[c] = thr[c] * dt[t]
    - pred_filt[t, c] = clip(pred[t, c], pred_filt[t-1, c] - delta_max[c],
                                         pred_filt[t-1, c] + delta_max[c])
    - 首拍不过滤 (没有 t-1 参考).

支持的 `which`: 'max' | 'P99.99' | 'P99.9' | 'P99' | 'none' (不过滤, 直接返回原值).
"""

from __future__ import annotations

import numpy as np

from bc.common.constants import PCPF_NAMES, phase_limits_array
from bc.data.phases import PHASE_NAMES, phase_ids_per_step


def _thresholds_matrix(which: str) -> np.ndarray:
    """返回 (3, 12) float 数组, 按 PHASE_NAMES 索引 (0=ramp_up, 1=flat_top, 2=ramp_down)."""
    limits = phase_limits_array(which=which)
    arr = np.full((3, 12), np.inf, dtype=np.float64)
    for pid, name in enumerate(PHASE_NAMES):
        if name in limits:
            arr[pid] = np.asarray(limits[name], dtype=np.float64)
    return arr


def apply_didt_filter(
    pred_kA: np.ndarray,
    time_s: np.ndarray,
    Ip_A: np.ndarray,
    which: str = "P99.9",
    phase_ids: np.ndarray | None = None,
    thresholds_per_phase: np.ndarray | None = None,
) -> np.ndarray:
    """Step-by-step causal dI/dt 硬过滤.

    Args:
        pred_kA:  (T, 12) 模型原始预测, kA
        time_s:   (T,) 物理时间, 秒 (严格单调递增)
        Ip_A:     (T,) 等离子体电流, 安培 (用于 phase 判定)
        which:    'max' | 'P99.99' | 'P99.9' | 'P99' | 'none'
        phase_ids:           optional (T,) int8, 若已算过可直接传入 (跳过 phase 检测)
        thresholds_per_phase: optional (3, 12) ndarray, 若已准备好可直接传入 (跳过 JSON 加载)

    Returns:
        pred_filt_kA: (T, 12) 过滤后的 kA 预测
    """
    pred_kA = np.asarray(pred_kA, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64)
    Ip_A = np.asarray(Ip_A, dtype=np.float64)

    T = pred_kA.shape[0]
    if T < 2 or which == "none":
        return pred_kA.copy()

    if thresholds_per_phase is None:
        thresholds_per_phase = _thresholds_matrix(which)
    else:
        thresholds_per_phase = np.asarray(thresholds_per_phase, dtype=np.float64)
        assert thresholds_per_phase.shape == (3, 12)

    if phase_ids is None:
        phase_ids = phase_ids_per_step(time_s, Ip_A, valid_len=T)
    # 无效段用 flat_top 兜底 (保守中等阈值)
    pid_safe = np.where(phase_ids >= 0, phase_ids, 1).astype(np.int64)

    # 每个 step t 的 12 路阈值
    thr_per_step = thresholds_per_phase[pid_safe]  # (T, 12)

    # causal clipping
    out = np.empty_like(pred_kA)
    out[0] = pred_kA[0]
    for t in range(1, T):
        dt = max(time_s[t] - time_s[t - 1], 1e-6)
        delta_max = thr_per_step[t] * dt  # (12,)
        lo = out[t - 1] - delta_max
        hi = out[t - 1] + delta_max
        out[t] = np.clip(pred_kA[t], lo, hi)
    return out


def compare_before_after(
    pred_kA: np.ndarray,
    time_s: np.ndarray,
    Ip_A: np.ndarray,
    which: str = "P99.9",
) -> dict:
    """对同一炮算出: 过滤前后 |dI/dt| 的 max/mean 差异 + 被 clip 的 step 比例."""
    pred_filt = apply_didt_filter(pred_kA, time_s, Ip_A, which=which)
    dt = np.diff(time_s)
    didt_orig = np.abs(np.diff(pred_kA, axis=0)) / np.maximum(dt[:, None], 1e-6)
    didt_filt = np.abs(np.diff(pred_filt, axis=0)) / np.maximum(dt[:, None], 1e-6)

    thr_mat = _thresholds_matrix(which) if which != "none" else None
    if thr_mat is not None:
        pid = phase_ids_per_step(time_s, Ip_A, valid_len=pred_kA.shape[0])
        step_pid = np.maximum(pid[:-1], pid[1:])
        step_pid_safe = np.where(step_pid >= 0, step_pid, 1).astype(np.int64)
        thr_per_step = thr_mat[step_pid_safe]
        n_clipped = int(((didt_orig > thr_per_step) & (step_pid >= 0)[:, None]).sum())
    else:
        n_clipped = 0
    return {
        "which": which,
        "didt_orig_max": float(didt_orig.max()) if didt_orig.size else float("nan"),
        "didt_filt_max": float(didt_filt.max()) if didt_filt.size else float("nan"),
        "didt_orig_mean": float(didt_orig.mean()) if didt_orig.size else float("nan"),
        "didt_filt_mean": float(didt_filt.mean()) if didt_filt.size else float("nan"),
        "steps_clipped": n_clipped,
        "total_steps": int(didt_orig.shape[0] * didt_orig.shape[1]),
    }
