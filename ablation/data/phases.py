"""Discharge phase detection: ramp-up / flat-top / ramp-down.

被 bc.analysis.analyze_didt 与 bc.evaluation.eval 共用, 保证两边用同一套规则.

算法:
    1. Ip_abs = |PCRL01| (A)
    2. Ip_max = max(Ip_abs).  若 Ip_max < IP_MIN_A 或 T < 3: 整炮归为 'ramp_up' (无平顶).
    3. Ip_thr = IP_PLATEAU_FRAC * Ip_max
    4. hits = 所有 Ip_abs >= Ip_thr 的索引.  k_in = hits[0], k_out = hits[-1].
    5. 平顶判据: 若 (time[k_out] - time[k_in]) >= MIN_PLATEAU_S:
           ramp_up   = [0, k_in)
           flat_top  = [k_in, k_out + 1)   # 半开, 包含 k_out
           ramp_down = [k_out + 1, T)
       否则视为无平顶:
           取 Ip 峰值索引 k_peak = argmax(Ip_abs)
           ramp_up   = [0, k_peak + 1)
           flat_top  = []
           ramp_down = [k_peak + 1, T)

返回的 phase_id 数组:  0=ramp_up, 1=flat_top, 2=ramp_down, -1=无效(超出 T_eff 的 pad).
"""

from __future__ import annotations

import numpy as np

PHASE_NAMES: list[str] = ["ramp_up", "flat_top", "ramp_down"]
PHASE_RAMP_UP: int = 0
PHASE_FLAT_TOP: int = 1
PHASE_RAMP_DOWN: int = 2

IP_PLATEAU_FRAC: float = 0.90
MIN_PLATEAU_S: float = 0.3
IP_MIN_A: float = 100_000.0


def detect_phase_slices(
    time_s: np.ndarray,
    Ip_A: np.ndarray,
    ip_plateau_frac: float = IP_PLATEAU_FRAC,
    min_plateau_s: float = MIN_PLATEAU_S,
    ip_min_A: float = IP_MIN_A,
) -> dict[str, slice]:
    """Return {'ramp_up': slice, 'flat_top': slice, 'ramp_down': slice}.

    slice 遵循 Python 半开区间约定. flat_top 可能是空 slice(0, 0).
    """
    T = int(time_s.shape[0])
    Ip_abs = np.abs(Ip_A)
    Ip_max = float(Ip_abs.max()) if T else 0.0

    # 无等离子体 / 太短: 整炮归 ramp_up
    if T < 3 or Ip_max < ip_min_A:
        return {
            "ramp_up": slice(0, T),
            "flat_top": slice(0, 0),
            "ramp_down": slice(T, T),
        }

    thr = ip_plateau_frac * Ip_max
    hits = np.where(Ip_abs >= thr)[0]
    if hits.size == 0:
        k_peak = int(Ip_abs.argmax())
        return {
            "ramp_up": slice(0, k_peak + 1),
            "flat_top": slice(0, 0),
            "ramp_down": slice(k_peak + 1, T),
        }

    k_in = int(hits[0])
    k_out = int(hits[-1])
    if (time_s[k_out] - time_s[k_in]) < min_plateau_s:
        k_peak = int(Ip_abs.argmax())
        return {
            "ramp_up": slice(0, k_peak + 1),
            "flat_top": slice(0, 0),
            "ramp_down": slice(k_peak + 1, T),
        }

    return {
        "ramp_up": slice(0, k_in),
        "flat_top": slice(k_in, k_out + 1),
        "ramp_down": slice(k_out + 1, T),
    }


def phase_ids_per_step(
    time_s: np.ndarray,
    Ip_A: np.ndarray,
    valid_len: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Return (T,) int array; entries are PHASE_*, pad 位置为 -1.

    valid_len: 若非 None, 仅对 [0, valid_len) 做 detection; [valid_len, T) 置 -1.
    """
    T = int(time_s.shape[0])
    out = np.full((T,), -1, dtype=np.int8)
    if valid_len is None:
        valid_len = T
    valid_len = min(valid_len, T)
    if valid_len < 2:
        return out

    slices = detect_phase_slices(time_s[:valid_len], Ip_A[:valid_len], **kwargs)
    for pid, name in enumerate(PHASE_NAMES):
        s = slices[name]
        if s.stop > s.start:
            out[s.start : s.stop] = pid
    return out


def phase_step_ids(phase_ids: np.ndarray) -> np.ndarray:
    """Step-level phase id for |dI/dt|: step k (1..T-1) 取 max(phase_ids[k-1], phase_ids[k]).

    这意味着跨 phase 边界的 step 归到更晚的阶段. 对 pad 位置 -1 传播.
    返回 (T-1,) int8.
    """
    if phase_ids.size < 2:
        return np.empty((0,), dtype=np.int8)
    a = phase_ids[:-1]
    b = phase_ids[1:]
    invalid = (a < 0) | (b < 0)
    step = np.maximum(a, b).astype(np.int8)
    step[invalid] = -1
    return step
