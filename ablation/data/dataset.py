"""PFDataset (ablation): 按炮加载, 整炮 pad 到 T_max=128, state 19 维 (不含 time/dt).

相比 bc_notime/data/dataset.py 的改动:
1. import 切到 ablation.*;
2. build_sample_arrays 额外输出 phase_ids (T_max,) int8: 0=ramp_up/1=flat_top/2=ramp_down/-1=pad.
   phase_ids 必须用**未归一化**的原始 PCRL01 (安培) 与 time_phys 计算 —— phase_ids_per_step
   的阈值 (IP_PLATEAU_FRAC*max, IP_MIN_A=1e5 A) 是物理量, 喂归一化值会出垃圾相位.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ablation.common.constants import (
    D_ACTION,
    D_STATE,
    DATASET_ROOT,
    PCPF_NAMES,
    T_MAX,
)
from ablation.data.phases import phase_ids_per_step


def _safe_diff_dt(time_phys: np.ndarray) -> np.ndarray:
    """Per-step dt; 首拍补 dt_median (鲁棒)."""
    T = time_phys.shape[0]
    if T < 2:
        return np.full((T,), 0.1, dtype=np.float32)
    dt = np.diff(time_phys, prepend=time_phys[0]).astype(np.float32)
    dt[0] = float(np.median(dt[1:]))
    return dt


def build_sample_arrays(
    shot_h5: Path,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
    T_max: int = T_MAX,
) -> dict[str, np.ndarray | int]:
    """Read one shot, build all numpy arrays needed by training/eval.

    Returns dict with shapes (T_max=128 default):
        state        (T_max, 19)  float32  (normalized if mean/std provided)
        action       (T_max, 12)  float32  (normalized if mean/std provided)
        action_mask  (T_max, 12)  bool
        token_mask   (T_max,)     bool
        time_phys    (T_max,)     float32  (un-normalized seconds)
        dt_phys      (T_max,)     float32  (un-normalized seconds)
        phase_ids    (T_max,)     int8     0/1/2 valid, -1 pad
        T            int          (effective length, before pad)
    """
    shot_h5 = Path(shot_h5)
    with h5py.File(shot_h5, "r") as f:
        time_phys = f["time"][:].astype(np.float32)
        T = int(time_phys.shape[0])

        R8 = f["R8"][:].astype(np.float32)
        Z8 = f["Z8"][:].astype(np.float32)
        lmsr = f["lmsr"][:].astype(np.float32)
        lmsz = f["lmsz"][:].astype(np.float32)
        pcrl = f["PCRL01"][:].astype(np.float32)  # 原始安培

        action = np.stack(
            [f[name][:].astype(np.float32) for name in PCPF_NAMES], axis=1
        )
        action_mask = np.stack(
            [f[f"mask_{name}"][:].astype(bool) for name in PCPF_NAMES], axis=1
        )

    dt_phys = _safe_diff_dt(time_phys)

    # ---- phase_ids: 用**原始** pcrl/time_phys (归一化前) ----
    phase_ids_raw = phase_ids_per_step(time_phys, pcrl, valid_len=min(T, T_max))

    state = np.concatenate(
        [
            R8,
            Z8,
            lmsr[:, None],
            lmsz[:, None],
            pcrl[:, None],
        ],
        axis=1,
    )

    state = np.nan_to_num(state, nan=0.0, copy=False)
    action = np.nan_to_num(action, nan=0.0, copy=False)

    if state_mean is not None and state_std is not None:
        state = (state - state_mean.astype(np.float32)) / (
            state_std.astype(np.float32) + 1e-8
        )
    if action_mean is not None and action_std is not None:
        action = (action - action_mean.astype(np.float32)) / (
            action_std.astype(np.float32) + 1e-8
        )

    T_eff = min(T, T_max)
    out_state = np.zeros((T_max, D_STATE), dtype=np.float32)
    out_action = np.zeros((T_max, D_ACTION), dtype=np.float32)
    out_amask = np.zeros((T_max, D_ACTION), dtype=bool)
    out_time = np.zeros((T_max,), dtype=np.float32)
    out_dt = np.zeros((T_max,), dtype=np.float32)
    out_phase = np.full((T_max,), -1, dtype=np.int8)
    token_mask = np.zeros((T_max,), dtype=bool)

    out_state[:T_eff] = state[:T_eff]
    out_action[:T_eff] = action[:T_eff]
    out_amask[:T_eff] = action_mask[:T_eff]
    out_time[:T_eff] = time_phys[:T_eff]
    out_dt[:T_eff] = dt_phys[:T_eff]
    out_phase[:T_eff] = phase_ids_raw[:T_eff]
    token_mask[:T_eff] = True

    return {
        "state": out_state,
        "action": out_action,
        "action_mask": out_amask,
        "token_mask": token_mask,
        "time_phys": out_time,
        "dt_phys": out_dt,
        "phase_ids": out_phase,
        "T": T_eff,
    }


def load_norm_stats(path: str | Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {
        "state_mean": d["state_mean"].astype(np.float32),
        "state_std": d["state_std"].astype(np.float32),
        "action_mean": d["action_mean"].astype(np.float32),
        "action_std": d["action_std"].astype(np.float32),
    }


def read_shots_txt(path: str | Path) -> list[int]:
    return [int(t) for t in Path(path).read_text().split() if t.strip().isdigit()]


# ----------------------------- torch wrapper ----------------------------- #


def _import_torch():
    try:
        import torch  # type: ignore
        return torch
    except ImportError as e:
        raise ImportError(
            "torch is required for PFDataset; on the login node where DCU "
            "runtime is missing you can still use build_sample_arrays() for "
            "numpy-only checks."
        ) from e


class PFDataset:
    """torch.utils.data.Dataset compatible. Lazy-imports torch so this module
    can also be imported on machines without a working torch install (login).
    """

    def __init__(
        self,
        shots: list[int],
        dataset_root: str | Path = DATASET_ROOT,
        norm_stats_path: str | Path | None = None,
        T_max: int = T_MAX,
    ):
        torch = _import_torch()
        self._torch = torch
        self.shots = list(map(int, shots))
        self.root = Path(dataset_root)
        self.T_max = T_max
        if norm_stats_path is not None:
            ns = load_norm_stats(norm_stats_path)
            self.state_mean = ns["state_mean"]
            self.state_std = ns["state_std"]
            self.action_mean = ns["action_mean"]
            self.action_std = ns["action_std"]
        else:
            self.state_mean = self.state_std = None
            self.action_mean = self.action_std = None

    def __len__(self) -> int:
        return len(self.shots)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shot = self.shots[idx]
        arrs = build_sample_arrays(
            self.root / f"{shot}.h5",
            self.state_mean,
            self.state_std,
            self.action_mean,
            self.action_std,
            T_max=self.T_max,
        )
        torch = self._torch
        out = {
            "state": torch.from_numpy(arrs["state"]),
            "action": torch.from_numpy(arrs["action"]),
            "action_mask": torch.from_numpy(arrs["action_mask"]),
            "token_mask": torch.from_numpy(arrs["token_mask"]),
            "time_phys": torch.from_numpy(arrs["time_phys"]),
            "dt_phys": torch.from_numpy(arrs["dt_phys"]),
            "phase_ids": torch.from_numpy(arrs["phase_ids"]),
            "shot": int(shot),
            "T": int(arrs["T"]),
        }
        return out
