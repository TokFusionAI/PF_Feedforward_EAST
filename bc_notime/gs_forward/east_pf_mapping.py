"""EAST PCPF (BC 12 路) 与 pf_active.xml 线圈名的映射.

独立于 freegs_adapter，避免 import freegs。
"""

from __future__ import annotations

import numpy as np

from bc_notime.common.constants import PCPF_NAMES

# BC 12 路 -> PF1..PF14（不含 PF9/10 直接预测）
PCPF_TO_PF: dict[str, str] = {
    "PCPF1": "PF1",
    "PCPF2": "PF2",
    "PCPF3": "PF3",
    "PCPF4": "PF4",
    "PCPF5": "PF5",
    "PCPF6": "PF6",
    "PCPF7": "PF7",
    "PCPF8": "PF8",
    "PCPF11": "PF11",
    "PCPF12": "PF12",
    "PCPF13": "PF13",
    "PCPF14": "PF14",
}

PF_NAMES_14: list[str] = [f"PF{i}" for i in range(1, 15)]
IC_NAMES: tuple[str, ...] = ("IC1", "IC2")


def pcpf12_to_pf14_amps(pcpf12_A: np.ndarray) -> dict[str, float]:
    """12 路 PCPF [A]（顺序同 PCPF_NAMES）→ PF1..PF14 + IC1/IC2（安培）.

    约定: PF9=PF7, PF10=PF8（串联）; IC1=IC2=0。
    """
    x = np.asarray(pcpf12_A, dtype=np.float64).ravel()
    if x.shape != (12,):
        raise ValueError(f"pcpf12_A must be (12,), got {x.shape}")
    d: dict[str, float] = {}
    for i, bc_name in enumerate(PCPF_NAMES):
        d[PCPF_TO_PF[bc_name]] = float(x[i])
    d["PF9"] = d["PF7"]
    d["PF10"] = d["PF8"]
    d["IC1"] = 0.0
    d["IC2"] = 0.0
    return d
