#!/usr/bin/env python3
"""从本地 EFIT h5 + PF_ATIME h5 构造 precursor.npz (绕过 MDS)。

precursor_export 需 MDS (计算节点 mds.ipp.ac.cn DNS 不可达); 本脚本从已有 h5 构造
同格式 precursor。PF_ATIME time 与 EFIT ATIME 同栅格 (训练数据已在 EFIT ATIME 重采样),
PCS 直接取 (长度一致) 或插值到 EFIT ATIME。

bundle = EFIT profile (EFIT h5) + PCS (PF_ATIME h5: PCRL01/lmsr/lmsz/PCPF),
save_precursor_npz 落盘, 供 paper_betan/export_all_slices + pick_best_slices 使用。

用法: python -m paper_betan.build_precursor_local --shot 159422
输出: results/freegsnke_precursors/<shot>/precursor.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from bc.gs_forward.precursor_export import save_precursor_npz  # noqa: E402
from bc.common.constants import PCPF_NAMES  # noqa: E402

EFIT_DIR = Path("/data/EFIT")
PF_DIR = Path("/data/PF_ATIME_dataset")
OUT = Path("results/freegsnke_precursors")


def build(shot: int, efit_dir: Path = EFIT_DIR, pf_dir: Path = PF_DIR) -> Path:
    # ---- EFIT h5: profile + R8/Z8 + ATIME ----
    with h5py.File(efit_dir / f"{int(shot)}.h5") as f:
        atime = np.asarray(f["ATIME"], dtype=np.float64).ravel()
        m = atime >= 0.0
        at = atime[m]
        T = int(at.size)

        def take(k):
            a = np.asarray(f[k], dtype=np.float64)
            return a[m] if a.shape[0] == atime.size else a

        R8 = take("R8"); Z8 = take("Z8")
        if R8.ndim == 2 and R8.shape[1] != 8 and R8.shape[0] == 8:
            R8 = R8.T; Z8 = Z8.T
        PPRIME, FFPRIM, FPOL = take("PPRIME"), take("FFPRIM"), take("FPOL")
        BCENTR = take("BCENTR").ravel(); BETAP = take("BETAP").ravel()
        RMAXIS = take("RMAXIS").ravel(); ZMAXIS = take("ZMAXIS").ravel()

    # ---- PF_ATIME h5: PCS (PCRL01/lmsr/lmsz/PCPF) ----
    with h5py.File(pf_dir / f"{int(shot)}.h5") as f:
        t = np.asarray(f["time"], dtype=np.float64).ravel()
        aligned = (t.size == T)
        if not aligned:
            lo, hi = float(max(t.min(), at.min())), float(min(t.max(), at.max()))
            at_c = np.clip(at, lo, hi)
        def pcs(k):
            v = np.asarray(f[k], dtype=np.float64).ravel()
            return v if aligned else np.interp(at_c, t, v)
        PCRL01 = pcs("PCRL01"); lmsr = pcs("lmsr"); lmsz = pcs("lmsz")
        PCPF = np.stack([pcs(n) for n in PCPF_NAMES], axis=1)  # (T, 12)

    bundle = {
        "shot": int(shot), "source": "local_efit_pf_h5",
        "efit_h5": str(efit_dir / f"{int(shot)}.h5"),
        "ATIME": at, "R8": R8, "Z8": Z8,
        "PPRIME": PPRIME, "FFPRIM": FFPRIM, "FPOL": FPOL,
        "BCENTR": BCENTR, "BETAP": BETAP, "RMAXIS": RMAXIS, "ZMAXIS": ZMAXIS,
        "PCRL01": PCRL01, "lmsr": lmsr, "lmsz": lmsz, "PCPF": PCPF,
        "PCPF_NAMES": list(PCPF_NAMES),
    }
    out = OUT / f"{int(shot)}" / "precursor.npz"
    save_precursor_npz(bundle, out)
    print(f"[build_precursor_local] shot {shot}: T={T} (aligned={aligned}) -> {out}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--efit-dir", default=str(EFIT_DIR))
    ap.add_argument("--pf-dir", default=str(PF_DIR))
    a = ap.parse_args()
    build(a.shot, Path(a.efit_dir), Path(a.pf_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
