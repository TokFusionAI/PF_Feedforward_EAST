#!/usr/bin/env python3
"""DCU only: 用最优模型 (s44) 对 134925 整炮推理, 导出**所有** EFIT 帧的 bc_pred 切片。

切片里的 ``pcpf12`` 即模型预测电流 (EFIT 第 k 行剖面/R8Z8), 是后续前向求解的唯一
模型相关输入。把「推理 (需 DCU/torch)」与「前向求解 (纯 CPU/numba, 2.6s/帧)」拆开:
本脚本只做推理 + 导出 (~2min on DCU); 排序与画图在登录节点跑 (无 torch)。

帧映射与 run_freegsnke_whole_shot 的 bc_pred 支路一致: k_bc = argmin|model_time - ATIME[k]|。

示例 (DCU 节点, 先 source DTK env)::

    python paper_betan/export_all_slices.py --shot 134925
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from bc.gs_forward.precursor_export import export_slice_npz  # noqa: E402 (模型无关)
from paper_betan.freegsnke_infer import infer_one_shot  # noqa: E402 (需 torch/DCU)
from paper_betan.model import MODEL_CKPT  # noqa: E402


def _load_bundle(precursor_npz: Path) -> dict:
    z = np.load(precursor_npz, allow_pickle=False)
    keys = ["ATIME", "R8", "Z8", "PPRIME", "FFPRIM", "FPOL", "BCENTR", "BETAP",
            "RMAXIS", "ZMAXIS", "PCRL01", "lmsr", "lmsz", "PCPF"]
    b = {k: np.asarray(z[k]) for k in keys}
    z.close()
    b["shot"] = int(Path(precursor_npz).parent.name)
    return b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=134925)
    ap.add_argument("--ckpt", type=str, default=str(MODEL_CKPT))
    ap.add_argument("--precursor", type=str,
                    default="results/freegsnke_precursors/134925/precursor.npz")
    ap.add_argument("--out-root", type=str,
                    default="results/paper_betan/freegsnke_whole_shot")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    shot = int(args.shot)
    precursor = Path(args.precursor).resolve()
    bundle = _load_bundle(precursor)
    atime = np.asarray(bundle["ATIME"], dtype=np.float64).ravel()
    te = int(atime.size)

    print(f"[infer] shot {shot} ckpt={Path(args.ckpt).name} ...", flush=True)
    inf = infer_one_shot(shot, args.ckpt, device=args.device)
    model_time = np.asarray(inf["time"], dtype=np.float64).ravel()
    pred_a = np.asarray(inf["pred_A"], dtype=np.float64)
    ip_bc_a = np.asarray(inf["Ip_A"], dtype=np.float64).ravel()
    print(f"[infer] T_model={model_time.size}  ATIME={te}", flush=True)

    slices_dir = out_root / str(shot) / "bc_pred" / "_slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for k in range(te):
        t_ef = float(atime[k])
        k_bc = int(np.argmin(np.abs(model_time - t_ef)))
        pred12 = np.asarray(pred_a[k_bc], dtype=np.float64).ravel()
        ip_b = float(ip_bc_a[k_bc])
        slice_path = slices_dir / f"slice_whole_bc_k{k:05d}.npz"
        export_slice_npz(bundle, k, slice_path,
                         pcpf12_override=pred12, ip_a_override=ip_b)
        n += 1
    print(f"[done] exported {n} bc_pred slices -> {slices_dir}", flush=True)

    # pcs 分支: 实际 PCS 电流 (EFIT 自洽 PCPF), 供 montage 参考行 / plot_montage 两行对比
    pcs_dir = out_root / str(shot) / "pcs" / "_slices"
    pcs_dir.mkdir(parents=True, exist_ok=True)
    pcpf_all = np.asarray(bundle["PCPF"], dtype=np.float64)  # (te, 12)
    n_pcs = 0
    for k in range(te):
        export_slice_npz(bundle, k, pcs_dir / f"slice_whole_pcs_k{k:05d}.npz",
                         pcpf12_override=pcpf_all[k])
        n_pcs += 1
    print(f"[done] exported {n_pcs} pcs slices -> {pcs_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
