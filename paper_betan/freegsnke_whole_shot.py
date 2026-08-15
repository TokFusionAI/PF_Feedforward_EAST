"""单炮 134925 whole-shot 式: 用 s44 推理 → 每相代表性帧 → 导出 bc_pred(s44)+pcs 切片 + manifest。

产出 scripts_notime/plot_{montage,flattop_pred_only,pred_only_wall_dist}.py 期望的布局:
  {out_root}/134925/{bc_pred,pcs}/_slices/slice_whole_{bc,pcs}_k{k:05d}.npz
  {out_root}/_montage/montage_pcs_vs_pred_shot134925.json
然后 F5/F6/F8 直接复用 scripts_notime/plot_*.py --whole-shot-root {out_root}。

帧映射: dataset 帧 k_ds (来自 phase_slices) → time → nearest_row(ATIME) → EFIT 帧 k_efit (鲁棒)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_betan.model import BEST_CKPT  # noqa: E402
from paper_betan.freegsnke_infer import infer_one_shot_best  # noqa: E402
from bc.gs_forward.precursor_export import export_slice_npz, nearest_row  # noqa: E402 (模型无关)
from bc.gs_forward.infer_one_shot import pick_representative_steps  # noqa: E402 (模型无关)
from bc.common.constants import PCPF_NAMES  # noqa: E402


def load_bundle(precursor_npz: Path) -> dict:
    z = np.load(precursor_npz, allow_pickle=False)
    keys = ["ATIME", "R8", "Z8", "PPRIME", "FFPRIM", "FPOL", "BCENTR", "BETAP",
            "RMAXIS", "ZMAXIS", "PCRL01", "lmsr", "lmsz", "PCPF"]
    b = {k: np.asarray(z[k]) for k in keys}
    z.close()
    b["shot"] = int(Path(precursor_npz).parent.name)
    b["PCPF_NAMES"] = list(PCPF_NAMES)
    b["source"] = "precursor_npz"
    return b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=134925)
    ap.add_argument("--ckpt", default=str(BEST_CKPT))
    ap.add_argument("--precursor", default="results/freegsnke_precursors/134925/precursor.npz")
    ap.add_argument("--out-root", default="results/paper_betan/freegsnke_whole_shot")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    inf = infer_one_shot_best(args.shot, args.ckpt, device=args.device)
    bundle = load_bundle(Path(args.precursor))
    picks = pick_representative_steps(inf["phase_slices"])  # {phase: k_ds}
    print(f"shot {args.shot}: T_eff={len(inf['time'])}  ATIME={len(bundle['ATIME'])}  phases={list(picks)}")

    root = Path(args.out_root)
    manifest = {"shot": args.shot, "converged_only": False}
    for phase, k_ds in picks.items():
        t_s = float(inf["time"][k_ds])
        k_efit = nearest_row(bundle["ATIME"], t_s)
        if k_efit >= len(inf["pred_A"]):
            print(f"  [warn] {phase}: k_efit={k_efit} >= T_eff={len(inf['pred_A'])}, clamp")
            k_efit = len(inf["pred_A"]) - 1
        bc_out = root / f"{args.shot}" / "bc_pred" / "_slices" / f"slice_whole_bc_k{k_efit:05d}.npz"
        pcs_out = root / f"{args.shot}" / "pcs" / "_slices" / f"slice_whole_pcs_k{k_efit:05d}.npz"
        export_slice_npz(bundle, k_efit, bc_out, pcpf12_override=inf["pred_A"][k_ds])     # s44 预测
        export_slice_npz(bundle, k_efit, pcs_out, pcpf12_override=bundle["PCPF"][k_efit])  # 实际 PCS
        manifest[phase] = {"k": int(k_efit), "time_s": t_s,
                           "pcs_lcfs_rmsep_m": 0.0, "bc_lcfs_rmsep_m": 0.0, "sum_rmsep_m": 0.0}
        print(f"  {phase}: k_ds={k_ds} t={t_s:.3f}s -> k_efit={k_efit}  (bc+pcs slices written)")

    man_path = root / "_montage" / f"montage_pcs_vs_pred_shot{args.shot}.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote manifest {man_path}")
    print(f"\n下一步 F5/F6/F8: python scripts_notime/plot_montage_paper.py "
          f"--shot {args.shot} --whole-shot-root {root} --precursor-root results/freegsnke_precursors "
          f"--out-dir results/paper_betan/figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
