#!/usr/bin/env python3
"""指定炮号：BC 整炮重推理 → 三阶段各取时间中点一步 → 与 EFIT ATIME 对齐导出 slice → 可选跑 freegsnke。

对每一阶段 ``k_bc``（BC 数据集时间轴中点）取 ``t_bc``，在 ``precursor.npz`` 的 ``ATIME`` 上
最近邻得 ``k_efit``，用 **第 k_efit 行** 的 PPRIME/FFPRIM/FPOL/BCENTR/R8Z8/…（与
``notebook/06`` / ``bc.precursor_export`` 一致），**PCPF 写入该步的模型预测电流**，
落盘 ``slice_pred_<phase>.npz`` 与 ``three_phases_manifest.json``。
其中 ``Ip_A`` 与 **BC 该步** ``state`` 中的 PCRL01 一致（写入 slice，供 freegsnke 使用），
剖面等仍来自 **EFIT 最近邻行** ``k_efit``。

完全离线 GS（不再连库）::

    python -m bc.batch_freegsnke.run_freegsnke_pred_three_phases --shot 100084 \\
      --precursor-npz results/freegsnke_precursors/100084/precursor.npz

若尚无 precursor，先在同一环境能连 MDS 时::

    python -m bc.precursor_export --shot 100084 --source auto

推理 + 导出 + 三发 ``run_freegsnke_eval``（需 torch + freegsnke）::

    python -m bc.batch_freegsnke.run_freegsnke_pred_three_phases --shot 100084 --run-gs

（``python -m bc.run_freegsnke_pred_three_phases`` 薄包装仍可用。）

输出根目录默认：``results/freegsnke_pred_three_phases/<shot>/``。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bc.gs_forward.infer_one_shot import infer_one_shot, pick_representative_steps
from bc.gs_forward.precursor_export import export_slice_npz, load_precursor_npz, nearest_row

_PHASE_ORDER = ("ramp_up", "flat_top", "ramp_down")
_REPO = Path(__file__).resolve().parent.parent.parent


def _default_precursor(shot: int) -> Path:
    return _REPO / "results" / "freegsnke_precursors" / str(shot) / "precursor.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument(
        "--precursor-npz",
        type=str,
        default=None,
        help="precursor_export 生成的全序列 npz；默认 results/freegsnke_precursors/{shot}/precursor.npz",
    )
    ap.add_argument("--ckpt", type=str, default=str(_REPO / "results/bc_v1/run1/checkpoints/best_val.pt"))
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="manifest 与 slice npz 目录；默认 results/freegsnke_pred_three_phases/{shot}",
    )
    ap.add_argument("--dataset-root", type=str, default=None, help="覆盖 ckpt 内 dataset_root（读 PF h5）")
    ap.add_argument("--device", type=str, default=None, help="torch device，默认自动 cuda/cpu")
    ap.add_argument(
        "--run-gs",
        action="store_true",
        help="对每个 phase 调用 python -m bc.run_freegsnke_eval --precursor-slice-npz ...",
    )
    ap.add_argument(
        "--gs-out-dir",
        type=str,
        default=None,
        help="run_freegsnke_eval 的 --out-dir；默认与 --out-dir 相同",
    )
    ap.add_argument("--rtol", type=float, default=8e-3)
    ap.add_argument("--nx", type=int, default=129)
    ap.add_argument("--ny", type=int, default=129)
    args = ap.parse_args()

    shot = int(args.shot)
    prec_path = Path(args.precursor_npz) if args.precursor_npz else _default_precursor(shot)
    if not prec_path.is_file():
        print(
            f"缺少 precursor 文件：{prec_path}\n"
            f"请先执行：python -m bc.precursor_export --shot {shot} --source auto",
            file=sys.stderr,
        )
        return 2

    out_root = Path(args.out_dir) if args.out_dir else _REPO / "results" / "freegsnke_pred_three_phases" / str(shot)
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        print(f"缺少 ckpt: {ckpt}", file=sys.stderr)
        return 2

    inf = infer_one_shot(
        shot,
        ckpt,
        device=args.device,
        dataset_root=args.dataset_root,
    )
    picks = pick_representative_steps(inf["phase_slices"])
    missing = [p for p in _PHASE_ORDER if p not in picks]
    if missing:
        print(f"该炮未检测到完整三阶段，缺少: {missing}；已有 picks={picks}", file=sys.stderr)
        return 3

    bundle = load_precursor_npz(prec_path)
    atime = np.asarray(bundle["ATIME"], dtype=np.float64).ravel()
    names = list(inf.get("pcpf_names", bundle.get("PCPF_NAMES", [])))

    slices_meta: list[dict[str, Any]] = []
    for phase in _PHASE_ORDER:
        k_bc = int(picks[phase])
        t_bc = float(np.asarray(inf["time"], dtype=np.float64).ravel()[k_bc])
        k_efit = nearest_row(atime, t_bc)
        t_efit = float(atime[k_efit])
        pred12 = np.asarray(inf["pred_A"][k_bc], dtype=np.float64).ravel().tolist()
        truth12 = np.asarray(inf["target_A"][k_bc], dtype=np.float64).ravel().tolist()
        ip_bc = float(inf["Ip_A"][k_bc])
        ip_efit_row = float(np.asarray(bundle["PCRL01"], dtype=np.float64).ravel()[k_efit])

        slice_path = out_root / f"slice_pred_{phase}_kbc{k_bc:04d}_kef{k_efit:04d}.npz"
        export_slice_npz(
            bundle,
            k_efit,
            slice_path,
            pcpf12_override=np.asarray(pred12, dtype=np.float64),
            ip_a_override=ip_bc,
        )

        row = {
            "phase": phase,
            "k_bc_step": k_bc,
            "t_bc_s": t_bc,
            "Ip_A_slice_npz_BcStep": ip_bc,
            "k_efit_row": k_efit,
            "t_efit_s": t_efit,
            "dt_efit_minus_bc_s": t_efit - t_bc,
            "Ip_A_PCRL01_efit_row": ip_efit_row,
            "pcpf_names": names,
            "pred_pcpf12_A": pred12,
            "truth_pcpf12_A": truth12,
            "slice_npz": str(slice_path.resolve()),
        }
        # 便于人工核对：EFIT 行内 PCS（数据库）与预测对比
        pcs_db = np.asarray(bundle["PCPF"][k_efit], dtype=np.float64).ravel().tolist()
        row["pcs_pcpf12_A_efit_row"] = pcs_db
        slices_meta.append(row)
        print(
            f"[{phase}] k_bc={k_bc} t_bc={t_bc:.6f}s  ->  k_efit={k_efit} t_efit={t_efit:.6f}s  "
            f"Δt={t_efit - t_bc:+.4e}s  slice={slice_path.name}",
            flush=True,
        )

    manifest = {
        "shot": shot,
        "ckpt": str(ckpt.resolve()),
        "precursor_npz": str(prec_path.resolve()),
        "out_dir": str(out_root.resolve()),
        "phase_representative_k_bc": picks,
        "slices": slices_meta,
    }
    man_path = out_root / "three_phases_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {man_path}", flush=True)

    if args.run_gs:
        gs_out = Path(args.gs_out_dir) if args.gs_out_dir else out_root
        py = sys.executable
        for row in slices_meta:
            phase = row["phase"]
            sp = row["slice_npz"]
            cmd = [
                py,
                "-m",
                "bc.run_freegsnke_eval",
                "--shot",
                str(shot),
                "--phase",
                phase,
                "--precursor-slice-npz",
                sp,
                "--out-dir",
                str(gs_out),
                "--rtol",
                str(args.rtol),
                "--nx",
                str(args.nx),
                "--ny",
                str(args.ny),
            ]
            if args.dataset_root:
                cmd.extend(["--dataset-root", args.dataset_root])
            print("RUN", " ".join(cmd), flush=True)
            r = subprocess.run(cmd, cwd=str(_REPO))
            if r.returncode != 0:
                print(f"[WARN] freegsnke_eval {phase} 退出码 {r.returncode}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
