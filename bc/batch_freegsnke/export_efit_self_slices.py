#!/usr/bin/env python3
"""从已有 ``precursor.npz`` 导出 **EFIT 全帧自洽** 单帧 slice（不经过 torch）。

每一帧：该行 **PCPF（PCS）+ PCRL01（Ip）+ PPRIME/FFPRIM/…** 均来自 **同一 ATIME 行**，
与 ``export_slice_npz`` 默认行为一致。用于 ``run_freegsnke_eval --precursor-slice-npz`` 前向求解，
与 EFIT R8/Z8 对照（应明显近于 ``slice_pred_*`` + BC 电流的情形）。

典型流程（与三阶段 manifest 对齐的 k_efit）::

    python -m bc.batch_freegsnke.export_efit_self_slices --shot 100084 \\
      --from-manifest results/freegsnke_pred_three_phases/100084/three_phases_manifest.json

导出目录默认 ``results/freegsnke_efit_self_slices/<shot>/``，文件名
``slice_efit_self_<phase>_kXXXX.npz``（``run_freegsnke_eval`` 可从文件名推断 ``out_phase``）。

可选一键跑三次 GS::

    python -m bc.batch_freegsnke.export_efit_self_slices --shot 100084 --from-manifest ... --run-gs \\
      --gs-out-dir results/freegsnke_eval_efit_self

（``python -m bc.export_efit_self_slices`` 薄包装仍可用。）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from bc.gs_forward.precursor_export import export_slice_npz, load_precursor_npz

_REPO = Path(__file__).resolve().parent.parent.parent


def _default_manifest(shot: int) -> Path:
    return _REPO / "results" / "freegsnke_pred_three_phases" / str(shot) / "three_phases_manifest.json"


def _default_precursor(shot: int) -> Path:
    return _REPO / "results" / "freegsnke_precursors" / str(shot) / "precursor.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--precursor-npz", type=str, default=None, help="默认 results/freegsnke_precursors/{shot}/precursor.npz")
    ap.add_argument(
        "--from-manifest",
        type=str,
        default=None,
        help="three_phases_manifest.json（读每条的 phase、k_efit_row）；默认若存在则用 pred_three_phases 下该炮 manifest",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="写出 slice 与 manifest；默认 results/freegsnke_efit_self_slices/{shot}",
    )
    ap.add_argument("--run-gs", action="store_true", help="对每个 slice 调用 bc.run_freegsnke_eval")
    ap.add_argument("--gs-out-dir", type=str, default=None, help="--out-dir 传给 run_freegsnke_eval；默认与 --out-dir 相同")
    ap.add_argument("--rtol", type=float, default=8e-3)
    ap.add_argument("--nx", type=int, default=129)
    ap.add_argument("--ny", type=int, default=129)
    args = ap.parse_args()

    shot = int(args.shot)
    prec = Path(args.precursor_npz) if args.precursor_npz else _default_precursor(shot)
    if not prec.is_file():
        print(f"缺少 precursor：{prec}", file=sys.stderr)
        return 2

    man_path = Path(args.from_manifest) if args.from_manifest else _default_manifest(shot)
    if not man_path.is_file():
        print(f"缺少 manifest：{man_path}\n请先生成 three_phases_manifest.json 或显式 --from-manifest", file=sys.stderr)
        return 2

    out_root = Path(args.out_dir) if args.out_dir else _REPO / "results" / "freegsnke_efit_self_slices" / str(shot)
    out_root.mkdir(parents=True, exist_ok=True)

    bundle = load_precursor_npz(prec)
    meta = json.loads(man_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = meta.get("slices") or []
    if not rows:
        print("manifest 中无 slices 列表", file=sys.stderr)
        return 3

    exported: list[dict[str, Any]] = []
    for row in rows:
        phase = str(row["phase"])
        k = int(row["k_efit_row"])
        t_efit = float(bundle["ATIME"].ravel()[k])
        out_npz = out_root / f"slice_efit_self_{phase}_k{k:04d}.npz"
        export_slice_npz(bundle, k, out_npz)
        exported.append(
            {
                "phase": phase,
                "k_efit": k,
                "t_efit_s": t_efit,
                "slice_npz": str(out_npz.resolve()),
            }
        )
        print(f"[{phase}] k_efit={k} t={t_efit:.6f}s -> {out_npz.name}", flush=True)

    out_meta = {
        "shot": shot,
        "precursor_npz": str(prec.resolve()),
        "source_manifest": str(man_path.resolve()),
        "description": "PCS + PCRL01 + EFIT profiles same ATIME row; forward GS benchmark vs EFIT R8/Z8",
        "slices": exported,
    }
    mp = out_root / "efit_self_slices_manifest.json"
    mp.write_text(json.dumps(out_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {mp}", flush=True)

    if args.run_gs:
        gs_out = Path(args.gs_out_dir) if args.gs_out_dir else out_root
        py = sys.executable
        for item in exported:
            cmd = [
                py,
                "-m",
                "bc.run_freegsnke_eval",
                "--shot",
                str(shot),
                "--phase",
                item["phase"],
                "--precursor-slice-npz",
                item["slice_npz"],
                "--out-dir",
                str(gs_out.resolve()),
                "--rtol",
                str(args.rtol),
                "--nx",
                str(args.nx),
                "--ny",
                str(args.ny),
            ]
            print("RUN", " ".join(cmd), flush=True)
            r = subprocess.run(cmd, cwd=str(_REPO))
            if r.returncode != 0:
                print(f"[WARN] {item['phase']} 退出码 {r.returncode}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
