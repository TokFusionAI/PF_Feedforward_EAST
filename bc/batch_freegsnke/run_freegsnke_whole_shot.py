#!/usr/bin/env python3
"""整炮（precursor 全 ATIME）逐帧 freegsnke 静态前向平衡。

对每一 EFIT 时间索引 ``k``：导出单帧 ``slice`` → 调用 ``bc.run_freegsnke_eval``（与现有 eval 管线一致）。

**线圈电流**

- ``pcs``：与 ``export_slice_npz(bundle, k)`` 相同，PCPF/Ip 均为该 EFIT 行（与 EFIT 自洽 PCS）。
- ``bc_pred``：剖面/R8Z8 等仍取 EFIT 第 ``k`` 行；PCPF 与 Ip 取 BC 模型时间上 **与 ``ATIME[k]`` 最近邻** 的一步（与 ``run_five_slices_per_phase_freegsnke`` 的代表步选取不同，此处按 EFIT 栅格遍历）。

**输出目录**（默认在仓库 ``results/freegsnke_whole_shot/{shot}/{coil_source}/``）::

    _slices/   （或 ``--slices-subdir``）
    by_t_efit/k{kkkkk}_t{t_efit}/   → ``07_overlay_...png``、``summary.json`` 等

须有已安装 ``freegsnke`` 的 Python 环境；BC 支路需要 torch + checkpoint。

示例::

    # PCS（整炮 EFIT 自洽电流）
    python -m bc.batch_freegsnke.run_freegsnke_whole_shot --shot 105653 \\
      --precursor-npz results/freegsnke_precursors/105653/precursor.npz --coil-source pcs

    # BC 预测电流 + 同帧 EFIT 剖面
    python -m bc.batch_freegsnke.run_freegsnke_whole_shot --shot 105653 \\
      --precursor-npz results/freegsnke_precursors/105653/precursor.npz \\
      --coil-source bc_pred \\
      --ckpt results/bc_v1/run1/checkpoints/best_val.pt

    # 只重算、且仅 07 图（默认即 overlay-only=True 逻辑由本脚本 --overlay-only 控制）
    python -m bc.batch_freegsnke.run_freegsnke_whole_shot ... --skip-existing --overlay-only
"""

# 多进程 freegsnke：``--eval-workers`` + ``--freegsnke-cpu``（子进程 ``CUDA_VISIBLE_DEVICES=``）避免与主进程 BC GPU 争抢。

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from bc.batch_freegsnke.batch_efit_self_overlay07 import (
    _dir_name_k_t_efit,
    _discharge_phase_for_precursor_k,
    _run_one_eval,
)
from bc.data.phases import phase_ids_per_step
from bc.gs_forward.infer_one_shot import infer_one_shot
from bc.gs_forward.precursor_export import export_slice_npz, load_precursor_npz

_REPO = Path(__file__).resolve().parent.parent.parent
MANIFEST_NAME = "whole_shot_freegsnke_manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument(
        "--precursor-npz",
        type=str,
        default=None,
        help="precursor.npz；默认 results/freegsnke_precursors/{shot}/precursor.npz",
    )
    ap.add_argument(
        "--coil-source",
        type=str,
        choices=("pcs", "bc_pred"),
        default="pcs",
        help="pcs: 该 EFIT 行 PCS；bc_pred: BC 网络在时间上最近邻到 ATIME[k] 的预测 + 该 EFIT 行剖面",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default=str(_REPO / "results" / "freegsnke_whole_shot"),
        help="输出根目录；实际写入 {out_root}/{shot}/{coil-source}/",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(_REPO / "results" / "bc_v1" / "run1" / "checkpoints" / "best_val.pt"),
        help="--coil-source bc_pred 时使用的 BC 权重",
    )
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument(
        "--mds-server",
        type=str,
        default=None,
        help="无本地 {shot}.h5 时从 MDS 建 BC 样本（覆盖 MDS_HOSTNAME）",
    )
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--k-start", type=int, default=0)
    ap.add_argument("--k-stop", type=int, default=None, help="不含；默认 len(ATIME)")
    ap.add_argument("--reuse-slices", action="store_true", help="已存在对应 npz 则跳过 export")
    ap.add_argument("--skip-existing", action="store_true", help="已有 07_overlay 则跳过 eval")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rtol", type=float, default=8e-3)
    ap.add_argument("--nx", type=int, default=129)
    ap.add_argument("--ny", type=int, default=129)
    ap.add_argument(
        "--by-t-efit-subdir",
        type=str,
        default="by_t_efit",
        help="各帧 eval 输出父目录名",
    )
    ap.add_argument(
        "--slices-subdir",
        type=str,
        default="_slices",
        help="导出 slice npz 的子目录",
    )
    ap.add_argument("--overlay-only", action="store_true", help="eval 只生成 07 + summary")
    ap.add_argument(
        "--full-plots",
        action="store_true",
        help="eval 生成 00–07（与 --overlay-only 互斥）",
    )
    ap.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="并行启动多少个 run_freegsnke_eval 子进程（线程池；建议与 --freegsnke-cpu 同用）",
    )
    ap.add_argument(
        "--freegsnke-cpu",
        action="store_true",
        help="子进程内屏蔽 GPU（CUDA_VISIBLE_DEVICES= 空），让 freegsnke 走 CPU，不与当前进程 BC 用的一张卡争抢",
    )
    ap.add_argument(
        "--freegsnke-omp-threads",
        type=int,
        default=None,
        help="写入子进程 OMP_NUM_THREADS；默认在 --freegsnke-cpu 时为 1，或用环境变量 FREEGSNKE_OMP_THREADS",
    )
    args = ap.parse_args()

    if args.full_plots and args.overlay_only:
        raise SystemExit("不可同时使用 --full-plots 与 --overlay-only")
    overlay_only = bool(args.overlay_only) or not bool(args.full_plots)

    prec_path = (
        Path(args.precursor_npz).resolve()
        if args.precursor_npz
        else (_REPO / "results" / "freegsnke_precursors" / str(int(args.shot)) / "precursor.npz")
    )
    if not prec_path.is_file():
        raise SystemExit(f"缺少 precursor：{prec_path}")

    shot_run = Path(args.out_root).resolve() / str(int(args.shot)) / str(args.coil_source)
    slices_dir = shot_run / args.slices_subdir
    by_root = shot_run / args.by_t_efit_subdir
    slices_dir.mkdir(parents=True, exist_ok=True)
    by_root.mkdir(parents=True, exist_ok=True)

    bundle = load_precursor_npz(prec_path)
    bundle["shot"] = int(args.shot)
    atime = np.asarray(bundle["ATIME"], dtype=np.float64).ravel()
    ipr = np.asarray(bundle["PCRL01"], dtype=np.float64).ravel()
    te = int(atime.size)
    if ipr.size != te:
        raise SystemExit(f"ATIME 长度 {te} 与 PCRL01 {ipr.size} 不一致")
    phase_ids = phase_ids_per_step(atime, ipr)

    k0 = max(0, int(args.k_start))
    k1 = int(args.k_stop) if args.k_stop is not None else te
    k1 = min(te, k1)
    if k0 >= k1:
        raise SystemExit(f"空索引范围 [{k0},{k1})")

    inf = None
    if args.coil_source == "bc_pred":
        ckpt = Path(args.ckpt)
        if not ckpt.is_file():
            raise SystemExit(f"缺少 ckpt：{ckpt}")
        dev = args.device
        if not dev:
            try:
                import torch

                dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"
        inf = infer_one_shot(
            int(args.shot),
            ckpt,
            device=dev,
            dataset_root=args.dataset_root,
            mds_server=args.mds_server,
        )
        time_s = np.asarray(inf["time"], dtype=np.float64).ravel()
        pred_a = np.asarray(inf["pred_A"], dtype=np.float64)
        ip_bc_a = np.asarray(inf["Ip_A"], dtype=np.float64).ravel()
        if pred_a.shape[0] != time_s.size or ip_bc_a.size != time_s.size:
            raise SystemExit("infer_one_shot 时间轴与 pred/Ip 长度不一致")

    if args.eval_workers < 1:
        raise SystemExit("--eval-workers 须 >= 1")

    freegsnke_env: dict[str, str] = {}
    if args.freegsnke_cpu:
        freegsnke_env["CUDA_VISIBLE_DEVICES"] = ""
        if args.freegsnke_omp_threads is not None:
            freegsnke_env["OMP_NUM_THREADS"] = str(int(args.freegsnke_omp_threads))
        else:
            freegsnke_env["OMP_NUM_THREADS"] = os.environ.get("FREEGSNKE_OMP_THREADS", "1")

    frames: list[dict] = []
    max_rc = 0
    prefix = "whole_bc" if args.coil_source == "bc_pred" else "whole_pcs"

    pending_eval: list[tuple[int, Path, Path, str]] = []

    for k in range(k0, k1):
        t_ef = float(atime[k])
        dname = _dir_name_k_t_efit(k, t_ef)
        slice_path = slices_dir / f"slice_{prefix}_k{k:05d}.npz"
        out_dir = by_root / dname
        discharge_phase = _discharge_phase_for_precursor_k(phase_ids, k)

        if args.coil_source == "pcs":
            if not args.reuse_slices or not slice_path.is_file():
                if not args.dry_run:
                    export_slice_npz(bundle, k, slice_path)
        else:
            k_bc = int(np.argmin(np.abs(time_s - t_ef)))
            pred12 = np.asarray(pred_a[k_bc], dtype=np.float64).ravel()
            ip_b = float(ip_bc_a[k_bc])
            if not args.reuse_slices or not slice_path.is_file():
                if not args.dry_run:
                    export_slice_npz(
                        bundle,
                        k,
                        slice_path,
                        pcpf12_override=pred12,
                        ip_a_override=ip_b,
                    )

        frames.append(
            {
                "k_efit": k,
                "t_efit": t_ef,
                "k_bc_nearest": int(np.argmin(np.abs(time_s - t_ef))) if inf is not None else None,
                "discharge_phase": discharge_phase,
                "slice_npz": str(slice_path.resolve().relative_to(shot_run.resolve())),
                "eval_out": str(out_dir.resolve().relative_to(shot_run.resolve())),
            }
        )

        if args.dry_run:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        target_png = out_dir / "07_overlay_coils_lcfs_r8z8.png"
        if args.skip_existing and target_png.is_file():
            print(f"[skip-existing] {target_png}", flush=True)
            continue

        pending_eval.append((k, slice_path, out_dir, discharge_phase))

    def _run_eval_job(item: tuple[int, Path, Path, str]) -> tuple[int, int]:
        _k, sp, od, dp = item
        rc = _run_one_eval(
            slice_path=sp,
            out_dir=od,
            shot=int(args.shot),
            phase="flat_top",
            rtol=float(args.rtol),
            nx=int(args.nx),
            ny=int(args.ny),
            dataset_root=args.dataset_root,
            dry_run=False,
            out_phase=dp,
            overlay_only=overlay_only,
            subprocess_env=freegsnke_env if freegsnke_env else None,
        )
        return _k, rc

    if not args.dry_run and pending_eval:
        nw = int(args.eval_workers)
        if nw <= 1:
            for item in pending_eval:
                k_i, rc = _run_eval_job(item)
                if rc != 0:
                    print(f"[WARN] k={k_i} 退出码 {rc}", file=sys.stderr, flush=True)
                    max_rc = max(max_rc, rc)
        else:
            with ThreadPoolExecutor(max_workers=nw) as ex:
                futs = [ex.submit(_run_eval_job, item) for item in pending_eval]
                for fu in as_completed(futs):
                    k_i, rc = fu.result()
                    if rc != 0:
                        print(f"[WARN] k={k_i} 退出码 {rc}", file=sys.stderr, flush=True)
                        max_rc = max(max_rc, rc)

    if not args.dry_run:
        man = shot_run / MANIFEST_NAME
        man.write_text(
            json.dumps(
                {
                    "shot": int(args.shot),
                    "coil_source": str(args.coil_source),
                    "precursor_npz": str(prec_path.resolve()),
                    "ckpt": str(Path(args.ckpt).resolve()) if args.coil_source == "bc_pred" else None,
                    "k_range": [k0, k1],
                    "overlay_only": overlay_only,
                    "frames": frames,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[manifest] {man}（{len(frames)} 帧）", flush=True)
    else:
        print(f"[dry-run] 将处理 {len(frames)} 帧 k∈[{k0},{k1})", flush=True)

    return min(max_rc, 255) if max_rc > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
