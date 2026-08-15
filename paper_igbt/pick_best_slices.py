#!/usr/bin/env python3
"""登录节点 (无 torch): 对 134925 已导出的 bc_pred 切片逐帧 freegsnke 前向, 每阶段挑
「LCFS 与 EFIT R8/Z8 最契合」的一帧, 写入 plot_pred_wall_dist 读取的 manifest。

前置: ``paper_igbt/export_all_slices.py`` (DCU 推理) 已导出全部 EFIT 帧 bc_pred 切片到
``{out_root}/{shot}/bc_pred/_slices/slice_whole_bc_k*.npz``。

为什么拆开: 前向求解 (NKGSsolver, numba/CPU) 约 2.6s/帧, 在登录节点跑 110 帧 ~5min;
而推理需要 DCU/torch。把两者解耦后, 重活 (前向) 跑在稳定快的登录节点, 不受 DCU 节点
CPU 争抢影响 (整炮 110 帧在争抢的 DCU 节点上曾 ~4.5min/帧)。

指标: ``lcfs_rmsep_to_r8z8_m`` —— 与 ``run_freegsnke_eval``/``summary.json`` 同口径
(``rmse_separatrix_to_r8z8(eq.separatrix(240), r8, z8)``), 即 ``montage_best_per_phase``
所用「每阶段最优」定义。成图时 plot_pred_wall_dist 会以全分辨率 (nx=129, maxit=80) 重解
这 3 帧, 故选片用同口径即可。

示例 (登录节点)::

    python paper_igbt/pick_best_slices.py --shot 134925
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_REPO / ".numba_cache"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
# 登录节点做个礼貌的单核公民: 前向 numba 并行对单帧几乎无收益 (实测 16/4/1 线程均 ~2.6s)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from bc.gs_forward.precursor_export import load_precursor_npz  # noqa: E402
from bc_notime.data.phases import PHASE_NAMES, phase_ids_per_step  # noqa: E402
from bc_notime.batch_freegsnke.batch_efit_self_overlay07 import (  # noqa: E402
    _discharge_phase_for_precursor_k,
)
# 复用 plot_pred_wall_dist 的前向求解 (含 EAST 机器/剖面构造) —— 保证排名口径与成图一致
from paper_igbt.plot_pred_wall_dist import _forward_solve_slice  # noqa: E402


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return float(np.hypot(px - ax, py - ay))
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return float(np.hypot(px - cx, py - cy))


def rmse_separatrix_to_r8z8(sep: np.ndarray, r8: np.ndarray, z8: np.ndarray) -> float:
    """与 bc.gs_forward.run_freegsnke_eval.rmse_separatrix_to_r8z8 完全一致 (就地复制避免副作用)。"""
    r8 = np.asarray(r8, dtype=np.float64).ravel()
    z8 = np.asarray(z8, dtype=np.float64).ravel()
    if sep is None or len(sep) < 3:
        return float("nan")
    d2 = []
    for i in range(sep.shape[0]):
        px, py = float(sep[i, 0]), float(sep[i, 1])
        dm = 1e30
        for k in range(8):
            ax_, ay_ = r8[k], z8[k]
            bx_, by_ = r8[(k + 1) % 8], z8[(k + 1) % 8]
            dm = min(dm, _point_to_segment_dist(px, py, ax_, ay_, bx_, by_))
        d2.append(dm * dm)
    return float(np.sqrt(np.mean(d2)))


def rmse_8ctrl_to_r8z8(sep: np.ndarray, r8: np.ndarray, z8: np.ndarray) -> float:
    """8↔8 控制点逐点 Euclidean RMSE (用户指定, 与 EFIT 同口径)。

    scan_data.compat.shape_params 与 EFIT_tools.shape_params 同算法: 从 LCFS 点序列
    提取 8 控制点 (外 argmax R / 上 argmax Z / 内 argmin R / 下 argmin Z + 4 象限平方点),
    方向自适应。FreeGSNKE 的 sep 与 EFIT 每帧 r8/z8 同 idx 逐点对应, 二维 Euclidean RMSE。
    替代旧 rmse_separatrix_to_r8z8 (240 点曲线到 8 点多边形的点到线段距离)。
    """
    from scan_data.compat import shape_params
    if sep is None or len(sep) < 3:
        return float("nan")
    r8 = np.asarray(r8, dtype=np.float64).ravel()
    z8 = np.asarray(z8, dtype=np.float64).ravel()
    sp = shape_params(np.asarray(sep[:, 0], dtype=np.float64),
                      np.asarray(sep[:, 1], dtype=np.float64))
    r8p = np.asarray(sp.R8, dtype=np.float64).ravel()
    z8p = np.asarray(sp.Z8, dtype=np.float64).ravel()
    if r8p.size != 8 or z8p.size != 8 or np.isnan(r8p).any() or np.isnan(z8p).any():
        return float("nan")
    d = np.hypot(r8p - r8, z8p - z8)
    return float(np.sqrt(np.mean(d ** 2)))


def _worker_solve(slice_path: str, pf_xml: str, wall_xml: str,
                  nx: int, maxit: int, rtol: float, q) -> None:
    """子进程: 前向求解单帧 + 算 lcfs_rmsep, 结果塞进 queue。父进程可超时 terminate。"""
    try:
        solved = _forward_solve_slice(Path(slice_path), pf_xml, wall_xml,
                                      nx=nx, ny=nx, rtol=rtol, maxit=maxit)
        if solved is None:
            q.put({"rmsep": None, "converged": False}); return
        eq, snap = solved
        sep = np.asarray(eq.separatrix(ntheta=240))
        rmsep = rmse_8ctrl_to_r8z8(sep, snap["r8"], snap["z8"])
        q.put({"rmsep": (float(rmsep) if np.isfinite(rmsep) else None), "converged": True})
    except Exception as e:  # 退化帧常见 RuntimeWarning/divide-by-zero, 捕获后跳过
        q.put({"rmsep": None, "converged": False, "error": f"{type(e).__name__}: {e}"})


def _solve_with_timeout(slice_path: Path, pf_xml, wall_xml, *,
                        nx: int, maxit: int, rtol: float, timeout: float) -> dict:
    """fork 子进程求解 (继承父进程已编译的 numba, 无重加载开销); 超时则 terminate 跳过。"""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker_solve,
                    args=(str(slice_path), str(pf_xml), str(wall_xml), nx, maxit, rtol, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(10)
        return {"rmsep": None, "converged": False, "timed_out": True}
    try:
        return q.get_nowait()
    except Exception:
        return {"rmsep": None, "converged": False, "timed_out": False}


def main() -> int:
    from bc.gs_forward.freegsnke_east_machine import PF_XML_DEFAULT, WALL_XML_DEFAULT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=134925)
    ap.add_argument("--precursor", type=str,
                    default="results/freegsnke_precursors/134925/precursor.npz")
    ap.add_argument("--out-root", type=str,
                    default="results/paper_igbt/freegsnke_whole_shot")
    ap.add_argument("--branch", type=str, default="bc_pred")
    ap.add_argument("--k-start", type=int, default=0)
    ap.add_argument("--k-stop", type=int, default=None, help="不含; 默认 len(ATIME)")
    ap.add_argument("--rtol", type=float, default=8e-3,
                    help="前向收敛相对容差; 默认 8e-3 (与 run_freegsnke_whole_shot/summary.json 同口径, "
                         "比成图的 1e-3 宽松→难收敛帧不至于迭代到 maxit 而拖慢排序)")
    ap.add_argument("--nx", type=int, default=65,
                    help="排序用网格分辨率 (粗→快, 单次迭代约 (65/129)^2≈1/4 成本); 成图仍用 129。"
                         "关键是给任何一帧设上限, 避免个别难收敛帧 (如早期 flat-top) 拖垮整批")
    ap.add_argument("--maxit", type=int, default=40,
                    help="NK 前向最大迭代数 (排序用); 与 nx 配合保证单帧不卡死")
    ap.add_argument("--per-frame-timeout", type=float, default=120.0,
                    help="单帧墙钟上限(秒); 超时 terminate 该子进程并跳过此帧 "
                         "(个别退化帧可数十分钟, 用此硬上限保证整批有界)")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    shot = int(args.shot)
    bundle = load_precursor_npz(Path(args.precursor).resolve())
    atime = np.asarray(bundle["ATIME"], dtype=np.float64).ravel()
    ipr = np.asarray(bundle["PCRL01"], dtype=np.float64).ravel()
    te = int(atime.size)
    phase_ids = phase_ids_per_step(atime, ipr)

    slices_dir = out_root / str(shot) / args.branch / "_slices"
    k0 = max(0, int(args.k_start))
    k1 = min(te, int(args.k_stop) if args.k_stop is not None else te)

    ranking: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any] | None] = {ph: None for ph in PHASE_NAMES}
    done = 0
    skipped = 0

    # 预热: 在父进程内先编译 numba (前几次 run 显示 k=0..12 都秒解, 用首个存在的切片即可),
    # 这样之后 fork 出去的子进程靠 COW 继承已编译代码, 无逐帧重加载开销。
    for kw in range(k0, k1):
        warm = slices_dir / f"slice_whole_bc_k{kw:05d}.npz"
        if warm.is_file():
            print(f"[warmup] 编译 numba (父进程, k={kw}) ...", flush=True)
            _forward_solve_slice(warm, PF_XML_DEFAULT, WALL_XML_DEFAULT,
                                 nx=args.nx, ny=args.nx, rtol=args.rtol, maxit=args.maxit)
            print("[warmup] done", flush=True)
            break

    for k in range(k0, k1):
        ph = _discharge_phase_for_precursor_k(phase_ids, k)
        if ph not in PHASE_NAMES:
            continue  # 放电外 / 无效帧
        slice_path = slices_dir / f"slice_whole_bc_k{k:05d}.npz"
        if not slice_path.is_file():
            continue  # 未导出, 跳过
        t_ef = float(atime[k])
        res = _solve_with_timeout(
            slice_path, PF_XML_DEFAULT, WALL_XML_DEFAULT,
            nx=args.nx, maxit=args.maxit, rtol=args.rtol, timeout=args.per_frame_timeout)
        rmsep = res.get("rmsep")

        if rmsep is None:
            skipped += 1
            tag = "TIMEOUT (跳过)" if res.get("timed_out") else "FAILED (跳过)"
            print(f"  k={k:3d} t={t_ef:7.3f}s {ph:9s} -> {tag}", flush=True)
            ranking.append({"k": k, "time_s": t_ef, "phase": ph, "lcfs_rmsep_m": None,
                            "converged": bool(res.get("converged", False)),
                            "timed_out": bool(res.get("timed_out", False))})
            continue

        done += 1
        ranking.append({"k": k, "time_s": t_ef, "phase": ph,
                        "lcfs_rmsep_m": float(rmsep), "converged": True})
        cur = best[ph]
        if cur is None or rmsep < cur["lcfs_rmsep_m"]:
            best[ph] = {"k": k, "time_s": t_ef, "lcfs_rmsep_m": float(rmsep)}
        flag = " <-- best" if (best[ph] and best[ph]["k"] == k) else ""
        print(f"  k={k:3d} t={t_ef:7.3f}s {ph:9s} rmsep={rmsep:.4f} m{flag}", flush=True)

    print(f"\n[solved] {done} frames, skipped {skipped}", flush=True)

    # 写 manifest (plot_pred_wall_dist 读这个)
    manifest: dict[str, Any] = {
        "shot": shot, "converged_only": False,
        "metric": "bc_pred 8-control-point Euclidean RMSE via shape_params (best per phase; paper_igbt)",
        "rank_solve": {"nx": args.nx, "maxit": args.maxit, "rtol": args.rtol,
                       "note": "粗网格/限迭代仅用于排序; 成图以 nx=129 rtol=1e-3 重解"},
    }
    # 对每相 bc_pred 选中的 k*, 额外求解 pcs 切片 -> 参考行 LCFS 误差 (与 montage PCS 行同口径)
    pcs_dir = out_root / str(shot) / "pcs" / "_slices"
    print("[best per phase + pcs reference]")
    for ph in PHASE_NAMES:
        b = best[ph]
        if b is None:
            manifest[ph] = None
            print(f"  {ph:9s}: (no valid frame)")
            continue
        k_star = int(b["k"])
        pcs_rmsep = None
        pcs_slice = pcs_dir / f"slice_whole_pcs_k{k_star:05d}.npz"
        if pcs_slice.is_file():
            pres = _solve_with_timeout(pcs_slice, PF_XML_DEFAULT, WALL_XML_DEFAULT,
                                       nx=args.nx, maxit=args.maxit, rtol=args.rtol,
                                       timeout=args.per_frame_timeout)
            if pres.get("rmsep") is not None:
                pcs_rmsep = float(pres["rmsep"])
        bc_r = float(b["lcfs_rmsep_m"])
        manifest[ph] = {
            "k": k_star,
            "time_s": float(b["time_s"]),
            "bc_lcfs_rmsep_m": bc_r,
            "pcs_lcfs_rmsep_m": pcs_rmsep,
            "sum_rmsep_m": (bc_r + pcs_rmsep) if pcs_rmsep is not None else None,
        }
        pcs_str = "NA" if pcs_rmsep is None else f"{pcs_rmsep:.4f} m"
        print(f"  {ph:9s}: k={k_star} t={b['time_s']:.3f}s  bc={bc_r:.4f} m  pcs={pcs_str}")

    man_dir = out_root / "_montage"
    man_dir.mkdir(parents=True, exist_ok=True)
    man_path = man_dir / f"montage_pcs_vs_pred_shot{shot}.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[manifest] {man_path}", flush=True)

    rank_path = man_dir / f"montage_pcs_vs_pred_ranking_shot{shot}.json"
    rank_path.write_text(json.dumps(
        {"shot": shot, "metric": "bc_pred 8-control-point Euclidean RMSE via shape_params",
         "best": {ph: best[ph] for ph in PHASE_NAMES}, "frames": ranking},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ranking]  {rank_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
