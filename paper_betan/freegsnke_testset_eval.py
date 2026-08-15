#!/usr/bin/env python3
"""测试集级 FreeGSNKE 评估 (多进程 + 分片版): 采样炮 × 每阶段几帧, 粗网格, 收敛+几何统计。

加速:
  - FreeGSNKE 单帧 numba 多线程无收益 (65² 网格小) -> 多进程并行多帧 (每进程 NUMBA_NUM_THREADS=1)。
  - 推理逐炮 DCU (~2s/炮) 是另一瓶颈 -> sbatch --array 分节点并行 (每 shard 1/n 炮)。

数据流 (绕过 MDS precursor): bundle 只从本地 EFIT h5 构造 (EFIT profile + R8/Z8 + ATIME);
PCS 填 dummy (bc_pred 分支 pcpf12/Ip 由模型 override, FreeGSNKE 不读 lmsr/lmsz)。模型推理
(transformer_bidir_on betan ckpt) 提供 PCPF + Ip。粗网格 (nx=65) FreeGSNKE 求解, 收敛信息
(rel_change/converged/method) + shape_params 8 控制点 (与 EFIT r8/z8 同口径) Euclidean RMSE
+ 逐控制点 ΔR/ΔZ (dr_i/dz_i, i=0..7)。

分片: --n-shards S --shard-id i -> 处理 shots[i::S]; 输出 per_frame_shard{i}.csv。
合并 + 画图: python -m paper_betan.aggregate_freegsnke_testset。

用法 (DCU 计算节点):
  python -m paper_betan.freegsnke_testset_eval --n-shots 999999 --frames-per-phase 3 \
      --n-proc 60 --n-shards 3 --shard-id 0
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from bc.gs_forward.precursor_export import export_slice_npz  # noqa: E402
from bc.gs_forward.mds_efit_snapshot import EFIT_DB_DEFAULT  # noqa: E402
from bc.data.phases import phase_ids_per_step  # noqa: E402
from scan_data.compat import shape_params  # noqa: E402

PHASE_NAMES = ["ramp_up", "flat_top", "ramp_down"]

# Pool worker 读的全局 (fork 前在 main 设)
_PF = None
_WALL = None
_NX = 65
_MAXIT = 40
_RTOL = 8e-3


def _perpoint_none() -> dict:
    """8 控制点逐点 ΔR/ΔZ 的 None 默认 (i=0..7), 保证 failed/timeout 行也有列。"""
    return {f"dr_{i}": None for i in range(8)} | {f"dz_{i}": None for i in range(8)}


def bundle_from_efit_h5(shot: int, efit_dir: Path = EFIT_DB_DEFAULT) -> dict:
    """只从本地 EFIT h5 构造 bundle (EFIT profile + R8/Z8 + ATIME); PCS 填 dummy。"""
    fp = Path(efit_dir) / f"{int(shot)}.h5"
    if not fp.is_file():
        raise FileNotFoundError(fp)
    with h5py.File(fp, "r") as f:
        atime = np.asarray(f["ATIME"][:], dtype=np.float64).ravel()
        m = atime >= 0.0
        atime_u = atime[m]
        T = int(atime_u.size)

        def take(k):
            a = np.asarray(f[k][:], dtype=np.float64)
            if a.shape[0] == atime.size:
                return a[m]
            if a.shape[0] == T:
                return a
            raise ValueError(f"{k} shape {a.shape} vs ATIME {atime.size}/{T}")

        R8 = take("R8"); Z8 = take("Z8")
        if R8.ndim == 2 and R8.shape[1] != 8 and R8.shape[0] == 8:
            R8 = R8.T; Z8 = Z8.T
        PPRIME, FFPRIM, FPOL = take("PPRIME"), take("FFPRIM"), take("FPOL")
        BCENTR = take("BCENTR").ravel(); BETAP = take("BETAP").ravel()
        RMAXIS = take("RMAXIS").ravel(); ZMAXIS = take("ZMAXIS").ravel()
    return {
        "shot": int(shot), "ATIME": atime_u, "source": "efit_h5",
        "R8": R8, "Z8": Z8, "PPRIME": PPRIME, "FFPRIM": FFPRIM, "FPOL": FPOL,
        "BCENTR": BCENTR, "BETAP": BETAP, "RMAXIS": RMAXIS, "ZMAXIS": ZMAXIS,
        "PCPF": np.zeros((T, 12)), "PCRL01": np.zeros(T),  # dummy (bc_pred override)
        "lmsr": np.zeros(T), "lmsz": np.zeros(T),  # FreeGSNKE 不读
    }


def solve_slice_with_conv(slice_npz: Path, pf_xml, wall_xml, *, nx, rtol, maxit) -> dict:
    """FreeGSNKE 前向求解单帧 + 收敛信息 (复现 _forward_solve_slice)。"""
    from bc.gs_forward.precursor_export import load_single_slice_npz
    from bc_notime.gs_forward.east_pf_mapping import pcpf12_to_pf14_amps
    from bc_notime.gs_forward.freegsnke_east_machine import (
        build_east_tokamak_freegsnke, currents_dict_to_vec, wall_outline_rz_from_xml)
    from freegsnke.equilibrium_update import Equilibrium
    from freegsnke.jtor_update import ConstrainBetapIp, GeneralPprimeFFprime
    from freegsnke.GSstaticsolver import NKGSsolver

    sl = load_single_slice_npz(slice_npz)
    snap = sl["snapshot"]
    pcpf12 = np.asarray(sl["pcpf12"], dtype=np.float64).ravel()
    ip_sel = float(sl["Ip_A"])
    pf_amps = pcpf12_to_pf14_amps(pcpf12)

    lim_r, lim_z = wall_outline_rz_from_xml(wall_xml)
    Rmin, Rmax = float(np.nanmin(lim_r)), float(np.nanmax(lim_r))
    Zmin, Zmax = float(np.nanmin(lim_z)), float(np.nanmax(lim_z))
    tok = build_east_tokamak_freegsnke(pf_xml, wall_xml, quiet=True)
    vec = currents_dict_to_vec(tok, pf_amps)
    tok.set_all_coil_currents(vec)

    psi_n = np.linspace(0.0, 1.0, len(snap["pprime"]), dtype=np.float64)
    bcentr = float(snap.get("bcentr", 0))
    rmaxis = float(snap.get("rmaxis", Rmin + 0.5 * (Rmax - Rmin)))
    fvac = bcentr * rmaxis if bcentr > 0 else 1.0
    Ip = ip_sel

    def _make():
        e = Equilibrium(tok, Rmin, Rmax, Zmin, Zmax, nx=nx, ny=nx, order=4)
        tok.set_all_coil_currents(vec)
        try:
            e.adjust_psi_plasma()
        except Exception:
            pass
        return e

    def _solve(eq_obj, prof):
        nk = NKGSsolver(eq_obj, seed=42)
        nk.forward_solve(eq_obj, prof, target_relative_tolerance=rtol,
                         max_solving_iterations=maxit, suppress=True, verbose=False)
        return float(nk.relative_change)

    for method, build_prof in (
        ("GeneralPprimeFFprime", lambda eq: GeneralPprimeFFprime(
            eq, Ip, fvac, psi_n, pprime_data=np.asarray(snap["pprime"], float),
            ffprime_data=np.asarray(snap["ffprim"], float), p_data=None, f_data=None,
            Raxis=rmaxis, Ip_logic=True)),
        ("ConstrainBetapIp", lambda eq: ConstrainBetapIp(
            eq, float(snap.get("betap", 0.5)), Ip, fvac, Raxis=rmaxis)),
    ):
        eq = _make()
        try:
            _solve(eq, build_prof(eq))
            return {"eq": eq, "rel_change": _solve(eq, build_prof(eq)) if False else None,
                    "converged": False, "method": method}
        except Exception:
            continue
    return {"eq": None, "rel_change": None, "converged": False, "method": "failed"}


def _solve_and_score(task: tuple) -> dict:
    """Pool worker: 求解 + shape_params 8 控制点 + rmse, 返小 dict (不返 eq)。"""
    slice_path, shot, k, phase, t, r8e, z8e = task
    r8e = np.asarray(r8e, float).ravel(); z8e = np.asarray(z8e, float).ravel()
    # 重新求解并拿 rel_change (上面的 solve_slice_with_conv 简化了, 这里完整重做以拿收敛)
    from bc.gs_forward.precursor_export import load_single_slice_npz
    from bc_notime.gs_forward.east_pf_mapping import pcpf12_to_pf14_amps
    from bc_notime.gs_forward.freegsnke_east_machine import (
        build_east_tokamak_freegsnke, currents_dict_to_vec, wall_outline_rz_from_xml)
    from freegsnke.equilibrium_update import Equilibrium
    from freegsnke.jtor_update import ConstrainBetapIp, GeneralPprimeFFprime
    from freegsnke.GSstaticsolver import NKGSsolver

    rec = {"shot": shot, "k": int(k), "phase": phase, "t": float(t),
           "rel_change": None, "iters": None, "converged": False,
           "method": "failed", "rmse_8ctrl_m": None, "dr_max": None, "dz_max": None,
           **_perpoint_none()}
    try:
        sl = load_single_slice_npz(Path(slice_path))
        snap = sl["snapshot"]
        pf_amps = pcpf12_to_pf14_amps(np.asarray(sl["pcpf12"], float).ravel())
        lim_r, lim_z = wall_outline_rz_from_xml(_WALL)
        Rmin, Rmax = float(np.nanmin(lim_r)), float(np.nanmax(lim_r))
        Zmin, Zmax = float(np.nanmin(lim_z)), float(np.nanmax(lim_z))
        tok = build_east_tokamak_freegsnke(_PF, _WALL, quiet=True)
        vec = currents_dict_to_vec(tok, pf_amps); tok.set_all_coil_currents(vec)
        psi_n = np.linspace(0, 1, len(snap["pprime"]))
        bcentr = float(snap.get("bcentr", 0)); rmaxis = float(snap.get("rmaxis", Rmin + 0.5*(Rmax-Rmin)))
        fvac = bcentr * rmaxis if bcentr > 0 else 1.0; Ip = float(sl["Ip_A"])

        def _make():
            e = Equilibrium(tok, Rmin, Rmax, Zmin, Zmax, nx=_NX, ny=_NX, order=4)
            tok.set_all_coil_currents(vec)
            try: e.adjust_psi_plasma()
            except Exception: pass
            return e
        def _solve(eq, prof):
            nk = NKGSsolver(eq, seed=42)
            nk.forward_solve(eq, prof, target_relative_tolerance=_RTOL,
                             max_solving_iterations=_MAXIT, suppress=True, verbose=False)
            return float(nk.relative_change)
        eq = None; rel = None; method = "failed"
        for m, bp in (("GeneralPprimeFFprime", lambda: GeneralPprimeFFprime(
                _make(), Ip, fvac, psi_n, pprime_data=np.asarray(snap["pprime"], float),
                ffprime_data=np.asarray(snap["ffprim"], float), p_data=None, f_data=None,
                Raxis=rmaxis, Ip_logic=True)),
            ("ConstrainBetapIp", lambda: ConstrainBetapIp(_make(), float(snap.get("betap", 0.5)), Ip, fvac, Raxis=rmaxis))):
            try:
                eq = _make(); prof = bp(); rel = _solve(eq, prof); method = m; break
            except Exception:
                continue
        rec["method"] = method; rec["rel_change"] = rel
        rec["converged"] = bool(rel is not None and rel < _RTOL)
        if eq is not None and rel is not None:
            sep = np.asarray(eq.separatrix(ntheta=240), float)
            sp = shape_params(sep[:, 0], sep[:, 1])
            r8p = np.asarray(sp.R8, float).ravel(); z8p = np.asarray(sp.Z8, float).ravel()
            if r8p.size == 8 and not np.isnan(r8p).any():
                d = np.hypot(r8p - r8e, z8p - z8e)
                rec["rmse_8ctrl_m"] = float(np.sqrt(np.mean(d ** 2)))
                rec["dr_max"] = float(np.max(np.abs(r8p - r8e)))
                rec["dz_max"] = float(np.max(np.abs(z8p - z8e)))
                # 逐控制点 ΔR/ΔZ (米), i=0..7, 供统计图 Panel (b) 用
                dr = (r8p - r8e).astype(float)
                dz = (z8p - z8e).astype(float)
                for i in range(8):
                    rec[f"dr_{i}"] = float(dr[i])
                    rec[f"dz_{i}"] = float(dz[i])
            for a in ("elong", "triu", "tril", "squo", "squi", "sqli", "sqlo"):
                v = getattr(sp, a, None)
                rec[f"pred_{a}"] = float(v) if isinstance(v, (int, float, np.floating)) else None
    except Exception as e:
        rec["method"] = f"err:{type(e).__name__}"
    return rec


def _timeout_rec(task):
    _, shot, k, phase, t, _, _ = task
    return {"shot": shot, "k": int(k), "phase": phase, "t": float(t),
            "rel_change": None, "iters": None, "converged": False,
            "method": "timeout", "rmse_8ctrl_m": None, "dr_max": None, "dz_max": None,
            **_perpoint_none()}


def _worker_run(task, q):
    """fork 子进程: 跑 _solve_and_score, 结果塞 queue (父进程超时可 terminate)。"""
    try:
        q.put(_solve_and_score(task))
    except Exception as e:
        r = _timeout_rec(task); r["method"] = f"err:{type(e).__name__}"; q.put(r)


def _solve_with_timeout(task, timeout):
    """fork 子进程跑单帧 solve, 超时强制 terminate (避免 FreeGSNKE 死循环卡死整体)。

    替代无 timeout 的 mp.Pool: 后者某帧死循环 (build_eq/forward_solve maxit 未拦) 会
    卡死整个 Pool 直到 SLURM time limit。fork+terminate 强制 kill 卡帧, 跳过记 timeout。
    """
    ctx = mp.get_context("fork")
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_worker_run, args=(task, q))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(10)
        return _timeout_rec(task)
    try:
        return q.get_nowait()
    except Exception:
        return _timeout_rec(task)


def main() -> int:
    global _PF, _WALL, _NX, _MAXIT, _RTOL
    from bc.gs_forward.freegsnke_east_machine import PF_XML_DEFAULT, WALL_XML_DEFAULT
    from paper_betan.freegsnke_infer import infer_one_shot
    from paper_betan.model import MODEL_CKPT

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(MODEL_CKPT))
    ap.add_argument("--current-source", choices=["pred", "pcs"], default="pred",
                    help="pred=模型预测电流; pcs=EFIT/PCS 实测电流(target_A), 跳过模型(无需GPU)")
    ap.add_argument("--test-shots", default="meta/split_by_order_betan/test_shots.txt")
    ap.add_argument("--n-shots", type=int, default=100, help="999999 = 全 test 集")
    ap.add_argument("--frames-per-phase", type=int, default=3)
    ap.add_argument("--nx", type=int, default=65)
    ap.add_argument("--maxit", type=int, default=40)
    ap.add_argument("--rtol", type=float, default=8e-3)
    ap.add_argument("--n-proc", type=int, default=60)
    ap.add_argument("--per-frame-timeout", type=float, default=120.0,
                    help="单帧 FreeGSNKE 墙钟上限(秒); 超时强制 kill 该帧 (记 timeout)")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--efit-dir", default=str(EFIT_DB_DEFAULT))
    ap.add_argument("--out-dir", default="results/paper_betan/freegsnke_testset")
    args = ap.parse_args()

    _PF, _WALL = PF_XML_DEFAULT, WALL_XML_DEFAULT
    _NX, _MAXIT, _RTOL = args.nx, args.maxit, args.rtol
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / f"_tmp_slices_shard{args.shard_id}"; tmp_dir.mkdir(parents=True, exist_ok=True)
    efit_dir = Path(args.efit_dir)

    all_shots = [int(x) for x in Path(args.test_shots).read_text().split() if x.strip().isdigit()]
    avail = [s for s in all_shots if (efit_dir / f"{s}.h5").is_file()]
    rng = random.Random(args.seed); rng.shuffle(avail)
    avail = avail[:args.n_shots]
    shots = avail[args.shard_id::args.n_shards]  # 分片
    print(f"[shard {args.shard_id}/{args.n_shards}] {len(shots)} shots "
          f"(of {len(avail)} sampled / {len(all_shots)} test), frames/phase={args.frames_per_phase}, "
          f"n_proc={args.n_proc}, nx={args.nx}, current_source={args.current_source}", flush=True)

    # ---- Phase 1: 逐炮推理 (DCU) + 导出 slice (串行; 推理是瓶颈) ----
    tasks = []
    t0 = time.monotonic()
    for si, shot in enumerate(shots, 1):
        try:
            bundle = bundle_from_efit_h5(shot, efit_dir)
            inf = infer_one_shot(shot, args.ckpt, run_model=(args.current_source == "pred"))
        except Exception as e:
            print(f"  [{si}/{len(shots)}] shot {shot} SKIP ({type(e).__name__}: {e})", flush=True)
            continue
        atime = np.asarray(bundle["ATIME"], float).ravel()
        mt = np.asarray(inf["time"], float).ravel()
        ip_A = np.asarray(inf["Ip_A"], float).ravel()
        # 电流源: pred=模型预测; pcs=EFIT/PCS 实测(target_A)。其余(slice 选择/Ip/profile/FreeGSNKE 设置)完全同口径。
        cur_A = np.asarray(inf["target_A"] if args.current_source == "pcs" else inf["pred_A"], float)
        ip_at = np.interp(atime, mt, ip_A)
        pids = phase_ids_per_step(atime, ip_at, valid_len=atime.size)
        for ph_idx, ph in enumerate(PHASE_NAMES):
            ks_all = [k for k in range(atime.size) if int(pids[k]) == ph_idx]
            if not ks_all:
                continue
            step = max(1, len(ks_all) // args.frames_per_phase)
            for k in ks_all[::step][:args.frames_per_phase]:
                k_bc = int(np.argmin(np.abs(mt - atime[k])))
                sp = tmp_dir / f"shot{shot}_k{k:05d}.npz"
                try:
                    export_slice_npz(bundle, k, sp, pcpf12_override=cur_A[k_bc],
                                     ip_a_override=float(ip_A[k_bc]))
                except Exception:
                    continue
                tasks.append((str(sp), shot, k, ph, float(atime[k]),
                              np.asarray(bundle["R8"][k], float).ravel(),
                              np.asarray(bundle["Z8"][k], float).ravel()))
        if si % 20 == 0 or si == len(shots):
            print(f"  [phase1] {si}/{len(shots)} shots, {len(tasks)} slices, "
                  f"elapsed={time.monotonic()-t0:.0f}s", flush=True)
    print(f"[phase1 done] {len(tasks)} slices from {len(shots)} shots in {time.monotonic()-t0:.0f}s", flush=True)

    # ---- 断点续跑: 读已有 per_frame_shard, 跳过已跑帧 ----
    shard_csv = out_dir / f"per_frame_shard{args.shard_id}.csv"
    rows_done, done_keys = [], set()
    if shard_csv.is_file():
        with open(shard_csv) as f:
            for r in csv.DictReader(f):
                rows_done.append(r); done_keys.add((int(r["shot"]), int(r["k"])))
        n_before = len(tasks)
        tasks = [t for t in tasks if (t[1], t[2]) not in done_keys]
        print(f"[resume] {len(rows_done)} frames done, {n_before}->{len(tasks)} remaining", flush=True)

    def _flush_shard(extra_rows):
        allr = rows_done + extra_rows
        if not allr:
            return
        keys = sorted({k for r in allr for k in r})
        with open(shard_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in allr:
                w.writerow({k: r.get(k) for k in keys})

    # ---- Phase 2: 串行 fork + per-frame timeout (强制 kill 卡帧, 替代 mp.Pool) ----
    t1 = time.monotonic()
    rows = []
    if tasks:
        for i, task in enumerate(tasks, 1):
            rec = _solve_with_timeout(task, timeout=args.per_frame_timeout)
            rows.append(rec)
            if i % 100 == 0 or i == len(tasks):
                cvg = sum(1 for r in rows if r.get("converged"))
                to = sum(1 for r in rows if r.get("method") == "timeout")
                print(f"  [phase2] {i}/{len(tasks)} solved (cvg={cvg} timeout={to}), "
                      f"elapsed={time.monotonic()-t1:.0f}s", flush=True)
                _flush_shard(rows)  # 增量保存 (断点续跑)
    print(f"[phase2 done] {len(rows)} frames solved in {time.monotonic()-t1:.0f}s", flush=True)

    # ---- 输出分片 (含断点续跑已跑帧) ----
    _flush_shard(rows)
    n_cvg = sum(1 for r in rows if r.get("converged"))
    print(f"[shard {args.shard_id}] {len(rows)} frames, converged={n_cvg} "
          f"({n_cvg/len(rows):.1%}) -> per_frame_shard{args.shard_id}.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
