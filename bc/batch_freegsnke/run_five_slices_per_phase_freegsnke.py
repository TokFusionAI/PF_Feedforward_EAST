#!/usr/bin/env python3
"""每阶段在 ``phase_slices`` 内取 N 个 BC 步，导出 ``slice_pred_*`` 并依次前向 freegsnke。

时刻选取策略（``--pick-policy``）::

    uniform   — 在阶段内按索引近似等间隔（旧行为）
    stability — 在阶段内按索引分 N 段，每段内选 **Ip 与预测 PCS 时间变化率最小** 的一步
                （|dIp/dt| 与 ||d(pred)/dt|| 加权；段内 argmin，兼顾覆盖与“稳态”）

输出目录::

    {out_base}/seed{seed}/{shot}/_slices/slice_pred_*.npz
    {out_base}/seed{seed}/{shot}/{phase}/t1/

    {out_base_efit_self}/seed{seed}/{shot}/_slices/slice_efit_self_*.npz
    {out_base_efit_self}/seed{seed}/{shot}/{phase}/t1_efit/

每个时间片两种 slice（预测 + EFIT 自洽 PCS），各跑一次 ``run_freegsnke_eval``；``--no-efit-self`` 只跑预测。

**单炮（默认测试）**：不显式指定批量选项时默认 ``--seed 2026 --shot 105653``（可用 ``--shot`` 覆盖）。

**单炮（显式）**：``--shot 103978``（可选 ``--precursor-npz``）。

**多炮（同一 seed）**：``--shots-file /path/to.txt``（每行一炮号），或
``--from-random-seed-shots`` 读取 ``results/random_freegsnke_shots/seed{seed}/shots.txt``。
二者优先于单独的 ``--shot``（显式 ``--shots-file`` 优先于 ``--from-random-seed-shots``）。

示例（推荐模块路径；``python -m bc.run_five_slices_per_phase_freegsnke`` 仍可用薄包装）::

    python -m bc.batch_freegsnke.run_five_slices_per_phase_freegsnke --shot 103978 --seed 4 \\
      --pick-policy stability --precursor-npz results/freegsnke_precursors/103978/precursor.npz

    python -m bc.batch_freegsnke.run_five_slices_per_phase_freegsnke --seed 2026 --from-random-seed-shots \\
      --pick-policy stability
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bc.gs_forward.infer_one_shot import infer_one_shot
from bc.gs_forward.precursor_export import export_slice_npz, load_precursor_npz, nearest_row

# bc/batch_freegsnke/本文件.py → 仓库根
_REPO = Path(__file__).resolve().parent.parent.parent
_PHASE_ORDER = ("ramp_up", "flat_top", "ramp_down")


def _read_shots_file(path: Path) -> list[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[int] = []
    for line in lines:
        s = line.split("#", 1)[0].strip()
        if s.isdigit():
            out.append(int(s))
    return out


def _pick_k_indices_in_slice_uniform(sl: slice, n: int) -> list[int]:
    """在 [start, stop) 内取至多 n 个整数索引，含端点近似均匀分布。"""
    if sl.stop <= sl.start:
        return []
    lo, hi = sl.start, sl.stop - 1
    if hi - lo + 1 <= n:
        return list(range(lo, sl.stop))
    if n == 1:
        return [(lo + hi) // 2]
    pts = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    out = sorted({max(lo, min(hi, int(round(p)))) for p in pts})
    return out


def _instability_per_step(
    time_s: np.ndarray,
    ip_a: np.ndarray,
    pred_a: np.ndarray,
    ip_w: float,
    coil_w: float,
) -> np.ndarray:
    """与 ``time_s`` 对齐的标量不稳定度，越大越“不稳”；长度 T。"""
    t = np.asarray(time_s, dtype=np.float64).ravel()
    ip = np.asarray(ip_a, dtype=np.float64).ravel()
    pred = np.asarray(pred_a, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 12:
        raise ValueError(f"pred_a 期望 (T,12)，得到 {pred.shape}")
    T = t.size
    if ip.shape[0] != T or pred.shape[0] != T:
        raise ValueError("time_s / Ip / pred_a 长度不一致")
    dip = np.abs(np.gradient(ip, t, edge_order=1))
    g2 = np.zeros(T, dtype=np.float64)
    for c in range(12):
        g2 += np.gradient(pred[:, c], t, edge_order=1) ** 2
    dcoil = np.sqrt(g2)
    return float(ip_w) * dip + float(coil_w) * dcoil


def _pick_k_indices_in_slice_stability_stratified(
    sl: slice,
    n: int,
    time_s: np.ndarray,
    ip_a: np.ndarray,
    pred_a: np.ndarray,
    *,
    ip_w: float,
    coil_w: float,
) -> list[int]:
    """阶段内按索引均分 N 个子区间，每区间内取不稳定度最小的 k。"""
    if sl.stop <= sl.start or n < 1:
        return []
    lo, stop = sl.start, sl.stop
    idx = np.arange(lo, stop, dtype=np.int64)
    if idx.size == 0:
        return []
    inst = _instability_per_step(time_s, ip_a, pred_a, ip_w, coil_w)
    scores = inst[idx]
    q = float(np.percentile(scores, 85))
    if q > 0:
        scores = scores / q
    if idx.size <= n:
        order = np.argsort(scores)
        picked = [int(idx[j]) for j in order[: min(n, idx.size)]]
        return sorted(picked)

    splits = np.array_split(np.arange(idx.size, dtype=np.int64), n)
    picked: list[int] = []
    for part in splits:
        if part.size == 0:
            continue
        local = part[np.argmin(scores[part])]
        picked.append(int(idx[local]))
    return picked


def _pick_k_indices_in_slice(
    sl: slice,
    n: int,
    policy: str,
    time_s: np.ndarray,
    ip_a: np.ndarray,
    pred_a: np.ndarray,
    *,
    ip_w: float,
    coil_w: float,
) -> list[int]:
    if policy == "uniform":
        return _pick_k_indices_in_slice_uniform(sl, n)
    if policy == "stability":
        return _pick_k_indices_in_slice_stability_stratified(
            sl, n, time_s, ip_a, pred_a, ip_w=ip_w, coil_w=coil_w
        )
    raise ValueError(f"未知 pick-policy: {policy!r}")


def _resolve_shots(args: argparse.Namespace) -> list[int]:
    if args.shots_file:
        p = Path(args.shots_file)
        if not p.is_file():
            raise FileNotFoundError(f"--shots-file 不存在：{p}")
        return _read_shots_file(p)
    if getattr(args, "from_random_seed_shots", False):
        p = _REPO / "results" / "random_freegsnke_shots" / f"seed{int(args.seed)}" / "shots.txt"
        if not p.is_file():
            raise FileNotFoundError(f"--from-random-seed-shots 需要存在文件：{p}")
        return _read_shots_file(p)
    if args.shot is not None:
        return [int(args.shot)]
    raise ValueError("请指定 --shot，或 --shots-file，或 --from-random-seed-shots")


def process_one_shot(
    *,
    shot: int,
    seed: int,
    args: argparse.Namespace,
) -> int:
    """单炮：写 slice、manifest、按需跑 freegsnke。返回子进程最大非零退出码（无子进程则 0）。"""
    n = int(args.n_per_phase)
    prec = Path(args.precursor_npz) if args.precursor_npz else _REPO / "results" / "freegsnke_precursors" / str(shot) / "precursor.npz"
    if not prec.is_file():
        print(f"[SKIP] shot={shot} 缺少 precursor：{prec}", file=sys.stderr)
        return 2

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        print(f"缺少 ckpt：{ckpt}", file=sys.stderr)
        return 2

    out_run = Path(args.out_base).resolve() / f"seed{seed}" / str(shot)
    slices_dir = out_run / "_slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    efit_out_run = Path(args.out_base_efit_self).resolve() / f"seed{seed}" / str(shot)
    efit_slices_dir = efit_out_run / "_slices"
    if not args.no_efit_self:
        efit_slices_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    inf = infer_one_shot(shot, ckpt, device=device, dataset_root=args.dataset_root)
    bundle = load_precursor_npz(prec)
    atime = np.asarray(bundle["ATIME"], dtype=np.float64).ravel()
    time_s = np.asarray(inf["time"], dtype=np.float64).ravel()
    ip_a = np.asarray(inf["Ip_A"], dtype=np.float64).ravel()
    pred_a = np.asarray(inf["pred_A"], dtype=np.float64)

    policy = str(args.pick_policy)
    ip_w = float(args.stability_ip_weight)
    coil_w = float(args.stability_coil_weight)

    manifest: dict[str, Any] = {
        "shot": shot,
        "seed": seed,
        "n_per_phase": n,
        "pick_policy": policy,
        "stability_ip_weight": ip_w,
        "stability_coil_weight": coil_w,
        "out_run": str(out_run),
        "out_run_efit_self": str(efit_out_run) if not args.no_efit_self else None,
        "precursor_npz": str(prec.resolve()),
        "phases": {},
    }

    max_rc = 0

    for phase in _PHASE_ORDER:
        if phase not in inf["phase_slices"]:
            print(f"[skip] shot={shot} 无阶段 {phase}", flush=True)
            continue
        sl = inf["phase_slices"][phase]
        k_list = _pick_k_indices_in_slice(sl, n, policy, time_s, ip_a, pred_a, ip_w=ip_w, coil_w=coil_w)
        if not k_list:
            print(f"[skip] shot={shot} {phase} 区间为空", flush=True)
            continue
        phase_rows: list[dict[str, Any]] = []
        inst_all = _instability_per_step(time_s, ip_a, pred_a, ip_w, coil_w)
        for ti, k_bc in enumerate(k_list, start=1):
            t_bc = float(time_s[k_bc])
            k_efit = nearest_row(atime, t_bc)
            t_efit = float(atime[k_efit])
            ip_bc = float(ip_a[k_bc])
            pred12 = np.asarray(pred_a[k_bc], dtype=np.float64).ravel()
            stab = float(inst_all[k_bc])
            fname = f"slice_pred_{phase}_t{ti:02d}_kbc{k_bc:04d}_kef{k_efit:04d}.npz"
            slice_path = slices_dir / fname
            export_slice_npz(
                bundle,
                k_efit,
                slice_path,
                pcpf12_override=pred12,
                ip_a_override=ip_bc,
            )
            exact_dir = out_run / phase / f"t{ti}"
            exact_dir.mkdir(parents=True, exist_ok=True)

            row: dict[str, Any] = {
                "t_index": ti,
                "k_bc": k_bc,
                "t_bc_s": t_bc,
                "k_efit": k_efit,
                "t_efit_s": t_efit,
                "stability_cost": stab,
                "bc_pred": {
                    "slice_npz": str(slice_path.resolve()),
                    "eval_out": str(exact_dir.resolve()),
                },
            }

            if not args.no_efit_self:
                efit_fname = f"slice_efit_self_{phase}_t{ti:02d}_k{k_efit:04d}.npz"
                efit_slice = efit_slices_dir / efit_fname
                export_slice_npz(bundle, k_efit, efit_slice)
                efit_dir = efit_out_run / phase / f"t{ti}_efit"
                efit_dir.mkdir(parents=True, exist_ok=True)
                row["efit_self"] = {
                    "slice_npz": str(efit_slice.resolve()),
                    "eval_out": str(efit_dir.resolve()),
                }

            phase_rows.append(row)

            if not args.dry_run:
                cmd_base = [
                    sys.executable,
                    "-m",
                    "bc.run_freegsnke_eval",
                    "--shot",
                    str(shot),
                    "--phase",
                    phase,
                    "--rtol",
                    str(args.rtol),
                    "--nx",
                    str(args.nx),
                    "--ny",
                    str(args.ny),
                ]
                if args.dataset_root:
                    cmd_base.extend(["--dataset-root", args.dataset_root])

                cmd = cmd_base + [
                    "--precursor-slice-npz",
                    str(slice_path.resolve()),
                    "--out-root-exact",
                    str(exact_dir.resolve()),
                ]
                print("RUN", " ".join(cmd), flush=True)
                r = subprocess.run(cmd, cwd=str(_REPO))
                if r.returncode != 0:
                    print(f"[WARN] shot={shot} {phase} t{ti} (bc_pred) 退出码 {r.returncode}", file=sys.stderr)
                    max_rc = max(max_rc, int(r.returncode))

                if not args.no_efit_self:
                    cmd_e = cmd_base + [
                        "--precursor-slice-npz",
                        str(efit_slice.resolve()),
                        "--out-root-exact",
                        str(efit_dir.resolve()),
                    ]
                    print("RUN", " ".join(cmd_e), flush=True)
                    r2 = subprocess.run(cmd_e, cwd=str(_REPO))
                    if r2.returncode != 0:
                        print(f"[WARN] shot={shot} {phase} t{ti}_efit 退出码 {r2.returncode}", file=sys.stderr)
                        max_rc = max(max_rc, int(r2.returncode))
        manifest["phases"][phase] = phase_rows

    man_path = out_run / "multi_phase_time_slices_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"shot={shot} 已写入 {man_path}", flush=True)
    return max_rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shot",
        type=int,
        default=105653,
        help="单炮号（与批量选项二选一）；默认 105653 便于与 seed2026 测试一致",
    )
    ap.add_argument(
        "--shots-file",
        type=str,
        default=None,
        help="多炮：每行一炮号；优先于 --from-random-seed-shots 与 --shot",
    )
    ap.add_argument(
        "--from-random-seed-shots",
        action="store_true",
        help=f"多炮：读取 {_REPO}/results/random_freegsnke_shots/seed{{seed}}/shots.txt",
    )
    ap.add_argument("--seed", type=int, default=2026, help="写入路径 seed{seed}/（默认 2026 单炮测试）")
    ap.add_argument("--n-per-phase", type=int, default=5, help="每阶段时间片数")
    ap.add_argument(
        "--pick-policy",
        type=str,
        choices=("uniform", "stability"),
        default="stability",
        help="时刻选取：uniform 等间隔索引；stability 分箱后每箱最稳（默认 stability）",
    )
    ap.add_argument(
        "--stability-ip-weight",
        type=float,
        default=1.0,
        help="stability 策略下 |dIp/dt| 权重（A/s 量级）",
    )
    ap.add_argument(
        "--stability-coil-weight",
        type=float,
        default=1.0,
        help="stability 策略下 ||d(pred_PCPF)/dt|| 权重",
    )
    ap.add_argument("--ckpt", type=str, default=str(_REPO / "results" / "bc_v1" / "run1" / "checkpoints" / "best_val.pt"))
    ap.add_argument("--precursor-npz", type=str, default=None, help="单炮时可覆盖默认 precursor 路径；多炮时每炮仍用默认 …/{shot}/precursor.npz")
    ap.add_argument(
        "--out-base",
        type=str,
        default=str(_REPO / "results" / "freegsnke_eval_random"),
        help="BC 预测支路根目录",
    )
    ap.add_argument(
        "--out-base-efit-self",
        type=str,
        default=str(_REPO / "results" / "freegsnke_eval_efit_self_random"),
        help="EFIT 自洽 PCS 支路根目录",
    )
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--rtol", type=float, default=8e-3)
    ap.add_argument("--nx", type=int, default=129)
    ap.add_argument("--ny", type=int, default=129)
    ap.add_argument("--dry-run", action="store_true", help="只写 manifest/slice，不调用 run_freegsnke_eval")
    ap.add_argument("--no-efit-self", action="store_true", help="不导出/不跑 EFIT 自洽 PCS slice")
    args = ap.parse_args()

    seed = int(args.seed)

    try:
        shots = _resolve_shots(args)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2

    exit_max = 0
    for shot in shots:
        print(f"======== shot {shot} (seed={seed}, n={len(shots)}) ========", flush=True)
        try:
            rc = process_one_shot(shot=shot, seed=seed, args=args)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] shot={shot}: {exc}", file=sys.stderr)
            exit_max = max(exit_max, 1)
            continue
        if rc != 0:
            exit_max = max(exit_max, rc)

    return min(exit_max, 255) if exit_max > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
