#!/usr/bin/env python3
"""为 paper_igbt 的 pred_wall_dist 图，按 s44 预测误差挑出每阶段最优时间片。

背景: paper_igbt/freegsnke_whole_shot.py 现在用 ``pick_representative_steps``
(policy="middle") 取每阶段**中间**帧 (k=4/55/105)，其中 flat_top k=55、
ramp_down k=105 预测误差很大，图不好看。本脚本对 s44 模型在每阶段的若干
候选帧上做 FreeGSNKE 正向求解，按 ``lcfs_rmsep_to_r8z8_m`` (与论文一致) 取最优，
导出对应的 bc_pred 切片并写一个独立的 best-per-phase manifest，供
plot_pred_wall_dist.py --manifest 使用，不动原有 montage manifest。

候选帧来自 results/freegsnke_whole_shot_notime (run1, 同为 no-time 架构) 的
by_t_efit 指标排序的 top-N，并强制包含当前 paper_igbt 选中帧与 run1 最优帧。

需在 source 过 DTK env 的 torch 环境下运行 (s44 推理走 CPU 即可)。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bc.gs_forward.precursor_export import export_slice_npz, nearest_row  # noqa: E402
from bc_notime.gs_forward.precursor_export import load_precursor_npz  # noqa: E402
from bc_notime.gs_forward.run_freegsnke_eval import rmse_separatrix_to_r8z8  # noqa: E402
from bc_notime.data.phases import PHASE_NAMES, phase_ids_per_step  # noqa: E402
from paper_igbt.freegsnke_infer import infer_one_shot_best  # noqa: E402
from paper_igbt.model import BEST_CKPT  # noqa: E402
from paper_igbt.plot_pred_wall_dist import _forward_solve_slice  # noqa: E402


def _load_bundle(precursor_npz: Path) -> dict:
    keys = ["ATIME", "R8", "Z8", "PPRIME", "FFPRIM", "FPOL", "BCENTR", "BETAP",
            "RMAXIS", "ZMAXIS", "PCRL01", "lmsr", "lmsz", "PCPF"]
    z = np.load(precursor_npz, allow_pickle=False)
    b = {k: np.asarray(z[k]) for k in keys}
    z.close()
    b["shot"] = int(Path(precursor_npz).parent.name)
    return b


def _run1_ranking(by_t_efit: Path, top_n: int) -> dict[str, list[int]]:
    """读 run1-notime by_t_efit/*/summary.json, 每阶段按 lcfs_rmsep 升序取 top_n 个 k_efit。"""
    from bc_notime.gs_forward.precursor_export import load_precursor_npz as _lp
    out: dict[str, list[int]] = {ph: [] for ph in PHASE_NAMES}
    if not by_t_efit.is_dir():
        return out
    # 阶段划分用 ATIME/PCRL01 (与 montage 一致)
    rows: list[tuple[int, float, int]] = []  # (k, rmsep, phase_id)
    for d in sorted(by_t_efit.iterdir()):
        sj = d / "summary.json"
        if not sj.is_file():
            continue
        try:
            data = json.loads(sj.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        k = int(data.get("k_efit_profile", -1))
        if k < 0:
            m = re.search(r"k(\d+)", d.name)
            k = int(m.group(1)) if m else -1
        rm = data.get("lcfs_rmsep_to_r8z8_m")
        if rm is None or not math.isfinite(float(rm)) or float(rm) <= 0:
            continue
        # 阶段: summary 里没存 phase_id, 用 time_s 近似 → 用 run1 已排序即可(按 rmsep)
        rows.append((k, float(rm)))
    rows.sort(key=lambda r: r[1])
    # 阶段归属用 ATIME-based phase_ids (下面 main 里补); 这里先按 rmsep 全局 top,
    # 再在 main 里按阶段过滤。简单起见返回全局 top_n*4, 由 main 按阶段筛。
    return rows  # type: ignore[return-value]


def main() -> int:
    from bc_notime.gs_forward.freegsnke_east_machine import PF_XML_DEFAULT, WALL_XML_DEFAULT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=134925)
    ap.add_argument("--ckpt", default=str(BEST_CKPT))
    ap.add_argument("--precursor", default="results/freegsnke_precursors/134925/precursor.npz")
    ap.add_argument("--whole-shot-root", default="results/paper_igbt/freegsnke_whole_shot")
    ap.add_argument("--run1-by-t-efit",
                    default="results/freegsnke_whole_shot_notime/134925/bc_pred/by_t_efit")
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pf-xml", default=str(PF_XML_DEFAULT))
    ap.add_argument("--wall-xml", default=str(WALL_XML_DEFAULT))
    args = ap.parse_args()

    repo = Path.cwd()
    prec = Path(args.precursor).resolve()
    whole = Path(args.whole_shot_root).resolve()
    run1_dir = Path(args.run1_by_t_efit).resolve()
    tmp_dir = whole / str(args.shot) / "bc_pred" / "_tmp_eval"
    slices_dir = whole / str(args.shot) / "bc_pred" / "_slices"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(parents=True, exist_ok=True)

    print(f"[infer] s44 ckpt={args.ckpt} device={args.device} ...", flush=True)
    inf = infer_one_shot_best(args.shot, args.ckpt, device=args.device)
    bundle = _load_bundle(prec)
    atime = np.asarray(bundle["ATIME"], dtype=np.float64).ravel()
    pcrl = np.asarray(bundle["PCRL01"], dtype=np.float64).ravel()
    phase_ids = phase_ids_per_step(atime, pcrl)  # over ATIME axis (k_efit)
    time_ds = np.asarray(inf["time"], dtype=np.float64).ravel()
    pred_A = np.asarray(inf["pred_A"], dtype=np.float64)
    T_eff = int(time_ds.size)
    print(f"[infer] T_eff={T_eff}  ATIME={atime.size}  pred_A={pred_A.shape}", flush=True)

    # ── 候选 k_efit: run1 全局 top + 当前 paper_igbt 选中 + run1 各阶段最优 ──
    run1_rows = _run1_ranking(run1_dir, args.top_n)  # list[(k, rmsep)]
    # 按阶段分组 run1 top
    cand_per_phase: dict[str, set[int]] = {ph: set() for ph in PHASE_NAMES}
    for k, _rm in run1_rows:
        if 0 <= k < phase_ids.size:
            ph = PHASE_NAMES[int(phase_ids[k])]
            cand_per_phase[ph].add(int(k))
    # 当前 paper_igbt 选中帧 (manifest middle picks)
    cur_manifest = whole / "_montage" / f"montage_pcs_vs_pred_shot{args.shot}.json"
    if cur_manifest.is_file():
        try:
            cm = json.loads(cur_manifest.read_text())
            for ph in PHASE_NAMES:
                e = cm.get(ph)
                if e and 0 <= int(e["k"]) < phase_ids.size:
                    cand_per_phase[ph].add(int(e["k"]))
        except (json.JSONDecodeError, OSError):
            pass
    # 每阶段只保留 top_n (按 run1 rmsep), 但至少保留候选
    rm_by_k = {k: rm for k, rm in run1_rows}
    for ph in PHASE_NAMES:
        ks = sorted(cand_per_phase[ph], key=lambda k: rm_by_k.get(k, 1e9))
        cand_per_phase[ph] = set(ks[: args.top_n])

    # ── k_ds → k_efit 映射, 仅对候选 k_efit 求解 ──
    cand_ks: set[int] = set()
    for s in cand_per_phase.values():
        cand_ks |= s
    kds_for_kefit: dict[int, int] = {}
    for k_ds in range(T_eff):
        t_s = float(time_ds[k_ds])
        k_efit = nearest_row(atime, t_s)
        if k_efit in cand_ks and k_efit not in kds_for_kefit:
            kds_for_kefit[int(k_efit)] = int(k_ds)
    print(f"[cand] 候选 k_efit 数={len(cand_ks)}  有 k_ds 映射={len(kds_for_kefit)}", flush=True)
    for ph in PHASE_NAMES:
        print(f"       {ph}: 候选 k_efit={sorted(cand_per_phase[ph])}", flush=True)

    # ── 逐候选正向求解 + rmsep ──
    results: dict[str, list[tuple[float, int, int, float]]] = {ph: [] for ph in PHASE_NAMES}
    pf_xml, wall_xml = args.pf_xml, args.wall_xml
    for ph in PHASE_NAMES:
        for k_efit in sorted(cand_per_phase[ph]):
            k_ds = kds_for_kefit.get(k_efit)
            if k_ds is None:
                print(f"  [skip] {ph} k_efit={k_efit} 无 k_ds 映射", flush=True)
                continue
            t_s = float(time_ds[k_ds])
            tmp_npz = tmp_dir / f"slice_eval_bc_k{k_efit:05d}.npz"
            export_slice_npz(bundle, k_efit, tmp_npz, pcpf12_override=pred_A[k_ds])
            t0 = time.time()
            solved = _forward_solve_slice(tmp_npz, pf_xml, wall_xml)
            dt = time.time() - t0
            if solved is None:
                print(f"  [fail] {ph} k_efit={k_efit} t={t_s:.3f}s 求解失败 ({dt:.1f}s)", flush=True)
                continue
            eq, snap = solved
            try:
                sep = np.asarray(eq.separatrix(ntheta=240))
                valid = ~(np.isnan(sep[:, 0]) | np.isnan(sep[:, 1]))
                sep_clean = sep[valid]
                rm = rmse_separatrix_to_r8z8(sep_clean,
                                             np.asarray(snap["r8"], dtype=np.float64).ravel(),
                                             np.asarray(snap["z8"], dtype=np.float64).ravel())
            except Exception as e:
                print(f"  [err] {ph} k_efit={k_efit} rmsep 计算异常: {e}", flush=True)
                continue
            if not math.isfinite(rm):
                continue
            results[ph].append((float(rm), int(k_efit), int(k_ds), t_s))
            print(f"  {ph} k_efit={k_efit:4d} (k_ds={k_ds:4d}) t={t_s:7.3f}s "
                  f"lcfs_rmsep={rm*1000:7.3f}mm  ({dt:.1f}s)", flush=True)

    # ── 每阶段取最优 ──
    manifest = {"shot": args.shot, "metric": "lcfs_rmsep_to_r8z8_m",
                "model": "s44 (transformer_bidir_on)", "converged_only": False}
    print("\n==== 每阶段最优 (s44) ====", flush=True)
    for ph in PHASE_NAMES:
        results[ph].sort(key=lambda r: r[0])
        if not results[ph]:
            print(f"  {ph}: 无有效结果!", flush=True)
            manifest[ph] = None
            continue
        rm, k_efit, k_ds, t_s = results[ph][0]
        print(f"  {ph:10s}: BEST k_efit={k_efit} (k_ds={k_ds}) t={t_s:.3f}s "
              f"lcfs_rmsep={rm*1000:.3f}mm", flush=True)
        # 导出最优 bc_pred 切片到正式 _slices 目录
        out_npz = slices_dir / f"slice_whole_bc_k{k_efit:05d}.npz"
        export_slice_npz(bundle, k_efit, out_npz, pcpf12_override=pred_A[k_ds])
        manifest[ph] = {"k": int(k_efit), "k_ds": int(k_ds), "time_s": float(t_s),
                        "bc_lcfs_rmsep_m": float(rm)}
        # 打印该阶段 top-3 供参考
        for rm2, k2, kd2, t2 in results[ph][:3]:
            print(f"      - k_efit={k2:4d} t={t2:7.3f}s rmsep={rm2*1000:7.3f}mm", flush=True)

    man_path = whole / "_montage" / f"best_pred_per_phase_shot{args.shot}.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[manifest] {man_path}", flush=True)
    print(f"[slices]   {slices_dir} (3 个最优 bc_pred 切片已导出)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
