#!/usr/bin/env python3
"""对某炮 ``efit_self`` 目录下已有 ``slice_efit_self_*.npz`` 批量跑 ``run_freegsnke_eval``，生成各时刻 ``07_overlay_coils_lcfs_r8z8.png``，可选拼接栅格图。

与 ``run_five_slices_per_phase_freegsnke`` 的目录约定一致::

    {out_base_efit_self}/seed{seed}/{shot}/{phase}/t{n}_efit/07_overlay_coils_lcfs_r8z8.png

其中 slice 文件名为 ``slice_efit_self_{phase}_t{nn}_{...}.npz``（``t`` 为两位数字），输出子目录为 ``t{int(nn)}_efit``（与批量脚本中 ``ti`` 一致、不带前导零）。

**全 EFIT 时间轴**：默认仅生成 ``07_overlay``（``--full-plots`` 可生成 00–07）；\
阶段名与 ``bc_notime.data.phases`` 一致，由 precursor 的 ``ATIME`` + ``PCRL01`` 划分 ramp_up / flat_top / ramp_down，并传入 ``--out-phase`` 用于图题。

::

    python -m bc_notime.batch_freegsnke.batch_efit_self_overlay07 --seed 2026 --shot 105653 \\
      --every-efit-frame --precursor-npz results/freegsnke_precursors/105653/precursor.npz

输出::

    {shot_run}/by_t_efit/k{kkkkk}_t<十进制 t_efit，保留小数点>/07_overlay_coils_lcfs_r8z8.png
    {shot_run}/_slices_all_efit/slice_k{kkkkk}.npz
    {shot_run}/all_efit_time_manifest.json

示例（代表时刻 + 拼图）::

    python -m bc_notime.batch_freegsnke.batch_efit_self_overlay07 --seed 2026 --shot 105653 --stitch

    # 只拼图（假定各目录已算过 forward）
    python -m bc_notime.batch_freegsnke.batch_efit_self_overlay07 --seed 2026 --shot 105653 --stitch-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from bc_notime.data.phases import PHASE_NAMES, phase_ids_per_step
from bc_notime.gs_forward.precursor_export import export_slice_npz, load_precursor_npz

_REPO = Path(__file__).resolve().parent.parent.parent

ALL_EFIT_MANIFEST = "all_efit_time_manifest.json"

_SLICE_RE = re.compile(
    r"^slice_efit_self_(?P<phase>ramp_up|flat_top|ramp_down)_t(?P<t>\d+)_k\d+\.npz$",
    re.IGNORECASE,
)
_PHASE_ORDER = {"ramp_up": 0, "flat_top": 1, "ramp_down": 2}


def _slice_sort_key(p: Path) -> tuple[int, int, str]:
    m = _SLICE_RE.match(p.name)
    if not m:
        return (99, 99, p.name)
    ph = m.group("phase").lower()
    t = int(m.group("t"), 10)
    return (_PHASE_ORDER.get(ph, 9), t, p.name)


def _jobs_from_manifest(manifest_path: Path) -> list[tuple[Path, Path, str | None]]:
    """返回 [(slice_npz, eval_out_dir, out_phase 或 None), ...] 按阶段与 t_index 排序。"""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    phases: dict[str, list] = data.get("phases") or {}
    out: list[tuple[int, int, Path, Path, str | None]] = []
    for phase in _PHASE_ORDER:
        rows = phases.get(phase)
        if not rows:
            continue
        for row in rows:
            es = row.get("efit_self") or {}
            sp = es.get("slice_npz")
            od = es.get("eval_out")
            if not sp or not od:
                continue
            ti = int(row.get("t_index", 0))
            sp_p = Path(sp)
            op = _inferred_out_phase_from_slice(sp_p) or phase
            out.append((_PHASE_ORDER[phase], ti, sp_p, Path(od), op))
    out.sort(key=lambda x: (x[0], x[1]))
    return [(a[2], a[3], a[4]) for a in out]


def _out_dir_for_slice(shot_run: Path, slice_path: Path) -> Path | None:
    m = _SLICE_RE.match(slice_path.name)
    if not m:
        return None
    phase = m.group("phase").lower()
    ti = int(m.group("t"), 10)
    return shot_run / phase / f"t{ti}_efit"


def _dir_name_k_t_efit(k: int, t_efit: float) -> str:
    """目录名：按 k 字典序即时间序列顺序；t_efit 用十进制秒（保留小数点与负号）。"""
    t = float(t_efit)
    ts = f"{t:.9f}".rstrip("0").rstrip(".") or "0"
    if ts == "-0":
        ts = "0"
    return f"k{k:05d}_t{ts}"


def _discharge_phase_for_precursor_k(phase_ids: np.ndarray, k: int) -> str:
    pid = int(phase_ids[k])
    if 0 <= pid < len(PHASE_NAMES):
        return PHASE_NAMES[pid]
    return "ramp_up"


def _inferred_out_phase_from_slice(slice_path: Path) -> str | None:
    m = _SLICE_RE.match(slice_path.name)
    if m:
        return m.group("phase").lower()
    return None


def _run_one_eval(
    *,
    slice_path: Path,
    out_dir: Path,
    shot: int,
    phase: str,
    rtol: float,
    nx: int,
    ny: int,
    dataset_root: str | None,
    dry_run: bool,
    out_phase: str | None = None,
    overlay_only: bool = False,
    subprocess_env: dict[str, str] | None = None,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "bc.run_freegsnke_eval",
        "--shot",
        str(int(shot)),
        "--phase",
        str(phase),
        "--precursor-slice-npz",
        str(slice_path.resolve()),
        "--out-root-exact",
        str(out_dir.resolve()),
        "--rtol",
        str(rtol),
        "--nx",
        str(nx),
        "--ny",
        str(ny),
    ]
    if out_phase is not None:
        cmd.extend(["--out-phase", str(out_phase)])
    if overlay_only:
        cmd.append("--overlay-only")
    if dataset_root:
        cmd.extend(["--dataset-root", dataset_root])
    print("RUN", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    env = {**os.environ, **dict(subprocess_env or {})}
    r = subprocess.run(cmd, cwd=str(_REPO), env=env)
    return int(r.returncode)


def _stitch_overlays(
    shot_run: Path,
    out_png: Path,
    ncols: int,
    *,
    jobs: list[tuple[Path, Path]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg

    if jobs is None:
        slices_dir = shot_run / "_slices"
        paths = sorted(slices_dir.glob("slice_efit_self_*.npz"), key=_slice_sort_key)
        job_pairs: list[tuple[Path, Path]] = []
        for sp in paths:
            od = _out_dir_for_slice(shot_run, sp)
            if od is not None:
                job_pairs.append((sp, od))
    else:
        job_pairs = list(jobs)

    overlay_paths: list[tuple[Path, Path]] = []
    for sp, od in job_pairs:
        png = od / "07_overlay_coils_lcfs_r8z8.png"
        if png.is_file():
            overlay_paths.append((sp, png))

    if not overlay_paths:
        raise SystemExit(f"未找到任何 07_overlay_coils_lcfs_r8z8.png（检查 {shot_run}）")

    n = len(overlay_paths)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 5.0 * nrows),
        squeeze=False,
    )
    for i, (sp, png) in enumerate(overlay_paths):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.imshow(mpimg.imread(str(png)))
        ax.axis("off")
        od = png.parent
        phase_note = ""
        sj = od / "summary.json"
        if sj.is_file():
            try:
                phase_note = str(json.loads(sj.read_text(encoding="utf-8")).get("phase", ""))
            except (json.JSONDecodeError, OSError):
                phase_note = ""
        m = _SLICE_RE.match(sp.name)
        if re.match(r"^k\d{5}_t", od.name):
            title = od.name
            if phase_note:
                title = f"{title} · {phase_note.replace('_', ' ')}"
        elif m:
            title = f"{m.group('phase')} t={int(m.group('t'), 10)}"
        else:
            title = sp.stem
        ax.set_title(title, fontsize=8)
    for j in range(len(overlay_paths), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    fig.suptitle(f"{shot_run.parent.name} / {shot_run.name} · overlay LCFS+R8Z8 (all slices)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[stitch] 写入 {out_png}（{len(overlay_paths)} 张）", flush=True)


def _write_all_efit_manifest(
    shot_run: Path,
    *,
    precursor_npz: Path,
    by_subdir: str,
    slices_subdir: str,
    work: list[tuple[Path, Path]],
    k_list: list[int],
    t_list: list[float],
    discharge_phases: list[str],
) -> None:
    rel_frames = []
    for i, (sp, od) in enumerate(work):
        rel_frames.append(
            {
                "k_efit": int(k_list[i]),
                "t_efit": float(t_list[i]),
                "discharge_phase": str(discharge_phases[i]),
                "slice_npz": str(sp.resolve().relative_to(shot_run.resolve())),
                "eval_out": str(od.resolve().relative_to(shot_run.resolve())),
            }
        )
    payload = {
        "version": 1,
        "precursor_npz": str(precursor_npz.resolve()),
        "by_t_efit_subdir": by_subdir,
        "slices_all_subdir": slices_subdir,
        "frames": rel_frames,
    }
    mp = shot_run / ALL_EFIT_MANIFEST
    mp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] 写入 {mp}（{len(rel_frames)} 帧）", flush=True)


def _jobs_from_all_efit_manifest(shot_run: Path) -> tuple[list[tuple[Path, Path, str | None]], bool]:
    mp = shot_run / ALL_EFIT_MANIFEST
    if not mp.is_file():
        return [], False
    data = json.loads(mp.read_text(encoding="utf-8"))
    root = shot_run.resolve()
    work: list[tuple[Path, Path, str | None]] = []
    for row in data.get("frames") or []:
        sp = (root / row["slice_npz"]).resolve()
        od = (root / row["eval_out"]).resolve()
        dp = row.get("discharge_phase")
        work.append((sp, od, str(dp) if dp else None))
    return work, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument(
        "--efit-self-root",
        type=str,
        default=str(_REPO / "results" / "freegsnke_eval_efit_self_random"),
    )
    ap.add_argument("--rtol", type=float, default=8e-3)
    ap.add_argument("--nx", type=int, default=129)
    ap.add_argument("--ny", type=int, default=129)
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目标目录已有 07_overlay_coils_lcfs_r8z8.png 则跳过 forward",
    )
    ap.add_argument("--stitch", action="store_true", help="结束后拼接所有 07 图为一张栅格 PNG")
    ap.add_argument(
        "--stitch-only",
        action="store_true",
        help="不跑 eval，只做拼图（需已有 07 图）",
    )
    ap.add_argument(
        "--stitch-out",
        type=str,
        default=None,
        help="拼图输出路径（默认 {shot_run}/00_all_overlays_grid.png）",
    )
    ap.add_argument("--stitch-ncols", type=int, default=5)
    ap.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="``multi_phase_time_slices_manifest.json``（BC 批量输出），"
        "用其中 efit_self 的 slice_npz + eval_out 驱动本脚本，可与按文件名推断二选一",
    )
    ap.add_argument(
        "--pred-run-root",
        type=str,
        default=str(_REPO / "results" / "freegsnke_eval_random"),
        help="未指定 --manifest 时尝试 ``{本根}/seed{seed}/{shot}/multi_phase_time_slices_manifest.json``",
    )
    ap.add_argument(
        "--ignore-manifest",
        action="store_true",
        help="忽略 multi_phase manifest，仅用 ``_slices/slice_efit_self_*.npz`` 推断任务",
    )
    ap.add_argument(
        "--every-efit-frame",
        action="store_true",
        help="按 precursor 的 ATIME 每一帧 export slice 并 eval；输出在 ``--by-t-efit-subdir`` 下按 k 与 t_efit 命名",
    )
    ap.add_argument(
        "--precursor-npz",
        type=str,
        default=None,
        help="``--every-efit-frame`` 时 precursor 路径；默认 results/freegsnke_precursors/{shot}/precursor.npz",
    )
    ap.add_argument(
        "--by-t-efit-subdir",
        type=str,
        default="by_t_efit",
        help="全时间轴模式下各帧 ``--out-root-exact`` 的父目录名（在 shot_run 下）",
    )
    ap.add_argument(
        "--slices-all-subdir",
        type=str,
        default="_slices_all_efit",
        help="全时间轴模式 slice npz 子目录（在 shot_run 下）",
    )
    ap.add_argument("--k-start", type=int, default=0, help="全时间轴：起始 EFIT 索引（含）")
    ap.add_argument(
        "--k-stop",
        type=int,
        default=None,
        help="全时间轴：结束 EFIT 索引（不含）；默认 len(ATIME)",
    )
    ap.add_argument(
        "--reuse-slices",
        action="store_true",
        help="全时间轴：若 slice_k*.npz 已存在则跳过 export_slice_npz",
    )
    ap.add_argument(
        "--phase",
        type=str,
        default="flat_top",
        choices=("ramp_up", "flat_top", "ramp_down"),
        help="传给 run_freegsnke_eval；slice 无阶段前缀时用于图注/summary 的名义阶段",
    )
    ap.add_argument(
        "--stitch-allow-huge",
        action="store_true",
        help="拼图帧数 >80 时仍执行（默认拒绝以防超大图）",
    )
    plot_grp = ap.add_mutually_exclusive_group()
    plot_grp.add_argument(
        "--overlay-only",
        action="store_true",
        help="调用 eval 时只生成 07 叠图（非 --every-efit-frame 时默认关闭）",
    )
    plot_grp.add_argument(
        "--full-plots",
        action="store_true",
        help="生成 00–07；与全时间轴默认「仅 07」相反",
    )
    args = ap.parse_args()

    if args.every_efit_frame and not args.stitch_only:
        overlay_only_effective = not bool(args.full_plots)
    else:
        overlay_only_effective = bool(args.overlay_only) and not bool(args.full_plots)

    shot_run = Path(args.efit_self_root).resolve() / f"seed{int(args.seed)}" / str(int(args.shot))
    shot_run.mkdir(parents=True, exist_ok=True)

    if args.every_efit_frame:
        if args.manifest:
            print("[warn] --every-efit-frame 忽略 multi_phase --manifest", flush=True)
        if args.stitch_only:
            work, ok = _jobs_from_all_efit_manifest(shot_run)
            if not ok or not work:
                raise SystemExit(
                    f"--stitch-only 需已有 {shot_run / ALL_EFIT_MANIFEST}（请先不带 --stitch-only 跑完全时间轴）"
                )
        else:
            prec = (
                Path(args.precursor_npz).resolve()
                if args.precursor_npz
                else (_REPO / "results" / "freegsnke_precursors" / str(int(args.shot)) / "precursor.npz")
            )
            if not prec.is_file():
                raise SystemExit(f"缺少 precursor：{prec}")
            bundle = load_precursor_npz(prec)
            bshot = int(bundle.get("shot") or 0)
            if bshot and bshot != int(args.shot):
                print(
                    f"[warn] precursor 内 shot={bshot} 与 --shot={args.shot} 不一致，仍按 --shot 写入 slice",
                    flush=True,
                )
            bundle["shot"] = int(args.shot)
            atime = np.asarray(bundle["ATIME"], dtype=np.float64).ravel()
            te = int(atime.size)
            k0 = max(0, int(args.k_start))
            k1 = int(args.k_stop) if args.k_stop is not None else te
            k1 = min(te, k1)
            if k0 >= k1:
                raise SystemExit(f"空 k 范围 [{k0},{k1})，ATIME 长度 {te}")

            slices_sub = shot_run / args.slices_all_subdir
            by_root = shot_run / args.by_t_efit_subdir
            slices_sub.mkdir(parents=True, exist_ok=True)
            by_root.mkdir(parents=True, exist_ok=True)

            ipr = np.asarray(bundle["PCRL01"], dtype=np.float64).ravel()
            if ipr.size != te:
                raise SystemExit(f"precursor ATIME 长度 {te} 与 PCRL01 长度 {ipr.size} 不一致")
            bundle_phase_ids = phase_ids_per_step(atime, ipr)

            work = []
            k_list: list[int] = []
            t_list: list[float] = []
            discharge_list: list[str] = []
            for k in range(k0, k1):
                t_efit = float(atime[k])
                dname = _dir_name_k_t_efit(k, t_efit)
                slice_path = slices_sub / f"slice_k{k:05d}.npz"
                out_dir = by_root / dname
                if not args.reuse_slices or not slice_path.is_file():
                    if not args.dry_run:
                        export_slice_npz(bundle, k, slice_path)
                dp = _discharge_phase_for_precursor_k(bundle_phase_ids, k)
                work.append((slice_path, out_dir, dp))
                k_list.append(k)
                t_list.append(t_efit)
                discharge_list.append(dp)

            if not args.dry_run:
                _write_all_efit_manifest(
                    shot_run,
                    precursor_npz=prec,
                    by_subdir=args.by_t_efit_subdir,
                    slices_subdir=args.slices_all_subdir,
                    work=[(a, b) for a, b, _ in work],
                    k_list=k_list,
                    t_list=t_list,
                    discharge_phases=discharge_list,
                )
            else:
                print(f"[dry-run] 将处理 {len(work)} 帧，k∈[{k0},{k1})", flush=True)

    else:
        manifest_path: Path | None = None
        if not args.ignore_manifest:
            if args.manifest:
                manifest_path = Path(args.manifest).resolve()
                if not manifest_path.is_file():
                    raise SystemExit(f"--manifest 不存在：{manifest_path}")
            else:
                cand = Path(args.pred_run_root).resolve() / f"seed{int(args.seed)}" / str(int(args.shot))
                mp = cand / "multi_phase_time_slices_manifest.json"
                if mp.is_file():
                    manifest_path = mp

        jobs_manifest: list[tuple[Path, Path, str | None]] | None = None
        if manifest_path is not None and manifest_path.is_file():
            jobs_manifest = _jobs_from_manifest(manifest_path)
            if not jobs_manifest:
                print(f"[warn] manifest 无 efit_self 项：{manifest_path}", flush=True)
                jobs_manifest = None

        if jobs_manifest is not None:
            work = jobs_manifest
        else:
            slices_dir = shot_run / "_slices"
            if not slices_dir.is_dir():
                raise SystemExit(f"缺少 slice 目录：{slices_dir}（或提供有效 --manifest）")
            slice_files = sorted(slices_dir.glob("slice_efit_self_*.npz"), key=_slice_sort_key)
            if not slice_files:
                raise SystemExit(f"{slices_dir} 下没有 slice_efit_self_*.npz（或提供有效 --manifest）")
            work = []
            for sp in slice_files:
                od = _out_dir_for_slice(shot_run, sp)
                if od is not None:
                    work.append((sp, od, _inferred_out_phase_from_slice(sp)))
            if not work:
                raise SystemExit("无法从 slice 文件名推断输出目录")

    max_rc = 0
    if not args.stitch_only:
        for sp, od, outp in work:
            od.mkdir(parents=True, exist_ok=True)
            target_png = od / "07_overlay_coils_lcfs_r8z8.png"
            if args.skip_existing and target_png.is_file():
                print(f"[skip-existing] {target_png}", flush=True)
                continue
            rc = _run_one_eval(
                slice_path=sp,
                out_dir=od,
                shot=int(args.shot),
                phase=str(args.phase),
                rtol=float(args.rtol),
                nx=int(args.nx),
                ny=int(args.ny),
                dataset_root=args.dataset_root,
                dry_run=bool(args.dry_run),
                out_phase=outp,
                overlay_only=overlay_only_effective,
            )
            if rc != 0:
                print(f"[WARN] 退出码 {rc}：{sp}", file=sys.stderr, flush=True)
                max_rc = max(max_rc, rc)

    if args.stitch or args.stitch_only:
        if len(work) > 80 and not args.stitch_allow_huge:
            raise SystemExit(
                f"将拼图任务含 {len(work)} 帧，过大；请加 --stitch-allow-huge 或缩小 k 范围/勿使用 --stitch"
            )
        default_stitch = (
            shot_run / "00_all_overlays_efit_time_grid.png"
            if args.every_efit_frame
            else shot_run / "00_all_overlays_grid.png"
        )
        out_s = args.stitch_out or str(default_stitch)
        stitch_jobs = [(a, b) for a, b, _ in work]
        _stitch_overlays(shot_run, Path(out_s), int(args.stitch_ncols), jobs=stitch_jobs)

    return min(max_rc, 255) if max_rc > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
