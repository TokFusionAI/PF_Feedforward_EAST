#!/usr/bin/env python3
"""每炮：三阶段各选「前向 freegsnke 与 EFIT R8/Z8 最契合」的一帧，拼 1×3 概览图。

在 ramp_up / flat_top / ramp_down 各自的时间索引集合内（与 ``phase_ids_per_step`` 一致），
对 ``by_t_efit/*/summary.json`` 比较指标，默认取 ``lcfs_rmsep_to_r8z8_m`` **最小**（非 NaN）的一帧，
展示该目录下的 ``07_overlay_coils_lcfs_r8z8.png``；下方附加 precursor 的 ``PCRL01``（等离子体电流 :math:`I_p`）随 ``ATIME`` 曲线，并在三阶段选优时刻画竖线与曲线散点。

输出两张图（若目录存在且该阶段有候选帧）::

    {out_dir}/montage_bestPerPhase_pcs_shot{shot}.png
    {out_dir}/montage_bestPerPhase_bc_pred_shot{shot}.png

并写入 ``montage_best_per_phase_shot{shot}.json`` 记录各阶段选定目录与误差。

默认假定整炮 eval 根：``{root}/{shot}/pcs`` 与 ``{root}/{shot}/bc_pred`` 下各有一``by_t_efit``。

示例::

    python -m bc.batch_freegsnke.montage_best_per_phase --shot 105653

    python -m bc.batch_freegsnke.montage_best_per_phase --shots-file shots.txt \\
      --whole-shot-root results/freegsnke_whole_shot --out-dir results/montage_best_lcfs
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from bc.data.phases import PHASE_NAMES, phase_ids_per_step
from bc.gs_forward.precursor_export import load_precursor_npz

_REPO = Path(__file__).resolve().parent.parent.parent
_DIR_K_RE = re.compile(r"^k(\d+)_t", re.IGNORECASE)
OVERLAY_NAME = "07_overlay_coils_lcfs_r8z8.png"


def _k_from_dirname(d: Path) -> int | None:
    m = _DIR_K_RE.match(d.name)
    return int(m.group(1), 10) if m else None


def _load_frame_metrics(by_t_efit: Path) -> list[dict[str, Any]]:
    """扫描 by_t_efit/* /summary.json。"""
    rows: list[dict[str, Any]] = []
    if not by_t_efit.is_dir():
        return rows
    for d in sorted(by_t_efit.iterdir()):
        if not d.is_dir():
            continue
        sj = d / "summary.json"
        if not sj.is_file():
            continue
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        k = int(data.get("k_efit_profile", -1))
        if k < 0:
            kk = _k_from_dirname(d)
            if kk is None:
                continue
            k = kk
        rmsep = data.get("lcfs_rmsep_to_r8z8_m")
        if rmsep is None:
            rmsep = float("nan")
        else:
            rmsep = float(rmsep)
        rel = data.get("forward_rel_residual")
        if rel is None:
            rel = float("nan")
        else:
            rel = float(rel)
        rows.append(
            {
                "eval_dir": str(d.resolve()),
                "k_efit": k,
                "time_s": float(data.get("time_s", float("nan"))),
                "lcfs_rmsep_to_r8z8_m": rmsep,
                "forward_rel_residual": rel,
                "forward_converged": bool(data.get("forward_converged", False)),
                "overlay_png": str((d / OVERLAY_NAME).resolve()),
            }
        )
    return rows


def _best_in_phase(
    rows: list[dict[str, Any]],
    phase_ids: np.ndarray,
    phase: str,
    *,
    metric: str,
    converged_only: bool,
) -> dict[str, Any] | None:
    pid = PHASE_NAMES.index(phase)
    candidates: list[dict[str, Any]] = []
    for r in rows:
        k = int(r["k_efit"])
        if k < 0 or k >= phase_ids.size:
            continue
        if int(phase_ids[k]) != pid:
            continue
        if converged_only and not r["forward_converged"]:
            continue
        if metric == "lcfs_rmsep":
            v = float(r["lcfs_rmsep_to_r8z8_m"])
        else:
            v = float(r["forward_rel_residual"])
        if not math.isfinite(v):
            continue
        candidates.append(r)

    if not candidates:
        return None

    reverse = False  # smaller is better
    def _metric_val(x: dict[str, Any]) -> float:
        if metric == "lcfs_rmsep":
            return float(x["lcfs_rmsep_to_r8z8_m"])
        return float(x["forward_rel_residual"])

    candidates.sort(key=_metric_val)
    return candidates[0]


def _interp_ip(atime: np.ndarray, ip: np.ndarray, t: float) -> float:
    m = np.isfinite(atime) & np.isfinite(ip)
    if not np.any(m):
        return float("nan")
    a = np.asarray(atime[m], dtype=np.float64)
    y = np.asarray(ip[m], dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    a = a[order]
    y = y[order]
    t = float(t)
    if t <= a[0]:
        return float(y[0])
    if t >= a[-1]:
        return float(y[-1])
    return float(np.interp(t, a, y))


_PHASE_VLINE_COLORS: dict[str, str] = {
    # 与上方 overlay（青蓝线圈、红 LCFS、青蓝 EFIT 边界）协调的区分色
    "ramp_up": "#0ea5e9",
    "flat_top": "#14b8a6",
    "ramp_down": "#e11d48",
}


def _render_ip_trace_strip(
    *,
    total_w_px: int,
    plot_h_px: int,
    dpi: int,
    atime: np.ndarray,
    ip: np.ndarray,
    picks: dict[str, dict[str, Any] | None],
    resample: int,
) -> Any:
    """用 Matplotlib 光栅化整炮 :math:`I_p(t)`，并在各阶段选优时刻画竖线与曲线散点。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    w_in = total_w_px / dpi
    h_in = plot_h_px / dpi
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi)
    ip_a = np.asarray(ip, dtype=np.float64)
    ip_ka = ip_a / 1000.0
    at_a = np.asarray(atime, dtype=np.float64)
    m = np.isfinite(at_a) & np.isfinite(ip_a)
    ax.plot(
        at_a[m],
        ip_ka[m],
        color="#475569",
        lw=1.25,
        alpha=0.92,
        zorder=1,
    )

    spots: list[tuple[float, float, str, str]] = []
    for ph in PHASE_NAMES:
        pick = picks.get(ph)
        if pick is None:
            continue
        ts = float(pick.get("time_s", float("nan")))
        if not math.isfinite(ts):
            continue
        color = _PHASE_VLINE_COLORS.get(ph, "0.35")
        ax.axvline(ts, color=color, ls="--", lw=1.35, alpha=0.88, zorder=2)
        iy = _interp_ip(at_a, ip_ka, ts)
        if math.isfinite(iy):
            ax.scatter(
                [ts],
                [iy],
                s=42,
                color=color,
                zorder=5,
                edgecolors="white",
                linewidths=0.65,
            )
            spots.append((ts, iy, color, ph))

    spots.sort(key=lambda x: x[0])

    for ts, iy, color, ph in spots:
        # 爬升/平顶：标注在标记点右下方（避开曲线）；下降：右上方
        dx_px = 6
        if ph == "ramp_down":
            dy_px = 8
            ha, va = "left", "bottom"
        else:
            dy_px = -8
            ha, va = "left", "top"
        ax.annotate(
            f"{ts:.2f} s",
            xy=(ts, iy),
            xytext=(dx_px, dy_px),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8,
            color=color,
            fontweight="medium",
            clip_on=False,
            zorder=6,
            path_effects=[
                pe.Stroke(linewidth=2.8, foreground="white"),
                pe.Normal(),
            ],
        )

    ax.set_xlabel(r"$t$ (s)", fontsize=9)
    ax.set_ylabel(r"$I_{\mathrm{p}}$ (kA)", fontsize=9)
    ax.grid(True, ls=":", alpha=0.4, color="0.65")
    ax.tick_params(labelsize=8)
    ax.margins(x=0.03, y=0.18)

    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    im = Image.open(buf).convert("RGB")
    if im.size[0] != total_w_px or im.size[1] != plot_h_px:
        im = im.resize((total_w_px, plot_h_px), resample)
    return im


def _merge_ip_panel_if_any(
    *,
    top_rgb: Any,
    atime: np.ndarray | None,
    ip: np.ndarray | None,
    picks: dict[str, dict[str, Any] | None],
    resample: int,
    dpi: int = 300,
) -> Any:
    """在 ``top_rgb`` 下方拼接 :math:`I_p(t)` 光栅图（宽度与顶部对齐）。"""
    from PIL import Image

    if atime is None or ip is None:
        return top_rgb
    aa = np.asarray(atime, dtype=np.float64).ravel()
    ii = np.asarray(ip, dtype=np.float64).ravel()
    if aa.size != ii.size or aa.size < 2:
        return top_rgb
    tw, th = top_rgb.size
    plot_h = int(max(320, min(900, round(th * 0.38))))
    bottom = _render_ip_trace_strip(
        total_w_px=tw,
        plot_h_px=plot_h,
        dpi=dpi,
        atime=aa,
        ip=ii,
        picks=picks,
        resample=resample,
    )
    vgap = 14
    out = Image.new("RGB", (tw, th + vgap + plot_h), (255, 255, 255))
    out.paste(top_rgb, (0, 0))
    out.paste(bottom, (0, th + vgap))
    return out


def _plot_montage(
    *,
    shot: int,
    branch: str,
    picks: dict[str, dict[str, Any] | None],
    out_png: Path,
    metric: str,
    atime: np.ndarray | None = None,
    ip: np.ndarray | None = None,
) -> None:
    """横向拼图 overlay PNG（像素级拼接），并在下方绘制整炮 :math:`I_p(t)` 与三阶段选优时刻。"""
    from PIL import Image

    try:
        _resample = Image.Resampling.LANCZOS
    except AttributeError:
        _resample = Image.LANCZOS  # type: ignore[attr-defined]

    gap_px = 8
    loaded: list[Image.Image | None] = []
    for phase in PHASE_NAMES:
        pick = picks.get(phase)
        if pick is None:
            loaded.append(None)
            continue
        png = Path(pick["overlay_png"])
        if not png.is_file():
            loaded.append(None)
            continue
        loaded.append(Image.open(png).convert("RGBA"))

    heights = [im.size[1] for im in loaded if im is not None]
    if not heights:
        max_h, ph_w = 600, 400
        total_w = 3 * ph_w + 2 * gap_px
        top_rgb = Image.new("RGB", (total_w, max_h), (240, 240, 240))
    else:
        max_h = max(heights)

        def _scale_to_h(im: Image.Image) -> Image.Image:
            if im.size[1] == max_h:
                return im
            new_w = max(1, int(round(im.size[0] * max_h / im.size[1])))
            return im.resize((new_w, max_h), _resample)

        scaled: list[Image.Image | None] = [_scale_to_h(im) if im is not None else None for im in loaded]
        widths_ok = [im.size[0] for im in scaled if im is not None]
        ph_w = int(np.median(widths_ok)) if widths_ok else 400

        panels: list[Image.Image] = []
        for im in scaled:
            if im is None:
                panels.append(Image.new("RGBA", (ph_w, max_h), (240, 240, 240, 255)))
            else:
                panels.append(im)

        total_w = sum(p.size[0] for p in panels) + gap_px * (len(panels) - 1)
        out = Image.new("RGBA", (total_w, max_h), (255, 255, 255, 255))
        x = 0
        for i, p in enumerate(panels):
            out.paste(p, (x, 0), p)
            x += p.size[0] + (gap_px if i < len(panels) - 1 else 0)

        top_rgb = Image.new("RGB", out.size, (255, 255, 255))
        top_rgb.paste(out, (0, 0), out)

    final_rgb = _merge_ip_panel_if_any(
        top_rgb=top_rgb,
        atime=atime,
        ip=ip,
        picks=picks,
        resample=_resample,
        dpi=300,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    final_rgb.save(out_png, dpi=(300, 300), optimize=True)
    print(f"[montage] {out_png}", flush=True)


def _process_shot(
    *,
    shot: int,
    precursor_npz: Path,
    pcs_by_t_efit: Path,
    bc_by_t_efit: Path,
    out_dir: Path,
    metric: str,
    converged_only: bool,
) -> int:
    if not precursor_npz.is_file():
        print(f"[skip] shot={shot} 无 precursor {precursor_npz}", file=sys.stderr, flush=True)
        return 2

    bundle = load_precursor_npz(precursor_npz)
    atime = np.asarray(bundle["ATIME"], dtype=np.float64).ravel()
    ipr = np.asarray(bundle["PCRL01"], dtype=np.float64).ravel()
    if ipr.size != atime.size:
        print(f"[skip] shot={shot} ATIME/PCRL01 长度不一致", file=sys.stderr, flush=True)
        return 2
    phase_ids = phase_ids_per_step(atime, ipr)

    manifest: dict[str, Any] = {"shot": shot, "metric": metric, "converged_only": converged_only, "pcs": {}, "bc_pred": {}}

    def run_branch(branch: str, by_root: Path, tag: str) -> None:
        rows = _load_frame_metrics(by_root)
        if not rows:
            print(f"[warn] shot={shot} {branch}: 无 summary（{by_root}）", flush=True)
            manifest[tag] = {ph: None for ph in PHASE_NAMES}
            return
        picks: dict[str, dict[str, Any] | None] = {}
        for ph in PHASE_NAMES:
            p = _best_in_phase(rows, phase_ids, ph, metric=metric, converged_only=converged_only)
            picks[ph] = p
            manifest[tag][ph] = p
        out_png = out_dir / f"montage_bestPerPhase_{branch}_shot{shot}.png"
        _plot_montage(
            shot=shot,
            branch=branch,
            picks=picks,
            out_png=out_png,
            metric=metric,
            atime=atime,
            ip=ipr,
        )

    if pcs_by_t_efit.parent.is_dir():
        run_branch("pcs", pcs_by_t_efit, "pcs")
    else:
        print(f"[warn] shot={shot} 无 PCS 目录 {pcs_by_t_efit.parent}", flush=True)
        manifest["pcs"] = {ph: None for ph in PHASE_NAMES}

    if bc_by_t_efit.parent.is_dir():
        run_branch("bc_pred", bc_by_t_efit, "bc_pred")
    else:
        print(f"[warn] shot={shot} 无 bc_pred 目录 {bc_by_t_efit.parent}", flush=True)
        manifest["bc_pred"] = {ph: None for ph in PHASE_NAMES}

    meta_path = out_dir / f"montage_best_per_phase_shot{shot}.json"

    def _json_safe(x: Any) -> Any:  # noqa: ANN401
        if x is None:
            return None
        if isinstance(x, dict):
            return {k: _json_safe(v) for k, v in x.items()}
        return x

    slim = {
        "shot": shot,
        "metric": metric,
        "converged_only": converged_only,
        "pcs": {ph: _json_safe(manifest["pcs"].get(ph)) for ph in PHASE_NAMES},
        "bc_pred": {ph: _json_safe(manifest["bc_pred"].get(ph)) for ph in PHASE_NAMES},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] {meta_path}", flush=True)
    return 0


def _read_shots_file(path: Path) -> list[int]:
    out: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s.isdigit():
            out.append(int(s))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=None)
    ap.add_argument("--shots-file", type=str, default=None, help="每行一炮号")
    ap.add_argument(
        "--precursor-root",
        type=str,
        default=str(_REPO / "results" / "freegsnke_precursors"),
        help="precursor 在 {root}/{shot}/precursor.npz",
    )
    ap.add_argument(
        "--whole-shot-root",
        type=str,
        default=str(_REPO / "results" / "freegsnke_whole_shot"),
        help="整炮 eval 根：{root}/{shot}/pcs|bc_pred/by_t_efit",
    )
    ap.add_argument(
        "--by-t-efit-subdir",
        type=str,
        default="by_t_efit",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="拼图与 json；默认 {whole-shot-root}/_montage",
    )
    ap.add_argument(
        "--metric",
        type=str,
        choices=("lcfs_rmsep", "forward_rel_residual"),
        default="lcfs_rmsep",
        help="阶段内选优指标：LCFS 对 R8/Z8 RMSE 或 forward 相对残差",
    )
    ap.add_argument(
        "--converged-only",
        action="store_true",
        help="只考虑 forward_converged==true 的帧",
    )
    args = ap.parse_args()

    shots: list[int] = []
    if args.shots_file:
        p = Path(args.shots_file).resolve()
        if not p.is_file():
            raise SystemExit(f"--shots-file 不存在：{p}")
        shots.extend(_read_shots_file(p))
    if args.shot is not None:
        shots.append(int(args.shot))
    if not shots:
        raise SystemExit("请指定 --shot 或 --shots-file")

    prec_root = Path(args.precursor_root).resolve()
    whole_root = Path(args.whole_shot_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (whole_root / "_montage")

    rc_all = 0
    for shot in sorted(set(shots)):
        prec = prec_root / str(shot) / "precursor.npz"
        pcs_bt = whole_root / str(shot) / "pcs" / args.by_t_efit_subdir
        bc_bt = whole_root / str(shot) / "bc_pred" / args.by_t_efit_subdir
        r = _process_shot(
            shot=shot,
            precursor_npz=prec,
            pcs_by_t_efit=pcs_bt,
            bc_by_t_efit=bc_bt,
            out_dir=out_dir,
            metric=str(args.metric),
            converged_only=bool(args.converged_only),
        )
        if r != 0:
            rc_all = max(rc_all, r)
    return min(rc_all, 255)


if __name__ == "__main__":
    raise SystemExit(main())
