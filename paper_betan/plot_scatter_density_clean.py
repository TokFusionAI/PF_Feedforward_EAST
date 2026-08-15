"""per_channel_scatter 清洁版: 丢弃每炮【末次破裂之后】的时段 (Ip 骤降悬崖起, 含悬崖点及之后),
再画密度散点。模型无关的物理判据 (破裂后等离子体已不存在, PF 目标无意义)。

判据 (可调):
  HIGH_FRAC=0.5  破裂前须到过 0.5*Ip_max (确认曾有等离子体)
  CRASH_FRAC=0.3 Ip 跌破 0.3*Ip_max 视为已崩溃
  破裂点 = flat_top 末尾之后首个 <CRASH_FRAC*Ip_max 的索引; 且该点与上一个高点之间是"悬崖"
           (这里 dt~200ms, 破裂=ms 级 → 单步悬崖; 正常 ramp-down 是渐进, 不会从 >0.5 瞬降到 <0.3,
            故天然区分)。
  drop: action_mask[crash_idx:, :] = False   (悬崖点及之后全丢)

复用 plot_scatter_density 的画图函数 (口径一致), 只是先把 records 的 action_mask 就地裁剪。
"""
from __future__ import annotations
import argparse, sys, copy
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_betan.plot_figures import PF_NAMES, _metrics_1d, load_records  # noqa
from paper_betan.plot_scatter_density import (  # noqa
    STYLES, _gather_channels, _rmse, _prep_grid, _prep_joint, render_grid, render_joint,
)

PRED = Path("results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/per_shot_preds.npz")
HIGH_FRAC = 0.5
CRASH_FRAC = 0.3


def detect_crash_idx(ip_abs, ip_max, phase_ids):
    """破裂悬崖索引 (含); None=无窗内破裂。"""
    if ip_max < 100_000:
        return None
    ft = phase_ids == 1
    if ft.sum() == 0:
        return None
    ft_end = int(np.where(ft)[0][-1])
    below = np.where(ip_abs[ft_end:] < CRASH_FRAC * ip_max)[0]
    if len(below) == 0:
        return None
    crash = ft_end + int(below[0])
    if ip_abs[:crash].max() < HIGH_FRAC * ip_max:
        return None
    return crash


def apply_disruption_mask(recs, drop_incl_crash=True):
    """就地裁剪 action_mask; 返回 (n_shots_affected, n_steps_dropped, 详情列表)。"""
    affected, dropped, details = 0, 0, []
    for r in recs:
        ip_abs = np.abs(r["Ip_A"])
        ip_max = float(ip_abs.max()) if ip_abs.size else 0.0
        crash = detect_crash_idx(ip_abs, ip_max, r["phase_ids"])
        if crash is None:
            continue
        start = crash if drop_incl_crash else crash + 1
        if start >= r["action_mask"].shape[0]:
            continue
        n = int(r["action_mask"][start:].sum())
        if n == 0:
            continue
        r["action_mask"][start:] = False
        affected += 1; dropped += n
        details.append((r["shot"], crash, n))
    return affected, dropped, details


def pf_metrics(records, ch):
    ts = np.concatenate([r["target_kA"][:, ch][r["action_mask"][:, ch]] for r in records])
    ps = np.concatenate([r["pred_kA"][:, ch][r["action_mask"][:, ch]] for r in records])
    return _metrics_1d(ps, ts)[0], _rmse(ps, ts), ts.size


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default=str(PRED))
    ap.add_argument("--out-dir", default="results/paper_betan/figures")
    ap.add_argument("--gridsize", type=int, default=180)
    ap.add_argument("--style", choices=STYLES, default=None)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--drop-incl-crash", action=argparse.BooleanOptionalAction, default=True,
                    help="True=连悬崖点一起丢(默认); --no-drop-incl-crash 只丢严格之后")
    ap.add_argument("--suffix", default="clean", help="输出文件后缀 (per_channel_scatter_hex_<suffix>.png)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = load_records(Path(args.preds))

    # before
    PF6 = 5
    print(f"loaded {len(recs)} shots")
    print("\n=== BEFORE disruption masking: per-channel R²/RMSE ===")
    before = {}
    for ch in range(12):
        r2, rmse, n = pf_metrics(recs, ch)
        before[ch] = (r2, n)
        print(f"  {PF_NAMES[ch]:5s} R²={r2:.4f}  RMSE={rmse:.3f}  n={n}")

    # apply
    aff, drp, det = apply_disruption_mask(recs, args.drop_incl_crash)
    print(f"\n=== disruption masking: affected {aff} shots, dropped {drp} masked-in steps ===")
    if det:
        print("  " + ", ".join(f"{s}(crash@{c},drop{n})" for s, c, n in det[:30]) + (" ..." if len(det) > 30 else ""))

    # after
    print("\n=== AFTER disruption masking: per-channel R²/RMSE ===")
    for ch in range(12):
        r2, rmse, n = pf_metrics(recs, ch)
        b2, bn = before[ch]
        print(f"  {PF_NAMES[ch]:5s} R²={r2:.4f} (Δ={r2-b2:+.4f})  RMSE={rmse:.3f}  n={n} (was {bn})")

    # plot
    styles = [args.style] if args.style else STYLES
    chans, glo, ghi = _prep_grid(recs)
    jprep = _prep_joint(recs, normalize=args.normalize)
    for s in styles:
        render_grid(chans, glo, ghi, out / f"per_channel_scatter_{s}_{args.suffix}.png",
                    gridsize=args.gridsize, style=s, sigma=args.sigma)
        render_joint(jprep, out / f"pooled_scatter_joint_{s}_{args.suffix}.png",
                     gridsize=args.gridsize + 40, style=s, sigma=args.sigma)
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
