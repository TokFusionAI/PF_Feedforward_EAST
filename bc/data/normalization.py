"""在 train_shots 上算 per-channel mean/std (Welford 在线统计), 写到 meta/norm_stats.npz.

state 21 维 (R8(8) + Z8(8) + lmsr + lmsz + PCRL01 + time + dt)
action 12 维 (PCPF1..8, PCPF11..14)

注意:
- state 末两维 time / dt 也用 z-score 归一化 (state_embed 见到的 scale 一致).
- model 的 sinusoidal PE 直接用未归一化的物理 time, 不受 norm_stats 影响.
- 只在源信号 mask = True 的位置统计, 避免 NaN/虚假 0 污染.
- R8 / Z8 没有 mask, 全部参与统计 (它们与 ATIME 天然对齐).

Usage:
    python -m bc.normalization \
        --shots-file meta/train_shots.txt \
        --dataset-root /data/PF_ATIME_dataset \
        --out meta/norm_stats.npz
"""

from __future__ import annotations

import argparse
import time as _time
from pathlib import Path

import h5py
import numpy as np

from bc.common.constants import (
    D_ACTION,
    D_STATE,
    DATASET_ROOT,
    PCPF_NAMES,
    STATE_LAYOUT,
)


class WelfordND:
    """Per-channel running mean/var via Welford (Chan parallel form)."""

    def __init__(self, dim: int):
        self.n = np.zeros(dim, dtype=np.float64)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray, mask: np.ndarray | None = None) -> None:
        """x: (N, dim) float; mask: (N, dim) bool, default all True."""
        if mask is None:
            mask = np.ones_like(x, dtype=bool)
        for c in range(x.shape[1]):
            m = mask[:, c]
            if not np.any(m):
                continue
            xc = x[m, c].astype(np.float64)
            n_a = self.n[c]
            n_b = xc.size
            mean_b = xc.mean()
            var_b = xc.var() if n_b > 1 else 0.0
            n_ab = n_a + n_b
            delta = mean_b - self.mean[c]
            self.mean[c] += delta * n_b / n_ab
            self.M2[c] += var_b * n_b + delta * delta * n_a * n_b / n_ab
            self.n[c] = n_ab

    @property
    def std(self) -> np.ndarray:
        var = np.where(self.n > 1, self.M2 / np.maximum(self.n - 1, 1), 0.0)
        return np.sqrt(var)


def _read_shots_txt(path: Path) -> list[int]:
    return [int(t) for t in path.read_text().split() if t.strip().isdigit()]


def _load_shot_arrays(shot_h5: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return state(T,21), action(T,12), action_mask(T,12) for one shot.
    state mask is computed inline in main() (R8/Z8/time/dt always valid;
    lmsr/lmsz/PCRL01 use their h5 masks).
    """
    with h5py.File(shot_h5, "r") as f:
        time_phys = f["time"][:].astype(np.float64)
        T = int(time_phys.shape[0])

        R8 = f["R8"][:].astype(np.float64)
        Z8 = f["Z8"][:].astype(np.float64)
        lmsr = f["lmsr"][:].astype(np.float64)
        lmsz = f["lmsz"][:].astype(np.float64)
        pcrl = f["PCRL01"][:].astype(np.float64)

        m_lmsr = f["mask_lmsr"][:].astype(bool)
        m_lmsz = f["mask_lmsz"][:].astype(bool)
        m_pcrl = f["mask_PCRL01"][:].astype(bool)

        if T >= 2:
            dt = np.diff(time_phys, prepend=time_phys[0])
            dt[0] = float(np.median(dt[1:]))
        else:
            dt = np.full((T,), 0.1, dtype=np.float64)

        state = np.concatenate(
            [
                R8,
                Z8,
                lmsr[:, None],
                lmsz[:, None],
                pcrl[:, None],
                time_phys[:, None],
                dt[:, None],
            ],
            axis=1,
        )
        # state 的 per-channel mask:
        #   R8/Z8 (16 cols) 全 True; lmsr/lmsz/PCRL01 用源 mask; time/dt 全 True.
        state_mask = np.ones((T, D_STATE), dtype=bool)
        state_mask[:, 16] = m_lmsr
        state_mask[:, 17] = m_lmsz
        state_mask[:, 18] = m_pcrl

        # NaN 防御 (mask=False 处理论上是 NaN, 不进 update; 仍 nan_to_num 防 indexing)
        state = np.nan_to_num(state, nan=0.0)

        action = np.stack([f[name][:].astype(np.float64) for name in PCPF_NAMES], axis=1)
        action_mask = np.stack(
            [f[f"mask_{name}"][:].astype(bool) for name in PCPF_NAMES], axis=1
        )
        action = np.nan_to_num(action, nan=0.0)

    return state, action, state_mask, action_mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots-file", type=Path, default=Path("meta/train_shots.txt"))
    ap.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    ap.add_argument("--out", type=Path, default=Path("meta/norm_stats.npz"))
    ap.add_argument("--limit", type=int, default=None, help="optional: only first N shots (debug)")
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args()

    shots = _read_shots_txt(args.shots_file)
    if args.limit is not None:
        shots = shots[: args.limit]
    print(f"Welford on {len(shots)} shots from {args.shots_file}")

    ws = WelfordND(D_STATE)
    wa = WelfordND(D_ACTION)
    t0 = _time.monotonic()
    n_ok = n_err = 0

    for i, shot in enumerate(shots, 1):
        try:
            state, action, state_mask, action_mask = _load_shot_arrays(
                args.dataset_root / f"{shot}.h5"
            )
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"  err shot={shot}: {type(e).__name__}: {e}")
            continue
        ws.update(state, mask=state_mask)
        wa.update(action, mask=action_mask)
        n_ok += 1
        if i % args.progress_every == 0:
            elapsed = _time.monotonic() - t0
            eta = elapsed / i * (len(shots) - i)
            print(
                f"  {i}/{len(shots)}  ok={n_ok} err={n_err}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
            )

    state_mean = ws.mean.astype(np.float32)
    state_std = ws.std.astype(np.float32)
    action_mean = wa.mean.astype(np.float32)
    action_std = wa.std.astype(np.float32)

    print(f"\nDone: ok={n_ok} err={n_err}  elapsed={_time.monotonic()-t0:.0f}s")

    print("\nstate per-channel stats (21 cols):")
    print(f"  {'idx':>3} {'name':<10} {'mean':>14} {'std':>14}  {'n_eff':>14}")
    for i, name in enumerate(STATE_LAYOUT):
        print(f"  {i:>3d} {name:<10} {state_mean[i]:>14.4f} {state_std[i]:>14.4f}  {ws.n[i]:>14.0f}")
    print("\naction per-channel stats (12 cols):")
    for i, name in enumerate(PCPF_NAMES):
        print(f"  {i:>3d} {name:<10} {action_mean[i]:>14.4f} {action_std[i]:>14.4f}  {wa.n[i]:>14.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        state_layout=np.asarray(STATE_LAYOUT),
        action_layout=np.asarray(PCPF_NAMES),
        n_shots_ok=np.int64(n_ok),
        n_shots_err=np.int64(n_err),
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
