"""Shared record loading and metric helpers for the paper evaluation scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bc.common.constants import PCPF_NAMES

PF_NAMES = [n.replace("PCPF", "PF") for n in PCPF_NAMES]


def _metrics_1d(preds, truths):
    """Return (R^2, RMSE, MAE) over the valid samples of one channel."""
    preds = np.asarray(preds, float)
    truths = np.asarray(truths, float)
    err = preds - truths
    ss = float(((truths - truths.mean()) ** 2).sum())
    r2 = 1.0 - float((err ** 2).sum()) / ss if ss > 0 else float("nan")
    return r2, float(np.sqrt((err ** 2).mean())), float(np.abs(err).mean())


def load_records(npz_path: Path) -> list[dict]:
    """Load per_shot_preds.npz into per-shot record dicts.

    Hoists the compressed arrays out of the loop (re-indexing a compressed
    npz member re-decompresses the whole array each time).
    """
    d = np.load(npz_path, allow_pickle=False)
    shots, T = d["shots"], d["T"]
    pred_all = d["pred_kA"]; tgt_all = d["target_kA"]; mask_all = d["action_mask"]
    time_all = d["time"]; dt_all = d["dt"]; ip_all = d["Ip_A"]; pid_all = d["phase_ids"]
    recs = []
    for i in range(len(shots)):
        Tt = int(T[i])
        pred = pred_all[i, :Tt].astype(np.float32)
        tgt = tgt_all[i, :Tt].astype(np.float32)
        mask = mask_all[i, :Tt]
        diff = pred - tgt
        loss = float((diff ** 2 * mask).sum() / max(mask.sum(), 1))
        recs.append(dict(
            shot=int(shots[i]), T=Tt,
            time=time_all[i, :Tt].astype(np.float64),
            dt=dt_all[i, :Tt].astype(np.float32),
            pred_kA=pred, target_kA=tgt, action_mask=mask, loss=loss,
            Ip_A=ip_all[i, :Tt].astype(np.float32),
            phase_ids=pid_all[i, :Tt].astype(np.int8),
        ))
    return recs
