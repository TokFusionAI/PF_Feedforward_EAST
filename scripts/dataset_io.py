"""Single-shot I/O + ATIME-aligned resampling for the PF_ATIME_dataset.

Pipeline per shot:
    1. ``load_shot_sources``: pull ATIME / R8 / Z8 from EFIT and PCPFk / lmsr /
       lmsz / PCRL01 (+ their ``_time`` leaves) from PCSEASTRaw.
    2. ``gate_check``: reject shots that cannot be resampled meaningfully.
    3. ``resample_shot``: clip to ``t >= 0`` on ATIME, ZOH PCPFk to ATIME,
       linearly interpolate lmsr/lmsz/PCRL01 to ATIME, slice R8/Z8.
    4. ``write_shot_h5``: atomic ``tmp + rename`` write with a flat layout.

The flat layout keeps each signal as its own top-level dataset so the file
mirrors the source library style; state/action are assembled at training time.
See ``plans/atime_aligned_dataset_build.md`` for the full spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PCPF_NAMES: list[str] = [f"PCPF{i}" for i in list(range(1, 9)) + list(range(11, 15))]
STATE_SCALAR_NAMES: list[str] = ["lmsr", "lmsz", "PCRL01"]

BUILD_VERSION = "v1"


@dataclass
class ShotSource:
    """Raw arrays pulled from EFIT + PCSEASTRaw for one shot."""

    shot: int
    atime: np.ndarray  # (T_raw,) float64
    R8: np.ndarray  # (T_raw, 8)
    Z8: np.ndarray  # (T_raw, 8)
    pcpf: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N_k,)
    pcpf_time: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N_s,)
    scalars_time: dict[str, np.ndarray] = field(default_factory=dict)
    src_efit: str = ""
    src_pcs: str = ""


@dataclass
class ShotArrays:
    """ATIME-aligned arrays ready to be written to the per-shot h5."""

    shot: int
    time: np.ndarray  # (T,) float64
    R8: np.ndarray  # (T, 8) float32
    Z8: np.ndarray  # (T, 8) float32
    scalars: dict[str, np.ndarray]  # name -> (T,) float32 (may contain NaN)
    scalars_mask: dict[str, np.ndarray]  # name -> (T,) bool
    pcpf: dict[str, np.ndarray]  # name -> (T,) float32
    pcpf_mask: dict[str, np.ndarray]  # name -> (T,) bool
    dt_median: float
    src_efit: str
    src_pcs: str

    @property
    def T(self) -> int:
        return int(self.time.size)

    @property
    def t_start(self) -> float:
        return float(self.time[0])

    @property
    def t_end(self) -> float:
        return float(self.time[-1])


def _safe_read(ds: h5py.Dataset) -> np.ndarray:
    """Read an h5 dataset, tolerating null dataspace / empty shapes.

    A small number of EAST shots have placeholder datasets with ``shape is None``
    (HDF5 null dataspace) or shape ``(0,)``; ``ds[:]`` would raise
    ``Empty datasets cannot be sliced``. We return an empty 1-D array in those
    cases so that ``gate_check`` can reject the shot cleanly.
    """

    if ds.shape is None:
        return np.zeros((0,), dtype=np.float64)
    if ds.size == 0:
        return np.zeros(ds.shape, dtype=ds.dtype)
    return np.asarray(ds[()])


def _r8z8_from_bdry(f) -> tuple[np.ndarray, np.ndarray]:
    """从 EFIT h5 的 BDRY + NBDRY 重算 R8/Z8 (Method A 方度插值, scan_data.compat.shape_params,
    与 EFIT 内部算法一致, 已在 2025 炮上验证 RMSE=0.000 cm)。

    用于缺 R8/Z8 字段的 EFIT h5 (如 2026 导出: 有 BDRY/PSIRZ 但漏了 R8/Z8 预计算)。
    """
    if "BDRY" not in f:
        raise KeyError("EFIT h5 缺 R8/Z8 且无 BDRY, 无法重算")
    import sys as _sys
    _repo = Path(__file__).resolve().parent.parent
    if str(_repo) not in _sys.path:
        _sys.path.insert(0, str(_repo))
    from scan_data.compat import shape_params
    bdry = f["BDRY"][()]
    nbdy = f["NBDRY"][()] if "NBDRY" in f else None
    T = bdry.shape[0]
    R8 = np.full((T, 8), np.nan, dtype=np.float64)
    Z8 = np.full((T, 8), np.nan, dtype=np.float64)
    for t in range(T):
        nb = int(nbdy[t]) if nbdy is not None else bdry.shape[1]
        nb = min(nb, bdry.shape[1])
        sp = shape_params(bdry[t, :nb, 0], bdry[t, :nb, 1])
        R8[t] = sp.R8
        Z8[t] = sp.Z8
    return R8, Z8


def load_shot_sources(shot: int, efit_dir: Path, pcs_dir: Path) -> ShotSource:
    """Load the raw arrays needed for one shot. I/O only, no validation."""

    efit_path = Path(efit_dir) / f"{shot}.h5"
    pcs_path = Path(pcs_dir) / f"{shot}.h5"

    with h5py.File(efit_path, "r") as f:
        atime = np.asarray(_safe_read(f["ATIME"]), dtype=np.float64)
        if "R8" in f and "Z8" in f:
            R8 = _safe_read(f["R8"])
            Z8 = _safe_read(f["Z8"])
        else:
            # 回退: 缺 R8/Z8 时从 BDRY+NBDRY 重算 (Method A, 与 EFIT 一致, 已验证 RMSE=0)
            R8, Z8 = _r8z8_from_bdry(f)

    pcpf: dict[str, np.ndarray] = {}
    pcpf_time: dict[str, np.ndarray] = {}
    scalars: dict[str, np.ndarray] = {}
    scalars_time: dict[str, np.ndarray] = {}
    with h5py.File(pcs_path, "r") as f:
        for name in PCPF_NAMES:
            if name in f and f"{name}_time" in f:
                pcpf[name] = _safe_read(f[name])
                pcpf_time[name] = _safe_read(f[f"{name}_time"])
        for name in STATE_SCALAR_NAMES:
            if name in f and f"{name}_time" in f:
                scalars[name] = _safe_read(f[name])
                scalars_time[name] = _safe_read(f[f"{name}_time"])

    return ShotSource(
        shot=shot,
        atime=atime,
        R8=R8,
        Z8=Z8,
        pcpf=pcpf,
        pcpf_time=pcpf_time,
        scalars=scalars,
        scalars_time=scalars_time,
        src_efit=str(efit_path),
        src_pcs=str(pcs_path),
    )


def gate_check(src: ShotSource) -> tuple[bool, str]:
    """Cheap sanity checks. Returns ``(ok, reason)``; reason is empty on pass."""

    if src.atime.ndim != 1 or src.atime.size < 2:
        return False, "atime_too_short"
    if src.R8.ndim != 2 or src.R8.shape[0] != src.atime.size or src.R8.shape[1] != 8:
        return False, "R8_shape_mismatch"
    if src.Z8.ndim != 2 or src.Z8.shape[0] != src.atime.size or src.Z8.shape[1] != 8:
        return False, "Z8_shape_mismatch"
    if not np.any(src.atime >= 0.0):
        return False, "no_atime_ge_zero"

    missing_pcpf = [n for n in PCPF_NAMES if n not in src.pcpf or n not in src.pcpf_time]
    if missing_pcpf:
        return False, f"missing_pcpf:{','.join(missing_pcpf)}"

    missing_scalars = [n for n in STATE_SCALAR_NAMES if n not in src.scalars or n not in src.scalars_time]
    if missing_scalars:
        return False, f"missing_scalars:{','.join(missing_scalars)}"

    for name in PCPF_NAMES:
        tf = src.pcpf_time[name]
        y = src.pcpf[name]
        if tf.ndim != 1 or y.ndim != 1 or tf.size < 2 or tf.size != y.size:
            return False, f"bad_pcpf_shape:{name}"

    for name in STATE_SCALAR_NAMES:
        tf = src.scalars_time[name]
        y = src.scalars[name]
        if tf.ndim != 1 or y.ndim != 1 or tf.size < 2 or tf.size != y.size:
            return False, f"bad_scalar_shape:{name}"

    return True, ""


def _linear_resample(
    t: np.ndarray, tf: np.ndarray, yf: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation; out-of-range samples become NaN."""

    tf64 = np.asarray(tf, dtype=np.float64)
    yf64 = np.asarray(yf, dtype=np.float64)
    order = np.argsort(tf64, kind="stable")
    tf64 = tf64[order]
    yf64 = yf64[order]

    mask = (t >= tf64[0]) & (t <= tf64[-1])
    out = np.full(t.shape, np.nan, dtype=np.float32)
    if np.any(mask):
        out[mask] = np.interp(t[mask], tf64, yf64).astype(np.float32)
    return out, mask


def _zoh_resample(
    t: np.ndarray, tf: np.ndarray, yf: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order hold: take the most recent sample with ``tf <= t[k]``.

    Samples before the first source sample become NaN with mask ``False``.
    """

    tf64 = np.asarray(tf, dtype=np.float64)
    yf64 = np.asarray(yf, dtype=np.float64)
    order = np.argsort(tf64, kind="stable")
    tf64 = tf64[order]
    yf64 = yf64[order]

    idx = np.searchsorted(tf64, t, side="right") - 1
    valid = idx >= 0
    out = np.full(t.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        out[valid] = yf64[idx[valid]].astype(np.float32)
    return out, valid


def resample_shot(src: ShotSource) -> ShotArrays:
    """Align everything to ATIME restricted to ``t >= 0``."""

    k0 = int(np.searchsorted(src.atime, 0.0, side="left"))
    if k0 >= src.atime.size:
        raise ValueError("resample_shot: no ATIME sample with t >= 0 (run gate_check first)")

    t = np.ascontiguousarray(src.atime[k0:], dtype=np.float64)
    R8 = np.ascontiguousarray(src.R8[k0:], dtype=np.float32)
    Z8 = np.ascontiguousarray(src.Z8[k0:], dtype=np.float32)

    scalars_out: dict[str, np.ndarray] = {}
    scalars_mask: dict[str, np.ndarray] = {}
    for name in STATE_SCALAR_NAMES:
        y, m = _linear_resample(t, src.scalars_time[name], src.scalars[name])
        scalars_out[name] = y
        scalars_mask[name] = m

    pcpf_out: dict[str, np.ndarray] = {}
    pcpf_mask: dict[str, np.ndarray] = {}
    for name in PCPF_NAMES:
        y, m = _zoh_resample(t, src.pcpf_time[name], src.pcpf[name])
        pcpf_out[name] = y
        pcpf_mask[name] = m

    if t.size >= 2:
        dt_median = float(np.median(np.diff(t)))
    else:
        dt_median = float("nan")

    return ShotArrays(
        shot=src.shot,
        time=t,
        R8=R8,
        Z8=Z8,
        scalars=scalars_out,
        scalars_mask=scalars_mask,
        pcpf=pcpf_out,
        pcpf_mask=pcpf_mask,
        dt_median=dt_median,
        src_efit=src.src_efit,
        src_pcs=src.src_pcs,
    )


def _chunk_1d(T: int) -> tuple[int]:
    return (min(max(T, 1), 128),)


def _chunk_2d(T: int, C: int) -> tuple[int, int]:
    return (min(max(T, 1), 128), C)


def write_shot_h5(
    path: Path,
    arr: ShotArrays,
    *,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
    build_timestamp: str | None = None,
) -> None:
    """Atomically write one shot's arrays to ``path`` (tmp + rename)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    T = arr.T
    ts = build_timestamp or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    kw1 = {"chunks": _chunk_1d(T)}
    kw2 = {"chunks": _chunk_2d(T, 8)}
    if compression:
        kw1["compression"] = compression
        kw2["compression"] = compression
        if compression_opts is not None and compression == "gzip":
            kw1["compression_opts"] = compression_opts
            kw2["compression_opts"] = compression_opts

    try:
        with h5py.File(tmp, "w") as f:
            f.create_dataset("time", data=arr.time.astype(np.float64), dtype="f8", **kw1)
            f.create_dataset("R8", data=arr.R8.astype(np.float32), dtype="f4", **kw2)
            f.create_dataset("Z8", data=arr.Z8.astype(np.float32), dtype="f4", **kw2)

            for name in STATE_SCALAR_NAMES:
                f.create_dataset(name, data=arr.scalars[name].astype(np.float32), dtype="f4", **kw1)
                f.create_dataset(
                    f"mask_{name}",
                    data=arr.scalars_mask[name].astype(np.bool_),
                    dtype="?",
                    **kw1,
                )

            for name in PCPF_NAMES:
                f.create_dataset(name, data=arr.pcpf[name].astype(np.float32), dtype="f4", **kw1)
                f.create_dataset(
                    f"mask_{name}",
                    data=arr.pcpf_mask[name].astype(np.bool_),
                    dtype="?",
                    **kw1,
                )

            attrs: dict[str, Any] = f.attrs
            attrs["shot"] = int(arr.shot)
            attrs["T"] = int(T)
            attrs["dt_median"] = float(arr.dt_median)
            attrs["t_start"] = float(arr.t_start)
            attrs["t_end"] = float(arr.t_end)
            attrs["PCPF_names"] = np.asarray(PCPF_NAMES, dtype=h5py.string_dtype("utf-8"))
            attrs["state_scalar_names"] = np.asarray(
                STATE_SCALAR_NAMES, dtype=h5py.string_dtype("utf-8")
            )
            attrs["resample_pcpf"] = "zoh"
            attrs["resample_state"] = "linear"
            attrs["src_efit"] = str(arr.src_efit)
            attrs["src_pcs"] = str(arr.src_pcs)
            attrs["build_version"] = BUILD_VERSION
            attrs["build_timestamp"] = ts
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    tmp.replace(path)


def coverage_dict(arr: ShotArrays) -> dict[str, float]:
    """Per-signal coverage (mean of its mask) for index aggregation."""

    out: dict[str, float] = {}
    for name in STATE_SCALAR_NAMES:
        out[f"coverage_{name}"] = float(arr.scalars_mask[name].mean())
    for name in PCPF_NAMES:
        out[f"coverage_{name}"] = float(arr.pcpf_mask[name].mean())
    return out


COVERAGE_COLUMNS: list[str] = (
    [f"coverage_{n}" for n in STATE_SCALAR_NAMES] + [f"coverage_{n}" for n in PCPF_NAMES]
)
