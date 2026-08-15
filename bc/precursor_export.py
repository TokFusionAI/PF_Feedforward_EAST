"""兼容入口：实现位于 ``bc.gs_forward.precursor_export``。

Notebook 等若 ``sys.path`` 含 ``bc/`` 且 ``import precursor_export``，仍解析到本文件。
"""
from __future__ import annotations

from bc.gs_forward import precursor_export as _prec

PrecursorUnavailableError = _prec.PrecursorUnavailableError
check_shot_available = _prec.check_shot_available
efit_snapshot_row = _prec.efit_snapshot_row
export_slice_npz = _prec.export_slice_npz
fetch_precursor_series = _prec.fetch_precursor_series
fetch_precursor_series_h5 = _prec.fetch_precursor_series_h5
fetch_precursor_series_mds = _prec.fetch_precursor_series_mds
load_precursor_npz = _prec.load_precursor_npz
load_single_slice_npz = _prec.load_single_slice_npz
main = _prec.main
nearest_row = _prec.nearest_row
save_precursor_npz = _prec.save_precursor_npz

__all__ = [
    "PrecursorUnavailableError",
    "check_shot_available",
    "efit_snapshot_row",
    "export_slice_npz",
    "fetch_precursor_series",
    "fetch_precursor_series_h5",
    "fetch_precursor_series_mds",
    "load_precursor_npz",
    "load_single_slice_npz",
    "main",
    "nearest_row",
    "save_precursor_npz",
]

if __name__ == "__main__":
    raise SystemExit(main())
