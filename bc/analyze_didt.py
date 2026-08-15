"""兼容入口：实现位于 ``bc.analysis.analyze_didt``。"""
from __future__ import annotations

from bc.analysis.analyze_didt import main

if __name__ == "__main__":
    raise SystemExit(main())
