"""兼容入口：实现位于 ``bc.training.benchmark_loader``。"""
from __future__ import annotations

from bc.training.benchmark_loader import main

if __name__ == "__main__":
    raise SystemExit(main())
