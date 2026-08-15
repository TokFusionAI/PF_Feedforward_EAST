"""兼容入口：实现位于 ``bc.training.train``。"""
from __future__ import annotations

from bc.training.train import main

if __name__ == "__main__":
    raise SystemExit(main())
