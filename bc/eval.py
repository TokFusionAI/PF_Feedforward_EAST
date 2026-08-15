"""兼容入口：实现位于 ``bc.evaluation.eval``。"""
from __future__ import annotations

from bc.evaluation.eval import main

if __name__ == "__main__":
    raise SystemExit(main())
