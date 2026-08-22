"""Central logging configuration.

Every module gets its own logger via logging.getLogger(__name__); this
module only wires up handlers/formatting once at process start (call from
scripts/ entry points, not from library code).
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "swing_scanner.log", encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
