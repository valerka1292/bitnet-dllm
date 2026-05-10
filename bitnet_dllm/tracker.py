from __future__ import annotations
from abc import ABC, abstractmethod
import json


class Tracker(ABC):
    @abstractmethod
    def log(self, data: dict, step: int | None = None) -> None: ...

    @abstractmethod
    def log_line(self, message: str) -> None: ...


class ConsoleTracker(Tracker):
    def log(self, data: dict, step: int | None = None) -> None:
        parts = []
        for k, v in data.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        prefix = f"[s{step}] " if step is not None else ""
        print(f"{prefix}{'  '.join(parts)}")

    def log_line(self, message: str) -> None:
        print(message)
