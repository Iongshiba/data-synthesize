from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ColorPreset:
    name: str
    rgb: Tuple[float | None, float | None, float | None]


COLOR_PRESETS: tuple[ColorPreset, ...] = (
    ColorPreset("Default", (None, None, None)),
    ColorPreset("Warm Sunset", (0.94, 0.45, 0.20)),
    ColorPreset("Ocean Blue", (0.24, 0.52, 0.84)),
    ColorPreset("Forest", (0.20, 0.55, 0.32)),
    ColorPreset("Pastel Violet", (0.68, 0.56, 0.80)),
    ColorPreset("Slate", (0.42, 0.48, 0.54)),
)
