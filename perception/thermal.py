"""Thermal inspection: find hotspots in radiometric thermal images (degrees Celsius per pixel).

A hotspot is a connected region hotter than the scene's ambient (median) temperature by a margin,
the standard first check for overheating motors, electrical joints, bearings and pipes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Hotspot:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    max_temp_c: float
    mean_temp_c: float
    delta_over_ambient_c: float
    area_px: int
    severity: str  # "warning" | "critical"


def find_hotspots(temps_c: np.ndarray, delta_warning: float = 10.0, delta_critical: float = 25.0,
                  min_area_px: int = 4) -> tuple[float, list[Hotspot]]:
    """Returns (ambient temperature, hotspots sorted by peak temperature)."""
    t = np.asarray(temps_c, np.float64)
    ambient = float(np.median(t))
    labels, n = ndimage.label(t > ambient + delta_warning)
    spots = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        region = labels[sl] == i
        area = int(region.sum())
        if area < min_area_px:
            continue
        vals = t[sl][region]
        peak = float(vals.max())
        delta = peak - ambient
        spots.append(Hotspot((sl[1].start, sl[0].start, sl[1].stop - 1, sl[0].stop - 1), peak,
                             float(vals.mean()), delta, area, "critical" if delta >= delta_critical else "warning"))
    return ambient, sorted(spots, key=lambda s: -s.max_temp_c)


def raw_to_celsius(raw: np.ndarray, gain: float = 0.01, offset_k: float = 0.0) -> np.ndarray:
    """Convert radiometric raw counts (e.g. centi-Kelvin from FLIR Lepton/Boson TLinear) to Celsius."""
    return np.asarray(raw, np.float64) * gain + offset_k - 273.15
