"""Detector names, declared apart from the detectors themselves.

`detect.py` populates its registry as a side effect of defining the detectors,
and defining them imports mediapipe. Listing what is installed must not cost
that, so the names live here and `detect.py` is checked against this list rather
than the other way round.
"""

from __future__ import annotations

DETECTOR_NAMES: tuple[str, ...] = ("mediapipe",)

DESCRIPTIONS: dict[str, str] = {
    "mediapipe": "multi-person pose landmarker, IMAGE mode, appearance histogram",
}

__all__ = ["DESCRIPTIONS", "DETECTOR_NAMES"]
