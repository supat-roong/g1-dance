"""Shot segmentation, because a hard cut breaks every tracker ever written.

A tracklet is only meaningful inside one continuous take. Across a cut the
pixels change completely, so any association made there is a guess. Splitting
first and matching shots by appearance afterwards (see propagate.py) keeps the
guessing in one place where it can be thresholded and, when it fails, reported
as absent instead of wrong.

A clip with no cuts is not a special case for callers: it is one shot covering
every frame.
"""

from __future__ import annotations

from pathlib import Path

from scenedetect import ContentDetector, detect

from ..ingest import probe_video

#: PySceneDetect's own default. Lower fires on fast camera motion, higher
#: misses cuts between visually similar takes.
DEFAULT_THRESHOLD = 27.0
#: Shots shorter than this are noise for our purpose: there is no dance to
#: retarget out of half a second, and short shots inflate the propagation work.
DEFAULT_MIN_SCENE_FRAMES = 15


def detect_shots(video: str | Path, threshold: float = DEFAULT_THRESHOLD,
                 min_scene_frames: int = DEFAULT_MIN_SCENE_FRAMES,
                 ) -> list[tuple[int, int]]:
    """Half-open frame ranges [start, end) covering the whole clip.

    Always returns at least one shot. An uncut clip yields exactly one.
    """
    video = Path(video)
    info = probe_video(video)
    n_frames = info.n_frames

    scenes = detect(str(video),
                    ContentDetector(threshold=threshold,
                                    min_scene_len=min_scene_frames))
    shots: list[tuple[int, int]] = []
    for start, end in scenes:
        s, e = int(start.frame_num), int(end.frame_num)
        if e > s:
            shots.append((s, e))
    return _normalise(shots, n_frames)


def _normalise(shots: list[tuple[int, int]], n_frames: int) -> list[tuple[int, int]]:
    """Force the shot list to tile [0, n_frames) with no gaps or overlaps."""
    if n_frames <= 0:
        raise ValueError("cannot segment a clip with no frames")
    if not shots:
        return [(0, n_frames)]
    shots = sorted(shots)
    out: list[tuple[int, int]] = []
    cursor = 0
    for start, end in shots:
        start = max(start, cursor)
        end = min(end, n_frames)
        if end <= start:
            continue
        if start > cursor:
            start = cursor          # close any gap onto the previous shot
        out.append((start, end))
        cursor = end
    if not out:
        return [(0, n_frames)]
    if cursor < n_frames:
        out[-1] = (out[-1][0], n_frames)
    return out


def shot_of(shots: list[tuple[int, int]], frame: int) -> int:
    """Index of the shot containing `frame`, or -1 if it is out of range."""
    for i, (start, end) in enumerate(shots):
        if start <= frame < end:
            return i
    return -1
