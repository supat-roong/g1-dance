"""Carry one chosen dancer across every cut, or admit they are not there.

This is the module the article's author replaced with scissors. He cropped the
intended dancer out of group shots by hand and spliced the takes together, and
called it the hardest part of the whole process. The automation is one idea:
the user identifies the subject once, in the shot where they are clearest, and
every other shot's tracklets are scored against that descriptor.

The important half of that is the refusal. A shot whose best match sits below
threshold is marked absent, not assigned to the runner up. A wrong assignment
puts another person's dance into the reference motion, where it survives
retargeting, survives training, and is only visible hours later as a robot
doing the wrong moves. An absent shot merely shortens the clip.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..ingest import probe_video
from ..motion.ir import _segments_from_valid
from .detect import detect_people
from .shots import detect_shots
from .tracklets import Tracklet, build_tracklets, similarity

#: Below this cosine similarity a shot is called absent rather than assigned.
#: Deliberately strict: missing a shot costs frames, a wrong shot costs a run.
SIMILARITY_THRESHOLD = 0.7

ASSIGNED = "assigned"
SUBJECT = "subject"
ABSENT = "absent"

#: Contact sheet geometry.
THUMB_SIZE = (128, 192)
SHEET_COLUMNS = 6


@dataclass(frozen=True)
class ShotAssignment:
    """Which tracklet, if any, is the subject in one shot."""

    shot: int
    start: int
    end: int
    tracklet: int | None        # index into that shot's tracklet list
    score: float
    status: str

    @property
    def present(self) -> bool:
        return self.status != ABSENT


@dataclass
class TrackResult:
    """The stage's artifact: where the subject is, and where they are not."""

    video: str
    fps: float
    width: int
    height: int
    bbox: np.ndarray                    # (T, 4) x1, y1, x2, y2
    valid: np.ndarray                   # (T,) bool
    shots: list[tuple[int, int]]
    assignments: list[ShotAssignment]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, np.float32)
        self.valid = np.asarray(self.valid, bool)
        self.validate()

    @property
    def n_frames(self) -> int:
        return len(self.bbox)

    @property
    def segments(self) -> list[tuple[int, int]]:
        """Contiguous runs the estimate stage can process without gaps."""
        return _segments_from_valid(self.valid)

    @property
    def coverage(self) -> float:
        return float(self.valid.mean()) if self.n_frames else 0.0

    def subject_window(self):
        """This result as the estimate stage's uniform second argument."""
        from ..motion.ir import SubjectWindow
        return SubjectWindow(bbox=self.bbox, valid=self.valid, fps=self.fps)

    @property
    def absent_shots(self) -> list[int]:
        return [a.shot for a in self.assignments if a.status == ABSENT]

    def validate(self) -> None:
        T = self.n_frames
        if self.bbox.shape != (T, 4):
            raise ValueError(f"bbox: expected {(T, 4)}, got {self.bbox.shape}")
        if self.valid.shape != (T,):
            raise ValueError(f"valid: expected {(T,)}, got {self.valid.shape}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, video=np.array(self.video), fps=self.fps,
            width=self.width, height=self.height, bbox=self.bbox,
            valid=self.valid, shots=np.array(self.shots, int).reshape(-1, 2),
            assign_tracklet=np.array(
                [-1 if a.tracklet is None else a.tracklet
                 for a in self.assignments], int),
            assign_score=np.array([a.score for a in self.assignments], np.float32),
            assign_status=np.array([a.status for a in self.assignments]),
            meta=np.array(repr(self.meta)))
        return path

    @classmethod
    def load(cls, path: str | Path) -> TrackResult:
        z = np.load(path, allow_pickle=False)
        shots = [(int(s), int(e)) for s, e in z["shots"]]
        assignments = [
            ShotAssignment(shot=i, start=shots[i][0], end=shots[i][1],
                           tracklet=None if int(t) < 0 else int(t),
                           score=float(s), status=str(st))
            for i, (t, s, st) in enumerate(
                zip(z["assign_tracklet"], z["assign_score"], z["assign_status"]))]
        return cls(video=str(z["video"]), fps=float(z["fps"]),
                   width=int(z["width"]), height=int(z["height"]),
                   bbox=z["bbox"], valid=z["valid"], shots=shots,
                   assignments=assignments,
                   meta=ast.literal_eval(str(z["meta"])))


def propagate(subject: Tracklet, tracklets_by_shot: dict[int, list[Tracklet]],
              shots: list[tuple[int, int]], video_meta: dict,
              threshold: float = SIMILARITY_THRESHOLD,
              overrides: dict[int, int | None] | None = None) -> TrackResult:
    """Assign the subject in every shot, or mark the shot absent.

    `overrides` is the review step's escape hatch: shot index to tracklet
    index, or None to force absent. A human correction always wins.
    """
    overrides = dict(overrides or {})
    n_frames = int(video_meta["n_frames"])
    height = int(video_meta["height"])
    bbox = np.zeros((n_frames, 4), np.float32)
    valid = np.zeros(n_frames, bool)
    assignments: list[ShotAssignment] = []

    for index, (start, end) in enumerate(shots):
        candidates = tracklets_by_shot.get(index, [])
        chosen, score, status = _choose(index, candidates, subject, threshold,
                                        overrides)
        if chosen is not None:
            tracklet = candidates[chosen]
            lo = max(start, tracklet.start)
            hi = min(end, tracklet.end)
            if hi > lo:
                frames = np.arange(lo, hi)
                for axis in range(4):
                    bbox[lo:hi, axis] = np.interp(
                        frames, tracklet.frames, tracklet.bboxes[:, axis])
                valid[lo:hi] = True
            else:
                chosen, status = None, ABSENT
        assignments.append(ShotAssignment(shot=index, start=start, end=end,
                                          tracklet=chosen, score=float(score),
                                          status=status))

    meta = dict(video_meta.get("meta", {}))
    meta.update({"threshold": float(threshold),
                 "subject_shot": int(subject.shot),
                 "subject_full_body_score": float(subject.full_body_score)})
    return TrackResult(video=str(video_meta["path"]), fps=float(video_meta["fps"]),
                       width=int(video_meta["width"]), height=height,
                       bbox=bbox, valid=valid, shots=list(shots),
                       assignments=assignments, meta=meta)


def _choose(index: int, candidates: list[Tracklet], subject: Tracklet,
            threshold: float, overrides: dict[int, int | None],
            ) -> tuple[int | None, float, str]:
    """Pick a tracklet for one shot: override, then subject, then best match."""
    if index in overrides:
        forced = overrides[index]
        if forced is None:
            return None, 0.0, ABSENT
        if not 0 <= forced < len(candidates):
            raise ValueError(
                f"shot {index} has {len(candidates)} tracklets, "
                f"override asked for {forced}")
        return forced, 1.0, ASSIGNED
    if not candidates:
        return None, 0.0, ABSENT
    if index == subject.shot:
        for i, t in enumerate(candidates):
            if t.id == subject.id:
                return i, 1.0, SUBJECT
    scores = [similarity(subject, t) for t in candidates]
    best = int(np.argmax(scores))
    if scores[best] < threshold:
        return None, float(scores[best]), ABSENT
    return best, float(scores[best]), ASSIGNED


# ------------------------------------------------------------------ driving
@dataclass(frozen=True)
class TrackAnalysis:
    """Everything the stage learned before a subject was chosen.

    Split out so `scout` can ask the same questions without committing to a
    subject or writing an artifact.
    """

    info: object                        # ingest.VideoInfo
    shots: list[tuple[int, int]]
    tracklets_by_shot: dict[int, list[Tracklet]]
    sample: list[int]

    @property
    def tracklets(self) -> list[Tracklet]:
        return [t for shot in sorted(self.tracklets_by_shot)
                for t in self.tracklets_by_shot[shot]]

    @property
    def video_meta(self) -> dict:
        i = self.info
        return {"path": str(i.path), "fps": i.fps, "width": i.width,
                "height": i.height, "n_frames": i.n_frames}


def analyse(video: str | Path, *, detector: str = "mediapipe", stride: int = 1,
            **detector_kwargs) -> TrackAnalysis:
    """Shots, detections and within-shot tracklets, in one decode pass."""
    video = Path(video)
    info = probe_video(video)
    shots = detect_shots(video)
    sample = _sample_frames(shots, max(1, int(stride)))
    detections = detect_people(video, sample, detector=detector, **detector_kwargs)

    by_frame = dict(zip(sample, detections))
    tracklets_by_shot: dict[int, list[Tracklet]] = {}
    next_id = 0
    for index, (start, end) in enumerate(shots):
        frames = [f for f in sample if start <= f < end]
        found = build_tracklets(frames, [by_frame[f] for f in frames],
                                shot=index, frame_height=info.height,
                                first_id=next_id)
        next_id = max([t.id for t in found], default=next_id - 1) + 1
        tracklets_by_shot[index] = found
    return TrackAnalysis(info=info, shots=shots,
                         tracklets_by_shot=tracklets_by_shot, sample=sample)

def track(video: str | Path, subject: tuple[int, int] | None = None,
          shot_overrides: dict[int, int | None] | None = None,
          ui: bool = False, *, out_dir: str | Path | None = None,
          detector: str = "mediapipe", stride: int = 1,
          threshold: float = SIMILARITY_THRESHOLD,
          contact_sheet: bool = True, refine_subject: bool = True,
          **detector_kwargs) -> TrackResult:
    """Full track stage, headless.

    `subject` is (shot index, tracklet index within that shot). Left as None
    the stage picks for itself and records that it did so in `meta`, so a
    caller never has to wonder whether a human was involved.
    """
    if ui:
        raise NotImplementedError(
            "the review UI lives in track/ui.py; use track(..., ui=False) with "
            "--subject and --shot-overrides for the headless path")

    video = Path(video)
    analysis = analyse(video, detector=detector, stride=stride, **detector_kwargs)
    shots = analysis.shots
    tracklets_by_shot = analysis.tracklets_by_shot

    chosen, auto, reason = _pick_subject(subject, tracklets_by_shot)
    result = propagate(chosen, tracklets_by_shot, shots, analysis.video_meta,
                       threshold=threshold, overrides=shot_overrides)
    result.meta.update({
        "detector": detector, "stride": int(stride),
        "subject": [int(chosen.shot), int(_index_of(tracklets_by_shot, chosen))],
        "subject_auto": bool(auto), "subject_reason": reason,
        "n_tracklets": int(sum(len(v) for v in tracklets_by_shot.values())),
    })

    if refine_subject:
        # Second pass over the frames the global detection missed. The subject
        # is not hard to see there, only small; see track/refine.py.
        from .refine import refine as _refine
        result, rstats = _refine(video, result, detector=detector)
        result.meta["refine"] = rstats.as_dict()

    if contact_sheet:
        directory = Path(out_dir) if out_dir else video.parent
        sheet = write_contact_sheet(video, tracklets_by_shot,
                                    directory / f"{video.stem}.contact.png")
        result.meta["contact_sheet"] = str(sheet)
    if out_dir:
        result.meta["tracks_path"] = str(
            result.save(Path(out_dir) / f"{video.stem}.tracks.npz"))
    return result


def _sample_frames(shots: list[tuple[int, int]], stride: int) -> list[int]:
    """Frames to detect on. Every shot keeps its first frame whatever the
    stride, so a short shot never goes unrepresented."""
    frames: set[int] = set()
    for start, end in shots:
        frames.update(range(start, end, stride))
        frames.add(start)
        if end - 1 >= start:
            frames.add(end - 1)
    return sorted(frames)


def _index_of(tracklets_by_shot: dict[int, list[Tracklet]], tracklet: Tracklet) -> int:
    for i, t in enumerate(tracklets_by_shot.get(tracklet.shot, [])):
        if t.id == tracklet.id:
            return i
    return -1


def _pick_subject(subject: tuple[int, int] | None,
                  tracklets_by_shot: dict[int, list[Tracklet]],
                  ) -> tuple[Tracklet, bool, str]:
    """Resolve the subject, auto picking when the caller gave none."""
    everything = [t for shot in sorted(tracklets_by_shot)
                  for t in tracklets_by_shot[shot]]
    if not everything:
        raise ValueError("no person was detected anywhere in this clip")
    if subject is not None:
        shot, index = int(subject[0]), int(subject[1])
        candidates = tracklets_by_shot.get(shot, [])
        if not 0 <= index < len(candidates):
            raise ValueError(
                f"shot {shot} has {len(candidates)} tracklets, "
                f"subject asked for {index}")
        return candidates[index], False, f"subject given as shot {shot}, tracklet {index}"

    scored = [(t.full_body_score * t.duration, t) for t in everything]
    best_value, best = max(scored, key=lambda pair: (pair[0], -pair[1].start))
    reason = (f"auto picked shot {best.shot} tracklet "
              f"{_index_of(tracklets_by_shot, best)}: full_body_score "
              f"{best.full_body_score:.2f} over {best.duration} frames "
              f"(rank score {best_value:.1f} of {len(everything)} tracklets)")
    return best, True, reason


# ------------------------------------------------------------- contact sheet
def write_contact_sheet(video: str | Path,
                        tracklets_by_shot: dict[int, list[Tracklet]],
                        path: str | Path) -> Path:
    """Numbered thumbnails, one per tracklet, for picking without a browser.

    The headless path is what CI uses, so it gets a real artifact rather than
    a log line: the numbers printed here are exactly the `--subject` and
    `--shot-overrides` arguments.
    """
    entries: list[tuple[int, int, Tracklet]] = [
        (shot, i, t) for shot in sorted(tracklets_by_shot)
        for i, t in enumerate(tracklets_by_shot[shot])]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not entries:
        cv2.imwrite(str(path), np.zeros((THUMB_SIZE[1], THUMB_SIZE[0], 3), np.uint8))
        return path

    wanted = {int(np.median(t.frames)): None for _, _, t in entries}
    frames = _read_frames(video, sorted(wanted))
    thumbs = [_thumbnail(frames.get(int(np.median(t.frames))), t, shot, i)
              for shot, i, t in entries]

    columns = min(SHEET_COLUMNS, len(thumbs))
    rows = (len(thumbs) + columns - 1) // columns
    tw, th = THUMB_SIZE
    sheet = np.zeros((rows * th, columns * tw, 3), np.uint8)
    for n, thumb in enumerate(thumbs):
        r, c = divmod(n, columns)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = thumb
    cv2.imwrite(str(path), sheet)
    return path


def _read_frames(video: str | Path, indices: list[int]) -> dict[int, np.ndarray]:
    """One forward decode pass for a handful of scattered frames."""
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return out
    try:
        want = set(indices)
        i = 0
        limit = max(indices) if indices else -1
        while want and i <= limit and cap.grab():
            if i in want:
                ok, image = cap.retrieve()
                if ok:
                    out[i] = image
                want.discard(i)
            i += 1
    finally:
        cap.release()
    return out


def _thumbnail(image: np.ndarray | None, tracklet: Tracklet, shot: int,
               index: int) -> np.ndarray:
    tw, th = THUMB_SIZE
    tile = np.zeros((th, tw, 3), np.uint8)
    if image is not None:
        h, w = image.shape[:2]
        box = tracklet.bbox_at(int(np.median(tracklet.frames)))
        x1 = int(max(0, min(w - 1, box[0])))
        y1 = int(max(0, min(h - 1, box[1])))
        x2 = int(max(x1 + 1, min(w, box[2])))
        y2 = int(max(y1 + 1, min(h, box[3])))
        crop = image[y1:y2, x1:x2]
        if crop.size:
            tile = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
    label = f"s{shot}t{index}"
    cv2.rectangle(tile, (0, th - 22), (tw, th), (0, 0, 0), -1)
    cv2.putText(tile, f"{label} {tracklet.full_body_score:.2f}", (4, th - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return tile
