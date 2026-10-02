"""The contract between the video half of the pipeline and the RL half.

Training never sees video, MediaPipe or SMPL. It sees a ReferenceMotion, and
nothing else. That is what lets the motion frontend be swapped without touching
the trainer.

PoseSequence is the narrower contract one stage earlier: what a frontend emits
before any robot-specific knowledge is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: MediaPipe pose landmark indices that carry skeletal meaning for retargeting
MP_POSE = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13,
    "right_elbow": 14, "left_wrist": 15, "right_wrist": 16, "left_hip": 23,
    "right_hip": 24, "left_knee": 25, "right_knee": 26, "left_ankle": 27,
    "right_ankle": 28, "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}
N_POSE_LANDMARKS = 33
N_HAND_LANDMARKS = 21

#: The SMPL-X parameter fields a PoseSequence may carry. All four or none.
_SMPLX_FIELDS = ("smplx_betas", "smplx_global_orient", "smplx_body_pose",
                 "smplx_transl")


def _segments_from_valid(valid: np.ndarray, min_len: int = 2) -> list[tuple[int, int]]:
    """Contiguous runs of True in `valid`, as half-open [start, end) ranges."""
    out, start = [], None
    for i, v in enumerate(np.asarray(valid, bool)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(valid) - start >= min_len:
        out.append((start, len(valid)))
    return out


@dataclass
class SubjectWindow:
    """Where the subject is, per frame: the estimate stage's other input.

    The track stage knows where the dancer is; an estimator may or may not care.
    Passing this as one object rather than three positionals is what lets every
    frontend share a signature, so a frontend that reads a parameter file instead
    of a video simply ignores it and no caller has to special-case it.
    """

    bbox: np.ndarray                # (T, 4) x1, y1, x2, y2, pixels
    valid: np.ndarray               # (T,) bool
    fps: float

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, np.float32)
        self.valid = np.asarray(self.valid, bool)
        if self.bbox.ndim != 2 or self.bbox.shape[1] != 4:
            raise ValueError(f"bbox: expected (T, 4), got {self.bbox.shape}")
        if self.valid.shape != (len(self.bbox),):
            raise ValueError(
                f"valid: expected {(len(self.bbox),)}, got {self.valid.shape}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    @property
    def n_frames(self) -> int:
        return len(self.bbox)


@dataclass
class PoseSequence:
    """Raw human motion from a frontend, before any robot knowledge is applied.

    Landmarks are in metres, hip centred, y up, in the subject's own frame.
    """

    fps: float
    landmarks: np.ndarray           # (T, 33, 3)
    visibility: np.ndarray          # (T, 33)
    hand_left: np.ndarray           # (T, 21, 3)
    hand_right: np.ndarray          # (T, 21, 3)
    hand_conf: np.ndarray           # (T, 2) left, right
    valid: np.ndarray               # (T,) bool
    meta: dict = field(default_factory=dict)
    #: (T, 33, 2) pixel positions of the body landmarks, or None.
    #:
    #: Optional because a frontend reading a parameter file never saw a frame and
    #: has no honest answer. It exists because the hand seam needs it: a hand
    #: estimator builds its region of interest from where the wrist and elbow are
    #: in the image, and the alternative, re-running pose inference inside the
    #: hand estimator, doubles the cost of the slowest stage in the pipeline.
    landmarks_px: np.ndarray | None = None
    #: SMPL-X parameters, when the frontend had them. All four or none.
    #:
    #: Joints are what most of this pipeline wants, and are always present. A
    #: retargeter built around the body model wants the parameters themselves:
    #: GMR reconstructs joint orientations from the pose, which joint positions
    #: cannot supply. Optional because MediaPipe has no such parameters and
    #: never will.
    smplx_betas: np.ndarray | None = None          # (T, B) or (B,)
    smplx_global_orient: np.ndarray | None = None  # (T, 3) axis-angle
    smplx_body_pose: np.ndarray | None = None      # (T, 63) axis-angle
    smplx_transl: np.ndarray | None = None         # (T, 3) metres

    def __post_init__(self) -> None:
        self.landmarks = np.asarray(self.landmarks, np.float32)
        self.visibility = np.asarray(self.visibility, np.float32)
        self.hand_left = np.asarray(self.hand_left, np.float32)
        self.hand_right = np.asarray(self.hand_right, np.float32)
        self.hand_conf = np.asarray(self.hand_conf, np.float32)
        self.valid = np.asarray(self.valid, bool)
        if self.landmarks_px is not None:
            self.landmarks_px = np.asarray(self.landmarks_px, np.float32)
        for name in _SMPLX_FIELDS:
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, np.asarray(v, np.float32))
        self.validate()

    @property
    def n_frames(self) -> int:
        return len(self.landmarks)

    def validate(self) -> None:
        T = self.n_frames
        for name, arr, shape in (
            ("landmarks", self.landmarks, (T, N_POSE_LANDMARKS, 3)),
            ("visibility", self.visibility, (T, N_POSE_LANDMARKS)),
            ("hand_left", self.hand_left, (T, N_HAND_LANDMARKS, 3)),
            ("hand_right", self.hand_right, (T, N_HAND_LANDMARKS, 3)),
            ("hand_conf", self.hand_conf, (T, 2)),
            ("valid", self.valid, (T,)),
        ):
            if arr.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {arr.shape}")
        if self.landmarks_px is not None:
            want = (T, N_POSE_LANDMARKS, 2)
            if self.landmarks_px.shape != want:
                raise ValueError(
                    f"landmarks_px: expected {want}, got {self.landmarks_px.shape}")
        # All four or none: a partial set is a frontend that half filled them,
        # and a consumer cannot tell that from a frontend that meant it.
        present = [n for n in _SMPLX_FIELDS if getattr(self, n) is not None]
        if present and len(present) != len(_SMPLX_FIELDS):
            missing = [n for n in _SMPLX_FIELDS if n not in present]
            raise ValueError(
                f"SMPL-X parameters are partial: missing {missing}. Provide all "
                f"of {list(_SMPLX_FIELDS)} or none.")
        if present:
            for name, width in (("smplx_global_orient", 3),
                                ("smplx_body_pose", 63), ("smplx_transl", 3)):
                got = getattr(self, name)
                if got.shape != (T, width):
                    raise ValueError(
                        f"{name}: expected {(T, width)}, got {got.shape}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    @property
    def segments(self) -> list[tuple[int, int]]:
        return _segments_from_valid(self.valid)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = ({} if self.landmarks_px is None
                 else {"landmarks_px": self.landmarks_px})
        extra.update({n: getattr(self, n) for n in _SMPLX_FIELDS
                      if getattr(self, n) is not None})
        np.savez_compressed(
            path, fps=self.fps, landmarks=self.landmarks,
            visibility=self.visibility, hand_left=self.hand_left,
            hand_right=self.hand_right, hand_conf=self.hand_conf,
            valid=self.valid, meta=np.array(repr(self.meta)), **extra)
        return path

    @classmethod
    def load(cls, path: str | Path) -> PoseSequence:
        z = np.load(path, allow_pickle=False)
        import ast
        # `landmarks_px` is absent from every file written before the hand seam
        # existed, so its absence is normal rather than an error.
        return cls(fps=float(z["fps"]), landmarks=z["landmarks"],
                   visibility=z["visibility"], hand_left=z["hand_left"],
                   hand_right=z["hand_right"], hand_conf=z["hand_conf"],
                   valid=z["valid"], meta=ast.literal_eval(str(z["meta"])),
                   landmarks_px=(z["landmarks_px"] if "landmarks_px" in z.files
                                 else None),
                   **{n: (z[n] if n in z.files else None)
                      for n in _SMPLX_FIELDS})


@dataclass
class ReferenceMotion:
    """Retargeted motion, expressed entirely in the robot's own terms.

    `qpos_ref` is ordered by `sim.model.BODY_JOINTS` and `finger_ref` by
    `sim.model.HAND_JOINTS`. `link_pos` is in the pelvis frame, which makes the
    link tracking reward invariant to root drift; monocular root estimates are
    the least reliable quantity in this pipeline, so no reward should depend on
    them more than it must.
    """

    fps: float
    dof_names: list[str]
    qpos_ref: np.ndarray            # (T, 29)
    qvel_ref: np.ndarray            # (T, 29)
    finger_names: list[str]
    finger_ref: np.ndarray          # (T, 12)
    root_pos: np.ndarray            # (T, 3)
    root_quat: np.ndarray           # (T, 4) wxyz
    link_names: list[str]
    link_pos: np.ndarray            # (T, K, 3) pelvis frame
    valid: np.ndarray               # (T,) bool
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("qpos_ref", "qvel_ref", "finger_ref", "root_pos",
                     "root_quat", "link_pos"):
            setattr(self, name, np.asarray(getattr(self, name), np.float32))
        self.valid = np.asarray(self.valid, bool)
        self.validate()

    @property
    def n_frames(self) -> int:
        return len(self.qpos_ref)

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    @property
    def segments(self) -> list[tuple[int, int]]:
        return _segments_from_valid(self.valid)

    def validate(self) -> None:
        T, D, F, K = (self.n_frames, len(self.dof_names),
                      len(self.finger_names), len(self.link_names))
        for name, arr, shape in (
            ("qpos_ref", self.qpos_ref, (T, D)),
            ("qvel_ref", self.qvel_ref, (T, D)),
            ("finger_ref", self.finger_ref, (T, F)),
            ("root_pos", self.root_pos, (T, 3)),
            ("root_quat", self.root_quat, (T, 4)),
            ("link_pos", self.link_pos, (T, K, 3)),
            ("valid", self.valid, (T,)),
        ):
            if arr.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {arr.shape}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if not np.isfinite(self.qpos_ref[self.valid]).all():
            raise ValueError("qpos_ref has non-finite values on valid frames")
        if not self.valid.any():
            raise ValueError("motion has no valid frames")

    def check_model(self, model_hash: str, body_joints: list[str],
                    hand_joints: list[str]) -> None:
        """Guard against a motion built for a different model. See spec 11."""
        if self.dof_names != list(body_joints):
            raise ValueError("motion DoF names do not match the model's BODY_JOINTS")
        if self.finger_names != list(hand_joints):
            raise ValueError("motion finger names do not match the model's HAND_JOINTS")
        stamped = self.meta.get("model_hash")
        if stamped and stamped != model_hash:
            raise ValueError(
                f"motion was built for model {stamped}, this model is {model_hash}. "
                "Re-run retargeting.")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, fps=self.fps, dof_names=np.array(self.dof_names),
            qpos_ref=self.qpos_ref, qvel_ref=self.qvel_ref,
            finger_names=np.array(self.finger_names), finger_ref=self.finger_ref,
            root_pos=self.root_pos, root_quat=self.root_quat,
            link_names=np.array(self.link_names), link_pos=self.link_pos,
            valid=self.valid, meta=np.array(repr(self.meta)))
        return path

    @classmethod
    def load(cls, path: str | Path) -> ReferenceMotion:
        import ast
        z = np.load(path, allow_pickle=False)
        return cls(
            fps=float(z["fps"]), dof_names=[str(s) for s in z["dof_names"]],
            qpos_ref=z["qpos_ref"], qvel_ref=z["qvel_ref"],
            finger_names=[str(s) for s in z["finger_names"]],
            finger_ref=z["finger_ref"], root_pos=z["root_pos"],
            root_quat=z["root_quat"], link_names=[str(s) for s in z["link_names"]],
            link_pos=z["link_pos"], valid=z["valid"],
            meta=ast.literal_eval(str(z["meta"])))
