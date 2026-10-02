"""SMPLest-X as an estimator: video plus a subject box in, SMPL-X parameters out.

SMPLest-X (TPAMI 2025) regresses expressive whole-body SMPL-X from a single
crop. It matters here because it is the only estimator in this project that
emits parameters rather than landmarks, which is what the `gmr` retargeter
consumes, so it is what makes a second full path from video to robot possible.

It is a per-frame model, so it has no world-grounded root. The root it predicts
is camera relative and will jitter; that is a property of the method, not of
this adapter, and it is why a world-grounded estimator remains the thing worth
having.

Four accommodations are needed and each is contained here rather than by editing
their source, so their checkout stays a clean clone:

1. **The checkpoint keys are `module.` prefixed**, from DDP training, and their
   loader uses `strict=False`. Loaded as published, zero of 519 parameters match
   and the network stays randomly initialised, which still runs and still
   produces plausible looking output. The prefix is stripped and the match count
   asserted, because this failure is invisible otherwise.
2. **Nine `.cuda()` calls** sit on the inference path. The tensor and module
   methods are redirected rather than the source patched.
3. **`smplx.create` defaults have moved.** The build this project resolves gives
   the body model 26 shape directions where their code passes 20, so the widths
   are pinned explicitly.
4. **Their script needs Ultralytics for person boxes**, which is AGPL-3.0 and
   which `track/detect.py` declined for this project. It is not needed: the
   estimator contract already carries a box per frame, so their model is driven
   directly and their script is not used.

The checkout is not vendored. `G1DANCE_SMPLESTX_ROOT` says where it is, and the
weights are 8.25 GB and separately downloaded.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..ir import MP_POSE, N_HAND_LANDMARKS, N_POSE_LANDMARKS, PoseSequence

#: Where the SMPLest-X checkout and its weights live.
ROOT = Path(os.environ.get("G1DANCE_SMPLESTX_ROOT", "third_party/SMPLest-X"))
WEIGHTS = Path(os.environ.get(
    "G1DANCE_SMPLESTX_WEIGHTS", str(ROOT / "smplest_x_h.pth.tar")))
CONFIG = Path(os.environ.get(
    "G1DANCE_SMPLESTX_CONFIG", str(ROOT / "config_base.py")))

#: SMPL-X joint index -> MediaPipe landmark slot, for the landmarks half of the
#: contract. Shared with `smpl_fe`, which already owns the table.
_SMPLX_BODY = {
    "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18,
    "right_elbow": 19, "left_wrist": 20, "right_wrist": 21, "left_hip": 1,
    "right_hip": 2, "left_knee": 4, "right_knee": 5, "left_ankle": 7,
    "right_ankle": 8, "left_foot_index": 10, "right_foot_index": 11,
    "nose": 15,
}


class SmplestXUnavailable(RuntimeError):
    """SMPLest-X is not present, or cannot run here."""


def _how_to_get_it() -> str:
    return (
        f"Clone https://github.com/MotrixLab/SMPLest-X to {ROOT}, download "
        f"smplest_x_h.pth.tar and config_base.py from "
        f"https://huggingface.co/waanqii/SMPLest-X, and place the SMPL-X body "
        f"models under {ROOT}/human_models/human_model_files/smplx/. Set "
        f"G1DANCE_SMPLESTX_ROOT to point elsewhere.")


def _load(device: str = "cpu"):
    """Build the model with every accommodation the module docstring lists."""
    import sys

    import torch
    if not ROOT.exists():
        raise SmplestXUnavailable(f"no SMPLest-X checkout at {ROOT}. "
                                  + _how_to_get_it())
    for f in (WEIGHTS, CONFIG):
        if not f.exists():
            raise SmplestXUnavailable(f"missing {f}. " + _how_to_get_it())

    dev = torch.device(device)
    torch.Tensor.cuda = lambda self, *a, **k: self.to(dev)          # noqa: E731
    torch.nn.Module.cuda = lambda self, *a, **k: self.to(dev)       # noqa: E731

    import smplx as smplx_pkg
    if not getattr(smplx_pkg.create, "_g1dance_pinned", False):
        original = smplx_pkg.create

        def create(*a, **k):
            k.setdefault("num_betas", 10)
            k.setdefault("num_expression_coeffs", 10)
            return original(*a, **k)

        create._g1dance_pinned = True
        smplx_pkg.create = create

    root = str(ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from human_models.human_models import SMPLX
    from main.config import Config
    from models.SMPLest_X import get_model

    cfg = Config.load_config(str(CONFIG))
    cfg.update_config({"model": {**dict(cfg.model),
                                 "human_model_path":
                                     str(ROOT / "human_models"
                                         / "human_model_files")}})
    SMPLX(cfg.model.human_model_path)
    model = get_model(cfg, "test").eval().to(dev)

    blob = torch.load(str(WEIGHTS), map_location="cpu")["network"]
    state = {k.replace("module.", "", 1): v for k, v in blob.items()}
    own = set(model.state_dict())
    matched = sum(1 for k in state if k in own)
    if matched < 0.9 * len(state):
        raise SmplestXUnavailable(
            f"only {matched} of {len(state)} checkpoint parameters match the "
            "model. Loading it would leave the network randomly initialised, "
            "which still runs and still produces plausible looking output.")
    model.load_state_dict(state, strict=False)
    return model, cfg, dev


def estimate(source, subject=None, *, device: str = "cpu",
             bbox_ratio: float = 1.25, **_) -> PoseSequence:
    """Run SMPLest-X over the tracked subject's crop, frame by frame."""
    import cv2
    import torch
    from torchvision import transforms

    if subject is None:
        raise ValueError(
            "the smplest_x estimator needs a subject window; run `track` first")
    model, cfg, dev = _load(device)
    from utils.data_utils import generate_patch_image, process_bbox

    from ..bodymodel import smplx_joints

    n = subject.n_frames
    valid = np.zeros(n, bool)
    root_pose = np.zeros((n, 3), np.float32)
    body_pose = np.zeros((n, 63), np.float32)
    betas = np.zeros((n, 10), np.float32)
    transl = np.zeros((n, 3), np.float32)
    to_tensor = transforms.ToTensor()
    sub_valid = np.asarray(subject.valid, bool)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {source}")
    try:
        for t in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            if not sub_valid[t]:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            x1, y1, x2, y2 = np.asarray(subject.bbox[t], float)
            box = process_bbox(np.array([x1, y1, x2 - x1, y2 - y1], float),
                               w, h, cfg.model.input_img_shape,
                               ratio=bbox_ratio)
            if box is None:
                continue
            patch, _, _ = generate_patch_image(
                cvimg=rgb, bbox=box, scale=1.0, rot=0.0, do_flip=False,
                out_shape=cfg.model.input_img_shape)
            img = (to_tensor(patch.astype(np.float32)) / 255).to(dev)
            with torch.no_grad():
                out = model({"img": img[None]}, {}, {}, "test")
            root_pose[t] = out["smplx_root_pose"][0].cpu().numpy()
            body_pose[t] = out["smplx_body_pose"][0].cpu().numpy().reshape(-1)[:63]
            betas[t] = out["smplx_shape"][0].cpu().numpy().reshape(-1)[:10]
            if "cam_trans" in out:
                transl[t] = out["cam_trans"][0].cpu().numpy()
            valid[t] = True
    finally:
        cap.release()

    if not valid.any():
        raise SmplestXUnavailable("SMPLest-X produced no usable frame")

    # Camera frame to a world with a floor in it. SMPLest-X predicts in the
    # camera's convention, x right, y down, z forward, so a standing person
    # arrives head-down relative to a z-up world and everything downstream sees
    # someone lying over. Measured on a real frame: the raw head-to-foot vector
    # is (0.068, -1.353, 0.371), up axis y and pointing the wrong way.
    root_pose = _to_z_up(root_pose)
    joints = smplx_joints(betas=betas, global_orient=root_pose,
                          body_pose=body_pose, transl=transl)
    _assert_upright(joints, valid)
    landmarks, visibility = _to_landmarks(joints, valid)
    return PoseSequence(
        fps=float(subject.fps), landmarks=landmarks, visibility=visibility,
        hand_left=np.zeros((n, N_HAND_LANDMARKS, 3), np.float32),
        hand_right=np.zeros((n, N_HAND_LANDMARKS, 3), np.float32),
        hand_conf=np.zeros((n, 2), np.float32), valid=valid,
        smplx_betas=betas, smplx_global_orient=root_pose,
        smplx_body_pose=body_pose, smplx_transl=transl,
        meta={"frontend": "smplest_x", "device": str(dev),
              "axes": "smpl_x_right_y_up_z_forward",
              "units": "metres_hip_centred",
              "root": "camera_relative_per_frame",
              "hands": "not_populated; smplest_x hand poses are parameters",
              "source": str(source),
              "coverage": float(valid.mean())})


def _to_z_up(global_orient: np.ndarray) -> np.ndarray:
    """Rotate the root orientation from the camera frame into a z-up world."""
    from scipy.spatial.transform import Rotation as R
    pre = R.from_euler("x", -np.pi / 2)
    return (pre * R.from_rotvec(np.asarray(global_orient, np.float64))
            ).as_rotvec().astype(np.float32)


def _assert_upright(joints: np.ndarray, valid: np.ndarray) -> None:
    """The frame convention above is inferred, so it is checked rather than trusted.

    A silently upside-down subject retargets to a robot under the floor, which
    the GMR backend refuses and the builtin one does not, so this is the place
    to catch it.
    """
    if not valid.any():
        return
    head = joints[valid][:, 15]
    foot = joints[valid][:, 10]
    rise = float(np.median(head[:, 2] - foot[:, 2]))
    if rise < 0.5:
        raise SmplestXUnavailable(
            f"the estimated subject is not upright: the head sits {rise:.3f} m "
            "above the foot in z, where a standing person is above one metre. "
            "The camera-to-world convention in _to_z_up no longer holds.")


def _to_landmarks(joints: np.ndarray, valid: np.ndarray):
    """SMPL-X joints into the 33 MediaPipe slots, hip centred.

    Slots this model has no joint for stay at the origin with zero visibility,
    which is the convention the body contract already uses for a landmark a
    frontend could not see.
    """
    n = len(joints)
    landmarks = np.zeros((n, N_POSE_LANDMARKS, 3), np.float32)
    visibility = np.zeros((n, N_POSE_LANDMARKS), np.float32)
    hip = 0.5 * (joints[:, 1] + joints[:, 2])
    for name, src in _SMPLX_BODY.items():
        dst = MP_POSE[name]
        landmarks[:, dst] = joints[:, src] - hip
        visibility[:, dst] = valid.astype(np.float32)
    return landmarks, visibility


estimate.needs_subject = True

__all__ = ["CONFIG", "ROOT", "WEIGHTS", "SmplestXUnavailable", "estimate"]
