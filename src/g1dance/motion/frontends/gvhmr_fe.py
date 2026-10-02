"""GVHMR as an estimator: the only one here that produces a world-grounded root.

Every other estimator in this project is per frame and camera relative, which is
why the reference root has been the binding constraint: the yaw rate cap in
`constraints.limit_root_yaw_rate` exists to remove an impossible signal rather
than to supply a real one. GVHMR (SIGGRAPH Asia 2024, TPAMI 2026) predicts
motion in gravity-view coordinates and emits `smpl_params_global`, which carries
translation and a world orientation.

This was written off earlier in the project as blocked four ways. Three of those
were wrong and are recorded so the reasoning is not repeated:

* **The weights are not only on Google Drive.** The full set is mirrored on
  Hugging Face, and the GVHMR checkpoint itself is 163 MB. ViTPose and HMR2 are
  2.5 GB each and are the expensive part.
* **SMPL is not needed.** It appears only in `render_incam` and `render_global`,
  which this adapter does not call. SMPL-X alone is enough to predict.
* **Ultralytics is not needed.** Their `run_preprocess` caches each stage to a
  file and skips the stage when the file exists, so writing the bounding boxes
  this pipeline already has means the YOLO tracker is never constructed. That
  matters because Ultralytics is AGPL-3.0, which `track/detect.py` declined.

What was true is the CUDA hardcoding, and it is handled the same way as for
SMPLest-X: the tensor and module methods are redirected rather than the source
patched.

Two further accommodations were needed, both from the checkout being ahead of
its own helpers. `hmr4d/build_gvhmr.py` and `hmr4d/configs/siga24_release.yaml`
both name things that no longer exist in the code they load, so the model is
built through the hydra store the demo actually uses, and
`hmr4d.model.gvhmr.gvhmr_pl_demo` is imported for its registration side effect,
which is what `tools/demo/demo.py` line 27 is doing.

`pytorch3d` does not build on macOS arm64. Its pure-Python `transforms` are
vendored into site-packages from upstream rather than reimplemented, and the
compiled parts are present as names that raise when touched.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np

from ..ir import MP_POSE, N_HAND_LANDMARKS, N_POSE_LANDMARKS, PoseSequence

ROOT = Path(os.environ.get("G1DANCE_GVHMR_ROOT", "third_party/GVHMR"))

#: SMPL-X joint index -> MediaPipe slot. Same table `smplestx_fe` uses.
_SMPLX_BODY = {
    "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18,
    "right_elbow": 19, "left_wrist": 20, "right_wrist": 21, "left_hip": 1,
    "right_hip": 2, "left_knee": 4, "right_knee": 5, "left_ankle": 7,
    "right_ankle": 8, "left_foot_index": 10, "right_foot_index": 11,
    "nose": 15,
}


#: COCO-17 index, which is what GVHMR's ViTPose emits -> MediaPipe slot.
_COCO_TO_MP = {
    0: "nose", 5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow",
    8: "right_elbow", 9: "left_wrist", 10: "right_wrist", 11: "left_hip",
    12: "right_hip", 13: "left_knee", 14: "right_knee", 15: "left_ankle",
    16: "right_ankle",
}


def _landmarks_px(kp2d: np.ndarray, n: int) -> np.ndarray:
    """ViTPose's 2D keypoints in the MediaPipe slots, full frame pixels.

    GVHMR computes these to condition its own network and discards them; kept,
    they are what the MediaPipe hand estimator needs to place its crop, which is
    what gives this path fingers. They are full resolution although ViTPose ran
    on a half size frame, because `get_batch` scales the boxes back by `img_ds`.
    Slots COCO has no point for stay at zero, and the hand estimator does not
    read them: its ROI is the wrist and elbow.
    """
    kp = np.asarray(kp2d, np.float32)
    px = np.zeros((n, N_POSE_LANDMARKS, 2), np.float32)
    for c, name in _COCO_TO_MP.items():
        px[:, MP_POSE[name]] = kp[:n, c, :2]
    return px


class GvhmrUnavailable(RuntimeError):
    """GVHMR is not present, or cannot run here."""


def _how_to_get_it() -> str:
    return (
        f"Clone https://github.com/zju3dv/GVHMR to {ROOT} and place the "
        f"checkpoints under {ROOT}/inputs/checkpoints: gvhmr, hmr2, vitpose. "
        "They are mirrored at https://huggingface.co/camenduru/GVHMR. Set "
        "G1DANCE_GVHMR_ROOT to point elsewhere.")


def _prepare(device: str = "cpu"):
    """Import GVHMR with the accommodations the docstring lists."""
    import sys

    import torch
    if not ROOT.exists():
        raise GvhmrUnavailable(f"no GVHMR checkout at {ROOT}. " + _how_to_get_it())
    ckpt = ROOT / "inputs" / "checkpoints" / "gvhmr" / "gvhmr_siga24_release.ckpt"
    if not ckpt.exists():
        raise GvhmrUnavailable(f"missing {ckpt}. " + _how_to_get_it())

    dev = torch.device(device)
    torch.Tensor.cuda = lambda self, *a, **k: self.to(dev)       # noqa: E731
    torch.nn.Module.cuda = lambda self, *a, **k: self.to(dev)    # noqa: E731
    # HMR2 is a Lightning module, and Lightning's mixin defines its own `cuda`
    # that wins in the method resolution order and calls
    # `torch.cuda.current_device()` directly, so patching `nn.Module` alone is
    # silently not enough.
    try:
        from lightning_fabric.utilities.device_dtype_mixin import (
            _DeviceDtypeModuleMixin)
        _DeviceDtypeModuleMixin.cuda = lambda self, *a, **k: self.to(dev)
    except ImportError:
        pass

    root = str(ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    _stub_unused_imports()
    return dev


#: Modules `hmr4d/utils/preproc/__init__.py` pulls in that this adapter never
#: uses. Python initialises a parent package before any submodule, so reaching
#: past the package does not avoid them and they have to be stood in for.
#:
#: `ultralytics` is the YOLO tracker, which supplies bounding boxes this
#: pipeline already has and which is AGPL-3.0, a licence `track/detect.py`
#: declined for this project. `pycolmap` is for visual odometry, which a static
#: camera does not need. Both stand-ins raise if anything actually reaches for
#: them, so an unused import cannot quietly become a used one.
_UNUSED = ("ultralytics", "pycolmap")


def _stub_unused_imports() -> None:
    import sys
    import types

    for name in _UNUSED:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)

        def _raise(*_a, _n=name, **_k):
            raise NotImplementedError(
                f"{_n} is deliberately unavailable here; see gvhmr_fe._UNUSED")

        def _getattr(attr, _r=_raise, _n=name):
            # Dunders must fail as absent, not as callables: the import
            # machinery asks for __path__ and __spec__ and cannot be handed a
            # function.
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            return _r

        module.__getattr__ = _getattr
        module.YOLO = _raise
        sys.modules[name] = module


def _build_model(dev):
    """Through the hydra store, because their own builder names a stale path."""
    import hydra
    from hydra import compose, initialize_config_module

    from hmr4d.configs import register_store_gvhmr
    import hmr4d.model.gvhmr.gvhmr_pl_demo  # noqa: F401  registers the store entry

    with initialize_config_module(version_base="1.3",
                                  config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo",
                      overrides=["video_name=g1dance", "static_cam=True",
                                 "verbose=False", "use_dpvo=False"])
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    # `cfg.ckpt_path` as it stands, joined to nothing: this runs inside
    # `_in_checkout`, so the working directory is already the checkout and their
    # config's relative path is correct. Joining ROOT applies it twice, and
    # resolving ROOT here resolves it against the directory it already names.
    model.load_pretrained_model(str(cfg.ckpt_path))
    return model.eval().to(dev), cfg


@contextlib.contextmanager
def _in_checkout():
    """Their code resolves checkpoints relative to the working directory.

    `inputs/checkpoints/vitpose/...` and friends are relative paths that assume
    the process was started from the checkout, so the run happens there and the
    caller's directory is restored afterwards. Every path this adapter passes in
    is made absolute first.
    """
    previous = Path.cwd()
    os.chdir(ROOT.resolve())
    try:
        yield
    finally:
        os.chdir(previous)


def estimate(source, subject=None, *, device: str = "cpu",
             static_cam: bool = True, **_) -> PoseSequence:
    """Run GVHMR over the clip, using this pipeline's boxes rather than YOLO."""
    import torch

    if subject is None:
        raise ValueError(
            "the gvhmr estimator needs a subject window; run `track` first")
    source = Path(source).resolve()
    dev = _prepare(device)
    from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
    # The submodules directly, not the package: `preproc/__init__.py` imports
    # their YOLO Tracker, which imports Ultralytics, which is AGPL-3.0. Reaching
    # past the package means it is never imported rather than merely unused.
    from hmr4d.utils.preproc.vitfeat_extractor import Extractor
    from hmr4d.utils.preproc.vitpose import VitPoseExtractor
    from hmr4d.utils.video_io_utils import get_video_lwh

    from ..bodymodel import smplx_joints

    src = str(source)
    length, width, height = get_video_lwh(src)
    n = min(int(length), subject.n_frames)

    # The boxes this pipeline already has, in their layout. Writing these is
    # what means the AGPL tracker is never constructed.
    xyxy = torch.from_numpy(np.asarray(subject.bbox[:n], np.float32))
    bbx_xys = get_bbx_xys_from_xyxy(xyxy, base_enlarge=1.2).float()

    with _in_checkout():
        vit = VitPoseExtractor()
        kp2d = vit.extract(src, bbx_xys)
        del vit
        extractor = Extractor()
        features = extractor.extract_video_features(src, bbx_xys)
        del extractor
        model, _ = _build_model(dev)
    data = {
        "length": torch.tensor(n),
        "bbx_xys": bbx_xys,
        "kp2d": kp2d,
        "K_fullimg": estimate_K(width, height).repeat(n, 1, 1),
        "cam_angvel": _static_cam_angvel(n),
        "f_imgseq": features,
    }
    with torch.no_grad():
        pred = model.predict(data, static_cam=static_cam)

    g = pred["smpl_params_global"]
    body_pose = _np(g["body_pose"])[:n].reshape(n, -1)[:, :63]
    global_orient = _np(g["global_orient"])[:n].reshape(n, 3)
    transl = _np(g["transl"])[:n].reshape(n, 3)
    betas = _np(g["betas"])[:n].reshape(n, -1)[:, :10]

    # Gravity-view coordinates are y-up: measured on a real clip, the head is
    # 1.261 m above the foot in y and `transl` carries the height there too.
    # Both the orientation and the translation are rotated, because rotating
    # only the orientation would leave the body displaced along the old axis.
    global_orient, transl = _to_z_up(global_orient, transl)

    valid = np.ones(n, bool) & np.asarray(subject.valid, bool)[:n]
    joints = smplx_joints(betas=betas, global_orient=global_orient,
                          body_pose=body_pose, transl=transl)
    _assert_upright(joints, valid)
    landmarks, visibility = _to_landmarks(joints, valid)
    return PoseSequence(
        fps=float(subject.fps), landmarks=landmarks, visibility=visibility,
        hand_left=np.zeros((n, N_HAND_LANDMARKS, 3), np.float32),
        hand_right=np.zeros((n, N_HAND_LANDMARKS, 3), np.float32),
        hand_conf=np.zeros((n, 2), np.float32), valid=valid,
        landmarks_px=_landmarks_px(_np(kp2d), n),
        smplx_betas=betas, smplx_global_orient=global_orient,
        smplx_body_pose=body_pose, smplx_transl=transl,
        meta={"frontend": "gvhmr", "device": str(dev),
              "axes": "smpl_x_right_y_up_z_forward",
              "units": "metres_hip_centred",
              "root": "world_grounded",
              "static_cam": bool(static_cam),
              "hands": "not_populated",
              "landmarks_px": "vitpose_coco17",
              "source": src, "coverage": float(valid.mean())})


def _to_z_up(global_orient: np.ndarray, transl: np.ndarray):
    """Gravity-view y-up into the z-up world the robot and GMR live in."""
    from scipy.spatial.transform import Rotation as R
    pre = R.from_euler("x", np.pi / 2)
    orient = (pre * R.from_rotvec(np.asarray(global_orient, np.float64))
              ).as_rotvec().astype(np.float32)
    return orient, pre.apply(np.asarray(transl, np.float64)).astype(np.float32)


def _assert_upright(joints: np.ndarray, valid: np.ndarray) -> None:
    """The frame convention above is inferred, so it is checked, not trusted."""
    if not valid.any():
        return
    rise = float(np.median(joints[valid][:, 15, 2] - joints[valid][:, 10, 2]))
    if rise < 0.5:
        raise GvhmrUnavailable(
            f"the estimated subject is not upright: the head sits {rise:.3f} m "
            "above the foot in z, where a standing person is above one metre. "
            "The gravity-view convention in _to_z_up no longer holds.")


def _static_cam_angvel(n: int):
    import torch

    from hmr4d.utils.geo_transform import compute_cam_angvel
    return compute_cam_angvel(torch.eye(3).repeat(n, 1, 1))


def _np(x) -> np.ndarray:
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, np.float32)


def _to_landmarks(joints: np.ndarray, valid: np.ndarray):
    """SMPL-X joints into the 33 MediaPipe slots, hip centred."""
    n = len(joints)
    landmarks = np.zeros((n, N_POSE_LANDMARKS, 3), np.float32)
    visibility = np.zeros((n, N_POSE_LANDMARKS), np.float32)
    hip = 0.5 * (joints[:, 1] + joints[:, 2])
    for name, s in _SMPLX_BODY.items():
        d = MP_POSE[name]
        landmarks[:, d] = joints[:, s] - hip
        visibility[:, d] = valid.astype(np.float32)
    return landmarks, visibility


__all__ = ["ROOT", "GvhmrUnavailable", "estimate"]
