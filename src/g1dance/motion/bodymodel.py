"""Run a SMPL-X body model, which is what turns parameters into joints.

`smpl_fe.py` opened by refusing to do this: "What this module will not do is run
a body model", because the mesh is separately licensed and was not a dependency.
That refusal was right at the time and is what made the frontend useless for
every modern estimator, since GVHMR, SMPLest-X and DanceHMR all emit parameters
rather than joint positions, and GMR consumes parameters too.

The licence has not changed, only whether the file is present. It is
registration gated and non-commercial, so it is never committed and never
downloaded automatically; `.gitignore` excludes `assets/smplx/` and the loader
below raises with instructions rather than fetching anything. A result produced
with it is reproducible from the path and the parameters, the way the rest of
this pipeline works.

Joints come out in SMPL-X's own frame, which `smpl_fe.py` already documents as
x right, y up, z forward, in metres. That is the `PoseSequence` frame already,
so nothing is rotated here; the only adjustment downstream is hip centring.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

#: The directory `smplx.create` is pointed at. It must *contain* a `smplx/`
#: folder, which is why this is the assets root rather than the model folder.
ASSET_ROOT = Path(os.environ.get(
    "G1DANCE_BODY_MODELS",
    Path(__file__).resolve().parents[3] / "assets"))

#: The number of SMPL-X joints `smpl_fe.SMPLX_JOINT_NAMES` names. The model also
#: returns face and fingertip landmarks past this point, which nothing here uses.
N_SMPLX_JOINTS = 55

_HOW_TO_GET_IT = (
    "Register at https://smpl-x.is.tue.mpg.de, download models_smplx_v1_1.zip, "
    "and place SMPLX_{gender}.npz at {path}. It is non-commercial and "
    "registration gated, so it is never fetched automatically and never "
    "committed.")


class BodyModelError(RuntimeError):
    """The body model is missing, or cannot be run."""


def body_model_path(gender: str = "neutral",
                    asset_root: Path | str | None = None) -> Path:
    root = Path(asset_root) if asset_root else ASSET_ROOT
    return root / "smplx" / f"SMPLX_{gender.upper()}.npz"


def ensure_body_model(gender: str = "neutral",
                      asset_root: Path | str | None = None) -> Path:
    """Path to the model file, or an error saying exactly how to obtain it."""
    p = body_model_path(gender, asset_root)
    if p.exists() and p.stat().st_size > 0:
        return p
    raise BodyModelError(
        f"the SMPL-X body model is not at {p}. "
        + _HOW_TO_GET_IT.format(gender=gender.upper(), path=p))


def _require_smplx():
    try:
        import smplx
    except ImportError as exc:
        raise BodyModelError(
            "the `smplx` package is not installed; it is the loader for the "
            "body model. `uv pip install smplx`") from exc
    return smplx


def smplx_joints(betas: np.ndarray, global_orient: np.ndarray,
                 body_pose: np.ndarray, transl: np.ndarray | None = None,
                 left_hand_pose: np.ndarray | None = None,
                 right_hand_pose: np.ndarray | None = None,
                 gender: str = "neutral", chunk: int = 64,
                 asset_root: Path | str | None = None) -> np.ndarray:
    """SMPL-X parameters to `(T, 55, 3)` joint positions in metres.

    Axis-angle throughout, `body_pose` 63 wide for the 21 body joints and each
    hand pose 45 wide, which means the model is built with `use_pca=False`: PCA
    hand components are a different parameterisation and silently produce a
    different hand from the same numbers.

    Evaluated in chunks because the model allocates per batch and a long clip at
    10475 vertices a frame is a great deal of memory for something we then throw
    away, keeping only 55 points of it.
    """
    import torch

    smplx = _require_smplx()
    root = Path(asset_root) if asset_root else ASSET_ROOT
    ensure_body_model(gender, root)

    betas = np.asarray(betas, np.float32)
    global_orient = np.asarray(global_orient, np.float32).reshape(-1, 3)
    body_pose = np.asarray(body_pose, np.float32).reshape(len(global_orient), -1)
    if body_pose.shape[1] != 63:
        raise BodyModelError(
            f"body_pose must be 63 wide (21 joints, axis-angle), got "
            f"{body_pose.shape[1]}")
    T = len(global_orient)
    if betas.ndim == 1:
        betas = np.tile(betas[None], (T, 1))
    n_betas = betas.shape[1]

    def _opt(a, width):
        if a is None:
            return np.zeros((T, width), np.float32)
        return np.asarray(a, np.float32).reshape(T, width)

    transl = _opt(transl, 3)
    lh = _opt(left_hand_pose, 45)
    rh = _opt(right_hand_pose, 45)

    out = np.empty((T, N_SMPLX_JOINTS, 3), np.float32)
    model = smplx.create(str(root), model_type="smplx", gender=gender,
                         use_pca=False, num_betas=n_betas, ext="npz",
                         batch_size=min(chunk, T))
    with torch.no_grad():
        for i in range(0, T, chunk):
            j = slice(i, min(i + chunk, T))
            n = j.stop - j.start
            if n != model.batch_size:
                model = smplx.create(str(root), model_type="smplx",
                                     gender=gender, use_pca=False,
                                     num_betas=n_betas, ext="npz", batch_size=n)
            res = model(betas=torch.from_numpy(betas[j]),
                        global_orient=torch.from_numpy(global_orient[j]),
                        body_pose=torch.from_numpy(body_pose[j]),
                        transl=torch.from_numpy(transl[j]),
                        left_hand_pose=torch.from_numpy(lh[j]),
                        right_hand_pose=torch.from_numpy(rh[j]))
            out[j] = res.joints.numpy()[:, :N_SMPLX_JOINTS]
    return out


__all__ = ["ASSET_ROOT", "BodyModelError", "N_SMPLX_JOINTS",
           "body_model_path", "ensure_body_model", "smplx_joints"]
