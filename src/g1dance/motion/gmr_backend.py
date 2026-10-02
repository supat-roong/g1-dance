"""GMR as a retargeting backend, beside the one this repo wrote.

GMR is General Motion Retargeting (ICRA 2026, MIT), which solves the same
problem `retarget.py` does and solves it differently: constrained optimisation
through `mink` against the robot's own model, with non-uniform local scaling,
tuned by its authors for downstream RL tracking. It runs on CPU at roughly 600
frames a second here, so it is not the expensive part of anything.

Two facts made this cheap to add and are worth recording because neither was
guaranteed. Its Unitree G1 model has the 29 hinge joints in exactly
`sim.model.BODY_JOINTS` order, verified rather than assumed, so nothing is
remapped. And its output is root position, a wxyz quaternion and the 29 joints,
which is the same 36 column layout our own exporter writes.

What it needs and `retarget.py` does not is SMPL-X parameters. It reconstructs
joint orientations by composing the pose down the kinematic tree, and a joint
position cannot supply an orientation, so a `PoseSequence` carrying only
landmarks is refused rather than approximated. That is why `PoseSequence` has
optional parameter fields and why the MediaPipe frontend cannot feed this
backend.
"""

from __future__ import annotations

import numpy as np

from ..sim.model import BODY_JOINTS, HAND_JOINTS, KEY_BODIES, G1Model
from . import constraints as C
from .ir import PoseSequence, ReferenceMotion

#: GMR's own name for the robot. Its XML is `g1_mocap_29dof.xml`.
GMR_ROBOT = "unitree_g1"


class GmrUnavailable(RuntimeError):
    """GMR is not installed, or cannot run here."""


def _require_gmr():
    try:
        import general_motion_retargeting as gmr
        from general_motion_retargeting.utils.smpl import (
            get_smplx_data_offline_fast)
    except ImportError as exc:
        raise GmrUnavailable(
            "the `gmr` retargeter needs General Motion Retargeting: "
            "`uv pip install -e git+https://github.com/YanjieZe/GMR`") from exc
    return gmr, get_smplx_data_offline_fast


def _check_joint_order(gmr) -> None:
    """Refuse rather than silently produce a plausible looking wrong dance."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(gmr.ROBOT_XML_DICT[GMR_ROBOT]))
    hinges = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(m.njnt)
              if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if hinges != list(BODY_JOINTS):
        raise GmrUnavailable(
            "GMR's G1 joint order is not this project's BODY_JOINTS order; "
            "retargeting through it would put joints in the wrong columns")


def retarget_with_gmr(pose: PoseSequence, model: G1Model,
                      cfg=None) -> ReferenceMotion:
    """Retarget a parameter-carrying `PoseSequence` onto the G1 through GMR."""
    import torch

    from .bodymodel import ASSET_ROOT, ensure_body_model
    from . import fingers as F
    from .retarget import _as_config

    gmr, offline = _require_gmr()
    _check_joint_order(gmr)
    conf = _as_config(cfg)

    if pose.smplx_body_pose is None:
        raise GmrUnavailable(
            "the `gmr` retargeter needs SMPL-X parameters, which this pose "
            "sequence does not carry. It reconstructs joint orientations from "
            "the pose, and joint positions cannot supply one. Use the `smpl` "
            "estimator on a parameter file, or the `builtin` retargeter.")
    ensure_body_model("neutral")

    import smplx as smplx_pkg
    T = pose.n_frames
    betas = np.asarray(pose.smplx_betas, np.float32)
    if betas.ndim == 1:
        betas = np.tile(betas[None], (T, 1))
    body_model = smplx_pkg.create(str(ASSET_ROOT), "smplx", gender="neutral",
                                  use_pca=False, num_betas=betas.shape[1],
                                  batch_size=T, ext="npz")
    with torch.no_grad():
        out = body_model(
            betas=torch.from_numpy(betas),
            global_orient=torch.from_numpy(
                np.asarray(pose.smplx_global_orient, np.float32)),
            body_pose=torch.from_numpy(
                np.asarray(pose.smplx_body_pose, np.float32)),
            transl=torch.from_numpy(np.asarray(pose.smplx_transl, np.float32)),
            left_hand_pose=torch.zeros(T, 45), right_hand_pose=torch.zeros(T, 45),
            jaw_pose=torch.zeros(T, 3), leye_pose=torch.zeros(T, 3),
            reye_pose=torch.zeros(T, 3), return_full_pose=True)

    # GMR's loader reads a dict of arrays with AMASS spellings; it is given one
    # directly rather than through a file, so nothing is written to disk.
    data = {"mocap_frame_rate": np.float32(pose.fps),
            "pose_body": np.asarray(pose.smplx_body_pose, np.float32),
            "betas": betas[0],
            "root_orient": np.asarray(pose.smplx_global_orient, np.float32),
            "trans": np.asarray(pose.smplx_transl, np.float32)}
    frames, _ = offline(data, body_model, out, tgt_fps=int(round(pose.fps)))
    height = 1.66 + 0.1 * float(betas[0, 0])

    engine = gmr.GeneralMotionRetargeting(
        src_human="smplx", tgt_robot=GMR_ROBOT,
        actual_human_height=height, verbose=False)
    # `offset_to_ground` is not optional for us. SMPL-X's rest pelvis sits near
    # the origin with the feet almost a metre below it, so a source whose
    # translation is not already ground referenced puts the whole robot under
    # the floor: measured, a pelvis at z = 0.010 instead of 0.877. Our
    # environment spawns a robot standing on a plane, so the reference has to be
    # on one too.
    qpos = np.asarray([engine.retarget(f, offset_to_ground=True)
                       for f in frames], np.float64)
    n = len(qpos)

    root_pos, root_quat, q = qpos[:, :3], qpos[:, 3:7], qpos[:, 7:]
    # A sunk or floating reference is the failure this backend is most likely to
    # produce, because it depends on a convention in somebody else's input, and
    # it does not raise anywhere: the robot simply falls. Checked rather than
    # hoped for.
    median_h = float(np.median(root_pos[:, 2]))
    if not 0.3 <= median_h <= 1.5:
        raise GmrUnavailable(
            f"the retargeted pelvis sits at a median height of {median_h:.3f} m, "
            "which is not a robot standing on a floor. The usual cause is a "
            "source whose root translation is in a different frame or unit.")
    q = np.clip(q, *model.joint_range(BODY_JOINTS).T)

    out_fps = float(conf.control_fps)
    q = C.resample(q, pose.fps, out_fps)
    root_pos = C.resample(root_pos, pose.fps, out_fps)
    root_quat = C.resample_quat(root_quat, pose.fps, out_fps)
    root_quat = C.limit_root_yaw_rate(root_quat, out_fps, conf.max_root_yaw_rate)
    valid = C.resample_bool(np.asarray(pose.valid, bool)[:n], pose.fps, out_fps)
    finger, backend = F.retarget_fingers(pose, model, conf)
    finger = C.resample(finger, pose.fps, out_fps)[:len(q)]
    qvel = C.finite_difference(q, out_fps)

    link_pos = _link_positions(model, q)
    return ReferenceMotion(
        fps=out_fps, dof_names=list(BODY_JOINTS), qpos_ref=q, qvel_ref=qvel,
        finger_names=list(HAND_JOINTS), finger_ref=finger[:len(q)],
        root_pos=root_pos, root_quat=root_quat, link_names=list(KEY_BODIES),
        link_pos=link_pos, valid=valid[:len(q)],
        meta={"frontend": (dict(pose.meta) or {}).get("frontend", "unknown"),
              "retargeter": "gmr",
              "model_hash": model.hash(),
              "source_fps": float(pose.fps),
              "config": conf.as_meta(),
              "diagnostics": {"human_height_assumption": height,
                              "finger_backend": backend,
                              "gmr_robot": GMR_ROBOT},
              "source": dict(pose.meta) if pose.meta else {}})


def _link_positions(model: G1Model, q: np.ndarray) -> np.ndarray:
    """Key body positions in the pelvis frame, which is what the reward reads."""
    import mujoco
    m, d = model.model, model.data()
    out = np.zeros((len(q), len(KEY_BODIES), 3))
    for i in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = [0.0, 0.0, 0.793]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[model.body_qpos] = q[i]
        mujoco.mj_kinematics(m, d)
        out[i] = d.xpos[model.key_body_ids] - d.xpos[model.pelvis_id]
    return out


#: Declared so a caller can tell what this backend eats without calling it, the
#: way a controller declares whether it returns residuals or absolute targets.
#: A landmark-only sequence cannot be approximated into parameters, so the
#: conformance suite reads this rather than discovering it through an exception.
retarget_with_gmr.needs_smplx_params = True

__all__ = ["GMR_ROBOT", "GmrUnavailable", "retarget_with_gmr"]
