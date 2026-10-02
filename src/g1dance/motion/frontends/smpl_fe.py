"""SMPL and SMPL-X frontend: parametric body joints in, PoseSequence out.

This exists so that the retargeter and the trainer can be exercised, and a
dance reproduced, without a camera anywhere in the loop. AMASS dumps and the
`smpl_param.pt` that GEM writes are the two sources the design names, and both
speak joint positions rather than MediaPipe landmarks, so the whole job is a
mapping table plus honest bookkeeping about what the table cannot supply.

The table is the interesting part and it is deliberately partial. SMPL has no
eyes, ears, mouth corners or heels; inventing them would hand the retargeter
plausible numbers with no measurement behind them, which is worse than a gap
because a gap is visible. Unmapped landmarks come back at the origin with zero
visibility, which is the same signal MediaPipe gives for a landmark it could
not see, so downstream code needs no special case.

This module used to refuse to run a body model, because the mesh is separately
licensed and was not a dependency. It now runs one when the file present carries
parameters rather than joints, which is what every modern estimator emits, via
`motion/bodymodel.py`. The licence has not changed: the model is registration
gated, never committed and never downloaded automatically, and a file that needs
it when it is absent raises saying exactly how to obtain it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..ir import N_HAND_LANDMARKS, N_POSE_LANDMARKS, PoseSequence

#: MediaPipe's 33 pose landmarks, in index order. `ir.MP_POSE` carries only the
#: subset retargeting uses; the full list is needed to write the table below.
MP_LANDMARK_NAMES: tuple[str, ...] = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)
MP_INDEX = {name: i for i, name in enumerate(MP_LANDMARK_NAMES)}

#: SMPL's 24 body joints, in index order.
SMPL_JOINT_NAMES: tuple[str, ...] = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
)

#: SMPL-X's 55 joints. Indices 0 to 21 are identical to SMPL, which is why one
#: body table serves both; 22 onward diverge, replacing SMPL's two lumped hand
#: joints with three joints per finger.
_FINGERS = ("index", "middle", "pinky", "ring", "thumb")
SMPLX_JOINT_NAMES: tuple[str, ...] = SMPL_JOINT_NAMES[:22] + (
    "jaw", "left_eye_smplhf", "right_eye_smplhf") + tuple(
    f"{side}_{finger}{k}"
    for side in ("left", "right") for finger in _FINGERS for k in (1, 2, 3))

#: SMPL or SMPL-X joint name -> MediaPipe landmark name, for joints 0 to 21
#: that every layout shares.
#:
#: `head -> nose` is the one approximation carried deliberately: retargeting
#: uses the nose only to orient the head, and SMPL's head joint sits at the
#: skull centre, so the direction is right even though the point is not.
BODY_JOINT_MAP: dict[str, str] = {
    "left_shoulder": "left_shoulder", "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow", "right_elbow": "right_elbow",
    "left_wrist": "left_wrist", "right_wrist": "right_wrist",
    "left_hip": "left_hip", "right_hip": "right_hip",
    "left_knee": "left_knee", "right_knee": "right_knee",
    "left_ankle": "left_ankle", "right_ankle": "right_ankle",
    "left_foot": "left_foot_index", "right_foot": "right_foot_index",
    "head": "nose",
}

#: SMPL only. Its single lumped hand joint sits at the base of the fingers,
#: which is closest to what MediaPipe calls the index landmark.
SMPL_EXTRA_MAP: dict[str, str] = {
    "left_hand": "left_index", "right_hand": "right_index",
}

#: SMPL-X only. Distal finger joints stand in for MediaPipe's coarse hand
#: points, which sit near the finger tips.
SMPLX_EXTRA_MAP: dict[str, str] = {
    "left_index3": "left_index", "right_index3": "right_index",
    "left_pinky3": "left_pinky", "right_pinky3": "right_pinky",
    "left_thumb3": "left_thumb", "right_thumb3": "right_thumb",
}

#: MediaPipe landmarks no SMPL layout can supply. Listed rather than inferred
#: so the omission is a decision on the page, not an accident.
UNMAPPABLE: tuple[str, ...] = (
    "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_heel", "right_heel",
)

#: MediaPipe hand landmark index -> SMPL-X finger joint suffix. SMPL-X gives
#: three joints per finger (metacarpophalangeal, proximal, distal) and no tips,
#: so the four tip landmarks and the thumb carpometacarpal stay unmapped.
HAND_JOINT_MAP: dict[int, str] = {
    2: "thumb1", 3: "thumb2", 4: "thumb3",
    5: "index1", 6: "index2", 7: "index3",
    9: "middle1", 10: "middle2", 11: "middle3",
    13: "ring1", 14: "ring2", 15: "ring3",
    17: "pinky1", 18: "pinky2", 19: "pinky3",
}

#: keys that, if present, already hold joint positions we can use directly
_JOINT_KEYS = ("joints", "joints3d", "j3d", "keypoints3d", "kp3d")
_FPS_KEYS = ("mocap_framerate", "mocap_frame_rate", "fps", "frame_rate")


class SmplLayoutError(ValueError):
    """The joint array does not match any SMPL family layout we can map."""


def joint_names(layout: str) -> tuple[str, ...]:
    """Ordered joint names for `smpl` or `smplx`."""
    if layout == "smpl":
        return SMPL_JOINT_NAMES
    if layout in ("smplx", "smplh"):
        return SMPLX_JOINT_NAMES
    raise SmplLayoutError(f"unknown SMPL layout {layout!r}")


def detect_layout(n_joints: int) -> str:
    """Name the layout from the joint count alone.

    SMPL-H's 52 joints are SMPL-X's minus jaw and eyes, but everything we map
    lives at an index the two share, so they are treated alike.
    """
    if n_joints == 24:
        return "smpl"
    if n_joints in (52, 55):
        return "smplx"
    raise SmplLayoutError(
        f"{n_joints} joints matches no known layout "
        "(expected 24 for SMPL, 52 for SMPL-H, 55 for SMPL-X)")


def pose_mapping(layout: str) -> dict[int, int]:
    """Source joint index -> MediaPipe landmark index, for one layout."""
    names = joint_names(layout)
    index = {n: i for i, n in enumerate(names)}
    table = dict(BODY_JOINT_MAP)
    table.update(SMPL_EXTRA_MAP if layout == "smpl" else SMPLX_EXTRA_MAP)
    out: dict[int, int] = {}
    for joint, landmark in table.items():
        if joint not in index:       # SMPL-H lacks nothing we map, but be safe
            continue
        out[index[joint]] = MP_INDEX[landmark]
    return out


def hand_mapping(layout: str, side: str) -> dict[int, int]:
    """MediaPipe hand landmark index -> source joint index, for one hand."""
    if layout == "smpl":
        return {}
    index = {n: i for i, n in enumerate(SMPLX_JOINT_NAMES)}
    out = {0: index[f"{side}_wrist"]}
    for mp_i, suffix in HAND_JOINT_MAP.items():
        joint = f"{side}_{suffix}"
        if joint in index:
            out[mp_i] = index[joint]
    return out


def smpl_joints_to_landmarks(joints: np.ndarray, layout: str = "auto"
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Convert (T, J, 3) SMPL family joints to MediaPipe's 33 landmark layout.

    Returns (landmarks, visibility). Positions are hip centred, in whatever
    units the input used, which for every SMPL source is metres. Landmarks the
    layout cannot supply are zero with zero visibility.
    """
    joints = np.asarray(joints, np.float32)
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise SmplLayoutError(f"joints must be (T, J, 3), got {joints.shape}")
    if layout == "auto":
        layout = detect_layout(joints.shape[1])
    mapping = pose_mapping(layout)
    T = len(joints)
    landmarks = np.zeros((T, N_POSE_LANDMARKS, 3), np.float32)
    visibility = np.zeros((T, N_POSE_LANDMARKS), np.float32)
    src = np.fromiter(mapping.keys(), int, len(mapping))
    dst = np.fromiter(mapping.values(), int, len(mapping))
    landmarks[:, dst] = joints[:, src]
    visibility[:, dst] = 1.0
    hip = 0.5 * (landmarks[:, MP_INDEX["left_hip"]]
                 + landmarks[:, MP_INDEX["right_hip"]])
    landmarks[:, dst] -= hip[:, None, :]
    return landmarks, visibility


def smpl_joints_to_hands(joints: np.ndarray, layout: str = "auto"
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert finger joints to MediaPipe's 21 per hand, wrist centred.

    Returns (left, right, conf). SMPL has no fingers, so it yields zeros and
    zero confidence rather than pretending.
    """
    joints = np.asarray(joints, np.float32)
    if layout == "auto":
        layout = detect_layout(joints.shape[1])
    T = len(joints)
    out = {}
    conf = np.zeros((T, 2), np.float32)
    for col, side in enumerate(("left", "right")):
        hand = np.zeros((T, N_HAND_LANDMARKS, 3), np.float32)
        mapping = hand_mapping(layout, side)
        if mapping:
            dst = np.fromiter(mapping.keys(), int, len(mapping))
            src = np.fromiter(mapping.values(), int, len(mapping))
            hand[:, dst] = joints[:, src]
            hand[:, dst] -= hand[:, 0][:, None, :]
            conf[:, col] = 1.0
        out[side] = hand
    return out["left"], out["right"], conf


def _first_key(container, keys) -> str | None:
    for key in keys:
        if key in container:
            return key
    return None


def _explain_missing_joints(path: Path, present: list[str]) -> NotImplementedError:
    return NotImplementedError(
        f"{path} carries neither joint positions nor SMPL-X parameters this "
        f"module recognises (keys: {', '.join(sorted(present)[:12])}). Save "
        f"joints under one of {list(_JOINT_KEYS)} with shape (T, J, 3), or "
        "parameters under global_orient and body_pose, or an AMASS style "
        "poses array.")


#: Parameter spellings this module reads. `poses` is AMASS, which packs the root
#: orientation and the body pose into one array; the rest is the split form that
#: GVHMR, SMPLest-X and the other regressors emit.
_POSE_KEYS = ("poses", "pose")
_ORIENT_KEYS = ("global_orient", "root_orient", "global_rot")
_BODY_KEYS = ("body_pose", "pose_body")
_TRANS_KEYS = ("transl", "trans", "translation")
_BETA_KEYS = ("betas", "shape", "beta")
_LHAND_KEYS = ("left_hand_pose", "pose_hand_left", "lhand_pose")
_RHAND_KEYS = ("right_hand_pose", "pose_hand_right", "rhand_pose")


def _params_to_joints(container, keys: list[str]):
    """Joints and the parameters they came from, or `(None, None)`.

    Returns SMPL-X's 55 joints, which `detect_layout` recognises, so everything
    downstream treats the result exactly as it would a file that shipped joints.
    The parameters come back too because a retargeter built around the body
    model wants them: GMR reconstructs joint orientations from the pose, and a
    joint position cannot supply an orientation.
    """
    def get(names):
        k = _first_key(keys, names)
        return None if k is None else np.asarray(container[k], np.float32)

    orient, body = get(_ORIENT_KEYS), get(_BODY_KEYS)
    if orient is None or body is None:
        packed = get(_POSE_KEYS)
        if packed is None or packed.ndim != 2 or packed.shape[1] < 66:
            return None, None
        # AMASS packs root orientation first, then 21 body joints.
        orient, body = packed[:, :3], packed[:, 3:66]

    from ..bodymodel import smplx_joints
    T = len(orient)
    betas = get(_BETA_KEYS)
    betas = np.zeros((T, 10), np.float32) if betas is None else betas
    if betas.ndim == 1:
        betas = np.tile(betas[None], (T, 1))
    body = body.reshape(T, -1)[:, :63]
    transl = get(_TRANS_KEYS)
    transl = np.zeros((T, 3), np.float32) if transl is None else transl
    joints = smplx_joints(
        betas=betas, global_orient=orient, body_pose=body, transl=transl,
        left_hand_pose=get(_LHAND_KEYS), right_hand_pose=get(_RHAND_KEYS))
    params = {"smplx_betas": betas, "smplx_global_orient": orient,
              "smplx_body_pose": body, "smplx_transl": transl}
    return joints, params


def _load_npz(path: Path) -> tuple[np.ndarray, float | None]:
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.keys())
        joint_key = _first_key(keys, _JOINT_KEYS)
        params = None
        if joint_key is None:
            joints, params = _params_to_joints(z, keys)
            if joints is None:
                raise _explain_missing_joints(path, keys)
        else:
            joints = np.asarray(z[joint_key], np.float32)
        fps_key = _first_key(keys, _FPS_KEYS)
        fps = float(np.asarray(z[fps_key]).reshape(-1)[0]) if fps_key else None
    return joints, fps, params


def _load_pt(path: Path) -> tuple[np.ndarray, float | None]:
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        raise SmplLayoutError(
            f"{path} holds a {type(blob).__name__}, expected a dict of tensors")
    joint_key = _first_key(blob, _JOINT_KEYS)
    params = None
    if joint_key is None:
        joints, params = _params_to_joints(blob, list(blob))
        if joints is None:
            raise _explain_missing_joints(path, list(blob))
    else:
        joints = np.asarray(blob[joint_key], np.float32)
    fps_key = _first_key(blob, _FPS_KEYS)
    fps = float(np.asarray(blob[fps_key]).reshape(-1)[0]) if fps_key else None
    return joints, fps, params


def estimate_from_smpl(path: Path | str, fps: float | None = None,
                       *, layout: str = "auto") -> PoseSequence:
    """Read an AMASS style `.npz` or a GEM `smpl_param.pt` as a PoseSequence.

    `fps` overrides whatever the file claims, which matters because AMASS spells
    its frame rate three different ways across releases and some files omit it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        joints, file_fps, params = _load_npz(path)
    elif suffix in (".pt", ".pth"):
        joints, file_fps, params = _load_pt(path)
    else:
        raise NotImplementedError(
            f"unsupported SMPL container {path.suffix!r}; expected .npz or .pt")

    if joints.ndim == 2 and joints.shape[1] == 3:   # a single frame
        joints = joints[None]
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise SmplLayoutError(f"joints must be (T, J, 3), got {joints.shape}")

    resolved = layout if layout != "auto" else detect_layout(joints.shape[1])
    rate = float(fps or file_fps or 0.0)
    if rate <= 0:
        raise ValueError(
            f"{path} does not state a frame rate; pass fps= explicitly")

    landmarks, visibility = smpl_joints_to_landmarks(joints, resolved)
    left, right, hand_conf = smpl_joints_to_hands(joints, resolved)
    valid = np.isfinite(joints).all(axis=(1, 2))

    return PoseSequence(
        fps=rate, landmarks=landmarks, visibility=visibility,
        hand_left=left, hand_right=right, hand_conf=hand_conf, valid=valid,
        **(params or {}),
        meta={
            "has_smplx_params": params is not None,
            "frontend": "smpl",
            "layout": resolved,
            "n_source_joints": int(joints.shape[1]),
            "unmapped_landmarks": list(UNMAPPABLE),
            "axes": "smpl_x_right_y_up_z_forward",
            "units": "metres_hip_centred",
            "source": str(path),
        })


__all__ = ["BODY_JOINT_MAP", "HAND_JOINT_MAP", "MP_LANDMARK_NAMES",
           "SMPL_EXTRA_MAP", "SMPL_JOINT_NAMES", "SMPLX_EXTRA_MAP",
           "SMPLX_JOINT_NAMES", "SmplLayoutError", "UNMAPPABLE",
           "detect_layout", "estimate_from_smpl", "hand_mapping", "joint_names",
           "pose_mapping", "smpl_joints_to_hands", "smpl_joints_to_landmarks"]
