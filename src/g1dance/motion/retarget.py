"""Turn human landmarks into robot joint angles, without ever leaving the robot's
own kinematics.

Two things make this the highest risk module in the project. The first is that a
coordinate frame or joint ordering mistake here does not crash, it produces a
plausible looking wrong dance that only shows up after hours of training. The
second is that a human and a G1 are not the same shape, so copying world
positions is meaningless: a 1.75 m subject's wrist is simply outside the reach of
a 1.3 m robot. Both are addressed the same way, by refusing to work in any frame
that is not derived from the skeleton itself.

The frame chain, stated once so it can be checked:

* PoseSequence landmarks arrive in the subject frame: metres, hip centred,
  x to the image right, y up, z toward the camera. That is right handed.
* `R_SUBJECT_TO_ROBOT` rotates that into the robot's world convention, x forward,
  y left, z up. A subject facing the camera then faces robot +x.
* IK residuals are evaluated in that robot world frame. Robot quantities come
  out of forward kinematics in the pelvis frame and are rotated into it by the
  pelvis orientation, which is an unknown of the solve rather than something
  assumed from the landmarks. Assuming it is the trap: take the pelvis frame from
  the hips and shoulders, as most retargeters do, and a waist pitch error trades
  exactly against an equal and opposite hip pitch error, leaving an almost flat
  valley that a solver wanders around in. Solving for it leaves the hip landmarks
  to pin two of its three components, and only the third, pitch about the hip
  line, needs a prior.
* That prior is `body_frame`, rebuilt identically from the human landmarks and
  from robot forward kinematics: y along the hip line from right to left, z from
  the hip midpoint toward the shoulder midpoint, x completing the right handed
  triple. It is equivariant, so it is exact whenever the subject's build matches
  the robot's, and a defensible guess otherwise. No landmark observes the pelvis
  link, so some such choice is unavoidable; tests/test_retarget.py measures the
  resulting null space rather than pretending it is absent.
* The pelvis frame, used only for the ReferenceMotion outputs, is plain forward
  kinematics with the floating base at the identity.

Occlusion is handled by pinning rather than by hoping. A landmark below the
visibility threshold contributes nothing, so on real footage a limb regularly
spends a run of frames with no data behind it at all. `_JOINT_GROUPS` records
which bones observe which joints, and the joints of an unobserved limb are held
toward the default stance for as long as that lasts. It is the same argument the
wrist DoF already get: unobservable is not the same as arbitrary, and a limb left
free drifts through the occlusion and reappears somewhere else. On a frame where
every landmark is visible the term is identically zero, so it does not touch the
synthetic round trip.

Scale is handled by bone rather than globally. Each bone gets its own ratio of G1
length to the subject's median measured length, and the positional targets are
built by walking the chain out from the robot's own joint centre along the
subject's bone directions at G1 bone lengths, so the position term measures
accumulated chain error and never asks for a point outside the robot's reach.

Forward kinematics is `mujoco.mj_kinematics` on a scratch MjData. No physics is
stepped anywhere in this file.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

from ..sim.model import BODY_JOINTS, HAND_JOINTS, KEY_BODIES, G1Model
from . import constraints as C
from .ir import MP_POSE, PoseSequence, ReferenceMotion

log = logging.getLogger(__name__)

#: subject frame (x image right, y up, z toward camera) to robot world
#: (x forward, y left, z up). Both are right handed; this is a pure rotation.
R_SUBJECT_TO_ROBOT = np.array([[0.0, 0.0, 1.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0]])

#: MediaPipe hand landmark indices used for wrist orientation
HAND_WRIST, HAND_INDEX_MCP, HAND_MIDDLE_MCP, HAND_PINKY_MCP = 0, 5, 9, 17


@dataclass
class RetargetConfig:
    """Weights and thresholds for the IK. Stamped verbatim into motion meta."""

    control_fps: float = 50.0

    # residual weights
    w_direction: float = 1.0
    w_position: float = 1.0
    w_smooth: float = 0.35
    w_limit: float = 4.0
    w_regular: float = 0.02
    w_unobserved: float = 1.0
    w_hand: float = 0.6
    limit_margin: float = 0.05
    first_frame_smooth: float = 0.05

    # gating
    #: Landmarks below this confidence are dropped from the direction
    #: residuals. This is a trade, not a ceiling to maximise, and the optimum
    #: is interior.
    #:
    #: At 0.5 about 30 percent of arm frames are discarded and those joints are
    #: pinned to the default stance. Those are not random frames: they are
    #: where the dancer's arms cross the midline, which is where confidence
    #: falls and where most of the choreography is. Lowering the gate recovers
    #: them, and source-to-reference correlation rises monotonically, 0.692 at
    #: 0.5 to 0.816 at 0.1.
    #:
    #: But the reference is not the product. A reference the policy cannot
    #: track produces a robot less like the video, and reference motion content
    #: rises with the gate too. Measured end to end, source to trained policy,
    #: over 12 seeds each:
    #:
    #:     gate 0.5  reference 0.697  policy tracks 0.558  end to end 0.464
    #:     gate 0.3  reference 0.729  policy tracks 0.647  end to end 0.522
    #:     gate 0.1  reference 0.816  policy tracks 0.309  end to end 0.268
    #:
    #: 0.3 it is. Going to 0.1 raises the reference by 12 percent and costs 45
    #: percent of the policy's ability to follow it.
    visibility_threshold: float = 0.3
    hand_conf_threshold: float = 0.5
    use_hand_orientation: bool = True

    # solver
    max_nfev: int = 300
    xtol: float = 1e-8
    ftol: float = 1e-8
    n_restarts: int = 3
    seed: int = 0
    w_root_prior: float = 1.0
    max_root_delta: float = 1.2
    residual_warn: float = 0.08
    #: Cap on how fast the root may yaw, degrees per second, or 0 to disable.
    #:
    #: The retargeted yaw carries the dancer turning and monocular heading noise
    #: on top of it, and on kirari the noise reaches 4009 deg/s, which nothing
    #: physical can do. Milestone A never noticed because its own policy was
    #: trained on this convention; a general tracking policy consumes the root
    #: and refuses. With Humanoid-GPT on kirari_v4 over 12 seeds, end-to-end
    #: similarity is 0.503 uncapped, 0.549 here, 0.517 at 90 and 0.482 at 180,
    #: against 0.562 for discarding the yaw altogether. The curve is flat from
    #: 15 to 60, so this is the loosest cap that costs nothing: it keeps 239
    #: degrees of the 310 degree turn where discarding it keeps none.
    max_root_yaw_rate: float = 60.0

    # constraints, applied after the per frame solve
    min_foot_distance: float = 0.17
    resolve_collisions: bool = True
    #: 4 Hz rather than 6. With the lower visibility threshold the IK fits
    #: more marginal landmarks, and this more than absorbs the extra noise:
    #: mean jerk over the tracked DoF is 751 against 1411 for the old
    #: defaults, so the reference is both more faithful and smoother.
    lowpass_hz: float = 4.0
    filter_order: int = 2

    # fingers
    finger_backend: str = "auto"
    finger_lowpass_hz: float = 3.0

    def as_meta(self) -> dict:
        return asdict(self)


def _as_config(cfg) -> RetargetConfig:
    if cfg is None:
        return RetargetConfig()
    if isinstance(cfg, RetargetConfig):
        return cfg
    if isinstance(cfg, dict):
        known = {f for f in RetargetConfig.__dataclass_fields__}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unknown retarget config keys: {sorted(unknown)}")
        return RetargetConfig(**cfg)
    raise TypeError(f"cfg must be RetargetConfig, dict or None, got {type(cfg)!r}")


@dataclass
class RetargetDiagnostics:
    """Everything needed to judge whether a retarget is worth training on."""

    residual: np.ndarray                 # (T,) rms IK residual per frame
    converged: np.ndarray                # (T,) bool
    limit_violation_rate: float
    limit_violation_per_joint: np.ndarray
    n_not_converged: int
    n_foot_distance_fixed: int
    n_collision_frames: int
    bone_scales: dict
    finger_backend: str

    def summary(self) -> dict:
        """Literal only, so it survives `repr` round tripping into meta."""
        r = np.asarray(self.residual, float)
        finite = r[np.isfinite(r)]
        if finite.size == 0:
            finite = np.zeros(1)
        return {
            "ik_residual_mean": float(finite.mean()),
            "ik_residual_p95": float(np.percentile(finite, 95)),
            "ik_residual_max": float(finite.max()),
            "limit_violation_rate": float(self.limit_violation_rate),
            "n_not_converged": int(self.n_not_converged),
            "n_foot_distance_fixed": int(self.n_foot_distance_fixed),
            "n_collision_frames": int(self.n_collision_frames),
            "bone_scales": {k: float(v) for k, v in self.bone_scales.items()},
            "finger_backend": str(self.finger_backend),
        }

    def report(self) -> str:
        s = self.summary()
        return (f"IK residual mean {s['ik_residual_mean']:.4f} "
                f"p95 {s['ik_residual_p95']:.4f} max {s['ik_residual_max']:.4f}; "
                f"limit violations {100 * s['limit_violation_rate']:.2f}%; "
                f"non-converged {s['n_not_converged']}; "
                f"foot-distance fixes {s['n_foot_distance_fixed']}; "
                f"collision frames {s['n_collision_frames']}; "
                f"fingers {s['finger_backend']}")


# --------------------------------------------------------------------------
# skeleton definition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Bone:
    """A segment whose *direction* is matched between human and robot."""

    name: str
    human: tuple[str, str]                 # MP_POSE keys, tail then head
    robot: tuple[str, str]                 # body names, tail then head
    weight: float = 1.0


#: The direction targets. Wrist and ankle also get position targets; everything
#: else is direction only, because a human's proportions carry no information
#: about where a G1's joint should be.
#:
#: `hip_line` is the only direct observation of the pelvis. Its robot endpoints
#: are fixed in the pelvis frame, so it contributes nothing to the joint angles
#: and everything to the pelvis orientation, pinning two of that rotation's three
#: components. The third, pitch about the hip line, is not observable from
#: landmarks at all and is fixed by the root prior in `_Solver.residual`.
BONES: tuple[_Bone, ...] = (
    _Bone("hip_line", ("right_hip", "left_hip"),
          ("right_hip_pitch_link", "left_hip_pitch_link"), 1.0),
    _Bone("shoulder_line", ("right_shoulder", "left_shoulder"),
          ("right_shoulder_pitch_link", "left_shoulder_pitch_link"), 1.0),
    _Bone("left_upper_arm", ("left_shoulder", "left_elbow"),
          ("left_shoulder_pitch_link", "left_elbow_link")),
    _Bone("right_upper_arm", ("right_shoulder", "right_elbow"),
          ("right_shoulder_pitch_link", "right_elbow_link")),
    _Bone("left_forearm", ("left_elbow", "left_wrist"),
          ("left_elbow_link", "left_wrist_roll_link")),
    _Bone("right_forearm", ("right_elbow", "right_wrist"),
          ("right_elbow_link", "right_wrist_roll_link")),
    _Bone("left_thigh", ("left_hip", "left_knee"),
          ("left_hip_pitch_link", "left_knee_link")),
    _Bone("right_thigh", ("right_hip", "right_knee"),
          ("right_hip_pitch_link", "right_knee_link")),
    _Bone("left_shin", ("left_knee", "left_ankle"),
          ("left_knee_link", "left_ankle_roll_link")),
    _Bone("right_shin", ("right_knee", "right_ankle"),
          ("right_knee_link", "right_ankle_roll_link")),
    # The foot needs two non collinear directions to pin ankle roll as well as
    # pitch. Heel and toe sit below and either side of the ankle, so they span
    # the sagittal plane of the foot and their cross product is its lateral axis.
    _Bone("left_toe", ("left_ankle", "left_foot_index"),
          ("left_ankle_roll_link", "left_toe"), 0.7),
    _Bone("right_toe", ("right_ankle", "right_foot_index"),
          ("right_ankle_roll_link", "right_toe"), 0.7),
    _Bone("left_heel", ("left_ankle", "left_heel"),
          ("left_ankle_roll_link", "left_heel_pt"), 0.7),
    _Bone("right_heel", ("right_ankle", "right_heel"),
          ("right_ankle_roll_link", "right_heel_pt"), 0.7),
)
_BONE_INDEX = {b.name: i for i, b in enumerate(BONES)}

#: end effector position targets: the robot anchor body, then the bones walked
#: outward from it to the tip body.
_CHAINS = (
    ("left_wrist", "left_shoulder_pitch_link", "left_wrist_roll_link",
     ("left_upper_arm", "left_forearm")),
    ("right_wrist", "right_shoulder_pitch_link", "right_wrist_roll_link",
     ("right_upper_arm", "right_forearm")),
    ("left_ankle", "left_hip_pitch_link", "left_ankle_roll_link",
     ("left_thigh", "left_shin")),
    ("right_ankle", "right_hip_pitch_link", "right_ankle_roll_link",
     ("right_thigh", "right_shin")),
)

#: Hand orientation targets, used only when hand confidence is high. Without
#: them the three wrist DoF per arm are exactly unobservable, see the module
#: docstring and tests/test_retarget.py. Two near perpendicular directions are
#: needed: along the palm and across it. Two directions from the same origin to
#: neighbouring knuckles would be nearly parallel and would leave roll about the
#: palm axis as badly conditioned as it was before.
_HAND_DIRS = (
    ("palm_axis", HAND_WRIST, HAND_MIDDLE_MCP,
     "{side}_wrist_yaw_link", "{p}_middle_proximal"),
    ("palm_across", HAND_INDEX_MCP, HAND_PINKY_MCP,
     "{p}_index_proximal", "{p}_pinky_proximal"),
)


#: which bones observe each joint group. A joint whose bones all have zero
#: direction weight on a frame has no data at all on that frame: the landmarks
#: behind it are occluded, and only the smoothness term is holding it. That is
#: how a limb drifts through an occlusion and comes back somewhere else. The
#: groups are used to pin such a limb toward the default stance in proportion to
#: how unobserved it is, which is the same argument the wrist DoF already get in
#: `_Solver.residual`: unobservable is not the same as arbitrary.
#: The wrist DoF are deliberately absent: no pose bone observes them at all, so
#: they are pinned by `r_free` when there are no hand landmarks and driven by
#: `r_hand` when there are.
_JOINT_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = tuple(
    [(tuple(f"{s}_{j}_joint" for j in
            ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")),
      (f"{s}_upper_arm", f"{s}_forearm"))
     for s in ("left", "right")]
    + [(tuple(f"{s}_{j}_joint" for j in
              ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch",
               "ankle_roll")),
        (f"{s}_thigh", f"{s}_shin", f"{s}_toe", f"{s}_heel"))
       for s in ("left", "right")]
    + [(("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
        ("shoulder_line",))])


def _hand_body(tmpl: str, side: str) -> str:
    return tmpl.format(side=side, p="L" if side == "left" else "R")

_JOINT_INDEX = {j: i for i, j in enumerate(BODY_JOINTS)}
N_Q = len(BODY_JOINTS)


def foot_points(model: G1Model) -> dict[str, np.ndarray]:
    """Toe and heel offsets in each ankle_roll_link frame, taken from the model.

    Derived from the foot's own collision geoms rather than hardcoded, so a
    rebuilt model with a different sole cannot silently desynchronise the
    retargeter from the simulation.
    """
    m = model.model
    out = {}
    for side in ("left", "right"):
        bid = model.body_id(f"{side}_ankle_roll_link")
        pts = np.array([m.geom_pos[g] for g in range(m.ngeom)
                        if m.geom_bodyid[g] == bid])
        if len(pts) < 2:
            raise ValueError(f"{side}_ankle_roll_link has no sole geoms to "
                             "derive toe and heel points from")
        out[f"{side}_toe"] = pts[int(np.argmax(pts[:, 0]))]
        out[f"{side}_heel_pt"] = pts[int(np.argmin(pts[:, 0]))]
    return out


# --------------------------------------------------------------------------
# small rotation helpers
# --------------------------------------------------------------------------

def body_frame(hip_l, hip_r, sho_l, sho_r) -> np.ndarray:
    """Right handed frame from four joint centres, as columns [x, y, z].

    y is the hip line pointing left, z points from the hip midpoint to the
    shoulder midpoint, x = y x z completes the triple pointing forward. It is
    equivariant: building it from rotated points returns the rotated frame. That
    is what lets it serve as a pelvis orientation estimate on the human side and
    as its exact counterpart on the robot side.
    """
    y = np.asarray(hip_l, float) - np.asarray(hip_r, float)
    ny = np.linalg.norm(y)
    up = (np.asarray(sho_l, float) + np.asarray(sho_r, float)) * 0.5 \
        - (np.asarray(hip_l, float) + np.asarray(hip_r, float)) * 0.5
    nu = np.linalg.norm(up)
    if ny < 1e-8 or nu < 1e-8:
        return np.eye(3)
    y = y / ny
    x = np.cross(y, up / nu)
    nx = np.linalg.norm(x)
    if nx < 1e-8:                      # degenerate: hips parallel to the spine
        return np.eye(3)
    x = x / nx
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-9), 0.0)


def _exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues. Small by construction here, so no branch on large angles."""
    t = float(np.linalg.norm(w))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if t < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + (np.sin(t) / t) * K + ((1 - np.cos(t)) / (t * t)) * (K @ K)


def _log_vee(R: np.ndarray) -> np.ndarray:
    """sin(theta) * axis. Zero exactly when R is the identity, and cheap."""
    return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                           R[1, 0] - R[0, 1]])


# --------------------------------------------------------------------------
# robot side forward kinematics
# --------------------------------------------------------------------------

class _RobotKinematics:
    """Scratch-MjData forward kinematics for the points the IK needs.

    The floating base is held at the identity throughout, so every position read
    back is already in the pelvis frame.
    """

    def __init__(self, model: G1Model):
        self.g1 = model
        self.m = model.model
        self.d = model.data()
        self.d.qpos[:] = 0.0
        self.d.qpos[3] = 1.0
        self.body_qpos = model.body_qpos
        self.offsets = foot_points(model)
        names = set()
        for b in BONES:
            names.update(b.robot)
        for _, anchor, tip, _ in _CHAINS:
            names.update((anchor, tip))
        names.update(("left_hip_pitch_link", "right_hip_pitch_link",
                      "left_shoulder_pitch_link", "right_shoulder_pitch_link",
                      "left_wrist_roll_link", "right_wrist_roll_link"))
        for side in ("left", "right"):
            for _, _, _, tail, head in _HAND_DIRS:
                names.update((_hand_body(tail, side), _hand_body(head, side)))
        self.point_names = sorted(names | set(self.offsets))
        self.index = {n: i for i, n in enumerate(self.point_names)}
        self.body_ids = np.array(
            [model.body_id(self._base(n)) for n in self.point_names], int)
        self.local = np.array(
            [self.offsets.get(n, np.zeros(3)) for n in self.point_names], float)
        self.key_ids = model.key_body_ids
        self.pelvis_id = model.pelvis_id
        self.hip_l = self.index["left_hip_pitch_link"]
        self.hip_r = self.index["right_hip_pitch_link"]
        self.sho_l = self.index["left_shoulder_pitch_link"]
        self.sho_r = self.index["right_shoulder_pitch_link"]

    @staticmethod
    def _base(name: str) -> str:
        if name.endswith("_toe"):
            return name[: -len("_toe")] + "_ankle_roll_link"
        if name.endswith("_heel_pt"):
            return name[: -len("_heel_pt")] + "_ankle_roll_link"
        return name

    def fk(self, q: np.ndarray) -> np.ndarray:
        """(n_points, 3) pelvis frame positions for body joint angles `q`."""
        self.d.qpos[self.body_qpos] = q
        mujoco.mj_kinematics(self.m, self.d)
        xpos = self.d.xpos[self.body_ids]
        xmat = self.d.xmat[self.body_ids].reshape(-1, 3, 3)
        return xpos + np.einsum("nij,nj->ni", xmat, self.local)

    def frame(self, p: np.ndarray) -> np.ndarray:
        return body_frame(p[self.hip_l], p[self.hip_r],
                          p[self.sho_l], p[self.sho_r])

    def key_body_pos(self, q: np.ndarray) -> np.ndarray:
        """(K, 3) KEY_BODIES in the pelvis frame."""
        self.d.qpos[self.body_qpos] = q
        mujoco.mj_kinematics(self.m, self.d)
        R = self.d.xmat[self.pelvis_id].reshape(3, 3)
        return (self.d.xpos[self.key_ids] - self.d.xpos[self.pelvis_id]) @ R

    def bone_lengths(self, q: np.ndarray) -> dict[str, float]:
        p = self.fk(q)
        return {b.name: float(np.linalg.norm(p[self.index[b.robot[1]]]
                                             - p[self.index[b.robot[0]]]))
                for b in BONES}


# --------------------------------------------------------------------------
# human side targets
# --------------------------------------------------------------------------

def _landmark(pose: PoseSequence, key: str) -> np.ndarray:
    return pose.landmarks[:, MP_POSE[key], :] @ R_SUBJECT_TO_ROBOT.T


def _visible(pose: PoseSequence, key: str, thr: float) -> np.ndarray:
    return pose.visibility[:, MP_POSE[key]] >= thr


def measure_bone_scales(pose: PoseSequence, kin: _RobotKinematics,
                        cfg: RetargetConfig) -> dict[str, float]:
    """G1 length / subject length per bone, median over confident frames.

    Measured once for the whole clip. A per frame length would turn monocular
    depth noise into limb length noise, which is exactly the jitter the
    smoothness term exists to remove.
    """
    robot_len = kin.bone_lengths(kin.g1.default_qpos)
    scales = {}
    for b in BONES:
        a = _landmark(pose, b.human[0])
        c = _landmark(pose, b.human[1])
        ok = (pose.valid & _visible(pose, b.human[0], cfg.visibility_threshold)
              & _visible(pose, b.human[1], cfg.visibility_threshold))
        lens = np.linalg.norm(c - a, axis=-1)[ok]
        lens = lens[lens > 1e-4]
        scales[b.name] = (1.0 if lens.size == 0
                          else float(robot_len[b.name] / np.median(lens)))
    return scales


@dataclass
class _FrameTargets:
    """One frame of targets, in the robot world convention. No frame projection
    is applied: the pelvis rotation is an explicit unknown of the solve, so the
    residuals live in the subject's own frame."""

    dirs: np.ndarray                 # (n_bone, 3) unit
    dir_w: np.ndarray                # (n_bone,)
    chain: np.ndarray                # (n_chain, 2, 3) unit human bone directions
    chain_w: np.ndarray              # (n_chain,)
    hand_dirs: np.ndarray            # (4, 3) unit
    hand_w: np.ndarray               # (4,)
    frame: np.ndarray                # (3, 3) human body frame, pelvis estimate
    hip_mid: np.ndarray              # (3,) subject frame, hip centred origin
    unobs: np.ndarray                # (N_Q,) 1 where no bone observes the joint


def _build_targets(pose: PoseSequence, cfg: RetargetConfig) -> list[_FrameTargets]:
    T = pose.n_frames
    thr = cfg.visibility_threshold
    lm = {k: _landmark(pose, k) for k in MP_POSE}
    vis = {k: _visible(pose, k, thr).astype(float) for k in MP_POSE}

    frames = np.zeros((T, 3, 3))
    for t in range(T):
        frames[t] = body_frame(lm["left_hip"][t], lm["right_hip"][t],
                               lm["left_shoulder"][t], lm["right_shoulder"][t])
    hip_mid = 0.5 * (lm["left_hip"] + lm["right_hip"])

    dirs = np.zeros((T, len(BONES), 3))
    conf = np.zeros((T, len(BONES)))
    for i, b in enumerate(BONES):
        v = lm[b.human[1]] - lm[b.human[0]]
        dirs[:, i] = _unit(v)
        conf[:, i] = (np.minimum(vis[b.human[0]], vis[b.human[1]])
                      * (np.linalg.norm(v, axis=-1) > 1e-4))
    dir_w = conf * np.array([b.weight for b in BONES])

    chain = np.zeros((T, len(_CHAINS), 2, 3))
    chain_w = np.zeros((T, len(_CHAINS)))
    for i, (_, _, _, bones) in enumerate(_CHAINS):
        ok = np.ones(T)
        for j, name in enumerate(bones):
            k = _BONE_INDEX[name]
            chain[:, i, j] = dirs[:, k]
            ok = np.minimum(ok, conf[:, k])
        chain_w[:, i] = ok

    # how unobserved each joint is on each frame, from the bones that see it
    unobs = np.zeros((T, N_Q))
    for joints, bones in _JOINT_GROUPS:
        seen = np.max([conf[:, _BONE_INDEX[b]] for b in bones], axis=0)
        unobs[:, [_JOINT_INDEX[j] for j in joints]] = (1.0 - seen)[:, None]

    hand_dirs = np.zeros((T, 4, 3))
    hand_w = np.zeros((T, 4))
    if cfg.use_hand_orientation:
        for h, hl in enumerate((pose.hand_left, pose.hand_right)):
            conf = pose.hand_conf[:, h]
            pts = hl @ R_SUBJECT_TO_ROBOT.T
            for k, (_, tail, head, _, _) in enumerate(_HAND_DIRS):
                v = pts[:, head] - pts[:, tail]
                hand_dirs[:, 2 * h + k] = _unit(v)
                ok = (conf >= cfg.hand_conf_threshold) & \
                     (np.linalg.norm(v, axis=-1) > 1e-5)
                hand_w[:, 2 * h + k] = np.where(ok, 1.0, 0.0)

    return [_FrameTargets(dirs[t], dir_w[t], chain[t], chain_w[t], hand_dirs[t],
                          hand_w[t], frames[t], hip_mid[t], unobs[t])
            for t in range(T)]


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------

class _Solver:
    """Per frame least squares over 29 joint angles plus the pelvis rotation.

    The pelvis rotation has to be solved for rather than assumed. Every residual
    below is a direction or a relative position, so a rotation of the whole robot
    is free unless it is pinned: `hip_line` pins two of its components from the
    hip landmarks, and the root prior pins the third, which no set of landmarks
    can observe. Assuming the pelvis orientation instead, by taking it from the
    hips and shoulders, leaves an almost flat valley in which a waist pitch error
    trades against an equal and opposite hip pitch error; that valley is what
    makes a retarget look right and score wrong.
    """

    def __init__(self, model: G1Model, kin: _RobotKinematics,
                 cfg: RetargetConfig):
        self.g1 = model
        self.kin = kin
        self.cfg = cfg
        rng = model.joint_range(BODY_JOINTS)
        self.lo = np.concatenate([rng[:, 0], np.full(3, -cfg.max_root_delta)])
        self.hi = np.concatenate([rng[:, 1], np.full(3, cfg.max_root_delta)])
        self.default = np.clip(model.default_qpos, rng[:, 0], rng[:, 1])
        self.bone_tail = np.array([kin.index[b.robot[0]] for b in BONES], int)
        self.bone_head = np.array([kin.index[b.robot[1]] for b in BONES], int)
        self.chain_anchor = np.array(
            [kin.index[a] for _, a, _, _ in _CHAINS], int)
        self.chain_tip = np.array([kin.index[t] for _, _, t, _ in _CHAINS], int)
        self.chain_bones = np.array(
            [[_BONE_INDEX[n] for n in bones] for _, _, _, bones in _CHAINS], int)
        self.hand_root = np.array(
            [kin.index[_hand_body(tail, side)] for side in ("left", "right")
             for _, _, _, tail, _ in _HAND_DIRS], int)
        self.hand_tip = np.array(
            [kin.index[_hand_body(head, side)] for side in ("left", "right")
             for _, _, _, _, head in _HAND_DIRS], int)
        # joints with no effect on any landmark derived residual unless hand
        # landmarks are available, see the module docstring
        self.free_joints = np.array(
            [_JOINT_INDEX[f"{s}_wrist_{a}_joint"] for s in ("left", "right")
             for a in ("roll", "pitch", "yaw")], int)
        self.n_data = 3 * (len(BONES) + len(_CHAINS))

    def pose(self, x: np.ndarray, tgt: _FrameTargets, R_ref: np.ndarray):
        """(points, pelvis rotation) for a parameter vector."""
        p = self.kin.fk(x[:N_Q])
        return p, R_ref @ _exp_so3(x[N_Q:])

    def residual(self, x: np.ndarray, tgt: _FrameTargets, R_ref: np.ndarray,
                 q_prev: np.ndarray, R_prev: np.ndarray,
                 w_smooth: float) -> np.ndarray:
        cfg = self.cfg
        q = x[:N_Q]
        p, R = self.pose(x, tgt, R_ref)

        v = (p[self.bone_head] - p[self.bone_tail]) @ R.T
        r_dir = (_unit(v) - tgt.dirs) * (cfg.w_direction * tgt.dir_w)[:, None]

        # end effector position, anchored at the robot's own joint centre and
        # built from G1 bone lengths pointed along the subject's bone directions.
        # That is the (G1 length / subject length) scaling of the positional
        # target, written so that it is exactly achievable by the robot.
        lengths = np.linalg.norm(v, axis=-1)
        rel = (p[self.chain_tip] - p[self.chain_anchor]) @ R.T
        want = (lengths[self.chain_bones][:, :, None] * tgt.chain).sum(axis=1)
        r_pos = (rel - want) * (cfg.w_position * tgt.chain_w)[:, None]

        if tgt.hand_w.any():
            hv = (p[self.hand_tip] - p[self.hand_root]) @ R.T
            r_hand = (_unit(hv) - tgt.hand_dirs) * (cfg.w_hand * tgt.hand_w)[:, None]
        else:
            r_hand = np.zeros((0, 3))

        # root prior. The pelvis pitch about the hip line is unobservable from
        # landmarks, so it is resolved by the hips-and-shoulders estimate, which
        # is exact whenever the subject's proportions match the robot's and is a
        # defensible guess otherwise. Recomputed from q each evaluation so that
        # the true pose satisfies it exactly.
        r_root = cfg.w_root_prior * _log_vee(
            (tgt.frame @ self.kin.frame(p).T).T @ R)

        m = cfg.limit_margin
        r_lim = cfg.w_limit * np.concatenate([
            np.maximum(0.0, (self.lo[:N_Q] + m) - q),
            np.maximum(0.0, q - (self.hi[:N_Q] - m))])
        r_smooth = w_smooth * np.concatenate([q - q_prev, _log_vee(R_prev.T @ R)])
        r_reg = cfg.w_regular * (q - self.default)
        # a limb whose landmarks are all occluded on this frame has no data at
        # all, so the only things holding it are the smoothness term, which lets
        # it keep whatever velocity it had, and the weak global regulariser. It
        # drifts, and it comes back out of the occlusion somewhere else. Pin it
        # toward the default stance in proportion to how unobserved it is; on a
        # fully visible frame this term is identically zero.
        r_unobs = cfg.w_unobserved * tgt.unobs * (q - self.default)
        r_free = np.zeros(len(self.free_joints))
        if not tgt.hand_w.any():
            # the wrist DoF are unobservable without hand landmarks; hold them at
            # the default stance rather than leaving the solve rank deficient
            r_free = 10.0 * cfg.w_regular * (q[self.free_joints]
                                             - self.default[self.free_joints])
        return np.concatenate([r_dir.ravel(), r_pos.ravel(), r_hand.ravel(),
                               r_root, r_lim, r_smooth, r_reg, r_unobs, r_free])

    def data_residual(self, x: np.ndarray, tgt: _FrameTargets,
                      R_ref: np.ndarray) -> float:
        """RMS of the data terms only, the number worth reporting per frame."""
        r = self.residual(x, tgt, R_ref, x[:N_Q], R_ref @ _exp_so3(x[N_Q:]), 0.0)
        n = self.n_data + (12 if tgt.hand_w.any() else 0)
        return float(np.sqrt(np.mean(np.square(r[:n]))))

    def solve(self, tgt: _FrameTargets, q_prev: np.ndarray, R_prev: np.ndarray,
              w_smooth: float, starts: list[np.ndarray] | None = None):
        """Returns (q, pelvis rotation, data residual, converged)."""
        eps = 1e-6
        best = best_ref = None
        for q0 in (starts or [q_prev]):
            # the reference rotation is the landmark estimate at the warm start,
            # so the solved delta stays small and no rotation vector ever wraps
            R_ref = tgt.frame @ self.kin.frame(self.kin.fk(q0)).T
            x0 = np.clip(np.concatenate([q0, np.zeros(3)]),
                         self.lo + eps, self.hi - eps)
            res = least_squares(
                self.residual, x0, bounds=(self.lo, self.hi), method="trf",
                args=(tgt, R_ref, q_prev, R_prev, w_smooth), xtol=self.cfg.xtol,
                ftol=self.cfg.ftol, gtol=None, max_nfev=self.cfg.max_nfev)
            if best is None or res.cost < best.cost:
                best, best_ref = res, R_ref
        R = best_ref @ _exp_so3(best.x[N_Q:])
        return (best.x[:N_Q], R, self.data_residual(best.x, tgt, best_ref),
                bool(best.status > 0))


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def retarget(pose: PoseSequence, model: G1Model, cfg=None) -> ReferenceMotion:
    """Retarget a PoseSequence onto the G1, returning a ReferenceMotion."""
    motion, _ = retarget_with_diagnostics(pose, model, cfg)
    return motion


def retarget_with_diagnostics(pose: PoseSequence, model: G1Model, cfg=None
                              ) -> tuple[ReferenceMotion, RetargetDiagnostics]:
    """As `retarget`, but also hands back the per frame quality record."""
    from . import fingers as F

    cfg = _as_config(cfg)
    kin = _RobotKinematics(model)
    solver = _Solver(model, kin, cfg)
    scales = measure_bone_scales(pose, kin, cfg)
    targets = _build_targets(pose, cfg)

    T = pose.n_frames
    q = np.zeros((T, N_Q))
    R_root = np.tile(np.eye(3), (T, 1, 1))
    residual = np.full(T, np.nan)
    converged = np.zeros(T, bool)
    q_prev = solver.default.copy()
    R_prev = np.eye(3)
    rng = np.random.default_rng(cfg.seed)
    first = True
    for t in range(T):
        if not pose.valid[t]:
            q[t], R_root[t] = q_prev, R_prev
            converged[t] = True
            continue
        if first:
            # multi start on the first frame only. Every later frame is warm
            # started from its predecessor, which is both faster and what keeps
            # the sequence in one IK branch instead of flipping between them.
            span = solver.hi[:N_Q] - solver.lo[:N_Q]
            starts = [solver.default] + [
                np.clip(solver.default + 0.25 * span * rng.normal(size=N_Q),
                        solver.lo[:N_Q], solver.hi[:N_Q])
                for _ in range(max(0, cfg.n_restarts - 1))]
            w_smooth = cfg.w_smooth * cfg.first_frame_smooth
        else:
            starts = [q_prev]
            w_smooth = cfg.w_smooth
        q[t], R_root[t], residual[t], converged[t] = solver.solve(
            targets[t], q_prev, R_prev, w_smooth, starts)
        q_prev, R_prev = q[t], R_root[t]
        first = False

    # constraints, in the order of spec 6.1 step 3
    ranges = model.joint_range(BODY_JOINTS)
    q, viol_rate, viol_joint = C.clamp_to_limits(q, ranges)
    q, n_foot = C.enforce_min_foot_distance(
        q, model, min_distance=cfg.min_foot_distance)
    n_coll = 0
    if cfg.resolve_collisions:
        q, n_coll = C.resolve_self_collision(q, model)
    q = C.lowpass_zero_phase(q, pose.fps, cfg.lowpass_hz, cfg.filter_order)
    q = np.clip(q, ranges[:, 0], ranges[:, 1])

    finger, backend = F.retarget_fingers(pose, model, cfg)
    root_pos, root_quat = _root_trajectory(q, R_root, targets, kin, model, scales)

    # resample everything to the control rate before differentiating, so that
    # qvel_ref is the derivative of the signal the policy actually sees
    out_fps = float(cfg.control_fps)
    q = C.resample(q, pose.fps, out_fps)
    finger = C.resample(finger, pose.fps, out_fps)
    root_pos = C.resample(root_pos, pose.fps, out_fps)
    root_quat = C.resample_quat(root_quat, pose.fps, out_fps)
    # After resampling, not before: the cap is a promise about the signal the
    # environment and the tracker read, and resampling a capped source
    # reintroduced rate at the boundaries. Capping here makes the bound hold at
    # the control rate, which is where it is consumed.
    root_quat = C.limit_root_yaw_rate(root_quat, out_fps, cfg.max_root_yaw_rate)
    valid = C.resample_bool(pose.valid, pose.fps, out_fps)
    residual_rs = C.resample(residual[:, None], pose.fps, out_fps)[:, 0]
    converged_rs = C.resample_bool(converged, pose.fps, out_fps)

    qvel = C.finite_difference(q, out_fps)
    qvel = C.lowpass_zero_phase(qvel, out_fps, cfg.lowpass_hz, cfg.filter_order)

    link_pos = np.stack([kin.key_body_pos(q[t]) for t in range(len(q))])

    diag = RetargetDiagnostics(
        residual=residual_rs, converged=converged_rs,
        limit_violation_rate=viol_rate, limit_violation_per_joint=viol_joint,
        n_not_converged=int((~converged).sum()), n_foot_distance_fixed=n_foot,
        n_collision_frames=n_coll, bone_scales=scales, finger_backend=backend)

    meta = {
        # From the pose sequence, which is where the truth is: the frontend
        # records its own identity in `pose.meta`. RetargetConfig used to carry a
        # `frontend` field as well, which defaulted to "unknown" and was the
        # answer everybody actually read. Two answers to one question is worse
        # than one.
        "frontend": (dict(pose.meta).get("frontend", "unknown")
                     if pose.meta else "unknown"),
        "model_hash": model.hash(),
        "source_fps": float(pose.fps),
        "config": cfg.as_meta(),
        "diagnostics": diag.summary(),
        "source": dict(pose.meta) if pose.meta else {},
    }
    motion = ReferenceMotion(
        fps=out_fps, dof_names=list(BODY_JOINTS), qpos_ref=q, qvel_ref=qvel,
        finger_names=list(HAND_JOINTS), finger_ref=finger, root_pos=root_pos,
        root_quat=root_quat, link_names=list(KEY_BODIES), link_pos=link_pos,
        valid=valid, meta=meta)
    log.info("retarget: %s", diag.report())
    if diag.summary()["ik_residual_p95"] > cfg.residual_warn:
        log.warning("IK residual p95 %.4f exceeds %.4f; the reference may not "
                    "be worth training on", diag.summary()["ik_residual_p95"],
                    cfg.residual_warn)
    return motion, diag


def _root_trajectory(q: np.ndarray, R_root: np.ndarray,
                     targets: list[_FrameTargets], kin: _RobotKinematics,
                     model: G1Model, scales: dict[str, float]):
    """Pelvis pose in the robot world frame.

    Orientation comes straight out of the solve. Translation is the weakest
    quantity in the whole pipeline, so it is only scaled by the measured leg
    ratio and then shifted once, for the entire clip, to put the median lowest
    foot at the height it has in the default stance. Per frame height correction
    would delete exactly the vertical motion a dance is made of.
    """
    T = len(q)
    leg = float(np.mean([scales[b] for b in
                         ("left_thigh", "right_thigh", "left_shin", "right_shin")]))
    root_pos = np.zeros((T, 3))
    root_quat = np.zeros((T, 4))
    foot_h = np.zeros(T)
    toe_i = [kin.index["left_toe"], kin.index["right_toe"],
             kin.index["left_heel_pt"], kin.index["right_heel_pt"]]
    for t in range(T):
        p = kin.fk(q[t])
        R = R_root[t]
        hip_mid_pelvis = 0.5 * (p[kin.hip_l] + p[kin.hip_r])
        root_pos[t] = targets[t].hip_mid * leg - R @ hip_mid_pelvis
        mujoco.mju_mat2Quat(root_quat[t], np.ascontiguousarray(R).ravel())
        foot_h[t] = (root_pos[t] + p[toe_i] @ R.T)[:, 2].min()

    d = model.data()
    model.reset(d)
    off = foot_points(model)
    ref = min(float((d.xpos[model.body_id(f"{s}_ankle_roll_link")]
                     + d.xmat[model.body_id(f"{s}_ankle_roll_link")].reshape(3, 3)
                     @ off[f"{s}_{k}"])[2])
              for s in ("left", "right") for k in ("toe", "heel_pt"))
    root_pos[:, 2] += ref - float(np.median(foot_h))
    return root_pos, root_quat
