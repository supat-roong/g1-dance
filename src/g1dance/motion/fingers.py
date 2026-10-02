"""MediaPipe's 21 hand landmarks to the Inspire hand's 6 independent DoF.

Fingers are expressive garnish, not tracked data. Hand landmarks in real dance
footage drop out constantly, from motion blur, from the hands crossing, and from
the body occluding them, so everything here is written to degrade into a neutral
hand rather than to be precise.

Two backends, selected by config:

* `dexpilot`, via the `dex-retargeting` package, whose Inspire configuration
  solves exactly the six joints this model exposes and whose finger closing prior
  suits expressive poses. The wheel ships no URDFs; they are vendored from
  dex-urdf into `assets/hands/inspire_hand`.
* `coarse`, always available, always built, and what runs when dex-retargeting or
  its URDFs are missing. The pipeline must not depend on an optional package.

The coarse map is deliberately not a tuned heuristic. It measures one
dimensionless quantity on the human hand, the total flexion of a finger from the
metacarpal direction round to the last segment, and then asks the model for the
joint angle that reproduces the same flexion on the robot. That inversion is
calibrated by sweeping the model's own joint through its range with
`mj_kinematics`, so there are no magic constants to drift out of date and the
round trip in tests/test_fingers.py is exact rather than approximate.

The thumb has two DoF and only one curl, so it gets a two parameter least squares
on the thumb direction in the palm frame plus its flexion.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from ..sim.model import FINGER_ORDER, HAND_JOINTS, G1Model
from . import constraints as C

log = logging.getLogger(__name__)

ASSET_HANDS = Path(__file__).resolve().parents[3] / "assets" / "hands"

#: MediaPipe hand landmark indices, per finger: (mcp, pip, dip, tip)
MP_HAND = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
MP_HAND_WRIST = 0
#: the four single-DoF fingers, in the order they appear in FINGER_ORDER
CURL_FINGERS = ("pinky", "ring", "middle", "index")
#: fraction of each joint's range used as the neutral, slightly closed, hand
NEUTRAL_FRACTION = 0.15


def _ang(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _signed_ang(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> float:
    """Angle from a to b about n, in (-pi, pi]."""
    return float(np.arctan2(float(np.dot(np.cross(a, b), n)), float(np.dot(a, b))))


def flexion(p_root: np.ndarray, p_mcp: np.ndarray, p_pip: np.ndarray,
            p_tip: np.ndarray, normal: np.ndarray) -> float:
    """Total turning, in radians, from the metacarpal round to the last segment.

    Two angles rather than one, because a closed Inspire finger turns through
    more than pi and a single angle would fold back on itself. Both are signed
    about the knuckle axis, the line through the metacarpal heads that every
    finger flexes about, because an unsigned angle folds back at zero instead: a
    finger passing through straight would read as flexing again, and the
    calibration curve inverted below would not be monotone there.

    Applied identically to the human landmarks and to the model, which is what
    makes the two comparable despite the human having one more phalanx.
    """
    return (_signed_ang(p_mcp - p_root, p_pip - p_mcp, normal)
            + _signed_ang(p_pip - p_mcp, p_tip - p_pip, normal))


def hand_frame(wrist: np.ndarray, index_mcp: np.ndarray, middle_mcp: np.ndarray,
               pinky_mcp: np.ndarray) -> np.ndarray:
    """Palm frame as columns [x, y, z]: x along the palm, y the knuckle axis
    from the index side to the little finger side, z the palm normal.

    Built identically from human landmarks and from model bodies, so a direction
    expressed in it means the same thing on both sides.
    """
    x = middle_mcp - wrist
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        return np.eye(3)
    x = x / nx
    across = pinky_mcp - index_mcp
    z = np.cross(x, across)
    nz = np.linalg.norm(z)
    if nz < 1e-9:
        return np.eye(3)
    z = z / nz
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


# --------------------------------------------------------------------------
# model side
# --------------------------------------------------------------------------

def mimic_coefficients(model: G1Model) -> dict[str, tuple[str, np.ndarray]]:
    """dependent joint -> (independent joint, polynomial coefficients).

    `mj_kinematics` does not apply equality constraints, so the coupled finger
    joints have to be driven by hand. Read from the model rather than hardcoded
    because these ratios are Unitree's, not ours.
    """
    m = model.model
    out = {}
    for i in range(m.neq):
        if m.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        dep = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.eq_obj1id[i])
        ind = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.eq_obj2id[i])
        out[dep] = (ind, np.array(m.eq_data[i][:5], float))
    return out


def _mesh_tip(model: G1Model, body: str) -> np.ndarray:
    """Local offset of the fingertip, as the mesh vertex farthest from the joint.

    The model is the only place the hand's geometry is written down, so the tip
    is measured from it rather than being another constant to keep in step.
    """
    m = model.model
    bid = model.body_id(body)
    best = np.zeros(3)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        v = v @ R.reshape(3, 3).T + m.geom_pos[g]
        k = int(np.argmax(np.linalg.norm(v, axis=1)))
        if np.linalg.norm(v[k]) > np.linalg.norm(best):
            best = v[k]
    return best


class HandKinematics:
    """Forward kinematics for one Inspire hand, with the mimic joints applied."""

    def __init__(self, model: G1Model, side: str):
        if side not in ("left", "right"):
            raise ValueError(f"side must be left or right, got {side!r}")
        self.g1 = model
        self.m = model.model
        self.side = side
        self.prefix = "L" if side == "left" else "R"
        self.d = model.data()
        self.d.qpos[:] = 0.0
        self.d.qpos[3] = 1.0

        p = self.prefix
        self.joints = [f"{p}_{j}_joint" for j in FINGER_ORDER]
        self.qadr = model.qpos_adr(self.joints)
        self.ranges = model.joint_range(self.joints)

        mimic = mimic_coefficients(model)
        self.mimic = [(int(model.qpos_adr([dep])[0]),
                       int(model.qpos_adr([ind])[0]), coef)
                      for dep, (ind, coef) in mimic.items()
                      if dep.startswith(f"{p}_")]

        self.points: dict[str, tuple[int, np.ndarray]] = {
            "wrist": (model.body_id(f"{side}_wrist_yaw_link"), np.zeros(3))}
        for f in CURL_FINGERS:
            self.points[f"{f}_mcp"] = (model.body_id(f"{p}_{f}_proximal"),
                                       np.zeros(3))
            self.points[f"{f}_pip"] = (model.body_id(f"{p}_{f}_intermediate"),
                                       np.zeros(3))
            self.points[f"{f}_tip"] = (model.body_id(f"{p}_{f}_intermediate"),
                                       _mesh_tip(model, f"{p}_{f}_intermediate"))
        self.points["index_mcp"] = (model.body_id(f"{p}_index_proximal"),
                                    np.zeros(3))
        self.points["middle_mcp"] = (model.body_id(f"{p}_middle_proximal"),
                                     np.zeros(3))
        self.points["thumb_base"] = (model.body_id(f"{p}_thumb_proximal_base"),
                                     np.zeros(3))
        self.points["thumb_mcp"] = (model.body_id(f"{p}_thumb_proximal"),
                                    np.zeros(3))
        self.points["thumb_pip"] = (model.body_id(f"{p}_thumb_intermediate"),
                                    np.zeros(3))
        self.points["thumb_tip"] = (model.body_id(f"{p}_thumb_distal"),
                                    _mesh_tip(model, f"{p}_thumb_distal"))
        self.names = list(self.points)
        self.bids = np.array([self.points[n][0] for n in self.names], int)
        self.local = np.array([self.points[n][1] for n in self.names], float)
        self.calibration = self._calibrate()

    def fk(self, q6: np.ndarray) -> dict[str, np.ndarray]:
        """Positions of the hand's landmark-equivalent points, world frame."""
        self.d.qpos[self.qadr] = q6
        for dep, ind, coef in self.mimic:
            x = float(self.d.qpos[ind])
            self.d.qpos[dep] = coef[0] + coef[1] * x + coef[2] * x * x \
                + coef[3] * x ** 3 + coef[4] * x ** 4
        mujoco.mj_kinematics(self.m, self.d)
        xpos = self.d.xpos[self.bids]
        xmat = self.d.xmat[self.bids].reshape(-1, 3, 3)
        pos = xpos + np.einsum("nij,nj->ni", xmat, self.local)
        return dict(zip(self.names, pos))

    def frame(self, p: dict[str, np.ndarray]) -> np.ndarray:
        return hand_frame(p["wrist"], p["index_mcp"], p["middle_mcp"],
                          p["pinky_mcp"])

    def finger_flexion(self, p: dict[str, np.ndarray], f: str) -> float:
        return flexion(p["wrist"], p[f"{f}_mcp"], p[f"{f}_pip"], p[f"{f}_tip"],
                       self.frame(p)[:, 1])

    def thumb_flexion(self, p: dict[str, np.ndarray]) -> float:
        # unsigned: the thumb does not flex about the palm normal, and this term
        # only has to break the tie between two thumb poses that the direction
        # residual cannot, so a fold back at zero is harmless here
        return (_ang(p["thumb_mcp"] - p["thumb_base"],
                     p["thumb_pip"] - p["thumb_mcp"])
                + _ang(p["thumb_pip"] - p["thumb_mcp"],
                       p["thumb_tip"] - p["thumb_pip"]))

    def thumb_direction(self, p: dict[str, np.ndarray]) -> np.ndarray:
        v = p["thumb_tip"] - p["thumb_base"]
        return self.frame(p).T @ v / max(np.linalg.norm(v), 1e-9)

    def _calibrate(self, n: int = 201) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Flexion as a function of joint angle, for each single-DoF finger."""
        cal = {}
        for i, f in enumerate(CURL_FINGERS):
            lo, hi = self.ranges[i]
            qs = np.linspace(lo, hi, n)
            flex = np.empty(n)
            for k, v in enumerate(qs):
                q6 = np.zeros(6)
                q6[i] = v
                flex[k] = self.finger_flexion(self.fk(q6), f)
            # a left and a right hand flex in opposite senses about their own
            # knuckle axis, so accept either direction and store increasing
            if flex[-1] < flex[0]:
                flex, qs = flex[::-1], qs[::-1]
            if np.any(np.diff(flex) <= 0):
                raise ValueError(
                    f"{self.side} {f} flexion is not monotone in its joint "
                    "angle; the coarse map cannot be inverted")
            cal[f] = (flex, qs)
        return cal

    def neutral(self) -> np.ndarray:
        lo, hi = self.ranges[:, 0], self.ranges[:, 1]
        return lo + NEUTRAL_FRACTION * (hi - lo)


# --------------------------------------------------------------------------
# coarse backend
# --------------------------------------------------------------------------

def _human_flexions(lm: np.ndarray) -> dict[str, float]:
    """Per finger flexion of one frame of MediaPipe hand landmarks."""
    normal = hand_frame(lm[MP_HAND_WRIST], lm[MP_HAND["index"][0]],
                        lm[MP_HAND["middle"][0]], lm[MP_HAND["pinky"][0]])[:, 1]
    out = {}
    for f in CURL_FINGERS:
        mcp, pip, _, tip = MP_HAND[f]
        out[f] = flexion(lm[MP_HAND_WRIST], lm[mcp], lm[pip], lm[tip], normal)
    cmc, mcp, ip, tip = MP_HAND["thumb"]
    out["thumb"] = (_ang(lm[mcp] - lm[cmc], lm[ip] - lm[mcp])
                    + _ang(lm[ip] - lm[mcp], lm[tip] - lm[ip]))
    return out


def coarse_fingers(hand_lm: np.ndarray, conf: np.ndarray, side: str,
                   model: G1Model, cfg=None, kin: HandKinematics | None = None
                   ) -> np.ndarray:
    """(T, 6) Inspire targets for one hand, in FINGER_ORDER, no dependencies.

    Confidence gated: frames below the threshold are replaced by the neutral hand
    before smoothing, so a dropout relaxes the hand instead of freezing whatever
    garbage the detector last produced.
    """
    from .retarget import RetargetConfig, _as_config

    cfg = _as_config(cfg) if not isinstance(cfg, RetargetConfig) else cfg
    kin = kin or hand_kinematics(model, side)
    lm = np.asarray(hand_lm, float)
    T = len(lm)
    out = np.tile(kin.neutral(), (T, 1))
    good = np.asarray(conf, float) >= cfg.hand_conf_threshold
    q_prev = kin.neutral()
    for t in range(T):
        if not good[t] or not np.isfinite(lm[t]).all():
            q_prev = kin.neutral()
            continue
        hf = _human_flexions(lm[t])
        q6 = np.zeros(6)
        for i, f in enumerate(CURL_FINGERS):
            flex, qs = kin.calibration[f]
            q6[i] = float(np.interp(np.clip(hf[f], flex[0], flex[-1]), flex, qs))
        q6[4:] = _solve_thumb(kin, lm[t], hf["thumb"], q6, q_prev[4:])
        out[t] = np.clip(q6, kin.ranges[:, 0], kin.ranges[:, 1])
        q_prev = out[t]
    return out


def _solve_thumb(kin: HandKinematics, lm: np.ndarray, flex_h: float,
                 q6: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Two parameter solve for thumb pitch and yaw.

    The thumb's two DoF are not separable by a curl scalar: pitch flexes it and
    yaw swings it across the palm, and both change where the tip ends up. The
    palm-frame tip direction distinguishes them, the flexion pins the curl.
    """
    F = hand_frame(lm[MP_HAND_WRIST], lm[MP_HAND["index"][0]],
                   lm[MP_HAND["middle"][0]], lm[MP_HAND["pinky"][0]])
    cmc, _, _, tip = MP_HAND["thumb"]
    v = lm[tip] - lm[cmc]
    d_h = F.T @ v / max(np.linalg.norm(v), 1e-9)
    lo, hi = kin.ranges[4:, 0], kin.ranges[4:, 1]

    def residual(x):
        q = q6.copy()
        q[4:] = x
        p = kin.fk(q)
        return np.concatenate([kin.thumb_direction(p) - d_h,
                               [0.3 * (kin.thumb_flexion(p) - flex_h)]])

    x0 = np.clip(start, lo + 1e-6, hi - 1e-6)
    res = least_squares(residual, x0, bounds=(lo, hi), method="trf",
                        xtol=1e-10, ftol=1e-10, gtol=None, max_nfev=120)
    return res.x


# --------------------------------------------------------------------------
# dex-retargeting backend
# --------------------------------------------------------------------------

def dex_available(urdf_dir: Path = ASSET_HANDS) -> bool:
    """True when both the package and the vendored Inspire URDFs are present."""
    try:
        import dex_retargeting  # noqa: F401
    except ImportError:
        return False
    return all((urdf_dir / "inspire_hand" / f"inspire_hand_{s}.urdf").exists()
               for s in ("left", "right"))


def estimate_hand_frame(lm: np.ndarray) -> np.ndarray:
    """dex-retargeting's operator frame, as its examples compute it.

    Kept byte-for-byte equivalent to the upstream example rather than replaced by
    `hand_frame`, because the Inspire URDF's zero pose is expressed in this
    convention and a different frame would quietly rotate every target.
    """
    pts = lm[[0, 5, 9], :]
    x_vector = pts[0] - pts[2]
    centred = pts - pts.mean(axis=0, keepdims=True)
    _, _, v = np.linalg.svd(centred)
    normal = v[2, :]
    x = x_vector - np.dot(x_vector, normal) * normal
    x = x / max(np.linalg.norm(x), 1e-9)
    z = np.cross(x, normal)
    if np.dot(z, centred[1] - centred[2]) < 0:
        normal = -normal
        z = -z
    return np.stack([x, normal, z], axis=1)


@lru_cache(maxsize=4)
def _dex_retargeting(side: str, urdf_dir: str):
    from dex_retargeting.constants import (HandType, RetargetingType, RobotName,
                                           get_default_config_path)
    from dex_retargeting.retargeting_config import RetargetingConfig

    RetargetingConfig.set_default_urdf_dir(urdf_dir)
    hand = HandType.left if side == "left" else HandType.right
    path = get_default_config_path(RobotName.inspire, RetargetingType.dexpilot,
                                   hand)
    return RetargetingConfig.load_from_file(path).build(), hand


def _reset(retgt) -> None:
    """Clear both the warm start and the low pass state.

    `SeqRetargeting.reset` leaves its own filter primed with the previous clip's
    last frame, which would leak one motion into the next in a process that
    retargets more than one.
    """
    retgt.reset()
    if getattr(retgt, "filter", None) is not None:
        retgt.filter.reset()


def dexpilot_fingers(hand_lm: np.ndarray, conf: np.ndarray, side: str,
                     model: G1Model, cfg=None,
                     kin: HandKinematics | None = None,
                     urdf_dir: Path = ASSET_HANDS) -> np.ndarray:
    """(T, 6) Inspire targets for one hand through dex-retargeting's DexPilot."""
    from dex_retargeting.constants import OPERATOR2MANO

    from .retarget import RetargetConfig, _as_config

    cfg = _as_config(cfg) if not isinstance(cfg, RetargetConfig) else cfg
    kin = kin or hand_kinematics(model, side)
    retgt, hand = _dex_retargeting(side, str(urdf_dir))
    idx = retgt.optimizer.target_link_human_indices
    origin, task = np.asarray(idx[0]), np.asarray(idx[1])
    # our joints carry an L_/R_ prefix, dex-retargeting's do not; map by name so
    # that a reordering upstream cannot silently permute the fingers
    take = np.array([retgt.joint_names.index(f"{j}_joint")
                     for j in FINGER_ORDER], int)

    lm = np.asarray(hand_lm, float)
    T = len(lm)
    out = np.tile(kin.neutral(), (T, 1))
    good = np.asarray(conf, float) >= cfg.hand_conf_threshold
    _reset(retgt)
    for t in range(T):
        if not good[t] or not np.isfinite(lm[t]).all():
            _reset(retgt)
            continue
        frame = estimate_hand_frame(lm[t])
        pts = (lm[t] - lm[t][MP_HAND_WRIST]) @ frame @ OPERATOR2MANO[hand]
        ref = pts[task] - pts[origin]
        q = np.asarray(retgt.retarget(ref), float)[take]
        out[t] = np.clip(q, kin.ranges[:, 0], kin.ranges[:, 1])
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

#: (id(model), side) -> (model, kinematics). The model is held so that a freed
#: object cannot have its id reused by a different one.
_KIN_CACHE: dict[tuple[int, str], tuple[G1Model, "HandKinematics"]] = {}


def hand_kinematics(model: G1Model, side: str) -> HandKinematics:
    """Cached per model instance; calibration costs a few hundred FK calls."""
    key = (id(model), side)
    hit = _KIN_CACHE.get(key)
    if hit is not None and hit[0] is model:
        return hit[1]
    kin = HandKinematics(model, side)
    _KIN_CACHE[key] = (model, kin)
    return kin


def retarget_fingers(pose, model: G1Model, cfg=None) -> tuple[np.ndarray, str]:
    """(T, 12) in HAND_JOINTS order, plus the backend that produced it."""
    from .retarget import RetargetConfig, _as_config

    cfg = _as_config(cfg) if not isinstance(cfg, RetargetConfig) else cfg
    backend = cfg.finger_backend
    if backend == "auto":
        backend = "dexpilot" if dex_available() else "coarse"
    if backend == "dexpilot" and not dex_available():
        log.warning("dex-retargeting or its Inspire URDFs are unavailable, "
                    "falling back to the coarse finger map")
        backend = "coarse"

    T = pose.n_frames
    cols = []
    for h, side in enumerate(("left", "right")):
        kin = hand_kinematics(model, side)
        lm = pose.hand_left if side == "left" else pose.hand_right
        conf = pose.hand_conf[:, h]
        if backend == "none":
            q = np.tile(kin.neutral(), (T, 1))
        elif backend == "dexpilot":
            try:
                q = dexpilot_fingers(lm, conf, side, model, cfg, kin)
            except Exception as exc:                  # noqa: BLE001
                log.warning("dexpilot finger retarget failed (%s), using the "
                            "coarse map", exc)
                backend = "coarse"
                q = coarse_fingers(lm, conf, side, model, cfg, kin)
        else:
            q = coarse_fingers(lm, conf, side, model, cfg, kin)
        # heavy smoothing: finger targets are the noisiest signal in the clip
        q = C.lowpass_zero_phase(q, pose.fps, cfg.finger_lowpass_hz)
        cols.append(np.clip(q, kin.ranges[:, 0], kin.ranges[:, 1]))
    out = np.concatenate(cols, axis=1)
    assert out.shape == (T, len(HAND_JOINTS))
    return out, backend
