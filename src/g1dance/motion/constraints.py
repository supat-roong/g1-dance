"""Post-processing that turns a per frame IK solution into something trainable.

The IK solves each frame against the human, and nothing else. It has no opinion
about whether the robot's feet pass through each other, whether an arm is inside
the torso, or whether the joint trajectory is smooth enough for a 50 Hz actuator
to follow. Those are properties of the sequence, not of any one frame, so they
are fixed here.

The minimum inter-foot distance is not a theoretical nicety. It is the specific
fix the reference article's author needed after his G1 kicked its own ankles: a
human dancer's feet cross the midline freely because human legs are narrower at
the hip than a G1's, so an unconstrained retarget asks for a pose the robot
physically cannot hold.

Nothing here steps physics. Self collision is read from `mj_forward`, which
evaluates the collision pipeline without integrating anything.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation, Slerp

from ..sim.model import BODY_JOINTS, G1Model

_HIP_ROLL = [BODY_JOINTS.index("left_hip_roll_joint"),
             BODY_JOINTS.index("right_hip_roll_joint")]
_SHOULDER_ROLL = [BODY_JOINTS.index("left_shoulder_roll_joint"),
                  BODY_JOINTS.index("right_shoulder_roll_joint")]
#: sign that abducts each joint, i.e. moves the limb away from the midline
_ABDUCT = np.array([1.0, -1.0])


def clamp_to_limits(q: np.ndarray, ranges: np.ndarray
                    ) -> tuple[np.ndarray, float, np.ndarray]:
    """Clamp and report. The rate is a quality metric, not a nuisance value.

    A high violation rate means the subject is asking for poses the G1 does not
    have, which is information about the clip, so it is surfaced rather than
    silently absorbed.
    """
    q = np.asarray(q, float)
    lo, hi = ranges[:, 0], ranges[:, 1]
    below, above = q < lo - 1e-9, q > hi + 1e-9
    violated = below | above
    per_joint = violated.mean(axis=0) if len(q) else np.zeros(q.shape[1])
    return np.clip(q, lo, hi), float(violated.mean() if q.size else 0.0), per_joint


def _fk(model: G1Model, d, q: np.ndarray) -> None:
    d.qpos[model.body_qpos] = q
    mujoco.mj_kinematics(model.model, d)


def enforce_min_foot_distance(q: np.ndarray, model: G1Model,
                              min_distance: float = 0.17,
                              max_iter: int = 12) -> tuple[np.ndarray, int]:
    """Abduct the hips until the ankles are at least `min_distance` apart.

    Distance is measured horizontally between the two ankle_roll_link origins,
    because the failure being prevented is the feet sweeping through each other
    in the ground plane, not one foot passing over the other.
    """
    q = np.asarray(q, float).copy()
    d = model.data()
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    ids = [model.body_id(f"{s}_ankle_roll_link") for s in ("left", "right")]
    rng = model.joint_range(BODY_JOINTS)
    n_fixed = 0
    for t in range(len(q)):
        fixed = False
        for _ in range(max_iter):
            _fk(model, d, q[t])
            gap = float(np.linalg.norm(d.xpos[ids[0]][:2] - d.xpos[ids[1]][:2]))
            if gap >= min_distance:
                break
            # a hip roll of dtheta moves each ankle about leg_length * dtheta
            leg = max(0.1, abs(float(d.xpos[ids[0]][2])))
            step = min(0.15, (min_distance - gap) / (2.0 * leg))
            q[t, _HIP_ROLL] = np.clip(
                q[t, _HIP_ROLL] + _ABDUCT * step,
                rng[_HIP_ROLL, 0], rng[_HIP_ROLL, 1])
            fixed = True
        n_fixed += int(fixed)
    return q, n_fixed


def _hand_subtree(model: G1Model) -> np.ndarray:
    """Which hand, if any, each geom belongs to: 0 none, 1 left, 2 right."""
    m = model.model
    tag = np.zeros(m.nbody, int)
    for i, side in enumerate(("left", "right")):
        tag[model.body_id(f"{side}_wrist_yaw_link")] = i + 1
    for b in range(1, m.nbody):
        p = m.body_parentid[b]
        if tag[b] == 0 and tag[p] != 0:
            tag[b] = tag[p]
    return tag[m.geom_bodyid]


def count_self_contacts(model: G1Model, d, q: np.ndarray,
                        geom_hand: np.ndarray) -> int:
    """Self collisions on one pose, ignoring the floor and within-hand contacts.

    Fingers touching each other is not a self collision worth fixing; the
    training model disables those geoms anyway, per spec 7.1.
    """
    m = model.model
    d.qpos[model.body_qpos] = q
    d.qpos[2] = 5.0                      # lift clear of the floor plane
    mujoco.mj_forward(m, d)
    n = 0
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if m.geom_bodyid[g1] == 0 or m.geom_bodyid[g2] == 0:
            continue
        if geom_hand[g1] != 0 and geom_hand[g1] == geom_hand[g2]:
            continue
        n += 1
    return n


def resolve_self_collision(q: np.ndarray, model: G1Model, step: float = 0.05,
                           max_iter: int = 32, blend_steps: int = 10
                           ) -> tuple[np.ndarray, int]:
    """Clear self collisions on the frames that have them, cheaply.

    Two stages, in order of how much of the pose they damage. First abduct the
    shoulders, the hips, or both, sweeping the offset outward and keeping the
    configuration with the fewest contacts. The contact count is not monotone in
    that offset, since an arm swung out of the torso passes through the other
    arm on the way, so the whole sweep is searched rather than stopped at the
    first increase. If nothing clears, blend the frame toward the default stance,
    which is collision free by construction, and take the first clear step.

    Deliberately crude. A collision aware IK would be a research project; what
    matters for training is that no reference frame asks the policy to put two
    links in the same place, since that target is unreachable and the tracking
    reward would be saturated there for the whole clip.
    """
    q = np.asarray(q, float).copy()
    d = model.data()
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    geom_hand = _hand_subtree(model)
    rng = model.joint_range(BODY_JOINTS)
    default = np.clip(model.default_qpos, rng[:, 0], rng[:, 1])
    arms, hips = np.array(_SHOULDER_ROLL), np.array(_HIP_ROLL)
    groups = (arms, hips, np.concatenate([arms, hips]))
    n_frames = 0
    for t in range(len(q)):
        best_n = count_self_contacts(model, d, q[t], geom_hand)
        if best_n == 0:
            continue
        n_frames += 1
        best_q = q[t].copy()
        for k in range(1, max_iter + 1):
            for idx in groups:
                trial = q[t].copy()
                sign = _ABDUCT if len(idx) == 2 else np.tile(_ABDUCT, 2)
                trial[idx] = np.clip(trial[idx] + sign * (k * step),
                                     rng[idx, 0], rng[idx, 1])
                n = count_self_contacts(model, d, trial, geom_hand)
                if n < best_n:
                    best_q, best_n = trial, n
            if best_n == 0:
                break
        for k in range(1, blend_steps + 1):
            if best_n == 0:
                break
            a = k / blend_steps
            trial = (1.0 - a) * q[t] + a * default
            n = count_self_contacts(model, d, trial, geom_hand)
            if n < best_n:
                best_q, best_n = trial, n
        q[t] = best_q
    return q, n_frames


def lowpass_zero_phase(x: np.ndarray, fps: float, cutoff_hz: float,
                       order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth along time. Phase lag would misalign the reward."""
    x = np.asarray(x, float)
    if x.shape[0] < 4 or cutoff_hz <= 0:
        return x.copy()
    nyq = 0.5 * fps
    wn = min(0.99, cutoff_hz / nyq)
    if wn >= 0.99:
        return x.copy()
    b, a = signal.butter(order, wn)
    padlen = min(3 * max(len(a), len(b)), x.shape[0] - 1)
    return signal.filtfilt(b, a, x, axis=0, padlen=padlen)


def _out_times(n_in: int, fps_in: float, fps_out: float):
    if n_in < 2:
        return np.zeros(max(1, n_in)), np.zeros(max(1, n_in))
    t_in = np.arange(n_in) / fps_in
    n_out = int(round(t_in[-1] * fps_out)) + 1
    return t_in, np.arange(n_out) / fps_out


def resample(x: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    """Linear resample along time. Joint angles, not orientations."""
    x = np.asarray(x, float)
    if abs(fps_in - fps_out) < 1e-9 or x.shape[0] < 2:
        return x.copy()
    t_in, t_out = _out_times(x.shape[0], fps_in, fps_out)
    flat = x.reshape(x.shape[0], -1)
    out = np.stack([np.interp(t_out, t_in, flat[:, i])
                    for i in range(flat.shape[1])], axis=1)
    return out.reshape((len(t_out),) + x.shape[1:])


def resample_quat(quat: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    """Slerp, in wxyz in and wxyz out. Linear interpolation of quaternions
    would shorten the rotation arc and wobble the root."""
    quat = np.asarray(quat, float)
    if abs(fps_in - fps_out) < 1e-9 or quat.shape[0] < 2:
        return quat.copy()
    t_in, t_out = _out_times(quat.shape[0], fps_in, fps_out)
    r = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    out = Slerp(t_in, r)(np.clip(t_out, t_in[0], t_in[-1])).as_quat()
    return out[:, [3, 0, 1, 2]]


def resample_bool(v: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    """Nearest neighbour, so a resample never invents validity."""
    v = np.asarray(v)
    if abs(fps_in - fps_out) < 1e-9 or v.shape[0] < 2:
        return v.copy()
    t_in, t_out = _out_times(v.shape[0], fps_in, fps_out)
    idx = np.clip(np.round(t_out * fps_in).astype(int), 0, v.shape[0] - 1)
    return v[idx]


def finite_difference(x: np.ndarray, fps: float) -> np.ndarray:
    """Central differences, one sided at the ends, scaled to units per second."""
    x = np.asarray(x, float)
    if x.shape[0] < 2:
        return np.zeros_like(x)
    return np.gradient(x, axis=0) * fps


def limit_root_yaw_rate(quat: np.ndarray, fps: float,
                        max_deg_s: float | None) -> np.ndarray:
    """Cap how fast the root may yaw, leaving roll and pitch alone.

    The retargeted root yaw carries a real signal and an impossible one on top
    of it. On kirari it sweeps 310 degrees, which is the dancer turning, at rates
    reaching 4009 degrees per second, which is nothing: neither a G1 base nor a
    human can turn that fast. The transients are monocular heading noise, and
    they survived all of milestone A because that policy was trained on this
    convention and learned to reproduce it.

    They matter now because a general tracking policy consumes the reference root
    and was trained on real motion capture. Measured with Humanoid-GPT on
    kirari_v4 over 12 seeds, end-to-end similarity against the source video is
    0.503 uncapped, 0.549 at 60 deg/s, 0.517 at 90 and 0.482 at 180, against
    0.562 for discarding the yaw entirely. The cap recovers nearly all of what
    discarding it would, and keeps 239 degrees of the turn instead of none.

    The limit is causal rather than a low pass, so a genuine sustained turn is
    followed at the maximum rate rather than smeared symmetrically across its
    start. Roll and pitch pass through untouched because the reference carries
    the whole quaternion even though the environment reads only the yaw.
    """
    q = np.asarray(quat, np.float64)
    if not max_deg_s or len(q) < 2:
        return q.copy()
    step = np.radians(float(max_deg_s)) / float(fps)

    mats = np.empty((len(q), 3, 3))
    buf9 = np.zeros(9)
    buf4 = np.zeros(4)
    for i in range(len(q)):
        mujoco.mju_quat2Mat(buf9, np.ascontiguousarray(q[i], np.float64))
        mats[i] = buf9.reshape(3, 3)

    yaw = np.unwrap(np.arctan2(mats[:, 1, 0], mats[:, 0, 0]))
    capped = yaw.copy()
    for i in range(1, len(capped)):
        capped[i] = capped[i - 1] + np.clip(yaw[i] - capped[i - 1], -step, step)

    out = np.empty((len(q), 4))
    for i in range(len(q)):
        # strip the original yaw, then reapply the capped one, so whatever roll
        # and pitch the solve produced is preserved exactly
        rest = _rot_z(-yaw[i]) @ mats[i]
        mujoco.mju_mat2Quat(
            buf4, np.ascontiguousarray(_rot_z(capped[i]) @ rest).ravel())
        out[i] = buf4
    return out


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
