"""How close is the robot to the person in the video?

`control.evaluate` measures the policy against the retargeted reference, which cannot
see retargeting error: a policy can score perfectly against a reference that
does not resemble the source at all. This decomposes the chain into its two
links and reports them separately.

    source landmarks  --retarget-->  reference  --policy-->  robot

Comparison is on wrist and elbow trajectories in the pelvis frame, scaled by
limb length so a 1.7 m human and a 1.3 m robot are comparable. Absolute
position is meaningless across that gap; shape and timing are the question, so
the headline number is the correlation of each axis over time, with normalised
RMSE reported alongside.

Moved here from `tools/fidelity.py` when the pipeline record had to join these
with the stability metrics into one artifact: a package cannot import from
`tools/`, and a number the platform reports should not live outside it. The two
tools are thin wrappers now, so there is one implementation of the score.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .motion.ir import MP_POSE, PoseSequence, ReferenceMotion
from .motion.retarget import body_frame
from .sim.model import G1Model

#: The IR frame is x right, y up, z toward the camera (see motion/ir.py and
#: the mediapipe frontend). The MuJoCo pelvis frame is x forward, y left, z up.
#: A subject facing the camera has "toward camera" as their forward, so:
#:     robot_x = human_z,  robot_y = human_x,  robot_z = human_y
#: Derived from the conventions, then confirmed by searching all 48
#: permutation and sign combinations: this one scores a mean correlation of
#: 0.464 against 0.045 for the identity, and no other combination beats it.
#: Without it the comparison is meaningless, which is worth stating because a
#: frame bug here reads exactly like a broken retargeter.
IR_TO_PELVIS = (2, 0, 1)


def to_pelvis_axes(v: np.ndarray) -> np.ndarray:
    return v[:, list(IR_TO_PELVIS)]


PAIRS = {"left_wrist": ("left_wrist_yaw_link", "left_wrist"),
         "right_wrist": ("right_wrist_yaw_link", "right_wrist"),
         "left_elbow": ("left_elbow_link", "left_elbow"),
         "right_elbow": ("right_elbow_link", "right_elbow")}


def _pelvis_frame_human(seq: PoseSequence) -> dict[str, np.ndarray]:
    """Human landmarks in the subject's OWN body frame, shoulder-normalised.

    Hip-centring alone is not enough. It leaves the landmarks in camera axes,
    while the robot's are in its own frame, which rotates with the robot. The
    dancer turns through 148 degrees in this clip, so the two frames diverge by
    that much over the take and the comparison measures the turn rather than
    the dance. `body_frame` is the same construction the retargeter uses for
    its pelvis estimate, so this is its exact counterpart on the human side.
    """
    lm = np.asarray(seq.landmarks)
    hl, hr = lm[:, MP_POSE["left_hip"]], lm[:, MP_POSE["right_hip"]]
    sl, sr = lm[:, MP_POSE["left_shoulder"]], lm[:, MP_POSE["right_shoulder"]]
    hip = (hl + hr) * 0.5
    scale = np.linalg.norm((sl + sr) * 0.5 - hip, axis=1)
    scale = np.where(scale > 1e-6, scale, np.nan)
    out = {k: np.empty((len(lm), 3)) for _, k in PAIRS.values()}
    for t in range(len(lm)):
        R = body_frame(hl[t], hr[t], sl[t], sr[t])
        for _, human in PAIRS.values():
            out[human][t] = ((lm[t, MP_POSE[human]] - hip[t]) / scale[t]) @ R
    return out


def _pelvis_frame_robot(g1: G1Model, qpos: np.ndarray) -> dict[str, np.ndarray]:
    """Robot key points in the pelvis frame, normalised by its own torso."""
    d = g1.data()
    torso = g1.body_id("torso_link")
    acc = {b: [] for b, _ in PAIRS.values()}
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_forward(g1.model, d)
        p = d.xpos[g1.pelvis_id]
        R = d.xmat[g1.pelvis_id].reshape(3, 3)
        s = np.linalg.norm(d.xpos[torso] - p) or 1.0
        for body, _ in PAIRS.values():
            acc[body].append((d.xpos[g1.body_id(body)] - p) @ R / s)
    return {k: np.asarray(v) for k, v in acc.items()}


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def _standardize(a: np.ndarray) -> np.ndarray:
    sd = a.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (a - a.mean(0)) / sd


def _score(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mean per-axis correlation, and a scale-free shape distance.

    Both quantities compare `x` against target `y`. A human normalised by
    shoulder width and a robot normalised by torso length do not share a scale,
    so RMSE is taken after standardising each series per axis. Otherwise the
    number is dominated by the unit mismatch rather than by the difference in
    motion, which is what the comparison is for. 0 is identical, and roughly
    1.41 is what two unrelated signals score.
    """
    ok = np.isfinite(x).all(1) & np.isfinite(y).all(1)
    x, y = x[ok], y[ok]
    if len(x) < 8:
        return float("nan"), float("nan")
    cors = []
    for ax in range(3):
        xa, ya = x[:, ax], y[:, ax]
        if xa.std() < 1e-9 or ya.std() < 1e-9:
            continue
        cors.append(float(np.corrcoef(xa, ya)[0, 1]))
    xs, ys = _standardize(x), _standardize(y)
    rmse = float(np.sqrt(((xs - ys) ** 2).sum(1).mean() / 3.0))
    return (float(np.mean(cors)) if cors else float("nan")), rmse


def report(pose_path: Path, motion_path: Path, states: np.ndarray | None,
           g1: G1Model | None = None, start_frame: int = 0) -> dict:
    """`start_frame` is where the policy rollout began.

    Reference state initialisation starts an episode at a random frame, so the
    policy's step k corresponds to reference frame `start_frame + k`, not k.
    Comparing them from index 0 silently measures a time-shifted signal, which
    reads as near-zero correlation and looks exactly like a policy that is not
    tracking at all.
    """
    g1 = g1 or G1Model()
    seq = PoseSequence.load(pose_path)
    motion = ReferenceMotion.load(motion_path)
    human = _pelvis_frame_human(seq)

    # the reference, replayed kinematically at its own frames
    ref_q = np.zeros((motion.n_frames, g1.model.nq))
    ref_q[:, 2] = 0.757
    ref_q[:, 3] = 1.0
    ref_q[:, g1.body_qpos] = motion.qpos_ref
    ref_q[:, g1.hand_qpos] = motion.finger_ref
    robot_ref = _pelvis_frame_robot(g1, ref_q)

    # human is at source fps, robot at motion fps: resample human onto motion
    t_src = np.arange(seq.n_frames) / seq.fps
    t_dst = np.arange(motion.n_frames) / motion.fps
    human_rs = {k: np.stack([np.interp(t_dst, t_src, v[:, i]) for i in range(3)], 1)
                for k, v in human.items()}

    out: dict[str, dict] = {"retarget": {}, "policy": {}, "end_to_end": {}}
    robot_pol = _pelvis_frame_robot(g1, states) if states is not None else None
    for name, (body, hum) in PAIRS.items():
        # retargeting is compared over the whole clip; the policy only over the
        # window it actually ran, offset by where the episode started
        a, b = _align(robot_ref[body], human_rs[hum])
        c, r = _score(a, b)
        out["retarget"][name] = {"corr": c, "nrmse": r}
        if robot_pol is not None:
            ref_w = robot_ref[body][start_frame:]
            hum_w = human_rs[hum][start_frame:]
            a, b = _align(ref_w, robot_pol[body])
            c, r = _score(b, a)
            out["policy"][name] = {"corr": c, "nrmse": r}
            a, b = _align(hum_w, robot_pol[body])
            c, r = _score(b, a)
            out["end_to_end"][name] = {"corr": c, "nrmse": r}
    return out


def fmt(rep: dict) -> str:
    lines = ["  link            stage        corr   shape",
             "  ----------------------------------------"]
    for stage in ("retarget", "policy", "end_to_end"):
        if not rep.get(stage):
            continue
        for name, v in rep[stage].items():
            lines.append(f"  {name:14}  {stage:11} {v['corr']:6.3f} {v['nrmse']:7.3f}")
        vals = [v["corr"] for v in rep[stage].values() if np.isfinite(v["corr"])]
        if vals:
            lines.append(f"  {'MEAN':14}  {stage:11} {np.mean(vals):6.3f}")
        lines.append("")
    return "\n".join(lines)


def average(pose: str, motion: str, config: str, seeds: int = 12,
            min_steps: int = 40, controller=None) -> dict:
    g1 = G1Model()
    # `retarget` is included because the platform record attributes loss per
    # stage, and source-to-reference is the link the other two are measured
    # against. It does not depend on the rollout, so it is constant across seeds
    # and its standard deviation is zero by construction.
    acc: dict[str, list[float]] = {"retarget": [], "policy": [], "end_to_end": []}
    lens = []
    for s in range(seeds):
        st, start = rollout_states(motion, config, seed=s,
                                   controller=controller)
        if len(st) < min_steps:          # too short to correlate meaningfully
            continue
        lens.append(len(st))
        rep = report(Path(pose), Path(motion), st, g1, start_frame=start)
        for stage in acc:
            v = [x["corr"] for x in rep[stage].values() if np.isfinite(x["corr"])]
            if v:
                acc[stage].append(float(np.mean(v)))
    out = {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in acc.items()}
    out["episode_steps"] = (float(np.mean(lens)), float(np.std(lens)), len(lens))
    return out


def rollout_states(*args, **kwargs):
    """Deferred re-export: `viz.render` imports matplotlib-adjacent machinery."""
    from .viz.render import rollout_states as _rs
    return _rs(*args, **kwargs)
