"""The round trip: robot pose to landmarks to robot pose, with no video at all.

A coordinate frame flip or a DoF ordering mistake in the retargeter does not
raise and does not look obviously wrong. It produces a dance that is plausible
and incorrect, and the cost of discovering it is a training run. So the test
here runs the pipeline backwards: sample a valid G1 pose, run forward kinematics,
synthesise MediaPipe-layout landmarks at the anatomical points a real detector
would report, retarget them, and require the original joint angles back.

Two things are provably not recoverable, and both are asserted rather than
assumed away.

1. **The six wrist DoF**, when hand landmarks are absent. The 33 pose landmarks
   end at the wrist, and wrist roll, pitch and yaw move only what is beyond it,
   so they have exactly zero effect on every residual. `test_unobservable_...`
   asserts that the null space is exactly these six directions plus the one
   below, and `test_hand_landmarks_...` asserts that supplying hand landmarks
   collapses it, recovering the wrists to within a degree.

2. **The pelvis pitch gauge.** No landmark observes the pelvis link. Rotating it
   about the hip line while pitching the waist and both hips back by the same
   amount leaves every landmark in exactly the same place, so a one dimensional
   family of joint vectors explains the data equally well. The retargeter picks
   one member of it with its regularisation toward the default stance. That is a
   modelling choice, not an error, and the assertions below therefore measure the
   recovery error with this null space projected out, computing the null space
   numerically from the model rather than hardcoding which joints it touches.

Sampling is uniform in a band around the default stance rather than uniform over
the whole 29 dimensional joint box. A pose drawn uniformly from the box is a
self intersecting contortion which the arms and legs can often reach by a second
exact IK branch, for instance shoulder pitch plus pi with the roll negated; both
branches reproduce the landmarks perfectly and there is nothing to choose between
them. `test_landmarks_are_reproduced_at_large_amplitude` covers those poses with
the assertion that does hold for them, that the landmarks come back.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from g1dance.motion import constraints as C
from g1dance.motion import fingers as F
from g1dance.motion import retarget as R
from g1dance.motion.ir import (MP_POSE, N_HAND_LANDMARKS, N_POSE_LANDMARKS,
                               PoseSequence, ReferenceMotion)
from g1dance.sim.model import BODY_JOINTS, HAND_JOINTS, KEY_BODIES, G1Model

#: which model body each MediaPipe landmark is synthesised from. This is the
#: inverse of the correspondence the retargeter uses, written out separately so
#: that a change to one and not the other shows up as a test failure.
LANDMARK_BODY = {
    "left_shoulder": "left_shoulder_pitch_link",
    "right_shoulder": "right_shoulder_pitch_link",
    "left_elbow": "left_elbow_link", "right_elbow": "right_elbow_link",
    "left_wrist": "left_wrist_roll_link", "right_wrist": "right_wrist_roll_link",
    "left_hip": "left_hip_pitch_link", "right_hip": "right_hip_pitch_link",
    "left_knee": "left_knee_link", "right_knee": "right_knee_link",
    "left_ankle": "left_ankle_roll_link", "right_ankle": "right_ankle_roll_link",
}
#: nose, in the torso frame. There is no neck DoF in the 29, so the head carries
#: no joint information; it is synthesised only so the landmark array is complete
NOSE_IN_TORSO = np.array([0.05, 0.0, 0.42])

WRIST_JOINTS = [f"{s}_wrist_{a}_joint" for s in ("left", "right")
                for a in ("roll", "pitch", "yaw")]
WRIST_INDEX = np.array([BODY_JOINTS.index(j) for j in WRIST_JOINTS])

#: the arm and leg joints of each side, for the per-side comparison below
SIDE_INDEX = {
    f"{side}_{limb}": np.array([BODY_JOINTS.index(f"{side}_{j}_joint")
                                for j in joints])
    for side in ("left", "right")
    for limb, joints in (
        ("arm", ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")),
        ("leg", ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch",
                 "ankle_roll")))
}

#: the post-processing in `constraints` deliberately moves the pose away from the
#: target, to separate the feet and to resolve self collisions. It is switched
#: off for recovery tests and exercised on its own further down.
ROUND_TRIP = dict(control_fps=30.0, min_foot_distance=0.0,
                  resolve_collisions=False, finger_backend="none")
FPS = 30.0


@pytest.fixture(scope="module")
def model() -> G1Model:
    return G1Model()


# --------------------------------------------------------------------------
# synthesis: the inverse of the retargeter
# --------------------------------------------------------------------------

def forward_kinematics(model: G1Model, q: np.ndarray,
                       hand: dict[str, np.ndarray] | None = None):
    """Scratch MjData at a body pose, with the floating base at the identity."""
    d = model.data()
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    d.qpos[model.body_qpos] = q
    for side in ("left", "right"):
        kin = F.hand_kinematics(model, side)
        d.qpos[kin.qadr] = (np.zeros(6) if hand is None else hand[side])
        for dep, ind, coef in kin.mimic:
            x = float(d.qpos[ind])
            d.qpos[dep] = coef[0] + coef[1] * x + coef[2] * x * x
    mujoco.mj_kinematics(model.model, d)
    return d


def anatomical_points(model: G1Model, d) -> dict[str, np.ndarray]:
    """Where a perfect detector would put each landmark, in the pelvis frame."""
    pts = {k: d.xpos[model.body_id(b)].copy() for k, b in LANDMARK_BODY.items()}
    off = R.foot_points(model)
    for side in ("left", "right"):
        bid = model.body_id(f"{side}_ankle_roll_link")
        mat = d.xmat[bid].reshape(3, 3)
        pts[f"{side}_foot_index"] = d.xpos[bid] + mat @ off[f"{side}_toe"]
        pts[f"{side}_heel"] = d.xpos[bid] + mat @ off[f"{side}_heel_pt"]
    tid = model.body_id("torso_link")
    pts["nose"] = d.xpos[tid] + d.xmat[tid].reshape(3, 3) @ NOSE_IN_TORSO
    return pts


def hand_points(model: G1Model, d, side: str) -> np.ndarray:
    """21 MediaPipe-layout hand landmarks from the model's own hand."""
    kin = F.hand_kinematics(model, side)
    p = {n: d.xpos[b] + d.xmat[b].reshape(3, 3) @ loc
         for n, (b, loc) in kin.points.items()}
    lm = np.zeros((N_HAND_LANDMARKS, 3))
    lm[F.MP_HAND_WRIST] = p["wrist"]
    cmc, mcp, ip, tip = F.MP_HAND["thumb"]
    lm[cmc], lm[mcp] = p["thumb_base"], p["thumb_mcp"]
    lm[ip], lm[tip] = p["thumb_pip"], p["thumb_tip"]
    for f in F.CURL_FINGERS:
        a, b, c, e = F.MP_HAND[f]
        lm[a], lm[b], lm[e] = p[f"{f}_mcp"], p[f"{f}_pip"], p[f"{f}_tip"]
        # the human distal joint has no counterpart on a two link robot finger,
        # so it is placed on the segment, where it carries no extra information
        lm[c] = p[f"{f}_pip"] + 0.6 * (p[f"{f}_tip"] - p[f"{f}_pip"])
    return lm


def synth_pose(model: G1Model, q: np.ndarray, n_frames: int = 3,
               rot: np.ndarray | None = None, scale: float = 1.0,
               hand: dict[str, np.ndarray] | None = None) -> PoseSequence:
    """A PoseSequence a perfect detector would produce for a robot at `q`.

    `rot` and `scale` place the subject in an arbitrary orientation and give them
    an arbitrary build, both of which the retargeter is required to be invariant
    to: the first is why every residual is a direction, the second is why every
    length is normalised by a measured bone ratio.
    """
    d = forward_kinematics(model, q, hand)
    pts = anatomical_points(model, d)
    rot = np.eye(3) if rot is None else rot
    hip_mid = 0.5 * (pts["left_hip"] + pts["right_hip"])

    def to_subject(p: np.ndarray) -> np.ndarray:
        return R.R_SUBJECT_TO_ROBOT.T @ (rot @ ((p - hip_mid) * scale))

    frame = np.zeros((N_POSE_LANDMARKS, 3))
    for key, idx in MP_POSE.items():
        frame[idx] = to_subject(pts[key])
    landmarks = np.tile(frame, (n_frames, 1, 1))
    visibility = np.ones((n_frames, N_POSE_LANDMARKS))

    if hand is None:
        left = np.zeros((n_frames, N_HAND_LANDMARKS, 3))
        right = left.copy()
        conf = np.zeros((n_frames, 2))
    else:
        left = np.tile(np.array([to_subject(p)
                                 for p in hand_points(model, d, "left")]),
                       (n_frames, 1, 1))
        right = np.tile(np.array([to_subject(p)
                                  for p in hand_points(model, d, "right")]),
                        (n_frames, 1, 1))
        conf = np.ones((n_frames, 2))
    return PoseSequence(fps=FPS, landmarks=landmarks, visibility=visibility,
                        hand_left=left, hand_right=right, hand_conf=conf,
                        valid=np.ones(n_frames, bool))


def random_valid_pose(model: G1Model, rng: np.random.Generator,
                      amplitude: float = 0.3) -> np.ndarray:
    """A valid pose, uniform in a band of each joint's range around the stance.

    The elbows and knees are held away from full extension. At a straight elbow
    the forearm is parallel to the upper arm, so shoulder yaw rotates the arm
    about its own axis and moves no landmark at all; the same is true of hip yaw
    at a straight knee. Those are kinematic singularities of the human skeleton,
    not of this code, and a test that sampled them would be asserting that an
    unobservable quantity is observable.
    """
    rng_ = model.joint_range(BODY_JOINTS)
    lo, hi = rng_[:, 0], rng_[:, 1]
    q = np.clip(model.default_qpos
                + amplitude * rng.uniform(-1.0, 1.0, len(lo)) * (hi - lo) / 2,
                lo, hi)
    for name in ("left_elbow_joint", "right_elbow_joint",
                 "left_knee_joint", "right_knee_joint"):
        i = BODY_JOINTS.index(name)
        if abs(q[i]) < 0.2:
            q[i] = np.clip(0.2, lo[i], hi[i])
    return q


def random_valid_fingers(model: G1Model, rng: np.random.Generator) -> dict:
    out = {}
    for side in ("left", "right"):
        r = F.hand_kinematics(model, side).ranges
        out[side] = rng.uniform(r[:, 0], r[:, 1])
    return out


# --------------------------------------------------------------------------
# the unobservable subspace, computed rather than assumed
# --------------------------------------------------------------------------

def unobservable_basis(model: G1Model, pose: PoseSequence, q: np.ndarray,
                       cfg, tol: float = 1e-3) -> np.ndarray:
    """Orthonormal basis, over the 29 joints, of directions the data cannot see.

    The singular vectors of the data residual's Jacobian at the true pose whose
    singular value is numerically zero. Deriving it instead of listing joint
    names means a change to the residual set is reflected here automatically.
    """
    kin = R._RobotKinematics(model)
    solver = R._Solver(model, kin, cfg)
    tgt = R._build_targets(pose, cfg)[-1]
    R_ref = tgt.frame @ kin.frame(kin.fk(q)).T
    n_data = solver.n_data + 3 + (12 if tgt.hand_w.any() else 0)

    def data(x):
        return solver.residual(x, tgt, R_ref, x[:R.N_Q], R_ref, 0.0)[:n_data]

    x0 = np.concatenate([q, np.zeros(3)])
    jac = np.zeros((n_data, len(x0)))
    for i in range(len(x0)):
        step = np.zeros(len(x0))
        step[i] = 1e-6
        jac[:, i] = (data(x0 + step) - data(x0 - step)) / 2e-6
    _, sv, vt = np.linalg.svd(jac)
    null = vt[len(sv) - int((sv < tol).sum()):][:, :R.N_Q]
    if not len(null):
        return np.zeros((R.N_Q, 0))
    return np.linalg.svd(null.T, full_matrices=False)[0]


def observable_error(error: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return error - basis @ (basis.T @ error)


# --------------------------------------------------------------------------
# the round trip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hands", [False, True])
def test_round_trip_recovers_observable_joint_angles(model, hands):
    """The measured error distribution is reported in the assertion messages."""
    raw, projected = [], []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        q = random_valid_pose(model, rng)
        rot = _rotation(rng.normal(size=3) * 0.4)
        finger = random_valid_fingers(model, rng) if hands else None
        pose = synth_pose(model, q, rot=rot,
                          scale=float(rng.uniform(0.75, 1.45)), hand=finger)
        motion, diag = R.retarget_with_diagnostics(pose, model, ROUND_TRIP)
        assert diag.summary()["ik_residual_max"] < 1e-2

        error = motion.qpos_ref[-1].astype(float) - q
        raw.append(np.abs(error))
        projected.append(np.abs(observable_error(
            error, unobservable_basis(model, pose, q, R._as_config(ROUND_TRIP)))))

    projected = np.asarray(projected)
    summary = (f"observable joint error over {len(projected)} seeds: "
               f"median {np.median(projected):.2e} "
               f"p95 {np.percentile(projected, 95):.2e} "
               f"max {projected.max():.2e} rad")
    assert np.median(projected) < 2e-3, summary
    assert np.percentile(projected, 95) < 2e-2, summary
    assert projected.max() < 0.06, summary


def test_round_trip_error_is_the_same_on_the_left_and_the_right(model):
    """The G1's arms are mirror images, so a sign convention applied to one side
    and not the other is invisible in the aggregate error and obvious here.

    This exists because the real clip retargets the left arm roughly half as
    faithfully as the right. If that asymmetry were a bug in this module it
    would show up on synthetic landmarks, which are exactly symmetric by
    construction. It does not, so the asymmetry belongs to the clip.
    """
    err = {k: [] for k in SIDE_INDEX}
    for seed in range(20):
        rng = np.random.default_rng(seed)
        q = random_valid_pose(model, rng)
        pose = synth_pose(model, q, rot=_rotation(rng.normal(size=3) * 0.4),
                          scale=float(rng.uniform(0.75, 1.45)))
        error = observable_error(
            R.retarget(pose, model, ROUND_TRIP).qpos_ref[-1].astype(float) - q,
            unobservable_basis(model, pose, q, R._as_config(ROUND_TRIP)))
        for name, idx in SIDE_INDEX.items():
            err[name].append(np.abs(error[idx]))
    med = {k: float(np.median(v)) for k, v in err.items()}
    p95 = {k: float(np.percentile(v, 95)) for k, v in err.items()}
    summary = {"median": med, "p95": p95}
    for limb in ("arm", "leg"):
        lo, hi = sorted((med[f"left_{limb}"], med[f"right_{limb}"]))
        assert hi < 2.0 * lo, summary
        lo, hi = sorted((p95[f"left_{limb}"], p95[f"right_{limb}"]))
        assert hi < 3.0 * lo, summary
        assert p95[f"left_{limb}"] < 2e-2 and p95[f"right_{limb}"] < 2e-2, summary


def test_an_occluded_limb_is_pinned_rather_than_left_to_drift(model):
    """Without data a limb is held by the smoothness term alone, which lets it
    wander through the occlusion and reappear somewhere else. The unobserved
    term pins it, and is identically zero when every landmark is visible."""
    rng = np.random.default_rng(31)
    q = random_valid_pose(model, rng)
    pose = synth_pose(model, q, n_frames=6)
    cfg = R._as_config(ROUND_TRIP)

    visible = R._build_targets(pose, cfg)[-1]
    assert not visible.unobs.any()

    arm = [MP_POSE[k] for k in ("left_shoulder", "left_elbow", "left_wrist")]
    pose.visibility[:, arm] = 0.0
    hidden = R._build_targets(pose, cfg)[-1]
    left = SIDE_INDEX["left_arm"]
    assert np.allclose(hidden.unobs[left], 1.0)
    assert not hidden.unobs[SIDE_INDEX["right_arm"]].any()
    assert not hidden.unobs[SIDE_INDEX["left_leg"]].any()

    # and the solve keeps the blind arm at the stance instead of anywhere
    q_out = R.retarget(pose, model, ROUND_TRIP).qpos_ref[-1].astype(float)
    assert np.abs(q_out[left] - model.default_qpos[left]).max() < 0.1


def test_unobservable_subspace_is_the_wrist_dof_and_the_pelvis_gauge(model):
    """Exactly seven directions, six of them the wrist joints one for one."""
    rng = np.random.default_rng(3)
    q = random_valid_pose(model, rng)
    pose = synth_pose(model, q)
    basis = unobservable_basis(model, pose, q, R._as_config(ROUND_TRIP))
    assert basis.shape[1] == 7

    # each wrist joint lies entirely inside the null space
    for i, name in zip(WRIST_INDEX, WRIST_JOINTS):
        e = np.zeros(R.N_Q)
        e[i] = 1.0
        assert np.linalg.norm(basis @ (basis.T @ e)) > 0.999, name

    # and the seventh direction, with the wrists removed, is the pelvis pitch
    # gauge: waist pitch traded against both hip pitches
    keep = np.setdiff1d(np.arange(R.N_Q), WRIST_INDEX)
    gauge = basis[keep]
    gauge = gauge[:, np.argmax(np.linalg.norm(gauge, axis=0))]
    gauge /= np.linalg.norm(gauge)
    named = dict(zip([BODY_JOINTS[i] for i in keep], gauge))
    for joint in ("waist_pitch_joint", "left_hip_pitch_joint",
                  "right_hip_pitch_joint"):
        assert abs(named[joint]) > 0.2, named
    assert np.sign(named["left_hip_pitch_joint"]) == \
        np.sign(named["right_hip_pitch_joint"])


def test_hand_landmarks_make_the_wrist_dof_observable(model):
    """With hand landmarks the null space collapses to the pelvis gauge alone."""
    rng = np.random.default_rng(5)
    q = random_valid_pose(model, rng)
    finger = random_valid_fingers(model, rng)
    pose = synth_pose(model, q, hand=finger)
    assert unobservable_basis(model, pose, q,
                             R._as_config(ROUND_TRIP)).shape[1] == 1

    motion = R.retarget(pose, model, ROUND_TRIP)
    wrist_error = np.abs(motion.qpos_ref[-1].astype(float) - q)[WRIST_INDEX]
    assert wrist_error.max() < 0.05, dict(zip(WRIST_JOINTS, wrist_error))


def test_wrists_are_held_at_the_default_stance_without_hand_landmarks(model):
    """Unobservable is not the same as arbitrary: they must be pinned, or the
    solve is rank deficient and the reference jitters between frames."""
    rng = np.random.default_rng(7)
    pose = synth_pose(model, random_valid_pose(model, rng))
    motion = R.retarget(pose, model, ROUND_TRIP)
    default = model.default_qpos[WRIST_INDEX]
    assert np.abs(motion.qpos_ref[-1].astype(float)[WRIST_INDEX]
                  - default).max() < 0.05


def test_retarget_is_invariant_to_subject_orientation_and_build(model):
    """Same dancer, turned round and half a head taller, same joint angles.

    This is the assertion that a frame convention error cannot survive: the
    subject frame to robot frame rotation, the per bone scale normalisation and
    the solved pelvis rotation all have to be right together for it to hold.
    """
    rng = np.random.default_rng(11)
    q = random_valid_pose(model, rng)
    pose = synth_pose(model, q)
    basis = unobservable_basis(model, pose, q, R._as_config(ROUND_TRIP))
    base = R.retarget(pose, model, ROUND_TRIP).qpos_ref[-1].astype(float)
    for rotvec, scale in (([0.0, 0.0, 2.5], 1.0), ([0.3, -0.5, 1.2], 0.7),
                          ([0.0, 0.0, 0.0], 1.6)):
        other = R.retarget(
            synth_pose(model, q, rot=_rotation(np.array(rotvec)), scale=scale),
            model, ROUND_TRIP).qpos_ref[-1].astype(float)
        delta = other - base
        # the pelvis gauge is free to land anywhere along its null direction, so
        # only the part of the difference the landmarks can see is asserted here
        assert np.abs(observable_error(delta, basis)).max() < 0.01
        assert np.abs(delta).max() < 0.06


def test_landmarks_are_reproduced_at_large_amplitude(model):
    """Extreme poses may be reached by a second IK branch, but the retargeted
    pose must still explain the landmarks it was given."""
    for seed in range(6):
        rng = np.random.default_rng(100 + seed)
        q = random_valid_pose(model, rng, amplitude=0.7)
        pose = synth_pose(model, q, rot=_rotation(rng.normal(size=3) * 0.5))
        _, diag = R.retarget_with_diagnostics(pose, model, ROUND_TRIP)
        assert diag.summary()["ik_residual_max"] < 0.12
        assert diag.n_not_converged == 0


def _rotation(rotvec: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mat = np.zeros(9)
    mujoco.mju_axisAngle2Quat(quat, rotvec / max(np.linalg.norm(rotvec), 1e-12),
                              float(np.linalg.norm(rotvec)))
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


# --------------------------------------------------------------------------
# the rest of the ReferenceMotion
# --------------------------------------------------------------------------

def test_reference_motion_is_complete_and_model_checked(model, tmp_path):
    rng = np.random.default_rng(13)
    q = random_valid_pose(model, rng)
    pose = synth_pose(model, q, n_frames=40)
    # The frontend label is read off the pose sequence, which is where a frontend
    # records its own identity. RetargetConfig used to carry the field too and it
    # was the one that defaulted to "unknown".
    pose.meta = {**(pose.meta or {}), "frontend": "synthetic"}
    motion = R.retarget(pose, model)

    assert motion.fps == 50.0
    assert motion.n_frames == int(round((40 - 1) / FPS * 50)) + 1
    assert motion.dof_names == list(BODY_JOINTS)
    assert motion.finger_names == list(HAND_JOINTS)
    assert motion.link_names == list(KEY_BODIES)
    assert motion.valid.all()
    motion.check_model(model.hash(), BODY_JOINTS, HAND_JOINTS)
    assert motion.meta["frontend"] == "synthetic"
    assert motion.meta["config"]["control_fps"] == 50.0
    assert set(motion.meta["diagnostics"]) >= {"ik_residual_p95",
                                               "limit_violation_rate"}
    # meta is persisted through repr, so it must contain only literals
    assert ReferenceMotion.load(motion.save(tmp_path / "m.npz")).meta \
        == motion.meta


def test_link_pos_is_in_the_pelvis_frame(model):
    """A link target in world coordinates would make the reward chase the root
    estimate, which is the least reliable quantity in the pipeline."""
    rng = np.random.default_rng(17)
    pose = synth_pose(model, random_valid_pose(model, rng), n_frames=8)
    motion = R.retarget(pose, model, ROUND_TRIP)
    d = forward_kinematics(model, motion.qpos_ref[3].astype(float))
    for k, name in enumerate(KEY_BODIES):
        expect = d.xpos[model.body_id(name)]
        assert np.allclose(motion.link_pos[3, k], expect, atol=1e-5), name
    # the pelvis frame is not the world frame: a wrist is nowhere near the origin
    assert np.abs(motion.link_pos).max() > 0.1


def test_qvel_is_the_derivative_of_qpos(model):
    rng = np.random.default_rng(19)
    q0 = random_valid_pose(model, rng)
    q1 = random_valid_pose(model, np.random.default_rng(20))
    poses = [synth_pose(model, q0 + (q1 - q0) * t, n_frames=1)
             for t in np.linspace(0, 1, 30)]
    pose = PoseSequence(
        fps=FPS, landmarks=np.concatenate([p.landmarks for p in poses]),
        visibility=np.concatenate([p.visibility for p in poses]),
        hand_left=np.concatenate([p.hand_left for p in poses]),
        hand_right=np.concatenate([p.hand_right for p in poses]),
        hand_conf=np.concatenate([p.hand_conf for p in poses]),
        valid=np.ones(30, bool))
    motion = R.retarget(pose, model, ROUND_TRIP | {"control_fps": 50.0})
    finite = C.finite_difference(motion.qpos_ref.astype(float), motion.fps)
    interior = slice(4, -4)
    assert np.abs(motion.qvel_ref[interior] - finite[interior]).max() < 0.5
    assert np.abs(motion.qvel_ref).max() > 0.05


def test_invalid_frames_are_held_and_marked(model):
    rng = np.random.default_rng(23)
    pose = synth_pose(model, random_valid_pose(model, rng), n_frames=10)
    pose.valid[4:7] = False
    motion = R.retarget(pose, model, ROUND_TRIP)
    assert not motion.valid[4:7].any()
    assert motion.valid[[0, -1]].all()
    assert np.isfinite(motion.qpos_ref).all()


def test_diagnostics_are_reported(model):
    rng = np.random.default_rng(29)
    pose = synth_pose(model, random_valid_pose(model, rng), n_frames=6)
    _, diag = R.retarget_with_diagnostics(pose, model, ROUND_TRIP)
    assert diag.residual.shape == diag.converged.shape
    assert 0.0 <= diag.limit_violation_rate <= 1.0
    assert diag.limit_violation_per_joint.shape == (len(BODY_JOINTS),)
    assert "IK residual" in diag.report()
    assert set(diag.bone_scales) == {b.name for b in R.BONES}


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown retarget config"):
        R._as_config({"w_directon": 1.0})


# --------------------------------------------------------------------------
# constraints
# --------------------------------------------------------------------------

def test_clamp_reports_the_violation_rate(model):
    ranges = model.joint_range(BODY_JOINTS)
    q = np.tile(model.default_qpos, (4, 1))
    q[1, 0] = ranges[0, 1] + 1.0
    q[2, 0] = ranges[0, 0] - 1.0
    clamped, rate, per_joint = C.clamp_to_limits(q, ranges)
    assert clamped[1, 0] == ranges[0, 1] and clamped[2, 0] == ranges[0, 0]
    assert rate == pytest.approx(2 / q.size)
    assert per_joint[0] == pytest.approx(0.5)


def test_minimum_inter_foot_distance_is_enforced(model):
    """The fix the article's author needed when their G1 kicked its own feet."""
    q = np.tile(model.default_qpos, (3, 1))
    hip_roll = [BODY_JOINTS.index(f"{s}_hip_roll_joint") for s in ("left", "right")]
    q[:, hip_roll] = [-0.35, 0.35]           # knees crossed over the midline
    before = _foot_gap(model, q[0])
    assert before < 0.17
    fixed, n = C.enforce_min_foot_distance(q, model, min_distance=0.17)
    assert n == 3
    assert _foot_gap(model, fixed[0]) >= 0.17 - 1e-3


def _foot_gap(model: G1Model, q: np.ndarray) -> float:
    d = forward_kinematics(model, q)
    a = d.xpos[model.body_id("left_ankle_roll_link")]
    b = d.xpos[model.body_id("right_ankle_roll_link")]
    return float(np.linalg.norm(a[:2] - b[:2]))


def test_self_collision_is_detected_and_pushed_apart(model):
    q = np.tile(model.default_qpos, (2, 1))
    for side, sign in (("left", 1.0), ("right", -1.0)):
        q[:, BODY_JOINTS.index(f"{side}_shoulder_roll_joint")] = -sign * 1.4
    d = model.data()
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    geom_hand = C._hand_subtree(model)
    assert C.count_self_contacts(model, d, q[0], geom_hand) > 0
    fixed, n_frames = C.resolve_self_collision(q, model)
    assert n_frames == 2
    assert C.count_self_contacts(model, d, fixed[0], geom_hand) == 0


def test_lowpass_is_zero_phase(model):
    t = np.arange(200) / 50.0
    clean = np.sin(2 * np.pi * 0.5 * t)[:, None]
    noisy = clean + 0.05 * np.sin(2 * np.pi * 20.0 * t)[:, None]
    out = C.lowpass_zero_phase(noisy, 50.0, 6.0)
    # interior only: filtfilt's padding leaves a small transient at each end
    assert np.abs(out - clean)[20:-20].max() < 0.005  # noise gone, no phase shift
    assert np.abs(C.lowpass_zero_phase(clean[:3], 50.0, 6.0) - clean[:3]).max() == 0


def test_resample_to_the_control_rate():
    x = np.linspace(0.0, 1.0, 31)[:, None] * np.array([1.0, -2.0])
    out = C.resample(x, 30.0, 50.0)
    assert out.shape == (51, 2)
    assert out[0] == pytest.approx(x[0]) and out[-1] == pytest.approx(x[-1])
    assert np.allclose(out[:, 1], -2.0 * out[:, 0])

    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (31, 1))
    quat[:, 0] = np.cos(np.linspace(0, 1, 31))
    quat[:, 3] = np.sin(np.linspace(0, 1, 31))
    out = C.resample_quat(quat, 30.0, 50.0)
    assert out.shape == (51, 4)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    valid = np.zeros(31, bool)
    valid[10:20] = True
    assert C.resample_bool(valid, 30.0, 50.0).shape == (51,)
    assert not C.resample_bool(valid, 30.0, 50.0)[0]
