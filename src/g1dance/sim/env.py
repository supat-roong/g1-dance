"""The motion tracking environment: 50 Hz control over 200 Hz MuJoCo physics.

One environment serves both modes. Mode A ("upper_body") tracks the 17
waist-and-arm DoF against the reference while the legs balance with no reference
of their own. Mode B ("whole_body") tracks all 29. What differs between them is
which DoF the reward tracks, and that comes from the config; there is a single
control law and a single step function:

    target[d] = ref[d] + scale * action[d]     for the controlled body DoF
                ref[d] = the default stance where no reference exists
    target[f] = finger_ref[f]                  fingers, open loop, never acted

The fingers deserve their own sentence because leaving them out is the whole
reason adding Inspire hands costs the RL problem nothing. They are low inertia
and touch nothing during a dance, so they are played back open loop and appear
in neither the action space nor the observation. The action stays at 29 body
DoF rather than 41, exactly as it would be without hands.

Two performance commitments shape the code, because the target is a 5M step run
in minutes rather than hours on a laptop CPU:

  * The reference motion is resampled to the control rate **once**, at
    construction, into flat contiguous arrays. Nothing per step interpolates,
    searches or rebuilds anything.
  * Observations are assembled into a preallocated buffer by slice assignment.
    The upcoming-reference window is a single contiguous slice of a padded
    feature matrix, not a loop over frames.

The observation layout is deliberately the motion-conditioned one: the policy
sees a window of upcoming reference frames rather than a single target, so a
generalist policy conditioned on many motions stays reachable without
redesigning the observation.

What separates mode A from mode B is the **reward**, not the action space. That
is a correction to the original design, which fixed mode A's action at the 17
tracked DoF, and it was forced by measurement: with the legs held by position PD
alone this G1 does not stand. From the default stance with fixed targets and no
motion playing it passes an upright dot of 0.5 at 1.4 s and is on the floor by
2 s. The support polygon is not the reason, the CoM sits well inside it; the
ankle and waist PD simply cannot reject their own steady-state error, and the
waist pitch alone sags 0.42 rad under the weight of the torso and arms, which
walks the CoM forward. Trained anyway at 17 DoF the policy went nowhere: a mean
episode length of 52.7 control steps against 48.8 for a zero action after 100k
steps, and still ~57 at 547k. Given the legs it climbed from 67 to 105 control
steps inside 1M steps and was still rising.

So `env.control_set` selects what the action drives, independently of what the
reward tracks. `all`, the default, gives the policy every body DoF, which in
mode A means the 12 leg DoF as a residual around the default stance while the
tracking reward still sees only the upper 17; that is the article's IK-upper-body
over RL-lower-body decomposition. `tracked` restricts the action to the reward's
own set, giving a 17-DoF action in mode A, and is kept because it is what the
original design specified and because it is the control that demonstrates why
the legs have to be in the action. In both settings `len(env.tracked)` is 17 in
mode A and 29 in mode B, which is the invariant that actually distinguishes the
modes.

Two heights are measured from the model rather than guessed, since both were
wrong in the first draft. `spawn_height` 0.757 is the pelvis height at which the
lowest collidable point of the default stance sits exactly at z=0, found by
bisecting on MuJoCo's own floor contact distances and confirmed against the foot
sphere geometry. `nominal_height` 0.715 is where the pelvis actually settles once
gravity loads the leg PD, measured by holding the root level and in place so the
sag is isolated from the tipping and running to equilibrium. The earlier 0.78
dropped the robot 2.3 cm onto the floor at every reset, and the earlier 0.69
aimed the height reward 2.5 cm below anywhere it can stand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from g1dance.motion.ir import ReferenceMotion
from g1dance.sim.model import BODY_JOINTS, HAND_JOINTS, KEY_BODIES, G1Model
from g1dance.sim.rewards import RewardContext, RewardFunction

CONFIGS = Path(__file__).resolve().parents[3] / "configs"
_EMPTY: dict = {}

DEFAULT_ENV_CFG: dict = {
    "decimation": 4,
    "window": 4,
    "control_set": "all",           # "tracked" | "all"  -- see the note below
    "finger_collisions": False,     # spec 7.1: off for training, on to render
    "action_scale": 1.0,
    "max_episode_steps": 500,
    "spawn_height": 0.757,
    "nominal_height": 0.715,
    "root_ref": "nominal",          # "nominal" | "motion"
    "track_root_xy": False,
    "link_weight": {},              # key body name -> weight, default below
    "termination": {
        "min_height": 0.45,
        "min_upright": 0.5,
        "max_joint_error": 1.2,
        "max_link_error": 0.8,
    },
    "domain_rand": {
        "enabled": True,
        "friction": [0.7, 1.3],
        "mass_scale": [0.9, 1.1],
        "kp_scale": [0.9, 1.1],
        "kv_scale": [0.9, 1.1],
        "qpos_noise": 0.03,
        "qvel_noise": 0.15,
        "push_interval_s": 3.0,
        "push_vel": 0.25,
    },
}


def load_config(path: str | Path | dict) -> dict:
    """Read a mode config. A dict passes through, which keeps tests cheap."""
    if isinstance(path, dict):
        return path
    import yaml
    p = Path(path)
    if not p.exists() and (CONFIGS / str(path)).exists():
        p = CONFIGS / str(path)
    return yaml.safe_load(p.read_text())


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(
            base.get(k), dict) else v
    return out


# --------------------------------------------------------------------- #
# reference preparation
# --------------------------------------------------------------------- #
@dataclass
class _Track:
    """A ReferenceMotion resampled to the control rate and pre-flattened.

    Everything the step loop reads from the reference is here, already in
    BODY_JOINTS / KEY_BODIES order and already at 50 Hz, so the hot path is
    pure indexing.
    """

    rate: float
    ref_q: np.ndarray               # (T, 29) control reference, defaults filled
    ref_qd: np.ndarray              # (T, 29)
    finger: np.ndarray              # (T, 12)
    link: np.ndarray                # (T, K, 3) pelvis frame
    root_pos: np.ndarray            # (T, 3)
    root_yaw: np.ndarray            # (T,)
    valid: np.ndarray               # (T,) bool
    segments: list[tuple[int, int]]
    window: np.ndarray              # (T + W + 1, n_tracked + 3K) padded


def _interp(src: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear resample along axis 0, keeping the trailing shape."""
    flat = src.reshape(len(src), -1)
    out = np.empty((len(t_dst), flat.shape[1]), np.float64)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, flat[:, j])
    return out.reshape((len(t_dst),) + src.shape[1:])


def _yaw_from_quat(q: np.ndarray) -> np.ndarray:
    """Yaw of a wxyz quaternion array, unwrapped so it can be interpolated."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.unwrap(np.arctan2(2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z)))


def prepare_track(motion: ReferenceMotion, g1: G1Model, mode: str, rate: float,
                  window: int, link_weight: np.ndarray) -> _Track:
    """Resample a motion to the control rate, once, at construction time."""
    tracked = g1.tracked_index(mode)
    t_src = np.arange(motion.n_frames, dtype=np.float64) / motion.fps
    n = max(int(np.floor(t_src[-1] * rate)) + 1, 2)
    t_dst = np.arange(n, dtype=np.float64) / rate

    q = _interp(motion.qpos_ref, t_src, t_dst)
    qd = _interp(motion.qvel_ref, t_src, t_dst)
    finger = _interp(motion.finger_ref, t_src, t_dst)
    root_pos = _interp(motion.root_pos, t_src, t_dst)
    yaw = np.interp(t_dst, t_src, _yaw_from_quat(motion.root_quat))

    # A resampled frame is valid only if both of the source frames it was
    # blended from are valid. Interpolating across a gap would invent motion
    # exactly where the tracker admitted it had none.
    src_i = np.clip(np.searchsorted(t_src, t_dst, "right") - 1, 0,
                    motion.n_frames - 1)
    src_j = np.minimum(src_i + 1, motion.n_frames - 1)
    valid = motion.valid[src_i] & motion.valid[src_j]

    # Key link positions, reordered from the motion's own link ordering into
    # KEY_BODIES order. A key body the motion does not carry gets zero weight.
    k = len(KEY_BODIES)
    link = np.zeros((n, k, 3), np.float64)
    name_to_col = {nm: i for i, nm in enumerate(motion.link_names)}
    have = np.zeros(k, bool)
    for i, nm in enumerate(KEY_BODIES):
        if nm in name_to_col:
            link[:, i] = _interp(motion.link_pos[:, name_to_col[nm]], t_src,
                                 t_dst)
            have[i] = True
    link_weight = np.where(have, link_weight, 0.0)

    # Untracked DoF have no reference, so the control reference for them is the
    # default stance (spec 7.2). Doing the fill here rather than per step is
    # what makes the two modes share one control law at no runtime cost.
    ref_q = np.repeat(g1.default_qpos[None, :], n, axis=0)
    ref_qd = np.zeros((n, len(BODY_JOINTS)), np.float64)
    ref_q[:, tracked] = q[:, tracked]
    ref_qd[:, tracked] = qd[:, tracked]

    feat = np.concatenate([ref_q[:, tracked], link.reshape(n, -1)], axis=1)
    pad = np.repeat(feat[-1:], window + 1, axis=0)
    win = np.ascontiguousarray(np.concatenate([feat, pad], axis=0), np.float32)

    return _Track(
        rate=rate, ref_q=np.ascontiguousarray(ref_q),
        ref_qd=np.ascontiguousarray(ref_qd),
        finger=np.ascontiguousarray(finger),
        link=np.ascontiguousarray(link), root_pos=np.ascontiguousarray(root_pos),
        root_yaw=np.ascontiguousarray(yaw), valid=valid,
        segments=_segments(valid, max(window + 2, 4)), window=win)


def _segments(valid: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(valid) - start >= min_len:
        out.append((start, len(valid)))
    return out


def synthetic_motion(n_frames: int = 240, fps: float = 30.0,
                     amplitude: float = 0.5, period_s: float = 2.0,
                     g1: G1Model | None = None,
                     gap: tuple[int, int] | None = None) -> ReferenceMotion:
    """A slow sinusoidal arm wave around the default stance.

    Every test and smoke run needs *a* reference motion, and requiring a real
    video for that would make the environment untestable in CI and couple the
    RL half of the pipeline to the video half it was explicitly designed not to
    depend on. This produces a kinematically legal, smooth, exactly known
    motion with correct key-link positions from forward kinematics, so a test
    can assert tracking behaviour against ground truth.
    """
    g1 = g1 or G1Model()
    m, d = g1.model, g1.data()
    t = np.arange(n_frames, dtype=np.float64) / fps
    w = 2.0 * np.pi / period_s
    q = np.repeat(g1.default_qpos[None, :], n_frames, axis=0)
    qd = np.zeros_like(q)
    swing = {"left_shoulder_pitch_joint": 1.0, "right_shoulder_pitch_joint": 1.0,
             "left_elbow_joint": 0.6, "right_elbow_joint": -0.6,
             "waist_yaw_joint": 0.3}
    col = {nm: i for i, nm in enumerate(BODY_JOINTS)}
    for nm, gain in swing.items():
        j = col[nm]
        q[:, j] += amplitude * gain * np.sin(w * t)
        qd[:, j] = amplitude * gain * w * np.cos(w * t)
    lo, hi = g1.joint_range(BODY_JOINTS).T
    q = np.clip(q, lo + 1e-3, hi - 1e-3)

    link = np.zeros((n_frames, len(KEY_BODIES), 3), np.float64)
    for i in range(n_frames):
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = [0.0, 0.0, 0.793]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[g1.body_qpos] = q[i]
        mujoco.mj_kinematics(m, d)
        link[i] = d.xpos[g1.key_body_ids] - d.xpos[g1.pelvis_id]

    finger = 0.5 * (1.0 + np.sin(w * t))[:, None] * np.ones(len(HAND_JOINTS))
    valid = np.ones(n_frames, bool)
    if gap is not None:
        valid[gap[0]:gap[1]] = False
    root = np.zeros((n_frames, 3))
    root[:, 2] = 0.793
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, 1))
    return ReferenceMotion(
        fps=fps, dof_names=list(BODY_JOINTS), qpos_ref=q, qvel_ref=qd,
        finger_names=list(HAND_JOINTS), finger_ref=finger, root_pos=root,
        root_quat=quat, link_names=list(KEY_BODIES), link_pos=link, valid=valid,
        meta={"source": "synthetic"})


# --------------------------------------------------------------------- #
# the environment
# --------------------------------------------------------------------- #
class DanceTrackEnv:
    """Track a ReferenceMotion at 50 Hz. See the module docstring for the law."""

    def __init__(self, motion: str | Path | ReferenceMotion,
                 config: str | Path | dict | None = None,
                 mode: str | None = None, seed: int = 0,
                 randomize: bool | None = None) -> None:
        cfg = load_config(config) if config is not None else {}
        self.mode = mode or cfg.get("mode", "upper_body")
        self.cfg = _merge(DEFAULT_ENV_CFG, cfg.get("env", {}))
        if randomize is not None:
            self.cfg["domain_rand"]["enabled"] = bool(randomize)

        self.g1 = G1Model()
        self.model = self.g1.model
        self.data = self.g1.data()
        self.rng = np.random.default_rng(seed)

        self.decimation = int(self.cfg["decimation"])
        self.dt = self.model.opt.timestep * self.decimation
        self.control_rate = 1.0 / self.dt
        self.window = int(self.cfg["window"])

        self.tracked = self.g1.tracked_index(self.mode)
        self.n_body = len(BODY_JOINTS)
        # Which DoF the action drives, as opposed to which the reward tracks.
        # These are different questions and the module docstring records the
        # measurement that forced them apart.
        cs = str(self.cfg.get("control_set", "all"))
        if cs not in ("tracked", "all"):
            raise ValueError(f"control_set must be 'tracked' or 'all', got {cs!r}")
        self.controlled = (np.arange(self.n_body) if cs == "all"
                           else self.tracked)
        self.act_dim = len(self.controlled)

        if isinstance(motion, (str, Path)):
            motion = ReferenceMotion.load(motion)
        motion.check_model(self.g1.hash(), BODY_JOINTS, HAND_JOINTS)
        lw_cfg = self.cfg.get("link_weight") or {}
        base_lw = 0.0 if self.mode == "upper_body" else 1.0
        upper_links = {"left_wrist_yaw_link", "right_wrist_yaw_link",
                       "left_elbow_link", "right_elbow_link", "torso_link"}
        link_weight = np.array(
            [float(lw_cfg.get(nm, 1.0 if nm in upper_links else base_lw))
             for nm in KEY_BODIES], float)
        self.link_weight = link_weight
        self.track = prepare_track(motion, self.g1, self.mode,
                                   self.control_rate, self.window, link_weight)
        if not self.track.segments:
            raise ValueError(
                "motion has no valid segment long enough to train on "
                f"(need {self.window + 2} control frames)")
        self.reward_fn = RewardFunction.from_config(cfg.get("rewards", {}))

        # index maps and pristine model parameters, cached once
        self.body_qpos = self.g1.body_qpos
        self.body_dof = self.g1.body_dof
        self.body_act = self.g1.body_act
        self.hand_qpos = self.g1.hand_qpos
        self.hand_act = self.g1.hand_act
        self.key_ids = self.g1.key_body_ids
        self.pelvis_id = self.g1.pelvis_id
        self.foot_ids = np.array([self.g1.body_id("left_ankle_roll_link"),
                                  self.g1.body_id("right_ankle_roll_link")], int)
        self.floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                            "floor")
        self._geom_contype0 = self.model.geom_contype.copy()
        self._geom_conaff0 = self.model.geom_conaffinity.copy()
        gb = self.model.geom_bodyid
        self.foot_geoms = (np.isin(gb, self.foot_ids)).nonzero()[0]
        self._is_foot_geom = np.zeros(self.model.ngeom, bool)
        self._is_foot_geom[self.foot_geoms] = True
        self._foot_side = np.full(self.model.ngeom, -1, np.int64)
        for s, bid in enumerate(self.foot_ids):
            self._foot_side[gb == bid] = s
        self._n_finger_geoms = self._set_finger_collisions(
            bool(self.cfg.get("finger_collisions", False)))
        self._m0 = {
            "mass": self.model.body_mass.copy(),
            "friction": self.model.geom_friction.copy(),
            "gain": self.model.actuator_gainprm.copy(),
            "bias": self.model.actuator_biasprm.copy(),
        }
        self.joint_range = self.g1.joint_range(BODY_JOINTS)
        self.action_scale = (self.g1.action_scales
                             * float(self.cfg["action_scale"]))
        self.default_qpos = self.g1.default_qpos.copy()
        # one weight per TRACKED DoF, which is not the action dimension
        # once `control_set: all` widens the action
        self.joint_weight = np.ones(len(self.tracked))

        # observation layout, fixed at construction
        w_feat = self.track.window.shape[1]
        self.obs_dim = (2 * self.n_body + 3 + 3 + self.act_dim + 2 + 2
                        + self.window * w_feat)
        self.priv_extra = 3 + 1 + 6 + 5
        self.priv_dim = self.obs_dim + self.priv_extra
        self._obs = np.zeros(self.obs_dim, np.float32)
        self._priv = np.zeros(self.priv_dim, np.float32)
        self._w_feat = w_feat

        # per-step scratch, allocated once
        self._target = np.empty(self.n_body)
        self._action = np.zeros(self.act_dim)
        self._prev_action = np.zeros(self.act_dim)
        #: fraction of joint targets that hit a limit on the last step, and the
        #: running sum over the episode. See `step`.
        self.target_clip_frac = 0.0
        self._clip_sum = 0.0
        self._foot_pos = np.zeros((2, 3))
        self._foot_vel = np.zeros((2, 3))
        self._foot_contact = np.zeros(2, bool)
        self._foot_force = np.zeros(6)
        self._dr = np.zeros(5)
        self._cf = np.zeros(6)
        self._vel6 = np.zeros(6)
        self._jac = np.zeros((3, self.model.nv))
        self._ctx = None
        self.frame = 0
        self.seg = self.track.segments[0]
        self.t = 0
        self.reset()

    # ----------------------------------------------------------------- #
    def spec_dims(self) -> tuple[int, int, int]:
        return self.obs_dim, self.priv_dim, self.act_dim

    def seed(self, s: int) -> None:
        self.rng = np.random.default_rng(s)

    def close(self) -> None:
        pass

    def _set_finger_collisions(self, enabled: bool) -> int:
        """Spec 7.1's training model: the finger collision geoms are disabled.

        Fingers contact nothing in a dance, so their geoms are pure cost. They
        are worse than free here. The grafted Inspire thumb intersects its own
        wrist link in the model's rest pose, and with collisions on that shows
        up as roughly 3.4 robot-on-robot contacts per control step, charged to
        `self_collision` at every step of every episode, for a geometry
        artifact the policy has no action over because the fingers are open
        loop. Measured on `motion/kirari.npz`: 115 of 147 non-floor contacts
        over 30 steps were a finger geom against its own wrist.

        Finger bodies are found by ancestry rather than by name, because the
        Unitree meshes are named after files like `Link11_L` and carry no
        finger naming.
        """
        m = self.model
        roots = {int(m.jnt_bodyid[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in HAND_JOINTS}
        finger = np.zeros(m.nbody, bool)
        for b in range(m.nbody):
            a = b
            while a > 0:
                if a in roots:
                    finger[b] = True
                    break
                a = int(m.body_parentid[a])
        geoms = finger[m.geom_bodyid].nonzero()[0]
        if enabled:
            m.geom_contype[geoms] = self._geom_contype0[geoms]
            m.geom_conaffinity[geoms] = self._geom_conaff0[geoms]
        else:
            m.geom_contype[geoms] = 0
            m.geom_conaffinity[geoms] = 0
        return len(geoms)

    # ----------------------------------------------------------------- #
    def _randomize(self) -> None:
        dr = self.cfg["domain_rand"]
        m, m0 = self.model, self._m0
        if not dr.get("enabled", True):
            m.body_mass[:] = m0["mass"]
            m.geom_friction[:] = m0["friction"]
            m.actuator_gainprm[:] = m0["gain"]
            m.actuator_biasprm[:] = m0["bias"]
            self._dr[:] = [m0["friction"][0, 0], 1.0, 1.0, 1.0, 0.0]
            self._push_vel = 0.0
            self._push_every = 1 << 30
            return
        u = self.rng.uniform
        fric = u(*dr["friction"])
        mass = u(*dr["mass_scale"])
        kp = u(*dr["kp_scale"])
        kv = u(*dr["kv_scale"])
        m.geom_friction[:, 0] = m0["friction"][:, 0] * fric
        m.body_mass[:] = m0["mass"] * mass
        # position actuators: gain = kp, bias = (0, -kp, -kv)
        m.actuator_gainprm[:, 0] = m0["gain"][:, 0] * kp
        m.actuator_biasprm[:, 1] = m0["bias"][:, 1] * kp
        m.actuator_biasprm[:, 2] = m0["bias"][:, 2] * kv
        # body_mass feeds derived quantities (subtree masses, inertia scaling)
        # that MuJoCo caches; without this the change is half applied.
        mujoco.mj_setConst(m, self.data)
        self._push_vel = float(dr["push_vel"])
        self._push_every = max(int(float(dr["push_interval_s"])
                                   * self.control_rate), 1)
        self._dr[:] = [fric, mass, kp, kv, self._push_vel]

    def reset(self, start_frame: int | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """Reference state initialization: a random frame in a random segment.

        `start_frame` pins the frame instead, for playing a clip from a chosen
        point; it must lie in a valid segment, because a frame outside one has
        no reference to start from.
        """
        self._randomize()
        segs = self.track.segments
        if start_frame is None:
            lens = np.array([b - a for a, b in segs], float)
            seg = segs[int(self.rng.choice(len(segs), p=lens / lens.sum()))]
            lo, hi = seg[0], max(seg[1] - 2, seg[0])
            frame = int(self.rng.integers(lo, hi + 1))
        else:
            frame = int(start_frame)
            seg = next((s for s in segs if s[0] <= frame < s[1] - 1), None)
            if seg is None:
                raise ValueError(
                    f"start_frame {frame} is in no valid segment; "
                    f"segments are {segs}")
        self.seg = seg
        self.frame = frame

        d, g1 = self.data, self.g1
        mujoco.mj_resetData(self.model, d)
        dr = self.cfg["domain_rand"]
        qn = float(dr["qpos_noise"]) if dr.get("enabled", True) else 0.0
        vn = float(dr["qvel_noise"]) if dr.get("enabled", True) else 0.0
        f = self.frame
        if self.cfg["root_ref"] == "motion":
            d.qpos[:2] = self.track.root_pos[f, :2]
            d.qpos[2] = self.track.root_pos[f, 2]
            yaw = self.track.root_yaw[f]
            d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        else:
            d.qpos[:3] = [0.0, 0.0, float(self.cfg["spawn_height"])]
            d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        q = self.track.ref_q[f].copy()
        if qn:
            q += self.rng.normal(0.0, qn, self.n_body)
        np.clip(q, self.joint_range[:, 0], self.joint_range[:, 1], out=q)
        d.qpos[self.body_qpos] = q
        d.qpos[self.hand_qpos] = 0.0
        if vn:
            d.qvel[self.body_dof] = (self.track.ref_qd[f]
                                     + self.rng.normal(0.0, vn, self.n_body))
        else:
            d.qvel[self.body_dof] = self.track.ref_qd[f]
        d.ctrl[self.body_act] = q
        d.ctrl[self.hand_act] = self.track.finger[f]
        mujoco.mj_forward(self.model, d)

        self._action[:] = 0.0
        self._prev_action[:] = 0.0
        self.target_clip_frac = 0.0
        self._clip_sum = 0.0
        self.t = 0
        self._build_ctx()
        return self._observe()

    # ----------------------------------------------------------------- #
    def step(self, action: np.ndarray, absolute: bool = False) -> tuple:
        """One control step.

        `action` is a residual the policy adds to the reference, clipped to the
        unit box, which is what training wants. `absolute=True` instead takes
        29 joint position targets directly, which is what an external tracking
        policy emits: it works in its own units and owns the whole target, so
        putting it through the residual path would clip it to one action scale
        either side of the reference. That does not fail, it saturates quietly.
        """
        nxt = min(self.frame + 1, len(self.track.ref_q) - 1)
        tgt = self._target
        self._prev_action[:] = self._action
        if absolute:
            tgt[:] = np.asarray(action, np.float64)
            # Report the implied residual so the action-rate penalty and the
            # last-action observation channel stay meaningful for a controller
            # that never speaks in residuals.
            implied = ((tgt[self.controlled] - self.track.ref_q[nxt][self.controlled])
                       / self.action_scale[self.controlled])
            self._action[:] = np.clip(implied, -1.0, 1.0)
        else:
            a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
            self._action[:] = a
            tgt[:] = self.track.ref_q[nxt]
            tgt[self.controlled] += self.action_scale[self.controlled] * a
        # Saturation is counted, not absorbed. An external controller whose
        # targets are out of range looks fine and dances wrong, so the rate is
        # reported rather than silently applied.
        lo, hi = self.joint_range[:, 0], self.joint_range[:, 1]
        n_clipped = int(np.count_nonzero((tgt < lo) | (tgt > hi)))
        self.target_clip_frac = n_clipped / max(len(tgt), 1)
        self._clip_sum += self.target_clip_frac
        np.clip(tgt, lo, hi, out=tgt)

        d = self.data
        d.ctrl[self.body_act] = tgt
        d.ctrl[self.hand_act] = self.track.finger[nxt]
        if self._push_vel and self.t and self.t % self._push_every == 0:
            d.qvel[:2] += self.rng.normal(0.0, self._push_vel, 2)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, d)

        self.frame = nxt
        self.t += 1
        self._build_ctx()
        rew = float(self.reward_fn(self._ctx))
        term, reason = self._terminated()
        trunc = (not term) and (self.frame + 1 >= self.seg[1]
                                or self.t >= int(self.cfg["max_episode_steps"]))
        obs, priv = self._observe()
        # An `info` dict is built only when the episode ends. On every other
        # step it would be a dict allocated, pickled and sent down the pipe for
        # nothing, eight times per control step.
        info = _EMPTY
        if term or trunc:
            info = {"frame": self.frame, "ep_len": self.t,
                    "joint_err": self._joint_err, "link_err": self._link_err,
                    "target_clip_frac": self._clip_sum / max(self.t, 1),
                    "reason": reason if term else "truncated"}
        return obs, priv, rew, term, trunc, info

    # ----------------------------------------------------------------- #
    def _build_ctx(self) -> None:
        d, m = self.data, self.model
        f = self.frame
        q = d.qpos[self.body_qpos]
        qd = d.qvel[self.body_dof]
        quat = d.qpos[3:7]
        R = d.xmat[self.pelvis_id].reshape(3, 3)
        self._upright = float(R[2, 2])
        # world gravity (0, 0, -1) expressed in the pelvis frame is -R[2, :]
        self._gravity = -R[2, :]
        # a free joint stores angular velocity in the body frame already and
        # linear velocity in the world frame, so only the latter is rotated
        self._ang_vel = d.qvel[3:6]
        self._lin_vel = R.T @ d.qvel[0:3]
        yaw = float(np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                               1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2)))
        self._yaw = yaw

        pelvis = d.xpos[self.pelvis_id]
        link = (d.xpos[self.key_ids] - pelvis) @ R      # into the pelvis frame
        self._foot_pos[:] = d.xpos[self.foot_ids]
        for s_, bid in enumerate(self.foot_ids):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(bid),
                                     self._vel6, 0)
            self._foot_vel[s_] = self._vel6[3:]
        self._contacts()

        ref_link = self.track.link[f]
        dl = link - ref_link
        self._link_err = float(np.sqrt(
            np.einsum("ij,ij->i", dl, dl) @ self.link_weight
            / max(self.link_weight.sum(), 1e-9)))
        je = q[self.tracked] - self.track.ref_q[f][self.tracked]
        self._joint_err = float(np.abs(je).mean())

        ref_h = (self.track.root_pos[f, 2] if self.cfg["root_ref"] == "motion"
                 else float(self.cfg["nominal_height"]))
        self._ctx = RewardContext(
            q=q, qd=qd, ref_q=self.track.ref_q[f], ref_qd=self.track.ref_qd[f],
            tracked=self.tracked, link_pos=link, ref_link_pos=ref_link,
            link_weight=self.link_weight, joint_weight=self.joint_weight,
            root_height=float(pelvis[2]), ref_root_height=ref_h,
            upright=self._upright, root_xy=d.qpos[:2],
            ref_root_xy=self.track.root_pos[f, :2], yaw=yaw,
            ref_yaw=float(self.track.root_yaw[f]),
            foot_pos=self._foot_pos, foot_vel=self._foot_vel,
            foot_contact=self._foot_contact, action=self._action,
            prev_action=self._prev_action,
            torque=d.actuator_force[self.body_act], joint_range=self.joint_range,
            n_self_contacts=self._n_self,
            track_root_xy=bool(self.cfg["track_root_xy"]))

    def _contacts(self) -> None:
        """Foot contact flags, per-foot contact force, and self-collision count.

        Walked by hand rather than through sensors: the model carries none, and
        `cfrc_ext` is not filled by `mj_step` unless something asks for it.
        Contact counts here are in the low tens, so the loop is cheap.
        """
        d, m = self.data, self.model
        self._foot_contact[:] = False
        self._foot_force[:] = 0.0
        n_self = 0
        cf = self._cf
        for i in range(d.ncon):
            c = d.contact[i]
            g1_, g2 = c.geom1, c.geom2
            floor = (g1_ == self.floor_geom) or (g2 == self.floor_geom)
            if not floor:
                n_self += 1
                continue
            other = g2 if g1_ == self.floor_geom else g1_
            s = self._foot_side[other]
            if s < 0:
                continue
            self._foot_contact[s] = True
            mujoco.mj_contactForce(m, d, i, cf)
            self._foot_force[3 * s:3 * s + 3] += c.frame.reshape(3, 3).T @ cf[:3]
        self._n_self = n_self

    def _terminated(self) -> tuple[bool, str]:
        t = self.cfg["termination"]
        if self._ctx.root_height < float(t["min_height"]):
            return True, "height"
        if self._upright < float(t["min_upright"]):
            return True, "upright"
        if self._joint_err > float(t["max_joint_error"]):
            return True, "joint_divergence"
        if (self.link_weight.sum() > 0
                and self._link_err > float(t["max_link_error"])):
            return True, "link_divergence"
        if not np.isfinite(self.data.qpos).all():
            return True, "nonfinite"
        return False, ""

    # ----------------------------------------------------------------- #
    def _yaw_error(self) -> float:
        """Policy heading minus reference heading, wrapped to [-pi, pi]."""
        d = self._yaw - float(self.track.root_yaw[self.frame])
        return float((d + np.pi) % (2.0 * np.pi) - np.pi)

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        o, d = self._obs, self.data
        n, a = self.n_body, self.act_dim
        i = 0
        o[i:i + n] = d.qpos[self.body_qpos]; i += n
        o[i:i + n] = d.qvel[self.body_dof]; i += n
        o[i:i + 3] = self._ang_vel; i += 3
        o[i:i + 3] = self._gravity; i += 3
        o[i:i + a] = self._action; i += a
        ph = 2.0 * np.pi * (self.frame - self.seg[0]) / max(
            self.seg[1] - self.seg[0], 1)
        o[i] = np.sin(ph); o[i + 1] = np.cos(ph); i += 2
        # Heading error, as sine and cosine so it is continuous across the
        # wrap. Every other quantity here is yaw-invariant by construction:
        # projected gravity is -R[2, :], which does not change when the robot
        # spins about the vertical, and the reference window is in the pelvis
        # frame. That invariance is deliberate, because it keeps the reward
        # immune to monocular root drift, but it also means that without this
        # pair the policy cannot perceive its own heading error at all. The
        # track_yaw reward was unlearnable before it existed: raising its
        # weight from 1.5 to 3.0 moved mean heading error from 61.9 to 64.2
        # degrees, which is noise.
        dyaw = self._yaw_error()
        o[i] = np.sin(dyaw); o[i + 1] = np.cos(dyaw); i += 2
        f = self.frame + 1
        o[i:] = self.track.window[f:f + self.window].reshape(-1)

        p = self._priv
        p[:self.obs_dim] = o
        j = self.obs_dim
        p[j:j + 3] = self._lin_vel; j += 3
        p[j] = d.xpos[self.pelvis_id, 2]; j += 1
        p[j:j + 6] = self._foot_force; j += 6
        p[j:j + 5] = self._dr
        return o.copy(), p.copy()

    # ----------------------------------------------------------------- #
    @property
    def reward_components(self) -> dict[str, float]:
        return dict(zip(self.reward_fn.names, self.reward_fn.components))

    def reference_qpos(self, frame: int | None = None) -> np.ndarray:
        """The control reference at a frame; the kinematic upper bound in eval."""
        return self.track.ref_q[self.frame if frame is None else frame]


def make_dance_env(**kw) -> DanceTrackEnv:
    """Picklable factory for the spawned pool.

    The pool is spawned, so the worker must build the environment itself from
    plain arguments. Passing a motion path and a config path rather than a
    constructed environment is what keeps a compiled `mjModel` off the pipe.
    """
    return DanceTrackEnv(**kw)
