"""Drive this environment with a downloaded general tracking policy.

The policy is Humanoid-GPT (Apache-2.0, CVPR 2026), which the HumanTracker
benchmark ranks first among general G1 trackers. It is an ONNX graph taking a
136-dim observation and returning 29 continuous actions, and on this machine it
runs at 0.164 ms per step against a 20 ms control period, so the tracker is not
the expensive part of anything.

Everything below is a transcription of their own code, not an interpretation of
their paper, because a wrong convention here does not crash. Sources, all at
GalaxyGeneralRobotics/Humanoid-GPT:

* `tracking/infer_utils.py:_ensure_fast_buffers` fixes the observation layout,
  and `_fill_obs_inplace` fixes what goes in each slot. Both are reproduced
  field for field in `_OBS_LAYOUT` and `observe` below.
* `tracking/infer_utils.py:nn2motor_action` turns the network output into motor
  targets: nominal stance plus action times 0.25 times a per-joint scale. Note
  it is a residual about the *default stance*, not about the reference, which is
  why this controller reports absolute targets.
* `tracking/constants.py` gives `DEFAULT_QPOS` (whose base quaternion is
  commented `(w, x, y, z)`, confirming the root rotation order our export
  writes), `TORQUE_LIMIT`, `BASE_KPs`, and `ACTION_SCALE = TORQUE_LIMIT / KPs`.
  `JOINT_KP_KD_SCALE` is 1 for all 29 joints and `BASE_KP_KD_SCALE` defaults to
  1, so `KPs` is `BASE_KPs`.
* `tracking/infer_utils.py:update_coord_cmd` gives the yaw and xy commands.
* `scripts/inference.py` applies `apply_ema_qpos` at alpha 0.8 to the reference
  before anything reads it, so this does too.

Two fields are transcribed with less certainty than the rest, and are called out
because they are where a silent error would live. `ref_cvel` is the reference
pelvis spatial velocity in their gravity-view frame, which their pipeline
computes inside `qpos2kpt`; here it is finite-differenced from the reference root
and rotated into the yaw-removed frame, with linear before angular. `ref_gvec`
and `ref_height` come from `kpt2gv_pose[:, 0]`, the pelvis pose in that same
frame, which for a gravity-aligned yaw-removed frame reduces to the reference
root's own height and gravity direction. The behavioural test is what validates
these: a reference the robot should simply stand through must not fall over.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ..sim.model import BODY_JOINTS

#: Shared with track.detect, so one cache serves the whole project.
CACHE_DIR = Path(os.environ.get("G1DANCE_CACHE", Path.home() / ".cache" / "g1dance"))
MODEL_DIR = CACHE_DIR / "models"

#: Released checkpoints, committed directly in their repository rather than
#: published to a model hub, so a raw URL is the whole download story.
TRACKER_URLS = {
    "humanoid_gpt": (
        "https://raw.githubusercontent.com/GalaxyGeneralRobotics/Humanoid-GPT/"
        "HEAD/storage/ckpts/pns_wo_priv264.onnx"),
    "humanoid_gpt_216": (
        "https://raw.githubusercontent.com/GalaxyGeneralRobotics/Humanoid-GPT/"
        "HEAD/storage/ckpts/pns_wo_priv216.onnx"),
    # The paper's transformer, which takes a 16-frame window rather than one
    # frame. 43 MB and 1.8 ms per step against the MLP's 0.2, still eleven times
    # faster than the 50 Hz loop needs.
    "humanoid_gpt_trans": (
        "https://raw.githubusercontent.com/GalaxyGeneralRobotics/Humanoid-GPT/"
        "HEAD/storage/ckpts/pns_wo_priv216_trans.onnx"),
}


class TrackerDownloadError(RuntimeError):
    """The tracker is neither cached nor reachable."""


def ensure_tracker(name: str = "humanoid_gpt",
                   cache_dir: Path | None = None) -> Path:
    """Path to a cached ONNX tracker, downloading it once if need be.

    Mirrors `track.detect.ensure_pose_model`: the weights are somebody else's
    and are not vendored here, so the first call fetches them and every later
    call is free. Apache-2.0, so caching them is unproblematic.
    """
    if name not in TRACKER_URLS:
        raise TrackerDownloadError(
            f"unknown tracker {name!r}; known: {sorted(TRACKER_URLS)}")
    directory = Path(cache_dir) if cache_dir else MODEL_DIR
    dest = directory / f"{name}.onnx"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    directory.mkdir(parents=True, exist_ok=True)
    url = TRACKER_URLS[name]
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            tmp.write_bytes(r.read())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise TrackerDownloadError(
            f"the tracker is not cached at {dest} and could not be downloaded "
            f"from {url} ({exc}). Fetch it once while online, or place the "
            f"file there by hand.") from exc
    tmp.replace(dest)
    return dest

#: Humanoid-GPT nominal joint configuration, `DEFAULT_QPOS[7:]`.
HGPT_NOMINAL_QPOS = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.3, 0.0, 1.28, 0.0, 0.0, 0.0,
    0.2, -0.3, 0.0, 1.28, 0.0, 0.0, 0.0,
], np.float64)

HGPT_TORQUE_LIMIT = np.array([
    88., 139., 88., 139., 50., 50.,
    88., 139., 88., 139., 50., 50.,
    88., 50., 50.,
    25., 25., 25., 25., 25., 5., 5.,
    25., 25., 25., 25., 25., 5., 5.,
], np.float64)

HGPT_BASE_KPS = np.array([
    40.17923737, 99.09842682, 40.17923737, 99.09842682, 28.5012455,
    28.5012455, 40.17923737, 99.09842682, 40.17923737, 99.09842682,
    28.5012455, 28.5012455, 40.17923737, 28.5012455, 28.5012455,
    14.25062275, 14.25062275, 14.25062275, 14.25062275, 14.25062275,
    16.77832794, 16.77832794, 14.25062275, 14.25062275, 14.25062275,
    14.25062275, 14.25062275, 16.77832794, 16.77832794,
], np.float64)

#: `ACTION_SCALE = TORQUE_LIMIT / KPs`, with KPs == BASE_KPs at default scales.
HGPT_ACTION_SCALE = HGPT_TORQUE_LIMIT / HGPT_BASE_KPS
#: `g1_infer_env_config(...).action_scale`
HGPT_NN_ACTION_SCALE = 0.25
#: `apply_ema_qpos` default
HGPT_EMA_ALPHA = 0.8
HGPT_OBS_DIM = 136
HGPT_ACT_DIM = 29

#: field -> width, in the order `_ensure_fast_buffers` advances its slices.
_OBS_LAYOUT: tuple[tuple[str, int], ...] = (
    ("gyro", 3), ("gvec", 3), ("jnt_qpos_rel", 29), ("jnt_qvel", 29),
    ("last_action", 29), ("ref_jnt_rel", 29), ("ref_height", 1),
    ("ref_gvec", 3), ("ref_cvel", 6), ("yaw_cmd", 2), ("xy_cmd", 2),
)


def obs_slices() -> dict[str, slice]:
    """The observation layout, as named slices. Sums to `HGPT_OBS_DIM`."""
    out, i = {}, 0
    for name, width in _OBS_LAYOUT:
        out[name] = slice(i, i + width)
        i += width
    if i != HGPT_OBS_DIM:
        raise AssertionError(f"layout sums to {i}, expected {HGPT_OBS_DIM}")
    return out


OBS = obs_slices()


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion to rotation matrix, matching MuJoCo's convention."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _gravity_in_body(q: np.ndarray) -> np.ndarray:
    """World -z expressed in the body frame, their fast-path formula verbatim."""
    w, x, y, z = (float(v) for v in q)
    return np.array([-2.0 * (x * z - w * y),
                     -2.0 * (y * z + w * x),
                     -(1.0 - 2.0 * (x * x + y * y))])


def _yaw(q: np.ndarray) -> float:
    return float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ema_qpos(qpos: np.ndarray, alpha: float = HGPT_EMA_ALPHA) -> np.ndarray:
    """`apply_ema_qpos`: exponential smoothing with quaternion sign fixing."""
    q = np.asarray(qpos, np.float64)
    if len(q) <= 1:
        return q.copy()
    out = np.empty_like(q)
    out[0] = q[0]
    for t in range(1, len(q)):
        prev = out[t - 1, 3:7]
        curr = q[t, 3:7]
        if float(np.dot(prev, curr)) < 0:
            curr = -curr
        out[t, :3] = out[t - 1, :3] * alpha + q[t, :3] * (1 - alpha)
        out[t, 7:] = out[t - 1, 7:] * alpha + q[t, 7:] * (1 - alpha)
        b = prev * alpha + curr * (1 - alpha)
        out[t, 3:7] = b / max(float(np.linalg.norm(b)), 1e-8)
    return out


class _History:
    """A rolling window of observations, newest last.

    An unseeded window is filled with its first frame rather than zero padded,
    which is what their `_push_history` does. The difference matters: zeros would
    tell the policy the robot spent the frames before the episode at the origin
    with an identity orientation and no joint offsets, a state it never saw in
    training, so the opening steps of every episode would be off distribution.
    """

    def __init__(self, length: int, width: int):
        self.length = int(length)
        self.width = int(width)
        self.buf: np.ndarray | None = None

    def clear(self) -> None:
        self.buf = None

    def push(self, obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, np.float32).reshape(self.width)
        if self.buf is None:
            self.buf = np.tile(o, (self.length, 1))
        else:
            if self.length > 1:
                self.buf[:-1] = self.buf[1:]
            self.buf[-1] = o
        return self.buf


class HumanoidGPTController:
    """Humanoid-GPT driving `DanceTrackEnv` through the absolute target path.

    `nn2motor_action` is a residual about the default stance rather than about
    the reference, so the targets it produces are absolute in this environment's
    terms and must not go through the residual path.
    """

    absolute = True

    def __init__(self, onnx_path: str | Path | None = None,
                 providers: list[str] | None = None, ema: bool = True,
                 anchor: bool = True):
        self.path = Path(onnx_path) if onnx_path else ensure_tracker()
        self._providers = providers or ["CPUExecutionProvider"]
        self._ema = bool(ema)
        self._anchor = bool(anchor)
        self._sess = None
        self._input = None
        self._history_len: int | None = None
        self._history: _History | None = None

    # ------------------------------------------------------------------ setup
    def _session(self):
        if self._sess is None:
            import onnxruntime as ort
            if not self.path.exists():
                raise FileNotFoundError(
                    f"the Humanoid-GPT ONNX tracker is not at {self.path}. "
                    "Fetch storage/ckpts/pns_wo_priv264.onnx from "
                    "github.com/GalaxyGeneralRobotics/Humanoid-GPT and pass "
                    "its path.")
            self._sess = ort.InferenceSession(str(self.path),
                                              providers=self._providers)
            ins = self._sess.get_inputs()
            if len(ins) != 1:
                raise ValueError(
                    f"expected one input, got {[(i.name, i.shape) for i in ins]}")
            self._input = ins[0].name
            # The window is read off the graph rather than configured. Their MLP
            # checkpoints take (batch, 136) and the transformer takes
            # (batch, 16, 136); asking the file removes a constant that could
            # disagree with the weights it is fed to.
            self._history_len = self._history_len_from_shape(ins[0].shape)
        return self._sess

    @staticmethod
    def _history_len_from_shape(shape) -> int | None:
        """None for a single-frame input, else the window the graph declares."""
        dims = list(shape)
        if len(dims) not in (2, 3) or dims[-1] != HGPT_OBS_DIM:
            raise ValueError(
                f"expected an input ending in {HGPT_OBS_DIM}, got {dims}")
        if len(dims) == 2:
            return None
        n = dims[1]
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"history length must be a fixed integer, got {n!r}")
        return n

    def reset(self, env) -> None:
        self._session()
        if env.n_body != len(BODY_JOINTS):
            raise ValueError(
                f"this tracker is a 29-DoF G1 policy; env has {env.n_body}")
        self._last_action = np.zeros(HGPT_ACT_DIM, np.float64)
        self._obs = np.zeros(HGPT_OBS_DIM, np.float32)
        # Cleared per episode, or the first frame of one carries the tail of the
        # last, which reads as the robot having just been somewhere else.
        self._history = (_History(self._history_len, HGPT_OBS_DIM)
                         if self._history_len else None)
        # Assemble the reference in their qpos layout, then smooth it exactly as
        # their inference script does. `_Track` keeps the root orientation as an
        # unwrapped yaw rather than a quaternion, because the environment itself
        # reduces the reference root to yaw about vertical, so the quaternion is
        # rebuilt from it the same way `reset` does.
        ref = env.track
        yaw = np.asarray(ref.root_yaw, np.float64)
        root_q = np.stack([np.cos(yaw / 2), np.zeros_like(yaw),
                           np.zeros_like(yaw), np.sin(yaw / 2)], axis=1)
        root_pos = np.asarray(ref.root_pos, np.float64)
        if self._anchor:
            yaw, root_pos = self._anchored(env, yaw, root_pos)
            root_q = np.stack([np.cos(yaw / 2), np.zeros_like(yaw),
                               np.zeros_like(yaw), np.sin(yaw / 2)], axis=1)
        full = np.concatenate([root_pos, root_q,
                               np.asarray(ref.ref_q, np.float64)], axis=1)
        self._ref = ema_qpos(full) if self._ema else full
        self._dt = float(env.dt)
        self._n = len(self._ref)

    @staticmethod
    def _anchored(env, yaw: np.ndarray, root_pos: np.ndarray):
        """The reference root, moved rigidly so the start frame is the robot.

        With `root_ref: nominal` the robot spawns at the origin facing +x, but
        the episode starts at a random reference frame, whose root is wherever
        the source put it. Unanchored, `xy_cmd` and `yaw_cmd` open every episode
        with that offset: measured on kirari_solo, a median of 0.31 m and 71
        degrees for a GVHMR reference, so the tracker spends its first steps
        walking and turning toward a point instead of dancing. One rotation
        about vertical plus a translation, so every relative quantity the
        tracker reads, `ref_cvel` included, is unchanged.

        On by default. Unanchored, a GVHMR reference, whose dancer faces about
        90 degrees in its world, held the robot that far off heading for the
        whole episode, because in mode A it cannot turn more than about 30.
        The scores barely moved either way, being pelvis-frame. Where the robot
        spawns on the reference root (`root_ref: motion`) the rotation is zero.
        """
        f = min(int(env.frame), len(yaw) - 1)
        d = env.data
        dyaw = _yaw(d.qpos[3:7]) - float(yaw[f])
        R = _rot_z(dyaw)[:2, :2]
        out = root_pos.copy()
        out[:, :2] = (root_pos[:, :2] - root_pos[f, :2]) @ R.T + d.qpos[:2]
        return yaw + dyaw, out

    # -------------------------------------------------------------- the obs
    def observe(self, env) -> np.ndarray:
        d = env.data
        o = self._obs
        cur = min(env.frame, self._n - 1)
        nxt = min(env.frame + 1, self._n - 1)
        ref_n, ref_c = self._ref[nxt], self._ref[cur]

        o[OBS["gyro"]] = d.qvel[3:6]
        o[OBS["gvec"]] = _gravity_in_body(d.qpos[3:7])
        o[OBS["jnt_qpos_rel"]] = d.qpos[env.body_qpos] - HGPT_NOMINAL_QPOS
        o[OBS["jnt_qvel"]] = d.qvel[env.body_dof]
        o[OBS["last_action"]] = self._last_action
        o[OBS["ref_jnt_rel"]] = ref_n[7:] - HGPT_NOMINAL_QPOS
        o[OBS["ref_height"]] = ref_n[2]
        o[OBS["ref_gvec"]] = _gravity_in_body(ref_n[3:7])
        o[OBS["ref_cvel"]] = self._ref_cvel(cur, nxt)

        # update_coord_cmd, which reads ref_curr rather than ref_next
        yaw_ref, yaw_cur = _yaw(ref_c[3:7]), _yaw(d.qpos[3:7])
        yaw_d = _wrap_pi(yaw_ref - yaw_cur)
        o[OBS["yaw_cmd"]] = (np.cos(yaw_d), np.sin(yaw_d))
        xy_d = ref_c[:2] - d.qpos[:2]
        c, s = np.cos(-yaw_cur), np.sin(-yaw_cur)
        o[OBS["xy_cmd"]] = (c * xy_d[0] - s * xy_d[1],
                            s * xy_d[0] + c * xy_d[1])
        return o

    def _ref_cvel(self, cur: int, nxt: int) -> np.ndarray:
        """Reference pelvis spatial velocity in the yaw-removed frame.

        Linear before angular. Finite differenced, because their own value comes
        out of `qpos2kpt` which we do not run; see the module docstring.
        """
        a, b = self._ref[cur], self._ref[nxt]
        v = (b[:3] - a[:3]) / self._dt
        Ra, Rb = _quat_to_mat(a[3:7]), _quat_to_mat(b[3:7])
        dR = Rb @ Ra.T
        w = np.array([dR[2, 1] - dR[1, 2],
                      dR[0, 2] - dR[2, 0],
                      dR[1, 0] - dR[0, 1]]) / (2.0 * self._dt)
        Rz = _rot_z(-_yaw(b[3:7]))
        return np.concatenate([Rz @ v, Rz @ w])

    # -------------------------------------------------------------- the step
    def act(self, env, obs, priv) -> np.ndarray:
        sess = self._session()
        o = self.observe(env)
        x = self._history.push(o)[None] if self._history is not None else o[None]
        nn = sess.run(None, {self._input: x})[0][0]
        self._last_action[:] = nn
        return (HGPT_NOMINAL_QPOS
                + np.asarray(nn, np.float64) * HGPT_NN_ACTION_SCALE
                * HGPT_ACTION_SCALE)


__all__ = ["CACHE_DIR", "MODEL_DIR", "TRACKER_URLS", "TrackerDownloadError",
           "_History",
           "ensure_tracker", "HGPT_ACTION_SCALE", "HGPT_ACT_DIM", "HGPT_EMA_ALPHA",
           "HGPT_NN_ACTION_SCALE", "HGPT_NOMINAL_QPOS", "HGPT_OBS_DIM",
           "HumanoidGPTController", "OBS", "ema_qpos", "obs_slices"]
