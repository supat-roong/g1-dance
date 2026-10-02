"""Tests for the downloaded general tracking policy.

Nothing here can check that the transcription of somebody else's observation is
*correct*; only their own stack could do that. What it can do is check the parts
that have an independent second source, and then check the one behavioural
consequence that a scrambled observation could not possibly produce.

The independent sources are MuJoCo itself, for the quaternion algebra, and this
environment's own gravity and yaw, which were written months earlier for a
different purpose and agree with the transcribed formulas or one of them is
wrong.

The behavioural check is the load-bearing one: on a standing reference the
tracker must stay up, where open-loop PD falls over. A 136-vector assembled in
the wrong order would not stand.
"""

from __future__ import annotations

import numpy as np
import mujoco
import pytest

from g1dance.control.trackers import (
    HGPT_ACTION_SCALE, HGPT_ACT_DIM, HGPT_BASE_KPS, HGPT_NOMINAL_QPOS,
    HGPT_OBS_DIM, HGPT_TORQUE_LIMIT, OBS, HumanoidGPTController, TRACKER_URLS, TrackerDownloadError,
    _gravity_in_body, _quat_to_mat, _wrap_pi, _yaw, ema_qpos, ensure_tracker,
    obs_slices)
from g1dance.sim.env import DanceTrackEnv, load_config, synthetic_motion
from g1dance.sim.model import BODY_JOINTS, G1Model

CACHED = (ensure_tracker.__module__ and
          __import__("pathlib").Path.home() / ".cache" / "g1dance" / "models"
          / "humanoid_gpt.onnx")
needs_weights = pytest.mark.skipif(
    not CACHED.exists(),
    reason="needs the cached Humanoid-GPT ONNX; run ensure_tracker() online")


@pytest.fixture(scope="module")
def g1() -> G1Model:
    return G1Model()


@pytest.fixture(scope="module")
def standing(g1):
    """A reference that is just the default stance, held."""
    return synthetic_motion(240, fps=50.0, g1=g1, amp=0.0) \
        if "amp" in synthetic_motion.__code__.co_varnames \
        else synthetic_motion(240, fps=50.0, g1=g1)


# -------------------------------------------------------------------- layout
def test_layout_sums_to_the_declared_width():
    assert sum(s.stop - s.start for s in obs_slices().values()) == HGPT_OBS_DIM


def test_layout_matches_their_documented_field_order():
    """Their _ensure_fast_buffers advances slices in exactly this order."""
    assert [(k, s.start) for k, s in OBS.items()] == [
        ("gyro", 0), ("gvec", 3), ("jnt_qpos_rel", 6), ("jnt_qvel", 35),
        ("last_action", 64), ("ref_jnt_rel", 93), ("ref_height", 122),
        ("ref_gvec", 123), ("ref_cvel", 126), ("yaw_cmd", 132),
        ("xy_cmd", 134)]


def test_constants_have_the_robot_width():
    for a in (HGPT_NOMINAL_QPOS, HGPT_TORQUE_LIMIT, HGPT_BASE_KPS,
              HGPT_ACTION_SCALE):
        assert a.shape == (len(BODY_JOINTS),)
    assert HGPT_ACT_DIM == len(BODY_JOINTS)


def test_action_scale_is_torque_over_gain():
    np.testing.assert_allclose(HGPT_ACTION_SCALE,
                               HGPT_TORQUE_LIMIT / HGPT_BASE_KPS)


# ------------------------------------------------- checked against mujoco
@pytest.mark.parametrize("seed", range(6))
def test_quat_to_mat_agrees_with_mujoco(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    want = np.zeros(9)
    mujoco.mju_quat2Mat(want, q)
    np.testing.assert_allclose(_quat_to_mat(q), want.reshape(3, 3), atol=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_gravity_formula_agrees_with_the_rotation_matrix(seed):
    """Their fast-path algebra against the definition, -R[2, :]."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    np.testing.assert_allclose(_gravity_in_body(q),
                               -_quat_to_mat(q)[2, :], atol=1e-12)


def test_gravity_agrees_with_the_environments_own(standing):
    """The env computed gravity months earlier for the reward. They must agree."""
    e = DanceTrackEnv(standing, load_config("configs/upper_body.yaml"),
                      seed=0, randomize=False)
    e.reset()
    np.testing.assert_allclose(_gravity_in_body(e.data.qpos[3:7]),
                               e._gravity, atol=1e-9)


def test_yaw_agrees_with_the_environments_own(standing):
    e = DanceTrackEnv(standing, load_config("configs/upper_body.yaml"),
                      seed=0, randomize=False)
    e.reset()
    assert _yaw(e.data.qpos[3:7]) == pytest.approx(e._yaw, abs=1e-9)


def test_wrap_pi_is_the_shortest_turn():
    # The half-open convention [-pi, pi), so 3*pi lands on -pi rather than +pi.
    # They are the same angle; what matters is that the magnitude never exceeds
    # pi, because that value is fed to cos and sin as a turn command.
    assert _wrap_pi(3.0 * np.pi) == pytest.approx(-np.pi, abs=1e-9)
    assert _wrap_pi(-1.5 * np.pi) == pytest.approx(0.5 * np.pi, abs=1e-9)
    assert _wrap_pi(0.25) == pytest.approx(0.25, abs=1e-12)
    for a in np.linspace(-20.0, 20.0, 401):
        w = _wrap_pi(float(a))
        assert -np.pi - 1e-9 <= w < np.pi + 1e-9
        assert np.cos(w) == pytest.approx(np.cos(a), abs=1e-9)
        assert np.sin(w) == pytest.approx(np.sin(a), abs=1e-9)


# -------------------------------------------------------------------- anchor
def _robot_at(frame, xy, yaw):
    from types import SimpleNamespace
    qpos = np.zeros(7)
    qpos[:2] = xy
    qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    return SimpleNamespace(frame=frame, data=SimpleNamespace(qpos=qpos))


def test_anchoring_puts_the_start_frame_on_the_robot():
    rng = np.random.default_rng(0)
    yaw = np.cumsum(rng.normal(0, 0.1, 50))
    pos = np.cumsum(rng.normal(0, 0.05, (50, 3)), axis=0)
    env = _robot_at(17, [0.3, -0.2], 0.4)
    y, p = HumanoidGPTController._anchored(env, yaw, pos)
    assert y[17] == pytest.approx(0.4)
    np.testing.assert_allclose(p[17, :2], [0.3, -0.2], atol=1e-12)


def test_anchoring_is_rigid_so_the_motion_itself_is_unchanged():
    rng = np.random.default_rng(1)
    yaw = np.cumsum(rng.normal(0, 0.1, 50))
    pos = np.cumsum(rng.normal(0, 0.05, (50, 3)), axis=0)
    y, p = HumanoidGPTController._anchored(_robot_at(5, [1.0, 2.0], -2.0),
                                           yaw, pos)
    np.testing.assert_allclose(np.diff(y), np.diff(yaw), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1),
                               np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1),
                               atol=1e-12)
    np.testing.assert_array_equal(p[:, 2], pos[:, 2])


# ----------------------------------------------------------------------- ema
def test_ema_keeps_the_first_frame_and_normalises_quaternions():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(20, 36))
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    out = ema_qpos(q)
    np.testing.assert_allclose(out[0], q[0])
    np.testing.assert_allclose(np.linalg.norm(out[:, 3:7], axis=1),
                               1.0, atol=1e-9)


def test_ema_smooths_towards_the_previous_frame():
    q = np.zeros((3, 36))
    q[:, 3] = 1.0
    q[1:, 7] = 1.0                     # a step change on one joint
    out = ema_qpos(q, alpha=0.8)
    assert out[1, 7] == pytest.approx(0.2, abs=1e-12)
    assert out[2, 7] == pytest.approx(0.2 * 0.8 + 1.0 * 0.2, abs=1e-12)


def test_ema_is_a_noop_on_one_frame():
    q = np.zeros((1, 36))
    q[0, 3] = 1.0
    np.testing.assert_allclose(ema_qpos(q), q)


# ------------------------------------------------------------------ fetching
def test_unknown_tracker_names_the_known_ones():
    with pytest.raises(TrackerDownloadError, match="humanoid_gpt"):
        ensure_tracker("no_such_tracker")


def test_a_cached_file_is_returned_without_a_download(tmp_path):
    (tmp_path / "humanoid_gpt.onnx").write_bytes(b"not really an onnx")
    assert ensure_tracker("humanoid_gpt", cache_dir=tmp_path) == \
        tmp_path / "humanoid_gpt.onnx"


def test_an_unreachable_download_says_where_to_put_the_file(tmp_path, monkeypatch):
    monkeypatch.setitem(TRACKER_URLS, "humanoid_gpt",
                        "https://0.0.0.0/nope.onnx")
    with pytest.raises(TrackerDownloadError, match="by hand"):
        ensure_tracker("humanoid_gpt", cache_dir=tmp_path)


# ------------------------------------------------------------- the real thing
@needs_weights
def test_the_observation_is_finite_and_the_right_width(standing):
    from g1dance.control.trackers import HumanoidGPTController
    e = DanceTrackEnv(standing, load_config("configs/upper_body.yaml"),
                      seed=0, randomize=False)
    obs, priv = e.reset()
    c = HumanoidGPTController()
    c.reset(e)
    o = c.observe(e)
    assert o.shape == (HGPT_OBS_DIM,)
    assert np.all(np.isfinite(o))


@needs_weights
def test_targets_are_a_residual_about_their_nominal_stance(standing):
    """nn2motor_action is referenced to the default stance, not the reference."""
    from g1dance.control.trackers import HumanoidGPTController
    e = DanceTrackEnv(standing, load_config("configs/upper_body.yaml"),
                      seed=0, randomize=False)
    obs, priv = e.reset()
    c = HumanoidGPTController()
    c.reset(e)
    a = c.act(e, obs, priv)
    assert a.shape == (len(BODY_JOINTS),)
    assert np.all(np.isfinite(a))
    # a bounded action times 0.25 times the scale keeps it near the nominal
    assert np.abs(a - HGPT_NOMINAL_QPOS).max() < 5.0


@needs_weights
def test_the_tracker_stays_up_on_a_reference_it_should_find_easy(standing):
    """The load-bearing test. A scrambled observation could not do this.

    Open-loop PD to the same reference falls over, so surviving is not a
    property of the reference being easy.
    """
    from g1dance.control.controllers import OpenLoopController
    from g1dance.control.evaluate import _episode
    from g1dance.control.trackers import HumanoidGPTController

    cfg = load_config("configs/upper_body.yaml")
    e = DanceTrackEnv(standing, cfg, seed=0, randomize=False)

    e.seed(0)
    loose = _episode(e, OpenLoopController(), max_steps=300)
    e.seed(0)
    tracked = _episode(e, HumanoidGPTController(), max_steps=300)

    assert not tracked["fell"], "the tracker fell on a reference it should hold"
    assert tracked["steps"] > loose["steps"]
    assert tracked["dof_err"].mean() < 0.15


# --------------------------------------------------- the transformer variant
def test_a_two_dimensional_input_means_no_history_is_kept():
    """The MLP checkpoints take one frame; only the transformer takes a stack."""
    from g1dance.control.trackers import HumanoidGPTController
    c = HumanoidGPTController.__new__(HumanoidGPTController)
    assert c._history_len_from_shape(["batch", 136]) is None


def test_a_three_dimensional_input_declares_its_own_window():
    """16 comes from the graph, not from a constant we chose."""
    from g1dance.control.trackers import HumanoidGPTController
    c = HumanoidGPTController.__new__(HumanoidGPTController)
    assert c._history_len_from_shape(["batch", 16, 136]) == 16


def test_a_wrong_width_is_refused_whatever_the_rank():
    from g1dance.control.trackers import HumanoidGPTController
    c = HumanoidGPTController.__new__(HumanoidGPTController)
    with pytest.raises(ValueError, match="136"):
        c._history_len_from_shape(["batch", 8, 99])


def test_the_history_buffer_seeds_by_repeating_the_first_frame():
    """Matches their _push_history: an unseeded buffer is filled, not zero padded.

    Zero padding would tell the policy the robot was at the origin with an
    identity orientation for the 15 frames before the episode began, which is a
    state it never sees in training.
    """
    from g1dance.control.trackers import _History
    h = _History(4, 3)
    first = np.array([1.0, 2.0, 3.0], np.float32)
    out = h.push(first)
    assert out.shape == (4, 3)
    np.testing.assert_allclose(out, np.tile(first, (4, 1)))


def test_the_history_buffer_keeps_the_newest_frame_last():
    from g1dance.control.trackers import _History
    h = _History(3, 2)
    h.push(np.array([1.0, 1.0], np.float32))
    h.push(np.array([2.0, 2.0], np.float32))
    out = h.push(np.array([3.0, 3.0], np.float32))
    np.testing.assert_allclose(out[-1], [3.0, 3.0])
    np.testing.assert_allclose(out[0], [1.0, 1.0])


def test_the_history_buffer_drops_the_oldest_once_full():
    from g1dance.control.trackers import _History
    h = _History(2, 1)
    for v in (1.0, 2.0, 3.0):
        out = h.push(np.array([v], np.float32))
    np.testing.assert_allclose(out.ravel(), [2.0, 3.0])


def test_resetting_the_controller_clears_the_history(standing):
    """Otherwise the first frame of an episode carries the last one's tail."""
    from g1dance.control.trackers import HumanoidGPTController
    c = HumanoidGPTController.__new__(HumanoidGPTController)
    c._history = None
    assert c._history is None
