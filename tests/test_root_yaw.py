"""The root yaw rate cap.

The retargeted root yaw is the least trustworthy signal in the reference. On
kirari it sweeps 310 degrees, which is a real property of the dancer turning,
but its rate reaches 4009 degrees per second, which is not a property of
anything: a G1 base cannot yaw that fast, and neither can the dancer. The
transients are monocular heading noise riding on a real turn.

They went unnoticed through all of milestone A because mode A's own policy was
trained on this convention and learned to fake it. A general tracking policy,
which consumes the reference root and was trained on real motion capture,
refuses. Measured with Humanoid-GPT on kirari_v4 over 12 seeds, end to end
similarity by cap: 0.503 uncapped, 0.549 at 60 deg/s, 0.517 at 90, 0.482 at 180.
The curve is flat from 15 to 60 and degrades above it, so the default is the
loosest cap that costs nothing.

What is tested here is that the cap removes the transient, keeps the turn, and
leaves roll and pitch alone, because the export carries the whole quaternion
even though the environment reads only the yaw from it.
"""

from __future__ import annotations

import numpy as np
import mujoco
import pytest

from g1dance.motion import constraints as C


def _yaw(q: np.ndarray) -> np.ndarray:
    return np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))


def _quat_from_euler(roll, pitch, yaw) -> np.ndarray:
    """wxyz quaternions for Rz(yaw) @ Ry(pitch) @ Rx(roll), one per frame."""
    out = np.zeros((len(yaw), 4))
    buf = np.zeros(4)
    for i in range(len(yaw)):
        cz, sz = np.cos(yaw[i]), np.sin(yaw[i])
        cy, sy = np.cos(pitch[i]), np.sin(pitch[i])
        cx, sx = np.cos(roll[i]), np.sin(roll[i])
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        mujoco.mju_mat2Quat(buf, np.ascontiguousarray(Rz @ Ry @ Rx).ravel())
        out[i] = buf
    return out


def _yaw_only(yaw: np.ndarray) -> np.ndarray:
    z = np.zeros_like(yaw)
    return _quat_from_euler(z, z, yaw)


# ------------------------------------------------------------------ behaviour
def test_a_slow_turn_is_left_alone():
    fps = 50.0
    yaw = np.radians(np.linspace(0.0, 20.0, 51))          # 20 deg/s
    q = _yaw_only(yaw)
    out = C.limit_root_yaw_rate(q, fps, 60.0)
    np.testing.assert_allclose(_yaw(out), yaw, atol=1e-9)


def test_a_spike_is_cut_to_the_cap():
    fps = 50.0
    yaw = np.zeros(11)
    yaw[5:] = np.radians(90.0)                            # instantaneous 90 deg
    out = C.limit_root_yaw_rate(_yaw_only(yaw), fps, 60.0)
    rate = np.degrees(np.abs(np.diff(_yaw(out))) * fps)
    assert rate.max() <= 60.0 + 1e-6


def test_the_measured_transient_is_removed_and_the_turn_survives():
    """A real turn with noise on it: keep the first, drop the second."""
    fps = 50.0
    t = np.arange(300) / fps
    turn = np.radians(40.0) * t                           # 40 deg/s, feasible
    noisy = turn.copy()
    noisy[100] += np.radians(80.0)                        # a one-frame flip
    noisy[200] -= np.radians(80.0)
    out = _yaw(C.limit_root_yaw_rate(_yaw_only(noisy), fps, 60.0))
    assert np.degrees(np.abs(np.diff(out)) * fps).max() <= 60.0 + 1e-6
    # the turn is still most of the way there, unlike zeroing it
    assert np.degrees(out.max() - out.min()) > 0.7 * np.degrees(turn.max() - turn.min())


def test_roll_and_pitch_are_untouched():
    """The export carries the whole quaternion, so only yaw may change."""
    fps = 50.0
    n = 40
    rng = np.random.default_rng(0)
    roll = np.radians(rng.uniform(-10, 10, n))
    pitch = np.radians(rng.uniform(-10, 10, n))
    yaw = np.radians(np.cumsum(rng.uniform(-20, 20, n)))      # wild
    q = _quat_from_euler(roll, pitch, yaw)
    out = C.limit_root_yaw_rate(q, fps, 30.0)

    def rp(quat):
        got = []
        for k in range(len(quat)):
            m = np.zeros(9)
            mujoco.mju_quat2Mat(m, quat[k])
            R = m.reshape(3, 3)
            y = np.arctan2(R[1, 0], R[0, 0])
            Rz = np.array([[np.cos(-y), -np.sin(-y), 0],
                           [np.sin(-y), np.cos(-y), 0], [0, 0, 1]])
            got.append(Rz @ R)
        return np.array(got)

    np.testing.assert_allclose(rp(out), rp(q), atol=1e-8)


def test_output_quaternions_stay_unit():
    fps = 50.0
    yaw = np.radians(np.cumsum(np.full(60, 30.0)))
    out = C.limit_root_yaw_rate(_yaw_only(yaw), fps, 10.0)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-9)


def test_the_cap_is_per_second_not_per_frame():
    yaw = np.radians(np.linspace(0.0, 100.0, 101))
    slow = _yaw(C.limit_root_yaw_rate(_yaw_only(yaw), 25.0, 50.0))
    fast = _yaw(C.limit_root_yaw_rate(_yaw_only(yaw), 100.0, 50.0))
    # at 4x the rate the per-frame step is 4x smaller, so the same cap in
    # deg/s must hold in both, and the higher rate lags further behind
    for got, fps in ((slow, 25.0), (fast, 100.0)):
        assert np.degrees(np.abs(np.diff(got)) * fps).max() <= 50.0 + 1e-6


def test_applying_it_twice_changes_nothing_more():
    fps = 50.0
    yaw = np.radians(np.cumsum(np.full(80, 25.0)))
    once = C.limit_root_yaw_rate(_yaw_only(yaw), fps, 10.0)
    twice = C.limit_root_yaw_rate(once, fps, 10.0)
    np.testing.assert_allclose(_yaw(twice), _yaw(once), atol=1e-9)


@pytest.mark.parametrize("cap", [0.0, None])
def test_a_falsy_cap_disables_it(cap):
    fps = 50.0
    yaw = np.radians(np.cumsum(np.full(30, 500.0)))
    q = _yaw_only(yaw)
    np.testing.assert_allclose(C.limit_root_yaw_rate(q, fps, cap), q, atol=1e-12)


def test_one_frame_is_a_noop():
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(C.limit_root_yaw_rate(q, 50.0, 10.0), q)


# --------------------------------------------------------------- the real data
def test_the_reference_transient_is_real_and_the_cap_removes_it():
    """Guards the finding itself, on the clip it was measured on."""
    from pathlib import Path

    from g1dance.motion.ir import ReferenceMotion
    p = Path("motion/kirari_v4.npz")
    if not p.exists():
        pytest.skip("motion/ is gitignored; needs the local kirari_v4 reference")
    m = ReferenceMotion.load(p)
    q = np.asarray(m.root_quat)
    before = np.degrees(np.abs(np.diff(_yaw(q))) * m.fps)
    assert before.max() > 1000.0, "the transient this cap exists for is gone"
    after_q = C.limit_root_yaw_rate(q, m.fps, 60.0)
    after = np.degrees(np.abs(np.diff(_yaw(after_q))) * m.fps)
    # Relative slack, not absolute: the cap is applied to the yaw and then the
    # quaternion is rebuilt, so a round trip through mat2Quat and back shows up
    # at the sixth significant figure on real data.
    assert after.max() <= 60.0 * (1 + 1e-4)
    # and the turn is mostly kept: 310 degrees down to roughly 240, not to zero
    kept = np.degrees(_yaw(after_q).max() - _yaw(after_q).min())
    assert kept > 200.0
