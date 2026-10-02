"""SMPLest-X as an estimator.

It matters because it is the only estimator here that emits SMPL-X parameters
rather than landmarks, which is what the `gmr` backend consumes: it is what
makes a second complete path from video to robot possible.

Four accommodations were needed and three of them fail silently if they regress,
so they are what is tested. The loudest is the checkpoint prefix: published as
it is, and loaded the way their own script loads it, zero of 519 parameters
match and a randomly initialised 687M network runs happily and produces
plausible looking output.

Everything skips without the checkout, which is 8.25 GB of weights and is not
vendored.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1dance.motion.frontends import available_frontends, get_frontend
from g1dance.motion.frontends.smplestx_fe import (
    CONFIG, ROOT, WEIGHTS, SmplestXUnavailable, _assert_upright, _to_z_up)

HAVE = ROOT.exists() and WEIGHTS.exists() and CONFIG.exists()
needs_checkout = pytest.mark.skipif(
    not HAVE, reason=f"no SMPLest-X checkout at {ROOT}; see smplestx_fe.py")


# ------------------------------------------------------------------ registry
def test_smplest_x_is_registered():
    assert "smplest_x" in available_frontends()


def test_it_has_the_uniform_estimator_signature():
    import inspect
    fn = get_frontend("smplest_x")
    assert list(inspect.signature(fn).parameters)[:2] == ["source", "subject"]


def test_a_missing_checkout_says_how_to_get_one(tmp_path, monkeypatch):
    from g1dance.motion.frontends import smplestx_fe
    monkeypatch.setattr(smplestx_fe, "ROOT", tmp_path / "absent")
    with pytest.raises(SmplestXUnavailable, match="MotrixLab/SMPLest-X"):
        smplestx_fe._load()


def test_it_refuses_to_run_without_a_subject_window():
    with pytest.raises(ValueError, match="subject window"):
        get_frontend("smplest_x")("x.mp4", None)


# ------------------------------------------------- the camera frame convention
def test_the_camera_to_world_rotation_stands_the_subject_up():
    """Raw output is y-down; a world with a floor is z-up."""
    from scipy.spatial.transform import Rotation as R
    # a subject upright in the camera frame has its head at -y
    raw = np.zeros((3, 3), np.float32)
    out = _to_z_up(raw)
    m = R.from_rotvec(out[0]).as_matrix()
    # the camera's -y must come out as the world's +z
    np.testing.assert_allclose(m @ np.array([0.0, -1.0, 0.0]),
                               np.array([0.0, 0.0, 1.0]), atol=1e-6)


def test_an_upside_down_subject_is_refused_not_retargeted():
    """Silently inverted, this retargets to a robot under the floor."""
    joints = np.zeros((4, 55, 3), np.float32)
    joints[:, 15, 2] = 0.1          # head barely above
    joints[:, 10, 2] = 0.0
    with pytest.raises(SmplestXUnavailable, match="not upright"):
        _assert_upright(joints, np.ones(4, bool))


def test_an_upright_subject_passes_the_check():
    joints = np.zeros((4, 55, 3), np.float32)
    joints[:, 15, 2] = 1.4
    _assert_upright(joints, np.ones(4, bool))


# --------------------------------------------------------------- the real run
@pytest.mark.slow
@needs_checkout
def test_the_checkpoint_actually_loads_into_the_model():
    """The silent failure: their keys are module.-prefixed and strict=False.

    Loaded as published, zero of 519 parameters match and the network stays
    random. Nothing raises, and the output still looks like a person.
    """
    import torch
    from g1dance.motion.frontends import smplestx_fe
    model, _, _ = smplestx_fe._load()
    blob = torch.load(str(WEIGHTS), map_location="cpu")["network"]
    raw_match = sum(1 for k in blob if k in model.state_dict())
    stripped = {k.replace("module.", "", 1): v for k, v in blob.items()}
    fixed_match = sum(1 for k in stripped if k in model.state_dict())
    assert raw_match == 0, "the prefix problem is gone; simplify the loader"
    assert fixed_match >= 0.9 * len(stripped)


@pytest.mark.slow
@needs_checkout
def test_it_produces_parameters_the_gmr_backend_can_consume():
    from g1dance.motion.ir import SubjectWindow
    from g1dance.track import TrackResult
    from pathlib import Path
    if not Path("tracks/kirari.tracks.npz").exists():
        pytest.skip("tracks/ is gitignored; needs a locally produced track")
    tr = TrackResult.load("tracks/kirari.tracks.npz")
    w = SubjectWindow(bbox=tr.bbox[:3], valid=tr.valid[:3], fps=tr.fps)
    seq = get_frontend("smplest_x")("clips/kirari_solo.mp4", w)
    for f in ("smplx_betas", "smplx_global_orient", "smplx_body_pose",
              "smplx_transl"):
        assert getattr(seq, f) is not None, f
    assert seq.valid.any()
    assert dict(seq.meta)["root"] == "camera_relative_per_frame"


@pytest.mark.slow
@needs_checkout
def test_the_subject_it_produces_is_the_right_way_up():
    """The end-to-end form of the convention check, on a real frame."""
    from pathlib import Path

    from g1dance.motion.bodymodel import smplx_joints
    from g1dance.motion.ir import SubjectWindow
    from g1dance.track import TrackResult
    if not Path("tracks/kirari.tracks.npz").exists():
        pytest.skip("tracks/ is gitignored; needs a locally produced track")
    tr = TrackResult.load("tracks/kirari.tracks.npz")
    w = SubjectWindow(bbox=tr.bbox[:3], valid=tr.valid[:3], fps=tr.fps)
    seq = get_frontend("smplest_x")("clips/kirari_solo.mp4", w)
    j = smplx_joints(betas=np.asarray(seq.smplx_betas),
                     global_orient=np.asarray(seq.smplx_global_orient),
                     body_pose=np.asarray(seq.smplx_body_pose))
    assert float(np.median(j[:, 15, 2] - j[:, 10, 2])) > 1.0
