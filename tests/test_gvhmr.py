"""GVHMR: the only estimator here whose root carries translation.

Every other estimator in this project is per frame and camera relative, which is
why the reference root has been the binding constraint all along: the yaw rate
cap removes an impossible signal rather than supplying a real one. GVHMR
predicts in gravity-view coordinates and emits a global trajectory. Measured on
a real clip, its root travels 0.697 m in xy where a mode A reference is pinned.

This was written off earlier as blocked four ways and three of those were wrong.
The tests below pin the two things that would silently regress: the frame
convention, and the fact that the AGPL tracker is never imported.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1dance.motion.frontends import available_frontends, get_frontend
from g1dance.motion.frontends.gvhmr_fe import (
    ROOT, GvhmrUnavailable, _assert_upright, _stub_unused_imports, _to_z_up)

HAVE = (ROOT / "inputs" / "checkpoints" / "gvhmr"
        / "gvhmr_siga24_release.ckpt").exists()
needs_checkout = pytest.mark.skipif(
    not HAVE, reason=f"no GVHMR checkout at {ROOT}; see gvhmr_fe.py")


# ------------------------------------------------------------------ registry
def test_gvhmr_is_registered():
    assert "gvhmr" in available_frontends()


def test_it_has_the_uniform_estimator_signature():
    import inspect
    fn = get_frontend("gvhmr")
    assert list(inspect.signature(fn).parameters)[:2] == ["source", "subject"]


def test_a_missing_checkout_says_where_the_weights_are(tmp_path, monkeypatch):
    from g1dance.motion.frontends import gvhmr_fe
    monkeypatch.setattr(gvhmr_fe, "ROOT", tmp_path / "absent")
    with pytest.raises(GvhmrUnavailable, match="huggingface.co/camenduru/GVHMR"):
        gvhmr_fe._prepare()


def test_it_refuses_to_run_without_a_subject_window():
    with pytest.raises(ValueError, match="subject window"):
        get_frontend("gvhmr")("x.mp4", None)


# ----------------------------------------------------------- the AGPL question
def test_the_yolo_tracker_is_stood_in_for_rather_than_installed():
    """Ultralytics is AGPL-3.0, which track/detect.py declined for this project.

    Its boxes are not needed: the estimator contract carries one per frame. But
    `preproc/__init__.py` imports the tracker, and Python initialises a parent
    package before any submodule, so the import has to be intercepted.
    """
    import sys
    _stub_unused_imports()
    assert "ultralytics" in sys.modules
    with pytest.raises(NotImplementedError, match="deliberately unavailable"):
        sys.modules["ultralytics"].YOLO("yolov8x.pt")


def test_the_stand_in_does_not_break_the_import_machinery():
    """A __getattr__ that answers dunders breaks `import`; it must not."""
    import sys
    _stub_unused_imports()
    for name in ("ultralytics", "pycolmap"):
        with pytest.raises(AttributeError):
            sys.modules[name].__path__


# ------------------------------------------------- the gravity-view convention
def test_the_rotation_takes_gravity_view_y_up_into_a_z_up_world():
    orient = np.zeros((2, 3), np.float32)
    transl = np.tile(np.array([0.0, 1.3, 0.0], np.float32), (2, 1))
    _, moved = _to_z_up(orient, transl)
    # the height was in y and must end up in z
    np.testing.assert_allclose(moved[0], [0.0, 0.0, 1.3], atol=1e-5)


def test_the_translation_is_rotated_with_the_orientation():
    """Rotating only the orientation leaves the body displaced along the old axis."""
    from scipy.spatial.transform import Rotation as R
    orient = R.from_euler("z", 0.4).as_rotvec()[None].astype(np.float32)
    transl = np.array([[0.2, 1.0, -0.3]], np.float32)
    out_o, out_t = _to_z_up(orient, transl)
    assert not np.allclose(out_t, transl), "translation was left in the old frame"
    assert not np.allclose(out_o, orient)


def test_an_upside_down_subject_is_refused():
    joints = np.zeros((3, 55, 3), np.float32)
    joints[:, 15, 2] = 0.1
    with pytest.raises(GvhmrUnavailable, match="not upright"):
        _assert_upright(joints, np.ones(3, bool))


# --------------------------------------------------------------- the real run
@pytest.mark.slow
@needs_checkout
def test_it_produces_a_root_that_actually_moves():
    """The whole point: a world-grounded root, not a pinned one."""
    from pathlib import Path

    from g1dance.motion.ir import SubjectWindow
    from g1dance.track import TrackResult
    if not Path("tracks/kirari.tracks.npz").exists():
        pytest.skip("tracks/ is gitignored; needs a locally produced track")
    tr = TrackResult.load("tracks/kirari.tracks.npz")
    w = SubjectWindow(bbox=tr.bbox[:40], valid=tr.valid[:40], fps=tr.fps)
    seq = get_frontend("gvhmr")("clips/kirari_solo.mp4", w)
    assert dict(seq.meta)["root"] == "world_grounded"
    t = np.asarray(seq.smplx_transl)
    assert float(np.linalg.norm(t.max(0) - t.min(0))) > 0.05, \
        "the root does not move; this estimator exists because it should"
    for f in ("smplx_betas", "smplx_global_orient", "smplx_body_pose",
              "smplx_transl"):
        assert getattr(seq, f) is not None, f


def test_vitpose_keypoints_land_in_the_slots_the_hand_crop_reads():
    """The hand estimator places its crop from the wrist and elbow pixels."""
    from g1dance.motion.frontends.gvhmr_fe import _landmarks_px
    from g1dance.motion.ir import MP_POSE
    kp = np.zeros((3, 17, 3), np.float32)
    kp[:, 9, :2] = [100.0, 200.0]     # COCO left wrist
    kp[:, 8, :2] = [300.0, 50.0]      # COCO right elbow
    px = _landmarks_px(kp, 2)
    assert px.shape == (2, 33, 2)
    np.testing.assert_array_equal(px[:, MP_POSE["left_wrist"]], [[100, 200]] * 2)
    np.testing.assert_array_equal(px[:, MP_POSE["right_elbow"]], [[300, 50]] * 2)
