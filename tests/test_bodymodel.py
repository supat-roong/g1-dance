"""Running the SMPL-X body model, which is what turns parameters into joints.

`smpl_fe.py` used to refuse this, and that refusal is why the frontend could not
read anything a modern estimator produces: GVHMR, SMPLest-X and DanceHMR all
emit parameters, and GMR consumes them. The licence has not changed, only
whether the file is on this machine, so everything here skips cleanly when it is
absent and says how to obtain it.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1dance.motion.bodymodel import (
    BodyModelError, N_SMPLX_JOINTS, body_model_path, ensure_body_model,
    smplx_joints)

HAVE_MODEL = body_model_path().exists()
needs_model = pytest.mark.skipif(
    not HAVE_MODEL, reason="SMPL-X is registration gated; see bodymodel.py")


# ------------------------------------------------------------------- locating
def test_a_missing_model_says_exactly_how_to_get_it(tmp_path):
    with pytest.raises(BodyModelError, match="smpl-x.is.tue.mpg.de"):
        ensure_body_model("neutral", asset_root=tmp_path)


def test_the_error_names_the_path_it_looked_at(tmp_path):
    with pytest.raises(BodyModelError, match=str(tmp_path)):
        ensure_body_model("neutral", asset_root=tmp_path)


def test_the_path_follows_the_published_naming():
    assert body_model_path("neutral").name == "SMPLX_NEUTRAL.npz"
    assert body_model_path("male").name == "SMPLX_MALE.npz"


def test_a_licensed_model_is_never_committed():
    """The one mistake that cannot be undone by a later commit."""
    import subprocess
    out = subprocess.run(["git", "check-ignore", "assets/smplx/SMPLX_NEUTRAL.npz"],
                         capture_output=True, text=True)
    assert out.returncode == 0, "assets/smplx/ is not gitignored"


# -------------------------------------------------------------------- running
@needs_model
def test_parameters_become_the_named_joints():
    T = 4
    j = smplx_joints(betas=np.zeros((T, 10)), global_orient=np.zeros((T, 3)),
                     body_pose=np.zeros((T, 63)))
    assert j.shape == (T, N_SMPLX_JOINTS, 3)
    assert np.all(np.isfinite(j))


@needs_model
def test_the_joint_order_matches_the_name_table():
    """Index 1 is the left hip and index 2 the right, so +x is the subject's left."""
    from g1dance.motion.frontends.smpl_fe import SMPLX_JOINT_NAMES
    assert SMPLX_JOINT_NAMES[:3] == ("pelvis", "left_hip", "right_hip")
    j = smplx_joints(betas=np.zeros((1, 10)), global_orient=np.zeros((1, 3)),
                     body_pose=np.zeros((1, 63)))[0]
    assert j[1][0] > j[2][0], "left hip should sit on +x of the right hip"
    assert j[15][1] > j[0][1], "the head should sit above the pelvis"


@needs_model
def test_bending_a_joint_moves_the_limb_below_it():
    """The check that the pose is actually being applied, not ignored."""
    T = 2
    rest = smplx_joints(betas=np.zeros((T, 10)), global_orient=np.zeros((T, 3)),
                        body_pose=np.zeros((T, 63)))
    bent = np.zeros((T, 63), np.float32)
    bent[:, 17 * 3 + 2] = 1.2                       # left elbow
    moved = smplx_joints(betas=np.zeros((T, 10)),
                         global_orient=np.zeros((T, 3)), body_pose=bent)
    assert np.linalg.norm(moved[0, 20] - rest[0, 20]) > 0.1, "wrist did not move"
    assert np.linalg.norm(moved[0, 1] - rest[0, 1]) < 1e-5, "hip should not move"


@needs_model
def test_translation_moves_every_joint_together():
    T = 3
    base = smplx_joints(betas=np.zeros((T, 10)), global_orient=np.zeros((T, 3)),
                        body_pose=np.zeros((T, 63)))
    shift = np.tile(np.array([0.5, 0.0, -0.25], np.float32), (T, 1))
    moved = smplx_joints(betas=np.zeros((T, 10)), global_orient=np.zeros((T, 3)),
                         body_pose=np.zeros((T, 63)), transl=shift)
    # assert_allclose does not broadcast, so the expectation is tiled.
    want = np.broadcast_to(shift[:, None, :], moved.shape)
    np.testing.assert_allclose(moved - base, want, atol=1e-5)


@needs_model
def test_chunking_does_not_change_the_answer():
    """A long clip is evaluated in pieces; the seams must be invisible."""
    T = 10
    kw = dict(betas=np.zeros((T, 10)), global_orient=np.zeros((T, 3)),
              body_pose=np.tile(np.linspace(0, 0.4, 63).astype(np.float32), (T, 1)))
    np.testing.assert_allclose(smplx_joints(chunk=3, **kw),
                               smplx_joints(chunk=T, **kw), atol=1e-6)


@needs_model
def test_a_wrong_pose_width_is_refused_rather_than_broadcast():
    with pytest.raises(BodyModelError, match="63"):
        smplx_joints(betas=np.zeros((2, 10)), global_orient=np.zeros((2, 3)),
                     body_pose=np.zeros((2, 72)))


@needs_model
def test_one_set_of_betas_may_stand_for_the_whole_clip():
    T = 4
    j = smplx_joints(betas=np.zeros(10), global_orient=np.zeros((T, 3)),
                     body_pose=np.zeros((T, 63)))
    assert j.shape[0] == T


# ------------------------------------------------- through the smpl frontend
@needs_model
def test_an_amass_style_file_of_parameters_is_read(tmp_path):
    from g1dance.motion.frontends.smpl_fe import estimate_from_smpl
    T = 6
    np.savez(tmp_path / "a.npz", poses=np.zeros((T, 165), np.float32),
             betas=np.zeros(16, np.float32), trans=np.zeros((T, 3), np.float32),
             mocap_framerate=np.float32(30.0))
    seq = estimate_from_smpl(tmp_path / "a.npz")
    assert seq.n_frames == T and seq.valid.all()


@needs_model
def test_a_regressor_style_file_of_split_parameters_is_read(tmp_path):
    """The shape GVHMR and the other regressors emit."""
    from g1dance.motion.frontends.smpl_fe import estimate_from_smpl
    T = 6
    np.savez(tmp_path / "g.npz", global_orient=np.zeros((T, 3), np.float32),
             body_pose=np.zeros((T, 63), np.float32),
             betas=np.zeros((T, 10), np.float32),
             transl=np.zeros((T, 3), np.float32), fps=np.float32(30.0))
    seq = estimate_from_smpl(tmp_path / "g.npz")
    assert seq.n_frames == T
    assert dict(seq.meta)["units"] == "metres_hip_centred"


def test_a_file_with_neither_joints_nor_parameters_still_explains_itself(tmp_path):
    from g1dance.motion.frontends.smpl_fe import estimate_from_smpl
    np.savez(tmp_path / "n.npz", something_else=np.zeros(4), fps=np.float32(30.0))
    with pytest.raises(NotImplementedError, match="parameters"):
        estimate_from_smpl(tmp_path / "n.npz")
