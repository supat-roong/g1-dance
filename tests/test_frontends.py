"""Tests for the frontend registry, the SMPL mapping table and crop geometry.

No network and no video. What is testable without either is exactly what is
worth testing: the registry contract the CLI depends on, the SMPL joint mapping
(a silent transposition there produces a dance with the arms swapped, which
looks plausible), and the crop arithmetic that decides where a hand is looked
for. The mediapipe models themselves are covered by the integration smoke test,
since running them needs a downloaded bundle and a Metal device.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1dance.motion.frontends import (
    available_frontends, get_frontend, register_frontend)
from g1dance.motion.frontends import mediapipe_fe as mp_fe
from g1dance.motion.frontends import smpl_fe
from g1dance.motion.ir import MP_POSE, N_HAND_LANDMARKS, N_POSE_LANDMARKS, PoseSequence


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def test_the_shipped_frontends_are_registered():
    """A registry, so this is a floor rather than an equality.

    It was an equality over two names until `smplest_x` was added, which is the
    wrong shape for a seam: a new entry should not break a test about the old
    ones. What matters is that nothing shipped has gone missing.
    """
    assert {"mediapipe", "smpl"} <= set(available_frontends())


def test_get_frontend_returns_something_with_the_uniform_signature():
    """Not the raw estimator: the registry adapts each one to `(source, subject)`.

    The adapter is what removed `if a.frontend == "smpl"` from the CLI, so the
    contract to assert here is the signature rather than object identity.
    """
    import inspect
    for name in ("mediapipe", "smpl"):
        fn = get_frontend(name)
        assert callable(fn)
        assert list(inspect.signature(fn).parameters)[:2] == ["source", "subject"]


def test_unknown_frontend_names_the_alternatives():
    with pytest.raises(ValueError, match="mediapipe, smpl"):
        get_frontend("openpose")


def test_registering_over_an_existing_name_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        register_frontend("smpl", lambda: None)


# --------------------------------------------------------------------------
# SMPL joint mapping
# --------------------------------------------------------------------------

def synthetic_joints(n_frames: int, n_joints: int) -> np.ndarray:
    """Joints whose position encodes their own index, so a swap is visible."""
    t = np.arange(n_frames, dtype=np.float32)[:, None]
    j = np.broadcast_to(np.arange(n_joints, dtype=np.float32),
                        (n_frames, n_joints))
    return np.stack([j, j * 10.0 + t, j * 100.0], axis=-1).astype(np.float32)


@pytest.mark.parametrize("layout,n_joints", [("smpl", 24), ("smplx", 55)])
def test_every_mapped_landmark_receives_its_source_joint(layout, n_joints):
    joints = synthetic_joints(5, n_joints)
    landmarks, visibility = smpl_fe.smpl_joints_to_landmarks(joints, layout)
    assert landmarks.shape == (5, N_POSE_LANDMARKS, 3)

    mapping = smpl_fe.pose_mapping(layout)
    src = np.fromiter(mapping.keys(), int, len(mapping))
    dst = np.fromiter(mapping.values(), int, len(mapping))
    hip = 0.5 * (joints[:, [k for k, v in mapping.items()
                            if v == MP_POSE["left_hip"]][0]]
                 + joints[:, [k for k, v in mapping.items()
                              if v == MP_POSE["right_hip"]][0]])
    expected = joints[:, src] - hip[:, None, :]
    assert np.allclose(landmarks[:, dst], expected, atol=1e-4)
    assert np.all(visibility[:, dst] == 1.0)


@pytest.mark.parametrize("layout,n_joints", [("smpl", 24), ("smplx", 55)])
def test_unmappable_landmarks_are_zero_with_zero_visibility(layout, n_joints):
    landmarks, visibility = smpl_fe.smpl_joints_to_landmarks(
        synthetic_joints(3, n_joints), layout)
    for name in smpl_fe.UNMAPPABLE:
        i = smpl_fe.MP_INDEX[name]
        assert visibility[:, i].max() == 0.0, name
        assert np.allclose(landmarks[:, i], 0.0), name


@pytest.mark.parametrize("layout", ["smpl", "smplx"])
def test_the_mapping_is_injective(layout):
    targets = list(smpl_fe.pose_mapping(layout).values())
    assert len(targets) == len(set(targets))


@pytest.mark.parametrize("layout", ["smpl", "smplx"])
def test_the_mapping_covers_every_landmark_retargeting_uses_but_the_heels(layout):
    covered = set(smpl_fe.pose_mapping(layout).values())
    needed = {v for k, v in MP_POSE.items() if not k.endswith("heel")}
    assert needed <= covered
    assert MP_POSE["left_heel"] not in covered


def test_the_mapping_does_not_swap_left_and_right():
    mapping = smpl_fe.pose_mapping("smpl")
    names = smpl_fe.SMPL_JOINT_NAMES
    for src, dst in mapping.items():
        source, target = names[src], smpl_fe.MP_LANDMARK_NAMES[dst]
        if source.startswith("left_"):
            assert target.startswith("left_"), (source, target)
        if source.startswith("right_"):
            assert target.startswith("right_"), (source, target)


def test_hip_centring_puts_the_hip_midpoint_at_the_origin():
    landmarks, _ = smpl_fe.smpl_joints_to_landmarks(
        synthetic_joints(4, 24), "smpl")
    mid = 0.5 * (landmarks[:, MP_POSE["left_hip"]]
                 + landmarks[:, MP_POSE["right_hip"]])
    assert np.allclose(mid, 0.0, atol=1e-4)


def test_smplx_fingers_map_to_wrist_centred_hand_landmarks():
    joints = synthetic_joints(3, 55)
    left, right, conf = smpl_fe.smpl_joints_to_hands(joints, "smplx")
    assert left.shape == (3, N_HAND_LANDMARKS, 3)
    assert np.allclose(conf, 1.0)
    assert np.allclose(left[:, 0], 0.0)              # wrist is the origin
    mapping = smpl_fe.hand_mapping("smplx", "left")
    wrist = joints[:, mapping[0]]
    for mp_i, src in mapping.items():
        assert np.allclose(left[:, mp_i], joints[:, src] - wrist, atol=1e-4)
    for tip in (1, 8, 12, 16, 20):                   # tips SMPL-X does not have
        assert np.allclose(left[:, tip], 0.0)


def test_smpl_has_no_fingers_and_says_so():
    left, right, conf = smpl_fe.smpl_joints_to_hands(synthetic_joints(3, 24), "smpl")
    assert np.allclose(left, 0.0) and np.allclose(right, 0.0)
    assert np.allclose(conf, 0.0)


@pytest.mark.parametrize("n,expected", [(24, "smpl"), (52, "smplx"), (55, "smplx")])
def test_layout_detection(n, expected):
    assert smpl_fe.detect_layout(n) == expected


def test_layout_detection_refuses_an_unknown_count():
    with pytest.raises(smpl_fe.SmplLayoutError, match="no known layout"):
        smpl_fe.detect_layout(17)


# --------------------------------------------------------------------------
# SMPL file reading
# --------------------------------------------------------------------------

def test_estimate_from_smpl_reads_an_npz_of_joints(tmp_path):
    path = tmp_path / "synthetic.npz"
    np.savez(path, joints=synthetic_joints(30, 55), mocap_framerate=120.0)
    seq = smpl_fe.estimate_from_smpl(path)
    assert isinstance(seq, PoseSequence)
    assert seq.fps == 120.0
    assert seq.n_frames == 30
    assert seq.valid.all()
    assert seq.meta["layout"] == "smplx"
    assert seq.meta["frontend"] == "smpl"


def test_estimate_from_smpl_honours_an_explicit_fps(tmp_path):
    path = tmp_path / "nofps.npz"
    np.savez(path, joints=synthetic_joints(10, 24))
    assert smpl_fe.estimate_from_smpl(path, fps=30.0).fps == 30.0
    with pytest.raises(ValueError, match="frame rate"):
        smpl_fe.estimate_from_smpl(path)


def test_estimate_from_smpl_marks_non_finite_frames_invalid(tmp_path):
    joints = synthetic_joints(12, 24)
    joints[5] = np.nan
    path = tmp_path / "holes.npz"
    np.savez(path, joints=joints, fps=30.0)
    seq = smpl_fe.estimate_from_smpl(path)
    assert not seq.valid[5]
    assert seq.valid.sum() == 11


def _has_body_model() -> bool:
    from g1dance.motion.bodymodel import body_model_path
    return body_model_path().exists()


def test_amass_parameters_are_read_through_the_body_model(tmp_path):
    """The AMASS layout proper: pose parameters, no joint positions.

    This used to raise. The body model is now run when one is present, which is
    the whole reason the SMPL frontend can read anything a modern estimator
    emits; when it is absent the error says how to obtain it, which is the
    branch asserted below.
    """
    path = tmp_path / "amass.npz"
    np.savez(path, poses=np.zeros((10, 165), np.float32),
             trans=np.zeros((10, 3), np.float32),
             betas=np.zeros(16, np.float32), mocap_framerate=120.0)
    if _has_body_model():
        seq = smpl_fe.estimate_from_smpl(path)
        assert seq.n_frames == 10 and seq.valid.all()
    else:
        from g1dance.motion.bodymodel import BodyModelError
        with pytest.raises(BodyModelError, match="smpl-x.is.tue.mpg.de"):
            smpl_fe.estimate_from_smpl(path)


def test_gem_parameters_are_read_through_the_body_model(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "smpl_param.pt"
    torch.save({"body_pose": torch.zeros(10, 63), "betas": torch.zeros(10, 10),
                "global_orient": torch.zeros(10, 3), "transl": torch.zeros(10, 3),
                "fps": torch.tensor(30.0)},
               path)
    if _has_body_model():
        assert smpl_fe.estimate_from_smpl(path).n_frames == 10
    else:
        from g1dance.motion.bodymodel import BodyModelError
        with pytest.raises(BodyModelError, match="smpl-x.is.tue.mpg.de"):
            smpl_fe.estimate_from_smpl(path)


def test_a_pt_file_carrying_joints_is_read(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "joints.pt"
    torch.save({"joints": torch.from_numpy(synthetic_joints(8, 24)), "fps": 30.0},
               path)
    seq = smpl_fe.estimate_from_smpl(path)
    assert seq.n_frames == 8 and seq.fps == 30.0 and seq.meta["layout"] == "smpl"


def test_unsupported_container_is_rejected(tmp_path):
    path = tmp_path / "motion.bvh"
    path.write_text("nope")
    with pytest.raises(NotImplementedError, match="unsupported SMPL container"):
        smpl_fe.estimate_from_smpl(path)


# --------------------------------------------------------------------------
# mediapipe crop geometry
# --------------------------------------------------------------------------

def test_expand_bbox_grows_and_clips():
    assert mp_fe.expand_bbox((100, 100, 200, 300), 640, 480, expand=0.2) == (
        90, 80, 210, 320)
    # a box against the frame edge is clipped, never pushed outside
    assert mp_fe.expand_bbox((0, 0, 50, 50), 640, 480, expand=1.0) == (0, 0, 75, 75)


def test_letterbox_preserves_aspect_ratio():
    crop = np.zeros((100, 50, 4), np.uint8)
    canvas, scale, pad_x, pad_y = mp_fe.letterbox(crop, 256)
    assert canvas.shape == (256, 256, 4)
    assert scale == pytest.approx(2.56)
    assert (pad_x, pad_y) == (64, 0)          # padded on the narrow axis only


def test_letterbox_refuses_an_empty_crop():
    with pytest.raises(ValueError, match="empty crop"):
        mp_fe.letterbox(np.zeros((0, 10, 4), np.uint8))


def test_unletterbox_round_trips_a_known_point():
    box = (100, 40, 300, 440)                 # 200 wide, 400 tall
    crop = np.zeros((400, 200, 4), np.uint8)
    _, scale, pad_x, pad_y = mp_fe.letterbox(crop, 256)
    for px, py in [(0.0, 0.0), (199.0, 399.0), (50.0, 120.0)]:
        norm = [((px * scale + pad_x) / 256.0), ((py * scale + pad_y) / 256.0)]
        got = mp_fe.unletterbox_points(np.array([norm]), box, scale, pad_x, pad_y)
        assert got[0] == pytest.approx([box[0] + px, box[1] + py], abs=1e-6)


def test_wrist_roi_is_square_and_reaches_past_the_wrist():
    pose = np.zeros((33, 2))
    pose[13] = (500.0, 100.0)                 # left elbow
    pose[15] = (500.0, 200.0)                 # left wrist, 100 px below
    vis = np.ones(33)
    vis[[17, 19, 21]] = 0.0                   # no coarse hand points
    roi = mp_fe.wrist_roi(pose, vis, "left", 2000, 2000)
    assert roi is not None
    x0, y0, x1, y1 = roi
    assert (x1 - x0) == pytest.approx(y1 - y0, abs=1)
    centre_y = 0.5 * (y0 + y1)
    assert centre_y > 200.0                   # pushed beyond the wrist
    assert y0 < 200.0 < y1                    # but still containing it


def test_wrist_roi_widens_to_hold_the_coarse_hand_points():
    pose = np.zeros((33, 2))
    pose[13], pose[15] = (100.0, 100.0), (100.0, 200.0)
    vis = np.ones(33)
    vis[[17, 19, 21]] = 0.0
    tight = mp_fe.wrist_roi(pose, vis, "left", 2000, 2000)
    vis[[17, 19, 21]] = 1.0
    pose[[17, 19, 21]] = (400.0, 200.0)       # hand flung sideways
    wide = mp_fe.wrist_roi(pose, vis, "left", 2000, 2000)
    assert (wide[2] - wide[0]) > (tight[2] - tight[0])
    assert wide[0] <= 400.0 <= wide[2]


def test_wrist_roi_gives_up_when_the_wrist_is_not_visible():
    pose = np.zeros((33, 2))
    pose[13], pose[15] = (100.0, 100.0), (100.0, 200.0)
    vis = np.ones(33)
    vis[15] = 0.1
    assert mp_fe.wrist_roi(pose, vis, "left", 640, 480) is None


def test_wrist_roi_clips_to_the_frame():
    pose = np.zeros((33, 2))
    pose[14], pose[16] = (600.0, 100.0), (630.0, 20.0)
    roi = mp_fe.wrist_roi(pose, np.ones(33), "right", 640, 480)
    assert roi is not None
    assert roi[0] >= 0 and roi[1] >= 0 and roi[2] <= 640 and roi[3] <= 480


def test_to_ir_frame_turns_mediapipe_y_down_into_y_up():
    got = mp_fe.to_ir_frame(np.array([[1.0, 2.0, 3.0]], np.float32))
    assert got == pytest.approx(np.array([[1.0, -2.0, -3.0]]))
    # a rotation, so lengths and handedness survive
    pts = np.random.default_rng(0).normal(size=(8, 3)).astype(np.float32)
    out = mp_fe.to_ir_frame(pts)
    assert np.allclose(np.linalg.norm(out, axis=-1), np.linalg.norm(pts, axis=-1),
                       atol=1e-5)
    assert np.linalg.det(np.diag([1.0, -1.0, -1.0])) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# model bundle bookkeeping
# --------------------------------------------------------------------------

def test_every_bundle_url_points_at_a_task_file():
    for name, (url, size) in mp_fe.MODEL_BUNDLES.items():
        assert url.startswith("https://storage.googleapis.com/mediapipe-models/")
        assert url.endswith(".task"), name
        assert size > 1_000_000, name


def test_missing_model_reports_the_url_instead_of_downloading(tmp_path):
    with pytest.raises(mp_fe.ModelUnavailableError, match="pose_landmarker_full.task"):
        mp_fe.ensure_model("pose_full", tmp_path, download=False)


def test_ensure_model_returns_a_cached_bundle_without_network(tmp_path):
    path = mp_fe.model_path("hand", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really a model")
    assert mp_fe.ensure_model("hand", tmp_path, download=False) == path


def test_unknown_bundle_names_the_alternatives():
    with pytest.raises(ValueError, match="available:"):
        mp_fe.model_path("holistic_v2")


def test_estimate_validates_its_mask_shape(tmp_path):
    with pytest.raises(ValueError, match="valid_mask"):
        mp_fe.estimate(tmp_path / "missing.mp4", np.zeros((10, 4)),
                       np.ones(7, bool), 25.0)


def test_estimate_validates_its_bbox_shape(tmp_path):
    with pytest.raises(ValueError, match="bbox_per_frame"):
        mp_fe.estimate(tmp_path / "missing.mp4", np.zeros((10, 2)), None, 25.0)
