"""Track stage tests on a synthetic clip with known cuts and known identities.

No network and no real footage: the clip is drawn here, so the ground truth
(where the cuts are, who is who, and which shot the subject genuinely leaves)
is known exactly rather than eyeballed. Detection itself is stubbed by a colour
blob detector registered into the same registry MediaPipe uses, which is the
point of the registry: association, descriptors and propagation are testable
without a 6 MB model download.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from g1dance.track import (
    ABSENT, SUBJECT, TrackResult, build_tracklets, detect_shots,
    detect_people, register_detector, similarity, track,
)
from g1dance.track.detect import Detection, crop_hist
from g1dance.track.propagate import analyse, propagate

WIDTH, HEIGHT = 360, 640
SHOT_LEN = 45
RED = (40, 40, 220)
BLUE = (220, 60, 40)
BACKGROUNDS = [(40, 40, 40), (205, 205, 205), (60, 150, 60)]

#: Landmark template in bbox-normalised coordinates, in MediaPipe's order.
_TEMPLATE = np.array(
    [[0.50, 0.06]] * 11
    + [[0.30, 0.20], [0.70, 0.20], [0.18, 0.35], [0.82, 0.35],
       [0.12, 0.50], [0.88, 0.50], [0.10, 0.54], [0.90, 0.54],
       [0.09, 0.56], [0.91, 0.56], [0.11, 0.57], [0.89, 0.57],
       [0.38, 0.55], [0.62, 0.55], [0.36, 0.75], [0.64, 0.75],
       [0.35, 0.95], [0.65, 0.95], [0.33, 0.97], [0.67, 0.97],
       [0.40, 0.99], [0.60, 0.99]], np.float32)


def draw_figure(canvas: np.ndarray, x: int, y: int, colour: tuple[int, int, int],
                height: int = 400) -> None:
    """A blocky dancer, solid enough that its crop histogram means something."""
    w = height // 3
    cv2.circle(canvas, (x, y + height // 12), height // 12, colour, -1)
    cv2.rectangle(canvas, (x - w // 2, y + height // 6),
                  (x + w // 2, y + int(height * 0.58)), colour, -1)
    cv2.rectangle(canvas, (x - w // 2 - 12, y + int(height * 0.18)),
                  (x - w // 2 + 6, y + int(height * 0.55)), colour, -1)
    cv2.rectangle(canvas, (x + w // 2 - 6, y + int(height * 0.18)),
                  (x + w // 2 + 12, y + int(height * 0.55)), colour, -1)
    cv2.rectangle(canvas, (x - w // 2 + 4, y + int(height * 0.55)),
                  (x - 4, y + height), colour, -1)
    cv2.rectangle(canvas, (x + 4, y + int(height * 0.55)),
                  (x + w // 2 - 4, y + height), colour, -1)


#: (background index, [(colour, x_start, x_end[, height])]) per shot.
SHOTS = [
    (0, [(RED, 90, 140), (BLUE, 270, 240)]),
    (1, [(RED, 250, 200), (BLUE, 90, 130)]),
    (2, [(BLUE, 160, 200)]),
]


def write_clip(path, shots=SHOTS, fps: float = 30.0) -> str:
    """Render the synthetic clip. Cuts land at multiples of SHOT_LEN."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (WIDTH, HEIGHT))
    assert writer.isOpened(), "opencv could not open an mp4 writer"
    for background, people in shots:
        for i in range(SHOT_LEN):
            t = i / (SHOT_LEN - 1)
            canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUNDS[background], np.uint8)
            for colour, x0, x1, *rest in people:
                draw_figure(canvas, int(round(x0 + (x1 - x0) * t)), 110, colour,
                            height=rest[0] if rest else 400)
            writer.write(canvas)
    writer.release()
    return str(path)


# ----------------------------------------------------------- stub detector
_COLOUR_RANGES = {
    "red": [((0, 120, 80), (10, 255, 255)), ((170, 120, 80), (180, 255, 255))],
    "blue": [((100, 120, 80), (130, 255, 255))],
}


class BlobDetector:
    """Finds the drawn figures by hue, and fakes landmarks from the box.

    Stands in for MediaPipe so the rest of the stage can be tested offline.
    """

    def __init__(self, min_area: int = 2000):
        self.min_area = min_area

    def detect_frame(self, image: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        out: list[Detection] = []
        for ranges in _COLOUR_RANGES.values():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            merged = [c for c in contours if cv2.contourArea(c) >= self.min_area]
            if not merged:
                continue
            points = np.vstack(merged)
            x, y, w, h = cv2.boundingRect(points)
            bbox = np.array([x, y, x + w, y + h], np.float32)
            landmarks = _TEMPLATE * np.array([w, h], np.float32) + np.array([x, y], np.float32)
            out.append(Detection(bbox=bbox, landmarks=landmarks,
                                 visibility=np.ones(33, np.float32), score=1.0,
                                 appearance=crop_hist(image, bbox)))
        return out


register_detector("test_blob")(BlobDetector)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return write_clip(tmp_path_factory.mktemp("track") / "synthetic.mp4")


@pytest.fixture(scope="module")
def analysis(clip):
    return analyse(clip, detector="test_blob", stride=1)


# ------------------------------------------------------------------- shots
def test_shot_detection_finds_the_known_cuts(clip):
    shots = detect_shots(clip)
    assert len(shots) == 3
    assert [s for s, _ in shots] == [0, SHOT_LEN, 2 * SHOT_LEN]
    assert shots[-1][1] == 3 * SHOT_LEN


def test_shots_tile_the_clip_without_gaps(clip):
    shots = detect_shots(clip)
    for (_, end), (start, _) in zip(shots, shots[1:]):
        assert end == start


def test_uncut_clip_is_one_shot(tmp_path):
    path = write_clip(tmp_path / "single.mp4", shots=SHOTS[:1])
    assert detect_shots(path) == [(0, SHOT_LEN)]


# --------------------------------------------------------------- tracklets
def test_tracklets_associate_two_people_per_shot(analysis):
    per_shot = analysis.tracklets_by_shot
    assert [len(per_shot[i]) for i in range(3)] == [2, 2, 1]
    for shot in (0, 1):
        for tracklet in per_shot[shot]:
            assert tracklet.n_frames == SHOT_LEN, "a tracklet was broken mid shot"


def test_tracklet_ids_are_unique_across_shots(analysis):
    ids = [t.id for t in analysis.tracklets]
    assert len(ids) == len(set(ids))


def test_full_body_score_is_high_for_a_whole_dancer(analysis):
    assert all(t.full_body_score > 0.8 for t in analysis.tracklets)


def test_descriptors_separate_the_two_dancers(analysis):
    shot0 = analysis.tracklets_by_shot[0]
    shot1 = analysis.tracklets_by_shot[1]
    same = max(similarity(shot0[0], t) for t in shot1)
    other = min(similarity(shot0[0], t) for t in shot1)
    assert same > 0.9 > other
    assert other < 0.5


def test_association_refuses_a_bad_match():
    """Two detections nowhere near each other must start separate tracklets."""
    def det(x):
        bbox = np.array([x, 0, x + 50, 200], np.float32)
        return Detection(bbox=bbox,
                         landmarks=_TEMPLATE * np.array([50, 200], np.float32)
                         + np.array([x, 0], np.float32),
                         visibility=np.ones(33, np.float32))

    tracklets = build_tracklets([0, 1, 2], [[det(0)], [det(300)], [det(300)]],
                                shot=0, frame_height=400, min_length=1)
    assert len(tracklets) == 2


# ------------------------------------------------------------- propagation
def _subject_index(analysis, shot: int, colour_bbox_left: bool) -> int:
    """Tracklets are ordered by first frame then id, so pick by position."""
    tracklets = analysis.tracklets_by_shot[shot]
    lefts = [t.bboxes[0, 0] for t in tracklets]
    return int(np.argmin(lefts) if colour_bbox_left else np.argmax(lefts))


def test_propagation_follows_the_subject_across_a_cut(analysis):
    red_shot0 = _subject_index(analysis, 0, colour_bbox_left=True)
    subject = analysis.tracklets_by_shot[0][red_shot0]
    result = propagate(subject, analysis.tracklets_by_shot, analysis.shots,
                       analysis.video_meta)

    assert result.assignments[0].status == SUBJECT
    assert result.assignments[1].status == "assigned"
    # in shot 1 the red dancer is the right hand figure, not the left one
    assert result.assignments[1].tracklet == _subject_index(
        analysis, 1, colour_bbox_left=False)
    assert result.assignments[1].score > 0.9


def test_absent_when_the_subject_leaves(analysis):
    subject = analysis.tracklets_by_shot[0][_subject_index(analysis, 0, True)]
    result = propagate(subject, analysis.tracklets_by_shot, analysis.shots,
                       analysis.video_meta)

    assert result.assignments[2].status == ABSENT
    assert result.assignments[2].tracklet is None
    assert result.absent_shots == [2]
    assert not result.valid[2 * SHOT_LEN:].any()
    assert result.valid[:2 * SHOT_LEN].all()


def test_absent_shot_is_never_guessed_at_a_lower_threshold(analysis):
    """Raising the threshold drops shots; it must never invent a match."""
    subject = analysis.tracklets_by_shot[0][_subject_index(analysis, 0, True)]
    strict = propagate(subject, analysis.tracklets_by_shot, analysis.shots,
                       analysis.video_meta, threshold=0.999)
    assert [a.status for a in strict.assignments] == [SUBJECT, ABSENT, ABSENT]


def test_shot_override_wins_over_the_score(analysis):
    subject = analysis.tracklets_by_shot[0][_subject_index(analysis, 0, True)]
    result = propagate(subject, analysis.tracklets_by_shot, analysis.shots,
                       analysis.video_meta, overrides={2: 0, 1: None})
    assert result.assignments[1].status == ABSENT
    assert result.assignments[2].tracklet == 0


def test_bbox_is_filled_for_every_valid_frame(analysis):
    subject = analysis.tracklets_by_shot[0][_subject_index(analysis, 0, True)]
    result = propagate(subject, analysis.tracklets_by_shot, analysis.shots,
                       analysis.video_meta)
    boxes = result.bbox[result.valid]
    assert (boxes[:, 2] > boxes[:, 0]).all()
    assert (boxes[:, 3] > boxes[:, 1]).all()
    assert result.segments == [(0, 2 * SHOT_LEN)]


# --------------------------------------------------------------- detection
def test_detect_people_returns_one_list_per_requested_frame(clip):
    frames = [0, 10, 50, 100]
    found = detect_people(clip, frames, detector="test_blob")
    assert [len(f) for f in found] == [2, 2, 2, 1]


# ------------------------------------------------------------------- stage
def test_track_auto_picks_and_says_so(clip, tmp_path):
    result = track(clip, detector="test_blob", out_dir=tmp_path)
    assert result.meta["subject_auto"] is True
    assert "auto picked" in result.meta["subject_reason"]
    assert result.coverage > 0.5


def test_track_writes_a_contact_sheet_and_npz(clip, tmp_path):
    result = track(clip, subject=(0, 0), detector="test_blob", out_dir=tmp_path)
    sheet = tmp_path / "synthetic.contact.png"
    assert sheet.exists() and cv2.imread(str(sheet)) is not None
    assert (tmp_path / "synthetic.tracks.npz").exists()

    loaded = TrackResult.load(tmp_path / "synthetic.tracks.npz")
    assert np.array_equal(loaded.valid, result.valid)
    assert np.allclose(loaded.bbox, result.bbox)
    assert loaded.shots == result.shots
    assert [a.status for a in loaded.assignments] == [
        a.status for a in result.assignments]


def test_track_honours_an_explicit_subject(clip, tmp_path):
    result = track(clip, subject=(0, 1), detector="test_blob", out_dir=tmp_path,
                   contact_sheet=False)
    assert result.meta["subject_auto"] is False
    assert result.meta["subject"] == [0, 1]


def test_track_rejects_an_impossible_subject(clip, tmp_path):
    with pytest.raises(ValueError, match="tracklets"):
        track(clip, subject=(0, 9), detector="test_blob", out_dir=tmp_path,
              contact_sheet=False)


def test_ui_is_not_this_module(clip):
    with pytest.raises(NotImplementedError):
        track(clip, ui=True, detector="test_blob")


# ------------------------------------------------------------ model cache
def test_cached_model_is_reused_without_a_download(tmp_path, monkeypatch):
    from g1dance.track import detect as detect_mod

    cached = tmp_path / "pose_landmarker_lite.task"
    cached.write_bytes(b"not really a model, but it is on disk")

    def explode(*args, **kwargs):
        raise AssertionError("a cached model must not be downloaded again")

    monkeypatch.setattr(detect_mod.urllib.request, "urlopen", explode)
    assert detect_mod.ensure_pose_model("lite", cache_dir=tmp_path) == cached


def test_offline_download_failure_is_actionable(tmp_path, monkeypatch):
    from g1dance.track import ModelDownloadError
    from g1dance.track import detect as detect_mod

    def offline(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(detect_mod.urllib.request, "urlopen", offline)
    with pytest.raises(ModelDownloadError, match="not cached"):
        detect_mod.ensure_pose_model("lite", cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.part")), "a partial download was left behind"


def test_unknown_detector_is_named():
    from g1dance.track import DetectError, get_detector

    with pytest.raises(DetectError, match="unknown detector"):
        get_detector("nope")
