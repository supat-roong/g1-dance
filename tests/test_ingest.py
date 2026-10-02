"""Ingest tests: the sidecar is the artifact, so the sidecar is what is checked.

Everything here runs on a clip generated on the spot. No network, because the
only part of this stage that needs one is yt-dlp, and testing yt-dlp is testing
yt-dlp. What is worth testing is the provenance record and the promise that
each way of failing reports itself as itself.
"""

from __future__ import annotations

import json
import subprocess

import cv2
import numpy as np
import pytest

from g1dance.ingest import (
    IngestError, NoVideoStreamError, UnreadableVideoError, ffmpeg_exe, ingest,
    is_url, load_sidecar, probe_video, sha256_file,
)

WIDTH, HEIGHT, FPS, N_FRAMES = 320, 240, 30.0, 60


def write_clip(path, n_frames: int = N_FRAMES, fps: float = FPS) -> str:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (WIDTH, HEIGHT))
    assert writer.isOpened(), "opencv could not open an mp4 writer"
    for i in range(n_frames):
        canvas = np.full((HEIGHT, WIDTH, 3), 30, np.uint8)
        cv2.rectangle(canvas, (i * 3 % 280, 80), (i * 3 % 280 + 40, 160),
                      (20, 30, 200), -1)
        writer.write(canvas)
    writer.release()
    return str(path)


@pytest.fixture
def clip(tmp_path):
    return write_clip(tmp_path / "source.mp4")


def test_is_url_only_for_http():
    assert is_url("https://example.com/v.mp4")
    assert not is_url("/tmp/v.mp4")
    assert not is_url("v.mp4")


def test_probe_reports_geometry(clip):
    info = probe_video(clip)
    assert (info.width, info.height) == (WIDTH, HEIGHT)
    assert info.fps == pytest.approx(FPS, rel=0.05)
    assert info.n_frames == N_FRAMES
    assert info.duration == pytest.approx(N_FRAMES / FPS, rel=0.05)


def test_ingest_writes_clip_and_sidecar(clip, tmp_path):
    out = tmp_path / "clips"
    result = ingest(clip, out, name="take1")

    assert result.path == out / "take1.mp4"
    assert result.sidecar == out / "take1.ingest.json"
    assert result.path.exists()

    sidecar = json.loads(result.sidecar.read_text())
    assert sidecar["source"] == clip
    assert sidecar["width"] == WIDTH and sidecar["height"] == HEIGHT
    assert sidecar["fps"] == pytest.approx(FPS, rel=0.05)
    assert sidecar["duration"] == pytest.approx(N_FRAMES / FPS, rel=0.05)
    assert sidecar["trim"] is None
    assert sidecar["sha256"] == sha256_file(result.path)


def test_sidecar_round_trips(clip, tmp_path):
    result = ingest(clip, tmp_path, name="take1")
    assert load_sidecar(result.sidecar)["sha256"] == result.sha256


def test_load_sidecar_rejects_other_json(tmp_path):
    path = tmp_path / "not.json"
    path.write_text('{"hello": 1}')
    with pytest.raises(IngestError, match="not an ingest sidecar"):
        load_sidecar(path)


def test_name_defaults_to_the_source_stem(clip, tmp_path):
    result = ingest(clip, tmp_path)
    assert result.path.name == "source.mp4"


def test_trim_shortens_the_clip(clip, tmp_path):
    result = ingest(clip, tmp_path, trim=(0.5, 1.5), name="cut")
    assert result.trim == (0.5, 1.5)
    assert result.duration == pytest.approx(1.0, abs=0.25)
    assert result.duration < N_FRAMES / FPS
    assert json.loads(result.sidecar.read_text())["trim"] == [0.5, 1.5]
    assert not list(tmp_path.glob(".*.src.mp4")), "the staged download was left behind"


def test_trim_must_run_forwards(clip, tmp_path):
    with pytest.raises(IngestError, match="must be after"):
        ingest(clip, tmp_path, trim=(2.0, 1.0), name="bad")


def test_missing_source_is_named_as_such(tmp_path):
    with pytest.raises(UnreadableVideoError, match="no such file"):
        ingest(tmp_path / "absent.mp4", tmp_path)


def test_unreadable_container_is_named_as_such(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video" * 100)
    with pytest.raises(IngestError):
        ingest(junk, tmp_path, name="junk_out")


def test_empty_file_is_unreadable(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(UnreadableVideoError, match="empty"):
        probe_video(empty)


def test_audio_only_is_named_as_such(tmp_path):
    audio = tmp_path / "audio.m4a"
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(audio)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not audio.exists():
        pytest.skip("this ffmpeg build cannot synthesise an audio only file")
    with pytest.raises(NoVideoStreamError):
        probe_video(audio)


def test_non_mp4_source_is_remuxed(tmp_path):
    avi = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"MJPG"), FPS,
                             (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("this opencv build cannot write MJPG avi")
    for _ in range(N_FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), 90, np.uint8))
    writer.release()

    result = ingest(avi, tmp_path / "clips", name="remuxed")
    assert result.path.suffix == ".mp4"
    assert probe_video(result.path).n_frames == N_FRAMES
