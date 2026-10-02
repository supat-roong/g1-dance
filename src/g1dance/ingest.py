"""Stage 1: get a clip onto disk and record exactly where it came from.

Source videos are copyrighted, so the repository keeps derived motion data and
a description of the source, never frames. That description is this stage's
real output: the sidecar JSON is what makes a run reproducible, and the sha256
is what tells us later whether the clip on disk is still the clip a motion was
built from.

A URL and a local path are equal citizens here. Everything downstream sees one
mp4 plus one sidecar, and never learns which of the two it started as.

Failures are specific on purpose. "ingest failed" sends the user looking at the
wrong thing; "the container has no video stream" does not.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urlparse

import cv2

#: yt-dlp gives up on a page in a great many ways. These are the ones worth
#: naming back to the user rather than dumping the raw log.
_FETCH_HINTS = (
    ("geo", "the video is geo restricted"),
    ("age", "the video is age restricted and needs cookies"),
    ("private", "the video is private"),
    ("members-only", "the video is members only"),
    ("unavailable", "the video is unavailable"),
    ("sign in", "the site demands a sign in"),
)


class IngestError(RuntimeError):
    """Base for every failure this stage can report."""


class FetchError(IngestError):
    """The source could not be downloaded or copied."""


class NoVideoStreamError(IngestError):
    """The container opened but carries no video track."""


class UnreadableVideoError(IngestError):
    """The file is missing, truncated, or not a media container at all."""


@dataclass(frozen=True)
class VideoInfo:
    """What every later stage needs to know about a clip's geometry."""

    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class IngestResult:
    """The clip plus the provenance record written beside it."""

    path: Path
    sidecar: Path
    source: str
    width: int
    height: int
    fps: float
    duration: float
    n_frames: int
    sha256: str
    trim: tuple[float, float] | None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        d["sidecar"] = str(self.sidecar)
        d["trim"] = list(self.trim) if self.trim else None
        return d


def is_url(source: str) -> bool:
    """True for things yt-dlp should handle rather than the filesystem."""
    return urlparse(str(source)).scheme in ("http", "https")


def ffmpeg_exe() -> str:
    """The ffmpeg binary imageio ships, so the user installs nothing."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: list[str], what: str, error: type[IngestError]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise error(f"{what}: could not run {cmd[0]!r} ({exc})") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        raise error(f"{what}:\n" + "\n".join(tail))
    return proc.stdout


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _describe_streams(path: Path) -> str:
    """ffmpeg's own report on a file. Non-zero exit is normal here, since we
    give it no output, so we read stderr rather than trust the return code."""
    try:
        proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                              capture_output=True, text=True)
    except OSError as exc:
        raise UnreadableVideoError(f"could not run ffmpeg: {exc}") from exc
    return proc.stderr or ""


def probe_video(path: str | Path) -> VideoInfo:
    """Geometry of a clip, with a specific error for each way it can fail.

    Shared with the track stage, which needs frame counts and frame size but
    has no business reopening this question itself.
    """
    path = Path(path)
    if not path.exists():
        raise UnreadableVideoError(f"no such file: {path}")
    if path.stat().st_size == 0:
        raise UnreadableVideoError(f"file is empty: {path}")

    report = _describe_streams(path)
    if "Stream #" in report and not re.search(r"Stream #\S+: Video:", report):
        raise NoVideoStreamError(f"{path} has no video stream (audio only?)")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            if "Stream #" not in report:
                raise UnreadableVideoError(
                    f"{path} is not a readable media container")
            raise NoVideoStreamError(f"{path} has no decodable video stream")
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        ok, _ = cap.read()
    finally:
        cap.release()

    if not ok or width <= 0 or height <= 0:
        raise NoVideoStreamError(f"{path} yielded no decodable frames")
    if fps <= 0:
        raise UnreadableVideoError(f"{path} reports a non-positive frame rate")
    if n_frames <= 0:
        n_frames = _count_frames(path)
    return VideoInfo(path=path, width=width, height=height, fps=fps,
                     n_frames=n_frames)


def _count_frames(path: Path) -> int:
    """Fallback for containers whose header lies about frame count."""
    cap = cv2.VideoCapture(str(path))
    n = 0
    try:
        while cap.grab():
            n += 1
    finally:
        cap.release()
    if n == 0:
        raise NoVideoStreamError(f"{path} yielded no decodable frames")
    return n


def _yt_dlp_cmd() -> list[str]:
    """Prefer the venv's console script, fall back to the module."""
    exe = Path(sys.executable).with_name("yt-dlp")
    return [str(exe)] if exe.exists() else [sys.executable, "-m", "yt_dlp"]


def _fetch(url: str, dest: Path) -> Path:
    """Download `url` to `dest`, an mp4 path, via yt-dlp."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = _yt_dlp_cmd() + [
        "--no-playlist", "--no-progress", "--merge-output-format", "mp4",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "-o", str(dest), url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise FetchError(f"could not run yt-dlp: {exc}") from exc
    if proc.returncode != 0 or not dest.exists():
        log = (proc.stderr or proc.stdout or "").strip()
        low = log.lower()
        hint = next((msg for key, msg in _FETCH_HINTS if key in low), None)
        tail = "\n".join(log.splitlines()[-6:])
        raise FetchError(
            f"could not fetch {url}"
            + (f": {hint}" if hint else "")
            + (f"\n{tail}" if tail else ""))
    return dest


def _copy_or_remux(src: Path, dest: Path) -> Path:
    """Local files are copied when already mp4, remuxed when not."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".mp4":
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        return dest
    _run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
          "-i", str(src), "-c", "copy", str(dest)],
         f"could not remux {src} into mp4", UnreadableVideoError)
    return dest


def _trim(src: Path, dest: Path, start: float, end: float) -> Path:
    """Re-encode the requested window. Stream copy would snap to keyframes,
    and a dance clip that starts a second early is a retargeting problem."""
    if end <= start:
        raise IngestError(f"trim end ({end}) must be after start ({start})")
    _run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
          "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
          "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
          "-an", str(dest)],
         f"could not trim {src} to [{start}, {end}]", IngestError)
    if not dest.exists() or dest.stat().st_size == 0:
        raise IngestError(
            f"trim [{start}, {end}] produced nothing; is it past the end?")
    return dest


def _default_name(source: str) -> str:
    stem = Path(urlparse(source).path or source).stem or "clip"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return slug or "clip"


def ingest(source: str, out_dir: str | Path, trim: tuple[float, float] | None = None,
           name: str | None = None) -> IngestResult:
    """Fetch or copy `source` into `out_dir` as `<name>.mp4` plus a sidecar.

    Returns the provenance record that was written, so a caller does not have
    to read the JSON back to learn what it got.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or _default_name(str(source))
    clip = out_dir / f"{name}.mp4"

    staged = out_dir / f".{name}.src.mp4" if trim else clip
    if is_url(str(source)):
        _fetch(str(source), staged)
    else:
        src = Path(source).expanduser()
        if not src.exists():
            raise UnreadableVideoError(f"no such file: {src}")
        _copy_or_remux(src, staged)

    try:
        probe_video(staged)
        if trim:
            _trim(staged, clip, float(trim[0]), float(trim[1]))
    finally:
        if trim and staged.exists():
            staged.unlink()

    info = probe_video(clip)
    result = IngestResult(
        path=clip, sidecar=out_dir / f"{name}.ingest.json", source=str(source),
        width=info.width, height=info.height, fps=info.fps,
        duration=info.duration, n_frames=info.n_frames,
        sha256=sha256_file(clip),
        trim=(float(trim[0]), float(trim[1])) if trim else None)
    result.sidecar.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    return result


def load_sidecar(path: str | Path) -> dict:
    """Read a `<name>.ingest.json` and fail loudly if it is not one."""
    data = json.loads(Path(path).read_text())
    missing = {"source", "fps", "sha256"} - set(data)
    if missing:
        raise IngestError(f"{path} is not an ingest sidecar (missing {sorted(missing)})")
    return data
