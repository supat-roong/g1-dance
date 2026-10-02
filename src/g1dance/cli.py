"""Command line entry point.

Each stage is a verb that reads and writes a file artifact, so any stage can be
rerun, inspected or replaced on its own. `run` chains them under one config.

Imports are deferred into each handler on purpose: the heavy optional
dependencies (MediaPipe, torch, dex-retargeting) belong to different stages,
and `g1dance --help` should not pay for any of them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULTS = dict(clips="clips", tracks="tracks", raw="raw", motion="motion", runs="runs")


def _p(root: Path, kind: str, name: str, ext: str) -> Path:
    d = root / DEFAULTS[kind]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}{ext}"


def cmd_ingest(a: argparse.Namespace) -> int:
    from .ingest import ingest
    trim = (a.start, a.end) if a.start is not None or a.end is not None else None
    res = ingest(a.source, a.root / DEFAULTS["clips"], trim=trim, name=a.name)
    print(f"ingested {res.path}  {res.width}x{res.height} @ {res.fps:.2f}fps  {res.duration:.1f}s")
    return 0


def cmd_track(a: argparse.Namespace) -> int:
    from .track import track
    res = track(a.video, subject=a.subject, ui=a.ui)
    out = _p(a.root, "tracks", a.name, ".tracks.npz")
    res.save(out)
    frac = float(res.valid.mean())
    print(f"tracked -> {out}  subject valid on {frac:.1%} of frames, "
          f"{len(res.shots)} shot(s)")
    return 0


def cmd_estimate(a: argparse.Namespace) -> int:
    from .motion.frontends import get_frontend
    from .track import TrackResult
    fe = get_frontend(a.frontend)
    # One convention for every frontend: the source it consumes, plus where the
    # subject is if we know. A frontend that does not need the window ignores it,
    # which is what removed the `if a.frontend == "smpl"` branch that used to
    # make adding a frontend a change to this file.
    source = a.source or a.video
    if source is None:
        raise SystemExit("estimate needs --source (or --video) to read from")
    tracks = _p(a.root, "tracks", a.name, ".tracks.npz")
    subject = (TrackResult.load(tracks).subject_window()
               if tracks.exists() else None)
    seq = fe(source, subject)
    from .motion.hands import attach, get_hand_estimator
    he = get_hand_estimator(a.hand_estimator)
    seq = attach(seq, he(source, subject, seq))
    # Isolated dropouts are detector noise, not the subject leaving frame, and
    # left alone each one splits the motion: a 78 s clip at 99.3 percent
    # coverage arrived as 15 segments because 29 scattered frames, in gaps of
    # one to four, cut it up. Segments matter because episodes truncate at
    # their boundaries, so the fragments are unusable training length.
    from .motion.clean import fill_gaps
    before = len(seq.segments)
    seq = fill_gaps(seq, max_gap_s=a.max_gap)
    out = _p(a.root, "raw", a.name, ".pose.npz")
    seq.save(out)
    merged = f", {before} -> {len(seq.segments)} segments" if before != len(seq.segments) else ""
    print(f"estimated -> {out}  {seq.n_frames} frames, "
          f"{seq.valid.mean():.1%} valid, {len(seq.segments)} segment(s){merged}")
    return 0


def cmd_retarget(a: argparse.Namespace) -> int:
    from .motion.ir import PoseSequence
    from .motion.retargeters import get_retargeter
    from .sim.model import G1Model
    seq = PoseSequence.load(_p(a.root, "raw", a.name, ".pose.npz"))
    motion = get_retargeter(a.retargeter)(seq, G1Model(), cfg=_load_cfg(a.config))
    out = _p(a.root, "motion", a.name, ".npz")
    motion.save(out)
    print(f"retargeted -> {out}  {motion.n_frames} frames @ {motion.fps}Hz, "
          f"{motion.duration:.1f}s, {len(motion.segments)} segment(s)")
    return 0


def cmd_preview(a: argparse.Namespace) -> int:
    from .motion.ir import ReferenceMotion
    from .viz.render import render_reference
    motion = ReferenceMotion.load(_p(a.root, "motion", a.name, ".npz"))
    out = a.out or (a.root / DEFAULTS["motion"] / f"{a.name}_preview.mp4")
    render_reference(motion, Path(out))
    print(f"preview -> {out}   (kinematic playback, no physics, no policy)")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    """Run a whole pipeline spec and write one record for it."""
    from .pipeline import PipelineSpec, run_pipeline
    from .pipeline.record import summarise
    spec = (PipelineSpec.load(a.spec) if a.spec else PipelineSpec())
    for stage in ("detector", "estimator", "hand_estimator", "retargeter",
                  "controller"):
        override = getattr(a, stage)
        if override:
            setattr(spec, stage, override)
    if a.env_config:
        spec.env_config = str(a.env_config)
    record = run_pipeline(spec, a.clip, name=a.name, exp=a.exp, root=a.root,
                          seeds=a.seeds, randomised=not a.no_randomised,
                          force=a.force)
    print()
    print(summarise(record))
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    """Print several run records side by side, without picking a winner."""
    from .pipeline.record import compare, read_record
    records = []
    for target in a.runs:
        p = Path(target)
        if p.is_dir():
            p = p / "record.json"
        if not p.exists():
            raise SystemExit(f"no record at {p}")
        records.append((Path(target).name, read_record(p)))
    print(compare(records))
    return 0


def cmd_components(a: argparse.Namespace) -> int:
    """List what is installed at each seam, without importing any of it."""
    from .motion.frontends import DESCRIPTIONS as FE_D, available_frontends
    from .motion.hands import (
        DESCRIPTIONS as HD_D, available_hand_estimators)
    from .motion.retargeters import DESCRIPTIONS as RT_D, available_retargeters
    from .control.controllers import DESCRIPTIONS as CT_D, available_controllers

    # The detector registry is populated by importing the module that defines the
    # detectors, which pulls in mediapipe. Its names are declared separately so
    # this listing stays free; see track.detect.DETECTOR_NAMES.
    from .track.detect_names import DESCRIPTIONS as DT_D, DETECTOR_NAMES

    stages = [
        ("detector", DETECTOR_NAMES, DT_D),
        ("estimator", available_frontends(), FE_D),
        ("hand estimator", available_hand_estimators(), HD_D),
        ("retargeter", available_retargeters(), RT_D),
        ("controller", available_controllers(), CT_D),
    ]
    for stage, names, desc in stages:
        print(f"{stage}:")
        for n in names:
            print(f"  {n:<16} {desc.get(n, '')}")
    return 0


def cmd_render(a: argparse.Namespace) -> int:
    """Render a run the way it is shown to people.

    Defaults come from the run's record when there is one, so a render shows
    what was measured: the same env config, the same controller and options,
    the same clip. Anything given on the command line wins.
    """
    from .pipeline.record import read_record
    from .viz.render import render_showcase
    run = a.root / DEFAULTS["runs"] / a.exp
    record = None
    if (run / "record.json").exists():
        record = read_record(run / "record.json")
    spec = (record or {}).get("spec", {})
    mpath = _p(a.root, "motion", a.name, ".npz")

    config = a.config or spec.get("env_config") or "configs/upper_body.yaml"
    cname = (a.controller or (spec.get("controller") or {}).get("name")
             or "humanoid_gpt")
    from .control.controllers import make_controller
    opts = dict((spec.get("controller") or {}).get("options") or {}) \
        if not a.controller else {}
    if a.anchor is not None:
        opts["anchor"] = a.anchor
    controller = make_controller(cname, **opts)

    video = a.video
    if video is None and record is not None and not a.no_video:
        clip = (record.get("artifacts") or {}).get("clip")
        video = Path(clip) if clip and Path(clip).exists() else None
    tracks = _p(a.root, "tracks", a.name, ".tracks.npz")
    pose = _p(a.root, "raw", a.name, ".pose.npz")

    dest = a.out or (run / ("dance_full.mp4" if a.full else "dance.mp4"))
    out, info = render_showcase(
        mpath, config, dest, controller=controller,
        video=video if not a.no_video else None,
        tracks=tracks if tracks.exists() else None,
        pose_path=pose if pose.exists() else None,
        seed=a.seed, start_frame=a.start_frame, full=a.full)
    ended = (f"terminated after {info['steps']} of {info['frames_available']}"
             f" frames ({info['reason']})" if info["terminated"]
             else f"{info['steps']} steps, no fall")
    print(f"render -> {out}   from frame {info['start_frame']}, {ended}"
          f"{', anchored' if info['anchored'] else ''}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("g1dance", description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=Path.cwd(),
                   help="working directory holding clips/, motion/, runs/")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="fetch or copy a clip")
    s.add_argument("source"); s.add_argument("--name", default=None)
    s.add_argument("--start", type=float, default=None)
    s.add_argument("--end", type=float, default=None)
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("track", help="pick and follow the subject")
    s.add_argument("video", type=Path); s.add_argument("--name", required=True)
    s.add_argument("--subject", type=int, default=None)
    s.add_argument("--ui", action="store_true")
    s.set_defaults(fn=cmd_track)

    s = sub.add_parser("estimate", help="pose and hands on the subject crop")
    s.add_argument("--name", required=True)
    s.add_argument("--video", type=Path, default=None)
    s.add_argument("--source", type=Path, default=None, help="for the smpl frontend")
    s.add_argument("--frontend", default="mediapipe",
                   help="estimator to use; see `g1dance components`")
    s.add_argument("--hand-estimator", default="mediapipe",
                   dest="hand_estimator",
                   help="hand estimator; see `g1dance components`")
    s.add_argument("--max-gap", type=float, default=0.2, dest="max_gap",
                   metavar="SECONDS",
                   help="interpolate across invalid runs no longer than this")
    s.set_defaults(fn=cmd_estimate)

    s = sub.add_parser("retarget", help="human motion to G1 joint targets")
    s.add_argument("--name", required=True)
    s.add_argument("--config", type=Path, default=None)
    s.add_argument("--retargeter", default="builtin",
                   help="retargeting backend; see `g1dance components`")
    s.set_defaults(fn=cmd_retarget)

    s = sub.add_parser("run", help="run a whole pipeline spec, and record it")
    s.add_argument("clip", type=Path)
    s.add_argument("--name", required=True)
    s.add_argument("--exp", required=True)
    s.add_argument("--spec", type=Path, default=None,
                   help="pipeline spec YAML; defaults to the built-in defaults")
    for stage in ("detector", "estimator", "hand_estimator", "retargeter",
                  "controller"):
        s.add_argument(f"--{stage.replace('_', '-')}", dest=stage, default=None,
                       help=f"override the spec's {stage}")
    s.add_argument("--env-config", type=Path, default=None, dest="env_config")
    s.add_argument("--seeds", type=int, default=12)
    s.add_argument("--no-randomised", action="store_true", dest="no_randomised",
                   help="skip the domain-randomised stability pass")
    s.add_argument("--force", action="store_true",
                   help="re-run every stage even if its artifacts are current")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("compare", help="print run records side by side")
    s.add_argument("runs", nargs="+",
                   help="run directories, or record.json paths")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("components", help="list what is installed at each seam")
    s.set_defaults(fn=cmd_components)

    s = sub.add_parser("preview", help="kinematic playback of a reference motion")
    s.add_argument("--name", required=True); s.add_argument("--out", type=Path, default=None)
    s.set_defaults(fn=cmd_preview)

    s = sub.add_parser("render", help="source, 2D pose, reference and policy, to mp4")
    s.add_argument("--name", required=True); s.add_argument("--exp", required=True)
    s.add_argument("--config", type=Path, default=None,
                   help="env config; defaults to the run record's, then upper_body")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=None)
    s.add_argument("--controller", default=None,
                   help="defaults to the run record's controller, then humanoid_gpt")
    s.add_argument("--video", type=Path, default=None,
                   help="source clip; defaults to the run record's")
    s.add_argument("--no-video", action="store_true", dest="no_video",
                   help="reference beside policy only")
    s.add_argument("--full", action="store_true",
                   help="play the whole clip from frame 0 as one episode")
    s.add_argument("--start-frame", type=int, default=None, dest="start_frame")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--anchor", action="store_true", dest="anchor", default=None,
                   help="anchor the tracker's reference to the robot at the start")
    g.add_argument("--no-anchor", action="store_false", dest="anchor")
    s.set_defaults(fn=cmd_render)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
