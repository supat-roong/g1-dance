"""Run a spec end to end, reusing whatever upstream work is still valid.

Caching is what makes comparison affordable. Swapping the controller should not
re-run MediaPipe over a 12 second clip, and the file-per-stage architecture
already permits that; what was missing was a key. Each artifact gets a sidecar
recording the hash of the spec that produced it, so a stage runs when its own
inputs changed and is skipped when they did not.

Nothing here decides whether a result is good. It produces the record; reading it
is the user's job, by design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .record import build_record, controller_provenance, summarise, write_record
from .spec import PipelineSpec

DIRS = dict(clips="clips", tracks="tracks", raw="raw", motion="motion",
            runs="runs")


@dataclass
class StageResult:
    path: Path
    reused: bool


def _sidecar(artifact: Path) -> Path:
    return artifact.with_suffix(artifact.suffix + ".stage.json")


def _is_fresh(artifact: Path, want: str) -> bool:
    side = _sidecar(artifact)
    if not (artifact.exists() and side.exists()):
        return False
    try:
        return json.loads(side.read_text()).get("hash") == want
    except (OSError, ValueError):
        return False


def _stamp(artifact: Path, stage: str, want: str, spec: PipelineSpec) -> None:
    _sidecar(artifact).write_text(json.dumps(
        {"stage": stage, "hash": want, "spec": spec.resolved()},
        indent=1, default=str) + "\n")


def _p(root: Path, kind: str, name: str, ext: str) -> Path:
    d = root / DIRS[kind]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}{ext}"


def run_pipeline(spec: PipelineSpec, source: str | Path, name: str, exp: str,
                 root: str | Path = ".", seeds: int = 12,
                 randomised: bool = True, force: bool = False,
                 log=print) -> dict:
    """Execute `spec` on `source` and return the record.

    `randomised` adds a second stability pass with domain randomisation on,
    because "upright on a nominal model" and "upright under the friction, mass
    and push variation the configs define" are different claims.
    """
    from ..metrics import average
    from ..motion.frontends import get_frontend
    from ..motion.ir import ReferenceMotion
    from ..motion.retargeters import get_retargeter
    from ..control.controllers import make_controller
    from ..control.evaluate import evaluate
    from ..sim.model import G1Model
    from ..track import TrackResult, track

    spec.validate()
    root = Path(root)
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"no clip at {source}")

    tracks_p = _p(root, "tracks", name, ".tracks.npz")
    pose_p = _p(root, "raw", name, ".pose.npz")
    motion_p = _p(root, "motion", name, ".npz")

    # ---------------------------------------------------------------- detect
    want = spec.stage_hash("detector", source)
    if force or not _is_fresh(tracks_p, want):
        log(f"track     {source}  [{spec.name('detector')}]")
        res = track(source, detector=spec.name("detector"),
                    **spec.options("detector"))
        res.save(tracks_p)
        _stamp(tracks_p, "detector", want, spec)
    else:
        log(f"track     reusing {tracks_p.name}")

    # -------------------------------------------------------------- estimate
    # Keyed on the hand stage, not the body one: both write into the same pose
    # artifact, so changing either has to invalidate it.
    want = spec.stage_hash("hand_estimator", source)
    if force or not _is_fresh(pose_p, want):
        log(f"estimate  [{spec.name('estimator')}] "
            f"hands [{spec.name('hand_estimator')}]")
        from ..motion.clean import fill_gaps
        from ..motion.hands import attach, get_hand_estimator
        fe = get_frontend(spec.name("estimator"))
        subject = TrackResult.load(tracks_p).subject_window()
        seq = fe(source, subject, **spec.options("estimator"))
        he = get_hand_estimator(spec.name("hand_estimator"))
        seq = attach(seq, he(source, subject, seq,
                             **spec.options("hand_estimator")))
        seq = fill_gaps(seq, max_gap_s=0.2)
        seq.save(pose_p)
        _stamp(pose_p, "hand_estimator", want, spec)
    else:
        log(f"estimate  reusing {pose_p.name}")

    # -------------------------------------------------------------- retarget
    want = spec.stage_hash("retargeter", source)
    if force or not _is_fresh(motion_p, want):
        log(f"retarget  [{spec.name('retargeter')}]")
        from ..motion.ir import PoseSequence
        cfg = {}
        if spec.retarget_config:
            import yaml
            cfg = yaml.safe_load(Path(spec.retarget_config).read_text()) or {}
        cfg = {**cfg, **spec.options("retargeter")}
        motion = get_retargeter(spec.name("retargeter"))(
            PoseSequence.load(pose_p), G1Model(), cfg=cfg or None)
        motion.save(motion_p)
        _stamp(motion_p, "retargeter", want, spec)
    else:
        log(f"retarget  reusing {motion_p.name}")

    # --------------------------------------------------------------- measure
    run_dir = root / DIRS["runs"] / exp
    run_dir.mkdir(parents=True, exist_ok=True)
    cname = spec.name("controller")
    log(f"evaluate  [{cname}]  {seeds} seeds, nominal")
    controller = make_controller(cname, **spec.options("controller"))
    nominal = evaluate(motion_p, spec.env_config, seeds=seeds,
                       out=run_dir / "report.json",
                       controllers={cname: controller}, randomize=False)
    random_rep = None
    if randomised:
        log(f"evaluate  [{cname}]  {seeds} seeds, domain randomised")
        random_rep = evaluate(motion_p, spec.env_config, seeds=seeds,
                              out=run_dir / "report_randomised.json",
                              controllers={cname: make_controller(
                                  cname, **spec.options("controller"))},
                              randomize=True)

    log(f"fidelity  {seeds} seeds")
    fid = average(str(pose_p), str(motion_p), spec.env_config, seeds,
                  controller=make_controller(cname, **spec.options("controller")))

    motion_meta = ReferenceMotion.load(motion_p).meta
    record = build_record(
        spec, source=source,
        artifacts={"clip": source, "tracks": tracks_p, "pose": pose_p,
                   "motion": motion_p, "run": run_dir},
        motion_meta=motion_meta,
        controller_info={"name": cname, **controller_provenance(controller)},
        eval_nominal=nominal, eval_random=random_rep, fidelity=fid, seeds=seeds)
    out = write_record(run_dir / "record.json", record)
    log(f"record    {out}")
    return record


__all__ = ["run_pipeline", "summarise"]
