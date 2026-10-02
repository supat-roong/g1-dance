"""The run record: what produced a result, and what the result was.

Provenance is aggregated rather than invented. The motion stage already writes
its frontend and that frontend's version, the pose bundle, the delegate, the axes
and units convention, the clip, the coverage, the full retarget config and a
per-frame diagnostics summary. All of it is carried through here verbatim.

The gap this closes is the control stage, which recorded provenance only for a
policy trained in this repo and nothing at all for a downloaded one, and the
metrics, which lived in two tools that nothing joined. A record carries both, so
"stable and performing the motion" can be read off one artifact.

The vector attributes loss per stage. `source_to_reference` is the estimator and
the retargeter, `reference_to_policy` is the controller, `end_to_end` is the
product. Reporting the chain rather than the product is what tells you which
component to change.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path

from .spec import STAGES

SCHEMA = 1


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def file_digest(path: str | Path | None) -> str | None:
    """Short content hash, so a weight file is identified by what it is."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _corr(entry) -> dict | None:
    """`metrics.average` returns (mean, sd, n) triples."""
    if entry is None:
        return None
    mean, sd, n = entry
    return {"mean": float(mean), "sd": float(sd), "n": int(n)}


def controller_provenance(controller) -> dict:
    """What a controller is, in enough detail to reproduce it.

    A downloaded tracker is identified by the hash of its weights rather than by
    its path, because a path says nothing about which version ran.
    """
    out: dict = {"class": type(controller).__name__,
                 "absolute": bool(getattr(controller, "absolute", False))}
    path = getattr(controller, "path", None)
    if path is not None:
        out["weights"] = str(path)
        out["weights_sha256_16"] = file_digest(path)
    providers = getattr(controller, "_providers", None)
    if providers:
        out["providers"] = list(providers)
    for attr in ("_ema", "_anchor"):
        if hasattr(controller, attr):
            out[attr.lstrip("_")] = getattr(controller, attr)
    return out


def build_record(spec, *, source, artifacts: dict, motion_meta: dict,
                 controller_info: dict, eval_nominal: dict,
                 eval_random: dict | None, fidelity: dict,
                 seeds: int) -> dict:
    """Assemble the record. Pure: every input is already computed."""
    diag = dict(motion_meta.get("diagnostics") or {})
    src = dict(motion_meta.get("source") or {})
    nominal = eval_nominal["results"]

    def role(results, name):
        return results.get(name)

    record = {
        "schema": SCHEMA,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "source": str(source),
        "seeds": int(seeds),
        "spec": spec.resolved(),
        # From STAGES rather than a literal: a hardcoded list here silently
        # omits any stage added later, which is how a record comes to describe
        # a pipeline that is not the one that ran.
        "stage_hashes": {s: spec.stage_hash(s, source) for s in STAGES},
        "artifacts": {k: str(v) for k, v in artifacts.items() if v is not None},
        "components": {
            "estimator": src,
            "retargeter": {"name": spec.name("retargeter"),
                           "config": motion_meta.get("config"),
                           "diagnostics": diag},
            "controller": controller_info,
            "model_hash": motion_meta.get("model_hash"),
        },
        "reference": {
            "coverage": src.get("coverage"),
            "frames": eval_nominal.get("n_frames"),
            "ik_residual_p95": diag.get("ik_residual_p95"),
            "collision_frames": diag.get("n_collision_frames"),
            "foot_distance_fixed": diag.get("n_foot_distance_fixed"),
            "not_converged": diag.get("n_not_converged"),
        },
        "metrics": {
            "source_to_reference": _corr(fidelity.get("retarget")),
            "reference_to_policy": _corr(fidelity.get("policy")),
            "end_to_end": _corr(fidelity.get("end_to_end")),
            "episode_steps": _corr(fidelity.get("episode_steps")),
            "stability": {
                "nominal": role(nominal, spec.name("controller")),
                "randomised": (role(eval_random["results"],
                                    spec.name("controller"))
                               if eval_random else None),
            },
            "baselines": {
                "reference": role(nominal, "reference"),
                "open_loop": role(nominal, "open_loop"),
            },
        },
    }
    return record


def write_record(path: str | Path, record: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=1, default=str) + "\n")
    return p


def read_record(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def summarise(record: dict) -> str:
    """The vector, as a table. No ranking: that judgement stays with the reader."""
    m = record["metrics"]
    ref = record["reference"]
    spec = record["spec"]
    lines = [
        f"  source        {record['source']}",
        "  components    " + " ".join(
            f"{st}={spec[st]['name']}" for st in STAGES if st in spec),
        f"  reference     coverage={_fmt(ref.get('coverage'))} "
        f"ik_p95={_fmt(ref.get('ik_residual_p95'))} "
        f"collisions={ref.get('collision_frames')}",
        "",
        "  stage                        corr     sd",
        "  ---------------------------------------",
    ]
    for label, key in (("source -> reference", "source_to_reference"),
                       ("reference -> policy", "reference_to_policy"),
                       ("end to end", "end_to_end")):
        v = m.get(key)
        if v:
            lines.append(f"  {label:<26} {v['mean']:6.3f} {v['sd']:6.3f}")
    lines += ["", "  stability            surv@5s   dof_err  link_err    slip",
              "  -------------------------------------------------------"]
    for label in ("nominal", "randomised"):
        s = m["stability"].get(label)
        if not s:
            continue
        lines.append(
            f"  {label:<18} {_fmt(s.get('survival_at_horizon')):>9} "
            f"{_fmt(s.get('dof_err_mean_rad')):>9} "
            f"{_fmt(s.get('link_err_m')):>9} {_fmt(s.get('foot_slip_mps')):>7}")
    for label in ("reference", "open_loop"):
        s = m["baselines"].get(label)
        if not s:
            continue
        lines.append(
            f"  {label:<18} {_fmt(s.get('survival_at_horizon')):>9} "
            f"{_fmt(s.get('dof_err_mean_rad')):>9} "
            f"{_fmt(s.get('link_err_m')):>9} {_fmt(s.get('foot_slip_mps')):>7}")
    return "\n".join(lines)


def compare(records: list[tuple[str, dict]]) -> str:
    """Several records side by side. Deliberately without a verdict.

    Reporting the vector rather than a score was a decision, so this prints the
    columns and stops. Which trade is worth making is not something a table can
    know: the standing example is a reference made more faithful by lowering
    the landmark visibility gate, which produced a robot less like the video.
    """
    if not records:
        return "  (no records)"
    rows: list[tuple[str, list[str]]] = []

    def add(label, get):
        rows.append((label, [_fmt(get(r)) for _, r in records]))

    for stage in STAGES:
        add(stage, lambda r, s=stage: _stage_label(r["spec"].get(s, {})))
    rows.append(("", ["" for _ in records]))
    add("coverage", lambda r: r["reference"].get("coverage"))
    add("ik residual p95", lambda r: r["reference"].get("ik_residual_p95"))
    add("collision frames", lambda r: r["reference"].get("collision_frames"))
    rows.append(("", ["" for _ in records]))
    for label, key in (("source -> reference", "source_to_reference"),
                       ("reference -> policy", "reference_to_policy"),
                       ("end to end", "end_to_end")):
        add(label, lambda r, k=key: (r["metrics"].get(k) or {}).get("mean"))
    rows.append(("", ["" for _ in records]))
    for label, pass_ in (("survival@5s nominal", "nominal"),
                         ("survival@5s randomised", "randomised")):
        add(label, lambda r, p=pass_: (r["metrics"]["stability"].get(p)
                                       or {}).get("survival_at_horizon"))
    for label, field in (("joint err", "dof_err_mean_rad"),
                         ("link err", "link_err_m"),
                         ("foot slip", "foot_slip_mps"),
                         ("target clip", "target_clip_frac")):
        add(label, lambda r, f=field: (r["metrics"]["stability"].get("nominal")
                                       or {}).get(f))

    names = [n for n, _ in records]
    w0 = max(len(lbl) for lbl, _ in rows) + 2
    # Width from the widest cell, not just the widest heading: a stage carrying
    # its options is easily longer than its column name, and columns that run
    # together are a table that cannot be read.
    widest = max((len(v) for _, vals in rows for v in vals), default=0)
    w = max(max(len(n) for n in names), widest, 9) + 2
    out = [" " * w0 + "".join(f"{n:>{w}}" for n in names),
           " " * w0 + "".join("-" * (w - 1) + " " for _ in names)]
    for lbl, vals in rows:
        if not lbl and not any(vals):
            out.append("")
            continue
        out.append(f"  {lbl:<{w0 - 2}}" + "".join(f"{v:>{w}}" for v in vals))
    return "\n".join(out)


def _stage_label(entry: dict) -> str:
    """`name`, or `name(opt=value)` when options distinguish it.

    Without this a comparison that varies only an option shows two identical
    stage rows and different results, which reads as the platform being
    unreliable rather than as the option mattering.
    """
    name = entry.get("name", "?")
    opts = entry.get("options") or {}
    if not opts:
        return str(name)
    inner = ",".join(f"{k}={v}" for k, v in sorted(opts.items()))
    return f"{name}({inner})"


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


__all__ = ["SCHEMA", "build_record", "compare", "controller_provenance",
           "file_digest", "read_record", "summarise", "write_record"]
