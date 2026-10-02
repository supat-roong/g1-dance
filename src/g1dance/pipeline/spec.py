"""Which component runs at each stage, and the hash that makes caching possible.

A spec is a selection, not a configuration. The knobs stay where they already
live, in `RetargetConfig` and the env YAML, so there is one place to change any
given number. What a spec adds is a name per stage and a stable hash of
everything upstream of a stage, which is what lets a controller swap reuse the
reference instead of re-running MediaPipe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

#: Stage order, which is also the order the cache keys nest in. A stage's hash
#: covers itself and everything before it, so changing the estimator invalidates
#: the reference but changing the controller does not.
STAGES: tuple[str, ...] = ("detector", "estimator", "hand_estimator",
                           "retargeter", "controller")


#: The package source each cached stage runs, relative to `g1dance/`. Its
#: content goes into that stage's key, so fixing a bug in a stage invalidates
#: what the buggy version cached: the SMPL-X parameters `hands.attach` dropped
#: would otherwise have been served from cache after the fix. Per stage rather
#: than one hash of the repository, because a commit to the controller should
#: not re-run a 25 minute GVHMR estimate. Vendored code under `third_party/`
#: is not covered; changing it needs `--force`.
STAGE_CODE: dict[str, tuple[str, ...]] = {
    "detector": ("track",),
    "estimator": ("motion/frontends", "motion/ir.py", "motion/bodymodel.py",
                  "motion/clean.py"),
    "hand_estimator": ("motion/hands.py",),
    "retargeter": ("motion/retarget.py", "motion/retargeters.py",
                   "motion/gmr_backend.py", "motion/constraints.py",
                   "motion/fingers.py", "sim/model.py"),
    "controller": (),
}


def code_digest(stage: str) -> str:
    """Content hash of the source `stage` runs; see `STAGE_CODE`."""
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for rel in STAGE_CODE[stage]:
        p = root / rel
        files = sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for f in files:
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


class SpecError(ValueError):
    """The spec is malformed or names something that does not exist."""


def _split(entry) -> tuple[str, dict]:
    """A stage entry is either a name or a dict carrying a name plus options."""
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict):
        if "name" not in entry:
            raise SpecError(f"stage entry {entry!r} has no 'name'")
        opts = {k: v for k, v in entry.items() if k != "name"}
        return str(entry["name"]), opts
    raise SpecError(f"stage entry must be a string or a dict, got {type(entry)!r}")


@dataclass
class PipelineSpec:
    """One combination of components, plus the configs they read."""

    detector: str | dict = "mediapipe"
    estimator: str | dict = "mediapipe"
    hand_estimator: str | dict = "mediapipe"
    retargeter: str | dict = "builtin"
    controller: str | dict = "humanoid_gpt"
    retarget_config: str | None = None
    env_config: str = "configs/upper_body.yaml"
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path) -> PipelineSpec:
        import yaml
        p = Path(path)
        if not p.exists():
            raise SpecError(f"no pipeline spec at {p}")
        data = yaml.safe_load(p.read_text()) or {}
        if not isinstance(data, dict):
            raise SpecError(f"{p} must contain a mapping")
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise SpecError(
                f"unknown spec keys in {p}: {sorted(unknown)}; "
                f"known: {sorted(known - {'meta'})}")
        spec = cls(**data)
        spec.meta = {**spec.meta, "spec_path": str(p)}
        return spec

    # -------------------------------------------------------------- resolve
    def name(self, stage: str) -> str:
        return _split(getattr(self, self._field(stage)))[0]

    def options(self, stage: str) -> dict:
        return _split(getattr(self, self._field(stage)))[1]

    @staticmethod
    def _field(stage: str) -> str:
        if stage not in STAGES:
            raise SpecError(f"unknown stage {stage!r}; known: {list(STAGES)}")
        return stage

    def resolved(self) -> dict:
        """Every stage as name plus options, with the configs named.

        This is what goes in the record: the question a comparison needs
        answered is which options were actually used, not which were written
        down.
        """
        out = {s: {"name": self.name(s), "options": self.options(s)}
               for s in STAGES}
        out["retarget_config"] = self.retarget_config
        out["env_config"] = self.env_config
        return out

    def validate(self) -> None:
        """Check every named component exists, before anything expensive runs."""
        from ..motion.frontends import available_frontends
        from ..motion.hands import available_hand_estimators
        from ..motion.retargeters import available_retargeters
        from ..control.controllers import available_controllers
        from ..track.detect_names import DETECTOR_NAMES
        for stage, have in (("detector", DETECTOR_NAMES),
                            ("estimator", available_frontends()),
                            ("hand_estimator", available_hand_estimators()),
                            ("retargeter", available_retargeters()),
                            ("controller", available_controllers())):
            n = self.name(stage)
            if n not in have:
                raise SpecError(
                    f"unknown {stage} {n!r}; available: {', '.join(sorted(have))}")
        for f in (self.retarget_config, self.env_config):
            if f is not None and not Path(f).exists():
                raise SpecError(f"config named by the spec does not exist: {f}")
        self._validate_compatible()

    def _validate_compatible(self) -> None:
        """Refuse a combination whose downstream stage needs what upstream lacks.

        Found the expensive way: GVHMR ran for minutes, then the MediaPipe hand
        estimator refused it for having no pixel landmarks. A frontend that did
        not declare its outputs is let through, and fails where it used to.
        """
        from ..motion import hands, retargeters
        from ..motion.frontends import provides
        est = self.name("estimator")
        have = provides(est)
        if have is None:
            return
        words = {"pixels": "pixel landmarks (landmarks_px)",
                 "smplx": "SMPL-X parameters"}
        for stage, needs in (("hand_estimator", hands.NEEDS),
                             ("retargeter", retargeters.NEEDS)):
            n = self.name(stage)
            missing = sorted(needs.get(n, frozenset()) - have)
            if missing:
                raise SpecError(
                    f"{stage} {n!r} needs {', '.join(words[m] for m in missing)}, "
                    f"which estimator {est!r} does not produce")

    # ----------------------------------------------------------------- hash
    def stage_hash(self, stage: str, source: str | Path | None = None) -> str:
        """Hash of this stage and everything upstream of it.

        `source` is included because the same components applied to a different
        clip are a different artifact. Config files are hashed by content rather
        than by path, so editing one invalidates the cache as it should.
        """
        upto = STAGES.index(self._field(stage))
        payload: dict = {"source": str(source) if source else None}
        for s in STAGES[:upto + 1]:
            payload[s] = {"name": self.name(s), "options": self.options(s),
                          "code": code_digest(s)}
        if upto >= STAGES.index("retargeter"):
            payload["retarget_config"] = _file_digest(self.retarget_config)
        if upto >= STAGES.index("controller"):
            payload["env_config"] = _file_digest(self.env_config)
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _file_digest(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return f"missing:{p}"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


__all__ = ["PipelineSpec", "STAGES", "SpecError"]
