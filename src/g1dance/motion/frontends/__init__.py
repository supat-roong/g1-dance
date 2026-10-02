"""Name keyed registry of motion frontends.

The pipeline's whole point is that nothing downstream of `PoseSequence` knows
where the motion came from. Keeping the frontends behind a registry keyed by
name means a config file string is the only thing that selects between reading
a video and reading an AMASS dump, and adding a third source never touches the
retargeter or the trainer.

Loaders are deliberately lazy. Importing this module must stay cheap and must
not pull in mediapipe or torch, because the CLI imports it merely to validate a
`--frontend` argument.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from ..ir import PoseSequence


@runtime_checkable
class Frontend(Protocol):
    """What every frontend must be: something that returns a PoseSequence.

    One signature for all of them. `source` is whatever the frontend consumes, a
    video for `mediapipe` and a parameter file for `smpl`. `subject` is a
    `SubjectWindow` saying where the dancer is per frame, which a frontend
    reading a parameter file ignores.

    The uniformity is the point. The CLI used to read

        if a.frontend == "smpl": seq = fe(a.source)
        else:                    seq = fe(a.video, tr.bbox, tr.valid, tr.fps)

    so adding a frontend meant editing the caller, and a registry whose callers
    know its entries' names is not a seam.
    """

    def __call__(self, source, subject=None, **kwargs) -> PoseSequence: ...


def _load_mediapipe() -> Frontend:
    from .mediapipe_fe import estimate as _estimate

    def estimate(source, subject=None, **opts) -> PoseSequence:
        if subject is None:
            raise ValueError(
                "the mediapipe frontend needs a subject window; run `track` "
                "first so it knows which person in the frame to follow")
        return _estimate(source, subject.bbox, subject.valid, subject.fps,
                         **opts)

    return estimate


def _load_smpl() -> Frontend:
    from .smpl_fe import estimate_from_smpl as _estimate

    def estimate(source, subject=None, **opts) -> PoseSequence:
        # A parameter file already is the subject; a tracking window has nothing
        # to add, so it is accepted and ignored rather than refused.
        return _estimate(source, **opts)

    return estimate


#: name -> zero argument loader. Loaders import their heavy dependencies only
#: when actually called, so `get_frontend` costs nothing until it is used.
def _load_gvhmr() -> Frontend:
    from .gvhmr_fe import estimate as _estimate

    def estimate(source, subject=None, **opts) -> PoseSequence:
        return _estimate(source, subject, **opts)

    return estimate


def _load_smplest_x() -> Frontend:
    from .smplestx_fe import estimate as _estimate

    def estimate(source, subject=None, **opts) -> PoseSequence:
        return _estimate(source, subject, **opts)

    return estimate


_LOADERS: dict[str, Callable[[], Frontend]] = {
    "mediapipe": _load_mediapipe,
    "smpl": _load_smpl,
    "smplest_x": _load_smplest_x,
    "gvhmr": _load_gvhmr,
}

#: one line per frontend, for `g1dance components`
DESCRIPTIONS: dict[str, str] = {
    "mediapipe": "pose landmarking on the tracked subject's crop",
    "smpl": "SMPL or SMPL-X joint positions from AMASS or a parameter file",
    "smplest_x": "expressive whole-body SMPL-X from a crop; feeds the gmr backend",
    "gvhmr": "world-grounded SMPL-X from video; the only root that carries translation",
}


#: What each frontend puts on the PoseSequence beyond the landmarks every one
#: carries: "pixels" is `landmarks_px`, "smplx" is the four parameter fields.
#: Static, so `PipelineSpec.validate` can refuse an impossible combination
#: before the expensive stage runs rather than after it. `smpl` carries
#: parameters only when it read a parameter file, so it claims them and the
#: retargeter's own check covers the case where it did not.
PROVIDES: dict[str, frozenset[str]] = {
    "mediapipe": frozenset({"pixels"}),
    "smpl": frozenset({"smplx"}),
    "smplest_x": frozenset({"smplx"}),
    "gvhmr": frozenset({"pixels", "smplx"}),
}


def provides(name: str) -> frozenset[str] | None:
    """What `name` produces, or None for a frontend that did not declare it."""
    return PROVIDES.get(name)


def register_frontend(name: str, loader: Callable[[], Frontend]) -> None:
    """Add a frontend under `name`. Refuses to shadow an existing one.

    Silently replacing a frontend would make two runs with the same config
    produce different motion, which is exactly the class of bug this pipeline
    is built to avoid.
    """
    if name in _LOADERS:
        raise ValueError(f"frontend {name!r} is already registered")
    _LOADERS[name] = loader


def available_frontends() -> list[str]:
    """Registered frontend names, sorted, for error messages and CLI help."""
    return sorted(_LOADERS)


def get_frontend(name: str) -> Frontend:
    """Resolve a frontend name to its callable, importing it on first use."""
    try:
        loader = _LOADERS[name]
    except KeyError:
        raise ValueError(
            f"unknown motion frontend {name!r}; "
            f"available: {', '.join(available_frontends())}"
        ) from None
    return loader()


__all__ = ["DESCRIPTIONS", "Frontend", "available_frontends", "get_frontend",
           "register_frontend"]
