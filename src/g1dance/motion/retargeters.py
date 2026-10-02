"""Name keyed registry of retargeting backends.

The other three stages were behind registries and this one was not, which made it
the only stage a caller had to name in code. It is also the stage the measured
ceiling lives behind: source to reference sits at 0.729 and its dominant cost is
`resolve_self_collision` firing on two thirds of the frames, so a platform that
could swap perception and control but not this would be swappable everywhere
except where it matters.

`builtin` is today's function, not a wrapper around it. Binding the name to the
same callable is deliberate: a wrapper is a place for numbers to drift.

Loaders are lazy for the same reason the frontend registry's are. Listing the
installed backends must not import mujoco or scipy, because `g1dance components`
and `--help` should cost nothing.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Retargeter(Protocol):
    """Anything that turns human motion into robot joint targets.

    `needs_smplx_params` is optional and defaults to False. A backend that sets
    it works from SMPL-X parameters rather than from landmarks, which cannot be
    approximated from each other: a joint position does not carry an
    orientation. Declaring it lets a caller check before paying for a call.
    """

    needs_smplx_params: bool

    def __call__(self, pose, model, cfg=None): ...


def needs_smplx_params(retargeter) -> bool:
    """Whether this backend requires SMPL-X parameters on the pose sequence."""
    return bool(getattr(retargeter, "needs_smplx_params", False))


def _load_builtin() -> Retargeter:
    from .retarget import retarget
    return retarget


def _load_gmr() -> Retargeter:
    from .gmr_backend import retarget_with_gmr
    return retarget_with_gmr


#: name -> zero argument loader, plus a description for `g1dance components`.
_LOADERS: dict[str, Callable[[], Retargeter]] = {
    "builtin": _load_builtin,
    "gmr": _load_gmr,
}
DESCRIPTIONS: dict[str, str] = {
    "builtin": "per-bone scaled IK against the robot's own kinematics",
    "gmr": "General Motion Retargeting; needs SMPL-X parameters, not landmarks",
}

#: What each backend needs from the frontend, in the vocabulary of
#: `frontends.PROVIDES`. Kept static beside `needs_smplx_params` because reading
#: that attribute means importing the backend, and validating a spec should not.
NEEDS: dict[str, frozenset[str]] = {
    "builtin": frozenset(),
    "gmr": frozenset({"smplx"}),
}

#: Present so the registry can be introspected the way the others can.
RETARGETERS = _LOADERS


def register_retargeter(name: str, loader: Callable[[], Retargeter],
                        description: str = "") -> None:
    """Add a backend under `name`. Refuses to shadow an existing one.

    Silently replacing one would make two runs with the same spec produce
    different motion, which is the class of bug this pipeline is built to avoid.
    """
    if name in _LOADERS:
        raise ValueError(f"retargeter {name!r} is already registered")
    _LOADERS[name] = loader
    DESCRIPTIONS[name] = description


def available_retargeters() -> list[str]:
    return sorted(_LOADERS)


def get_retargeter(name: str) -> Retargeter:
    try:
        loader = _LOADERS[name]
    except KeyError:
        raise ValueError(
            f"unknown retargeter {name!r}; "
            f"available: {', '.join(available_retargeters())}") from None
    return loader()


__all__ = ["DESCRIPTIONS", "RETARGETERS", "Retargeter", "available_retargeters",
           "get_retargeter", "needs_smplx_params", "register_retargeter"]
