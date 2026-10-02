"""Controllers: anything that can drive `DanceTrackEnv` for one episode.

The one that dances is a zero-shot tracker: somebody else's network, with its
own observation layout, its own action units and its own idea of what a
reference is. The others are baselines that bound it. The measurement harness
does not care which of them is driving.

The seam is deliberately not `act(obs)`. This environment's observation is
249-dim in its own arrangement, and a general tracker assembles its own
proprioception and reference window instead, so an observation is the wrong
currency. Joint targets are the right one, because the actuators are the one
thing both kinds of controller genuinely share.

Hence two declarations per controller: what it returns from `act`, and whether
that is a residual around the reference or an absolute target. `step` honours
the flag, and the flag is on the controller rather than passed by the caller so
a harness cannot pair them up wrongly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Controller(Protocol):
    """Something that drives the environment for one episode.

    `absolute` says how `act`'s return value is interpreted: False for a
    residual the environment adds to the reference and clips to the unit box,
    True for 29 joint position targets applied directly.
    """

    absolute: bool

    def reset(self, env) -> None:
        """Called once after `env.reset`, before the first `act`."""

    def act(self, env, obs: np.ndarray, priv: np.ndarray) -> np.ndarray:
        ...


class OpenLoopController:
    """Zero residual, so the position targets are exactly the reference.

    The lower bound every trained policy has to beat. On a whole-body dance the
    PD controller and gravity are the only things acting and it falls over
    almost immediately, which is the measurement.
    """

    absolute = False

    def reset(self, env) -> None:
        self._n = env.act_dim

    def act(self, env, obs, priv) -> np.ndarray:
        return np.zeros(self._n)


class ZeroActionController:
    """Hold the default stance, ignoring the reference entirely.

    Distinct from `OpenLoopController` despite the similar name used in the
    logs: this one does not track at all, which is the baseline the design doc
    quotes as surviving 0.10 of episodes. It exercises the absolute path with
    something whose right answer is known.
    """

    absolute = True

    def reset(self, env) -> None:
        self._q = np.asarray(env.g1.default_qpos, np.float64).copy()

    def act(self, env, obs, priv) -> np.ndarray:
        return self._q


#: name -> zero argument factory, for the harness and the command line.
CONTROLLERS: dict[str, Callable[..., Controller]] = {
    "open_loop": OpenLoopController,
    "zero_action": ZeroActionController,
}


def _humanoid_gpt(**kwargs) -> Controller:
    # Imported lazily: it pulls in onnxruntime, which the training path never
    # needs and should not pay for.
    from g1dance.control.trackers import HumanoidGPTController
    return HumanoidGPTController(**kwargs)


def _hgpt_variant(weights: str):
    """A named Humanoid-GPT checkpoint, downloaded on first use."""

    def make(**kwargs) -> Controller:
        from g1dance.control.trackers import HumanoidGPTController, ensure_tracker
        kwargs.setdefault("onnx_path", ensure_tracker(weights))
        return HumanoidGPTController(**kwargs)

    return make


CONTROLLERS["humanoid_gpt"] = _humanoid_gpt
CONTROLLERS["humanoid_gpt_216"] = _hgpt_variant("humanoid_gpt_216")
CONTROLLERS["humanoid_gpt_trans"] = _hgpt_variant("humanoid_gpt_trans")

#: Controllers that exist to bound the others rather than to track.
#:
#: `open_loop` is position PD straight to the reference and falls over on any
#: real dance; `zero_action` holds the default stance and ignores the reference
#: entirely. Both are supposed to fail at tracking, which is what makes them
#: useful, so the conformance gate that every tracker must stay upright does not
#: apply to them.
BASELINES: frozenset[str] = frozenset({"open_loop", "zero_action"})

#: one line per controller, for `g1dance components`
DESCRIPTIONS: dict[str, str] = {
    "open_loop": "zero residual, so the targets are exactly the reference",
    "zero_action": "hold the default stance, ignoring the reference",
    "humanoid_gpt": "zero-shot general tracker, MLP, one frame (pns_wo_priv264)",
    "humanoid_gpt_216": "the same, other MLP checkpoint (pns_wo_priv216)",
    "humanoid_gpt_trans": "the paper's transformer, 16-frame window",
}


def available_controllers() -> list[str]:
    return sorted(CONTROLLERS)


def make_controller(name: str, **kwargs) -> Controller:
    try:
        factory = CONTROLLERS[name]
    except KeyError:
        raise ValueError(
            f"unknown controller {name!r}; "
            f"available: {', '.join(available_controllers())}") from None
    return factory(**kwargs)


__all__ = ["BASELINES", "CONTROLLERS", "DESCRIPTIONS", "Controller",
           "OpenLoopController",
           "ZeroActionController", "available_controllers", "make_controller"]
