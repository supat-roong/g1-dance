"""Reward terms as a named registry, so mode A and mode B differ by YAML only.

The two milestones in this project share one environment and one control law;
the only thing that separates "track the arms while the legs stand still" from
"track everything" is which DoF the tracking terms look at and how heavily each
term is weighted. Encoding that as a registry keyed by name, with the weights
and the shaping constants coming from the config file, keeps the difference in
`configs/*.yaml` where it can be diffed, instead of behind an `if mode ==` that
has to be kept consistent in five places.

Sign convention, applied without exception: every term returns a quantity that
is *larger when the thing it measures is larger*. Tracking terms return
exp(-k * weighted squared error), which is in (0, 1] and larger when tracking is
better, so their weights are positive. Penalty terms return a non-negative
penalty magnitude, so their weights are negative in the YAML. Having the sign
live in exactly one place means a term can be read in isolation and a weight can
be retuned without re-deriving which direction it pushes.

`RewardContext` is a flat bundle of already-computed quantities rather than the
environment itself. That is what makes each term unit testable against a hand
computed number without standing up MuJoCo.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RewardContext:
    """Everything the terms need, computed once per control step.

    Index conventions: `q`, `qd`, `ref_q`, `ref_qd` and `torque` are ordered by
    `sim.model.BODY_JOINTS` (29 entries). `tracked` indexes into those 29 and
    selects the DoF this mode tracks. `link_pos` and `ref_link_pos` are (K, 3)
    in the pelvis frame, ordered by `sim.model.KEY_BODIES`.
    """

    q: np.ndarray
    qd: np.ndarray
    ref_q: np.ndarray
    ref_qd: np.ndarray
    tracked: np.ndarray
    link_pos: np.ndarray
    ref_link_pos: np.ndarray
    link_weight: np.ndarray
    joint_weight: np.ndarray
    root_height: float
    ref_root_height: float
    upright: float
    root_xy: np.ndarray
    ref_root_xy: np.ndarray
    yaw: float
    ref_yaw: float
    foot_pos: np.ndarray            # (2, 3) world
    foot_vel: np.ndarray            # (2, 3) world linear velocity
    foot_contact: np.ndarray        # (2,) bool
    action: np.ndarray
    prev_action: np.ndarray
    torque: np.ndarray
    joint_range: np.ndarray         # (29, 2)
    n_self_contacts: int = 0
    track_root_xy: bool = False


TermFn = Callable[..., float]
#: name -> term function. Populated by the `@term` decorator below.
REGISTRY: dict[str, TermFn] = {}


def term(name: str) -> Callable[[TermFn], TermFn]:
    def deco(fn: TermFn) -> TermFn:
        if name in REGISTRY:
            raise ValueError(f"duplicate reward term {name!r}")
        REGISTRY[name] = fn
        return fn
    return deco


def _weighted_sq(err: np.ndarray, w: np.ndarray) -> float:
    """sum(w * err^2) / sum(w), with an all-zero weight vector meaning zero.

    Normalizing by the weight sum is what lets the same `k` be reused across
    modes: mode A tracks 17 DoF and mode B tracks 29, and without the
    normalization the same shaping constant would mean two different things.
    """
    tot = float(w.sum())
    if tot <= 0.0:
        return 0.0
    return float(np.dot(w, err * err) / tot)


# --------------------------------------------------------------------- #
# tracking
# --------------------------------------------------------------------- #
@term("track_joint")
def track_joint(ctx: RewardContext, k: float = 2.0) -> float:
    """Joint angle tracking over this mode's tracked DoF."""
    e = ctx.q[ctx.tracked] - ctx.ref_q[ctx.tracked]
    return math.exp(-k * _weighted_sq(e, ctx.joint_weight))


@term("track_link")
def track_link(ctx: RewardContext, k: float = 20.0) -> float:
    """Key link position tracking, in the pelvis frame.

    Pelvis frame rather than world: the monocular root estimate is the least
    reliable quantity in this pipeline, and a world-frame link reward would
    inherit all of its drift.
    """
    d = ctx.link_pos - ctx.ref_link_pos
    sq = np.einsum("ij,ij->i", d, d)
    tot = float(ctx.link_weight.sum())
    if tot <= 0.0:
        return 1.0
    return math.exp(-k * float(np.dot(ctx.link_weight, sq) / tot))


@term("track_vel")
def track_vel(ctx: RewardContext, k: float = 0.05) -> float:
    """Joint velocity tracking. Low `k`: velocities are noisy after retargeting
    and this term exists to damp phase lag, not to be matched exactly."""
    e = ctx.qd[ctx.tracked] - ctx.ref_qd[ctx.tracked]
    return math.exp(-k * _weighted_sq(e, ctx.joint_weight))


@term("track_root")
def track_root(ctx: RewardContext, k_height: float = 20.0,
               k_upright: float = 5.0, k_xy: float = 2.0,
               k_yaw: float = 2.0) -> float:
    """Root height and uprightness always; world xy and yaw in mode B only.

    `ctx.track_root_xy` is set by the environment from the config, so mode A
    gets "stay upright at the right height" while mode B additionally gets
    "be where the dancer was".
    """
    e = (k_height * (ctx.root_height - ctx.ref_root_height) ** 2
         + k_upright * (1.0 - ctx.upright) ** 2)
    if ctx.track_root_xy:
        dxy = ctx.root_xy - ctx.ref_root_xy
        dyaw = (ctx.yaw - ctx.ref_yaw + math.pi) % (2.0 * math.pi) - math.pi
        e += k_xy * float(np.dot(dxy, dxy)) + k_yaw * dyaw * dyaw
    return math.exp(-e)


# --------------------------------------------------------------------- #
# regularization and safety (weights are negative in the YAML)
# --------------------------------------------------------------------- #
@term("alive")
def alive(ctx: RewardContext) -> float:
    """Constant per surviving control step; the counterweight to termination."""
    return 1.0


@term("feet_distance")
def feet_distance(ctx: RewardContext, min_dist: float = 0.15) -> float:
    """Hinge penalty on the horizontal inter-foot distance.

    Only the lower bound is penalized. An upper bound would fight the splits
    and wide stances that a dance legitimately contains.
    """
    d = ctx.foot_pos[0, :2] - ctx.foot_pos[1, :2]
    return max(0.0, min_dist - float(np.linalg.norm(d)))


@term("feet_slip")
def feet_slip(ctx: RewardContext) -> float:
    """Squared horizontal foot speed, counted only while that foot is loaded.

    A foot sliding in the air is a step; a foot sliding under load is the
    signature of a policy that has learned to skate rather than to balance.
    """
    v = ctx.foot_vel[:, :2]
    sq = np.einsum("ij,ij->i", v, v)
    return float(np.dot(sq, ctx.foot_contact.astype(np.float64)))


@term("track_yaw")
def track_yaw(ctx: RewardContext, k: float = 2.0) -> float:
    """Heading tracking, separated from world position on purpose.

    `track_root` bundles yaw with world xy behind `track_root_xy`, so mode A
    gets neither. They are not equally trustworthy: monocular root
    *translation* is the least reliable quantity in the pipeline, but the
    reference *heading* comes from the pelvis rotation the retargeter solves
    for explicitly, and is worth tracking on its own.

    It matters visually more than its weight suggests. With nothing rewarding
    heading the policy is free to rotate, and the milestone A run did: it
    stayed upright and moved, but ended up facing a different way from the
    dancer, which reads as a different dance however well the joints track.
    """
    d = (ctx.yaw - ctx.ref_yaw + math.pi) % (2.0 * math.pi) - math.pi
    return math.exp(-k * d * d)


@term("leg_posture")
def leg_posture(ctx: RewardContext) -> float:
    """Squared deviation of the UNTRACKED DoF from their reference.

    In mode A the legs carry no reference, so `ref_q` holds the default stance
    for them and this is a posture prior rather than a tracking term. It exists
    because the untracked DoF are otherwise completely unconstrained, and the
    policy exploits that: measured on the first milestone A run, it learned to
    survive by splaying into a wide brace and shuffling, reaching a foot slip
    of 0.700 against 0.077 for open loop while tracking 1.8 times worse.

    Deliberately not an exponential: this is a penalty that should keep biting
    as the pose gets worse, not saturate.
    """
    untracked = np.setdiff1d(np.arange(len(ctx.q)), ctx.tracked, assume_unique=False)
    if untracked.size == 0:
        return 0.0
    e = ctx.q[untracked] - ctx.ref_q[untracked]
    return float(np.dot(e, e) / untracked.size)


@term("stance_width")
def stance_width(ctx: RewardContext, max_dist: float = 0.35) -> float:
    """Hinge penalty on the horizontal inter-foot distance, upper bound.

    The counterpart to `feet_distance`, which only bounds it below. That term's
    docstring is right that an upper bound fights the wide stances a dance
    legitimately contains, so the bound is set well outside normal dance width
    and only catches the brace the policy adopts to avoid balancing.
    """
    d = ctx.foot_pos[0, :2] - ctx.foot_pos[1, :2]
    return max(0.0, float(np.linalg.norm(d)) - max_dist)


@term("self_collision")
def self_collision(ctx: RewardContext) -> float:
    """Number of robot-on-robot contacts this step.

    Beware that on real retargeted motion this fights the tracking terms
    directly. Measured on `motion/kirari.npz`, whose own retarget diagnostics
    report 302 self-colliding frames out of 600: replaying its reference poses
    kinematically produces 1.42 robot-on-robot contacts per frame, mostly the
    left wrist passing through the right arm and a shoulder into the torso. The
    policy is then penalized for going where `track_joint` and `track_link` are
    paying it to go. The fix belongs upstream in the retargeter rather than in
    this weight; until then, read a persistent nonzero value here as a property
    of the reference, not of the policy.
    """
    return float(ctx.n_self_contacts)


@term("action_rate")
def action_rate(ctx: RewardContext) -> float:
    """Mean squared change in the action between consecutive control steps."""
    d = ctx.action - ctx.prev_action
    return float(np.dot(d, d) / max(d.size, 1))


@term("torque")
def torque(ctx: RewardContext) -> float:
    """Mean squared actuator force over the body DoF."""
    t = ctx.torque
    return float(np.dot(t, t) / max(t.size, 1))


@term("joint_limit")
def joint_limit(ctx: RewardContext, margin: float = 0.05) -> float:
    """Total excursion past a soft limit inset from each joint's hard limit.

    `margin` is a fraction of the joint's own range, so a wrist and a knee are
    given proportionate headroom rather than the same absolute one.
    """
    lo = ctx.joint_range[:, 0]
    hi = ctx.joint_range[:, 1]
    inset = margin * (hi - lo)
    over = np.maximum(ctx.q - (hi - inset), 0.0)
    under = np.maximum((lo + inset) - ctx.q, 0.0)
    return float(over.sum() + under.sum())


# --------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------- #
@dataclass
class RewardFunction:
    """A weighted sum of registry terms, built from the `rewards:` config block.

    The config maps a term name to a dict carrying `weight` plus whatever
    keyword arguments that term takes:

        rewards:
          track_joint: {weight: 3.0, k: 2.0}
          torque:      {weight: -2.0e-5}

    A term named in the config that is not in the registry is an error rather
    than a silent no-op, because a typo in a reward weight is otherwise
    invisible until the run has already been paid for.
    """

    terms: list[tuple[str, TermFn, float, dict]] = field(default_factory=list)
    components: np.ndarray = field(default=None, repr=False)

    @classmethod
    def from_config(cls, cfg: dict) -> RewardFunction:
        terms = []
        for name, spec in (cfg or {}).items():
            if name not in REGISTRY:
                raise ValueError(
                    f"unknown reward term {name!r}; known terms are "
                    f"{sorted(REGISTRY)}")
            spec = dict(spec or {})
            weight = float(spec.pop("weight", 1.0))
            if weight == 0.0:
                continue
            terms.append((name, REGISTRY[name], weight, spec))
        return cls(terms=terms, components=np.zeros(len(terms), np.float64))

    @property
    def names(self) -> list[str]:
        return [n for n, _, _, _ in self.terms]

    def __call__(self, ctx: RewardContext) -> float:
        """Total reward. Per-term values land in `self.components`, in `names`
        order, so the caller can log them without a per-step dict allocation."""
        total = 0.0
        out = self.components
        for i, (_, fn, w, kw) in enumerate(self.terms):
            v = fn(ctx, **kw)
            out[i] = v
            total += w * v
        return total
