"""Evaluate a controller against the two baselines that bracket it.

A tracking number on its own says nothing. 6 degrees of mean joint error is
excellent or useless depending on what the motion demands, so every metric here
is reported alongside two references that bound it:

  * **kinematic reference** -- the retargeted motion replayed with no physics at
    all. Zero tracking error and no falls by construction. This is the upper
    bound nothing can beat, and it exists to show what the reward is actually
    asking for.
  * **open-loop PD playback** -- the same environment with a zero action, so the
    position targets are exactly the reference and the only thing acting is the
    PD controller and gravity. This is the lower bound a tracker has to beat.
    On a dance it falls over almost immediately, and the gap between it and the
    tracker is the measurement.

Reported per run: survival rate, per-DoF mean absolute angle error, key link
position error, foot slip, and time to fall.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from g1dance.control.controllers import OpenLoopController
from g1dance.sim.env import DanceTrackEnv, load_config
from g1dance.sim.model import BODY_JOINTS, KEY_BODIES


#: Fixed window for the comparable survival figure, in seconds.
#:
#: `survival_rate` counts an episode as survived when it did not terminate,
#: and an episode also ends by truncation at the end of its segment. That makes
#: the number depend on the reference's segment length rather than on the
#: policy. Measured: a 12 s reference has a 514 frame segment, so an episode
#: starting at a random frame averages 257 frames before the segment ends and
#: truncation is cheap; a 78 s reference has a 3899 frame segment, where an
#: episode can essentially never reach the end and must survive the full step
#: cap instead. Two policies of equal stability scored 0.83 and 0.17 that way,
#: while their times to fall were 2.96 s and 3.37 s.
#:
#: `survival_at_horizon` asks one question of every clip: did the policy stay
#: up for this long. Episodes whose segment ends before the horizon cannot
#: answer it and are excluded rather than counted as successes.
SURVIVAL_HORIZON_S = 5.0


def _episode(env: DanceTrackEnv, controller, max_steps: int,
             horizon: int | None = None) -> dict:
    """One episode, collecting the per-step quantities every metric needs.

    `controller` is anything satisfying `control.controllers.Controller`, so the same
    metrics apply to a policy trained here and to a downloaded tracker that has
    never seen this environment. `absolute` comes off the controller rather than
    from the caller, so the two cannot be paired up wrongly.
    """
    obs, priv = env.reset()
    controller.reset(env)
    absolute = bool(getattr(controller, "absolute", False))
    clip = 0.0
    tracked = env.tracked
    n = len(tracked)
    err = np.zeros(n)
    link_err = 0.0
    slip = 0.0
    steps = 0
    fell = False
    truncated = False
    for _ in range(max_steps):
        a = controller.act(env, obs, priv)
        obs, priv, _, term, trunc, info = env.step(a, absolute=absolute)
        clip += env.target_clip_frac
        ctx = env._ctx
        err += np.abs(ctx.q[tracked] - ctx.ref_q[tracked])
        link_err += env._link_err
        v = ctx.foot_vel[:, :2]
        slip += float(np.linalg.norm(v, axis=1) @ ctx.foot_contact)
        steps += 1
        if term:
            fell = True
            break
        if trunc:
            truncated = True
            break
    k = max(steps, 1)
    h = horizon if horizon is not None else int(round(SURVIVAL_HORIZON_S / env.dt))
    # Only episodes that could have reached the horizon get to vote on it: one
    # truncated earlier by its segment ending simply did not run the experiment.
    horizon_valid = steps >= h or fell
    return {"steps": steps, "fell": fell, "truncated": truncated,
            "horizon": h, "horizon_valid": bool(horizon_valid),
            "survived_horizon": bool(steps >= h and not (fell and steps < h)),
            "dof_err": err / k, "link_err": link_err / k, "slip": slip / k,
            "clip_frac": clip / k, "time": steps * env.dt}


def _kinematic_episode(env: DanceTrackEnv, max_steps: int) -> dict:
    """Replay the reference with forward kinematics only. The upper bound.

    Nothing here can fail, which is exactly the point: it fixes the zero of
    every column so the other two rows can be read as distances from it.
    """
    env.reset()
    m, d, g1 = env.model, env.g1.data(), env.g1
    n = len(env.tracked)
    steps = min(max_steps, env.seg[1] - env.frame - 1)
    link_err = 0.0
    for i in range(steps):
        f = env.frame + 1 + i
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = [0.0, 0.0, 0.793]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[g1.body_qpos] = env.track.ref_q[f]
        mujoco.mj_kinematics(m, d)
        rel = d.xpos[g1.key_body_ids] - d.xpos[g1.pelvis_id]
        dl = rel - env.track.link[f]
        link_err += float(np.sqrt(
            np.einsum("ij,ij->i", dl, dl) @ env.link_weight
            / max(env.link_weight.sum(), 1e-9)))
    k = max(steps, 1)
    return {"steps": steps, "fell": False, "dof_err": np.zeros(n),
            "link_err": link_err / k, "slip": 0.0, "clip_frac": 0.0,
            "time": steps * env.dt}


def _aggregate(eps: list[dict]) -> dict:
    survived = [not e["fell"] for e in eps]
    votes = [e for e in eps if e.get("horizon_valid")]
    at_h = [e["survived_horizon"] for e in votes]
    ttf = [e["time"] for e in eps if e["fell"]]
    dof = np.mean([e["dof_err"] for e in eps], axis=0)
    return {
        "episodes": len(eps),
        "survival_rate": float(np.mean(survived)),
        "survival_at_horizon": float(np.mean(at_h)) if at_h else None,
        "horizon_s": float(eps[0]["horizon"] * (eps[0]["time"] / max(eps[0]["steps"], 1)))
        if eps and eps[0].get("horizon") else None,
        "horizon_episodes": len(votes),
        "dof_err_mean_rad": float(dof.mean()),
        "dof_err_max_rad": float(dof.max()) if dof.size else 0.0,
        "dof_err_per_joint_rad": [round(float(v), 4) for v in dof],
        "link_err_m": float(np.mean([e["link_err"] for e in eps])),
        "foot_slip_mps": float(np.mean([e["slip"] for e in eps])),
        # Nonzero means the controller asked for joint targets outside the
        # model's range and they were clipped. For an external tracker that is
        # the difference between "tracks badly" and "was never applied".
        "target_clip_frac": float(np.mean([e.get("clip_frac", 0.0) for e in eps])),
        "time_to_fall_s": float(np.mean(ttf)) if ttf else None,
        "episode_len_s": float(np.mean([e["time"] for e in eps])),
    }


def evaluate(motion_path: str | Path, config_path: str | Path, seeds: int = 8,
             max_steps: int = 500, out: str | Path | None = None,
             mode: str | None = None, randomize: bool = False,
             controllers: dict[str, object] | None = None) -> dict:
    """Run every baseline plus each controller, `seeds` episodes each.

    `controllers` names what is being scored; every one is measured by the
    same metrics against the same baselines.
    """
    cfg = load_config(config_path)
    mode = mode or cfg.get("mode", "upper_body")
    env = DanceTrackEnv(motion_path, config_path, mode=mode,
                        randomize=randomize)

    # `open_loop` is always present because every other row is read as a
    # distance from it. Anything else is whatever the caller wants scored.
    named: dict[str, object] = {"open_loop": OpenLoopController()}
    for name, c in (controllers or {}).items():
        named[name] = c

    runs: dict[str, list[dict]] = {"reference": []}
    for name in named:
        runs[name] = []
    for s in range(seeds):
        env.seed(s)
        runs["reference"].append(_kinematic_episode(env, max_steps))
        for name, c in named.items():
            env.seed(s)
            runs[name].append(_episode(env, c, max_steps))
    env.close()

    report = {"motion": str(motion_path), "config": str(config_path),
              "mode": mode,
              "tracked_joints": [BODY_JOINTS[i] for i in env.tracked],
              "key_bodies": list(KEY_BODIES),
              "results": {k: _aggregate(v) for k, v in runs.items()}}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=1))
    return report


def format_report(report: dict) -> str:
    rows = ["  role            surv  surv@5s   dof_err   link_err   slip   clip"
            "    ttf   len",
            "  " + "-" * 79]
    # The three bracketing rows first and in a fixed order, then anything else
    # the caller asked to be scored. A controller that does not appear in the
    # table is the one failure this function can have.
    fixed = ("reference", "open_loop", "policy")
    order = ([k for k in fixed if k in report["results"]]
             + [k for k in report["results"] if k not in fixed])
    for k in order:
        r = report["results"][k]
        t = r["time_to_fall_s"]
        ttf = "   -  " if t is None else f"{t:5.2f}"
        sh = r.get("survival_at_horizon")
        shs = "   -  " if sh is None else f"{sh:5.2f}"
        rows.append(
            f"  {k:<14} {r['survival_rate']:5.2f}  {shs}   "
            f"{r['dof_err_mean_rad']:7.4f}  {r['link_err_m']:8.4f}  "
            f"{r['foot_slip_mps']:5.3f}  {r.get('target_clip_frac', 0.0):5.3f}"
            f"  {ttf}  {r['episode_len_s']:5.2f}")
    return "\n".join(rows)
