"""The render camera stands where the source camera stood."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from g1dance.sim.env import synthetic_motion
from g1dance.sim.model import G1Model
from g1dance.viz.render import facing_camera


def _facing(deg: float, n: int = 20):
    m = synthetic_motion(n, fps=50.0, g1=G1Model())
    h = np.radians(deg)
    q = np.tile([np.cos(h / 2), 0.0, 0.0, np.sin(h / 2)], (m.n_frames, 1))
    return replace(m, root_quat=q)


@pytest.mark.parametrize("heading, azimuth", [(0, 180), (90, 270), (-90, 90)])
def test_the_camera_is_in_front_of_the_dancer(heading, azimuth):
    """MuJoCo looks along (cos az, sin az), so in front means heading + 180."""
    az = facing_camera(_facing(heading)).azimuth % 360
    assert az == pytest.approx(azimuth % 360, abs=1e-3)


def test_headings_either_side_of_the_wrap_average_across_it():
    """A plain mean of 179 and -179 degrees would face the dancer's back."""
    m = _facing(179.0, 20)
    h = np.radians(np.r_[np.full(10, 179.0), np.full(m.n_frames - 10, -179.0)])
    q = np.stack([np.cos(h / 2), 0 * h, 0 * h, np.sin(h / 2)], axis=1)
    az = facing_camera(replace(m, root_quat=q)).azimuth % 360
    assert az == pytest.approx(0.0, abs=1.5) or az == pytest.approx(360.0, abs=1.5)


def test_the_skeleton_is_drawn_through_the_panel_transform():
    """Edges land where the mapping puts them, and empty slots draw nothing."""
    from g1dance.motion.ir import MP_POSE
    from g1dance.viz.render import _draw_skeleton
    img = np.zeros((100, 100, 3), np.uint8)
    px = np.zeros((33, 2))
    px[MP_POSE["left_shoulder"]] = [20, 20]
    px[MP_POSE["left_elbow"]] = [20, 60]
    _draw_skeleton(img, px, lambda xy: (int(xy[0]) + 10, int(xy[1])))
    assert img[40, 30].any()            # on the shifted shoulder-elbow edge
    assert not img[40, 20].any()        # where it would be unshifted
    assert not img[0, 0].any()          # no edge drawn to a zero slot


def test_steady_boxes_hold_one_size_and_smooth_the_centre():
    from g1dance.viz.render import steady_boxes
    rng = np.random.default_rng(0)
    n = 90
    side = 400 + 100 * rng.random(n)                  # a pumping detection
    cx = 500 + 20 * rng.standard_normal(n)            # a jittering one
    b = np.stack([cx - side / 2, 300 - side / 2, cx + side / 2, 300 + side / 2], 1)
    b[10] = np.nan                                    # a frame with no box
    s = steady_boxes(b, fps=30.0)
    sizes = s[:, 2] - s[:, 0]
    assert np.ptp(sizes) < 1e-9
    assert sizes[0] >= np.percentile(side, 90)
    assert np.all(np.isfinite(s))
    centre = 0.5 * (s[:, 0] + s[:, 2])
    assert np.std(np.diff(centre)) < 0.5 * np.std(np.diff(cx))


def test_a_rollout_says_whether_it_fell_or_ran_out_of_reference(tmp_path):
    """A render reports a fall rather than implying the clip played through."""
    from g1dance.control.controllers import make_controller
    from g1dance.viz.render import rollout_states
    m = synthetic_motion(400, fps=50.0, g1=G1Model())
    p = m.save(tmp_path / "m.npz")
    # Open loop PD falls over on this G1 inside two seconds.
    fell: dict = {}
    rollout_states(p, "configs/upper_body.yaml", start_frame=0,
                   max_steps=m.n_frames, outcome=fell,
                   controller=make_controller("open_loop"))
    assert fell["terminated"] is True and fell["reason"]
    # Ten steps from the end, nothing has time to fall: it runs out of clip.
    short: dict = {}
    rollout_states(p, "configs/upper_body.yaml",
                   start_frame=m.n_frames - 12, max_steps=m.n_frames,
                   outcome=short, controller=make_controller("open_loop"))
    assert short["terminated"] is False and short["reason"] == "truncated"
