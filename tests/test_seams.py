"""The four component seams, and that callers no longer know component names.

A registry whose callers branch on its entries' names is not a seam. Before this,
`cli.py` contained `if a.frontend == "smpl"`, so adding an estimator meant editing
the CLI. What is tested here is that each stage has one calling convention, that
the registries are listable without paying for a heavy import, and that the new
`builtin` retargeter is the old function rather than a reimplementation of it.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from g1dance.motion.ir import SubjectWindow
from g1dance.motion.retargeters import (
    RETARGETERS, available_retargeters, get_retargeter)


# ------------------------------------------------------------ subject window
def test_subject_window_validates_its_shapes():
    w = SubjectWindow(bbox=np.zeros((10, 4)), valid=np.ones(10, bool), fps=30.0)
    assert w.n_frames == 10


def test_subject_window_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="valid"):
        SubjectWindow(bbox=np.zeros((10, 4)), valid=np.ones(9, bool), fps=30.0)


def test_subject_window_refuses_a_bad_box_shape():
    with pytest.raises(ValueError, match="bbox"):
        SubjectWindow(bbox=np.zeros((10, 3)), valid=np.ones(10, bool), fps=30.0)


def test_subject_window_refuses_a_nonpositive_rate():
    with pytest.raises(ValueError, match="fps"):
        SubjectWindow(bbox=np.zeros((4, 4)), valid=np.ones(4, bool), fps=0.0)


def test_a_track_result_can_hand_over_a_subject_window():
    from g1dance.track import TrackResult
    tr = TrackResult(video="x.mp4", fps=30.0, width=640, height=480,
                     bbox=np.zeros((6, 4)), valid=np.ones(6, bool),
                     shots=[(0, 6)], assignments=[])
    w = tr.subject_window()
    assert isinstance(w, SubjectWindow)
    assert w.fps == 30.0 and w.n_frames == 6


# ------------------------------------------------------- retargeter registry
def test_builtin_retargeter_is_registered():
    assert "builtin" in RETARGETERS
    assert "builtin" in available_retargeters()


def test_unknown_retargeter_names_the_alternatives():
    with pytest.raises(ValueError, match="builtin"):
        get_retargeter("no_such_backend")


def test_the_builtin_retargeter_is_the_existing_function():
    """Not a reimplementation: the same callable, so numbers cannot drift."""
    from g1dance.motion.retarget import retarget
    assert get_retargeter("builtin") is retarget


# ------------------------------------------------------------ uniform seams
def test_every_estimator_takes_source_then_subject():
    """One convention, so a caller never needs to know which frontend it has."""
    import inspect

    from g1dance.motion.frontends import available_frontends, get_frontend
    for name in available_frontends():
        fn = get_frontend(name)
        params = list(inspect.signature(fn).parameters)
        assert params[:2] == ["source", "subject"], (name, params)


def test_the_cli_does_not_branch_on_a_component_name():
    """The regression this whole stage exists to prevent.

    Comments are stripped first: cli.py documents the branch it used to have, and
    a comment about history is not a branch.
    """
    import io
    import tokenize
    from pathlib import Path

    src = Path("src/g1dance/cli.py").read_text()
    code = "".join(
        "" if tok.type in (tokenize.COMMENT, tokenize.STRING) else tok.string
        for tok in tokenize.generate_tokens(io.StringIO(src).readline))
    for name in ("mediapipe", "smpl", "builtin", "humanoid_gpt", "open_loop"):
        for form in (f'=="{name}"', f"=='{name}'"):
            assert form not in code.replace(" ", ""), \
                f"cli.py branches on {name!r}"


# ----------------------------------------------------------------- listing
def test_components_can_be_listed_without_importing_torch_or_mediapipe():
    """Listing must stay cheap; `--help` paying for MediaPipe was the old sin."""
    code = (
        "import sys;"
        "from g1dance.motion.frontends import available_frontends;"
        "from g1dance.motion.retargeters import available_retargeters;"
        "from g1dance.control.controllers import available_controllers;"
        "available_frontends(); available_retargeters();"
        "available_controllers();"
        "heavy=[m for m in ('torch','mediapipe','scipy','mujoco') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "", f"listing imported heavy modules: {out}"


def test_the_components_verb_reports_all_four_registries():
    out = subprocess.run(
        [sys.executable, "-m", "g1dance.cli", "components"],
        capture_output=True, text=True, check=True).stdout
    for stage in ("detector", "estimator", "retargeter", "controller"):
        assert stage in out.lower(), stage
    for entry in ("mediapipe", "builtin", "humanoid_gpt"):
        assert entry in out, entry


def test_every_declared_detector_name_really_exists():
    """Two lists of the same thing drift; this catches a stale declared name.

    Only one direction can be asserted: other tests register their own detectors
    into the live registry, so the registry is legitimately a superset.
    """
    from g1dance.track.detect import DETECTORS
    from g1dance.track.detect_names import DETECTOR_NAMES
    missing = [n for n in DETECTOR_NAMES if n not in DETECTORS]
    assert not missing, f"declared but not registered: {missing}"


def test_the_retarget_config_no_longer_carries_a_frontend_field():
    """It defaulted to "unknown" while pose.meta held the real answer."""
    from g1dance.motion.retarget import RetargetConfig
    assert "frontend" not in RetargetConfig.__dataclass_fields__


def test_the_motion_records_the_frontend_that_produced_it():
    from pathlib import Path

    from g1dance.motion.ir import ReferenceMotion
    p = Path("motion/kirari.npz")
    if not p.exists():
        pytest.skip("motion/ is gitignored; needs a local reference")
    meta = ReferenceMotion.load(p).meta
    assert meta["source"]["frontend"] == "mediapipe"
