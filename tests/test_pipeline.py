"""The spec, the cache key, and the record.

The cache key is the part with teeth. A key that is too coarse re-runs MediaPipe
every time a controller changes, which makes comparison too slow to do; a key
that is too fine reuses a reference that a changed estimator should have
invalidated, which silently reports one combination's number under another's
name. So what is tested here is exactly which changes propagate to which stage.
"""

from __future__ import annotations

import json

import pytest

from g1dance.pipeline import PipelineSpec, SpecError
from g1dance.pipeline.record import SCHEMA, build_record, summarise
from g1dance.pipeline.spec import STAGES

CLIP = "clips/a.mp4"


# ------------------------------------------------------------------ the spec
def test_defaults_name_a_real_component_at_every_stage():
    PipelineSpec().validate()


def test_a_stage_entry_may_carry_options():
    s = PipelineSpec(controller={"name": "humanoid_gpt", "ema": False})
    assert s.name("controller") == "humanoid_gpt"
    assert s.options("controller") == {"ema": False}


def test_an_entry_without_a_name_is_refused():
    with pytest.raises(SpecError, match="name"):
        PipelineSpec(controller={"ema": False}).name("controller")


def test_an_unknown_component_is_refused_before_anything_expensive_runs():
    with pytest.raises(SpecError, match="available"):
        PipelineSpec(controller="no_such_controller").validate()


def test_an_unknown_stage_is_refused():
    with pytest.raises(SpecError, match="unknown stage"):
        PipelineSpec().name("frobnicator")


def test_a_missing_config_file_is_refused():
    with pytest.raises(SpecError, match="does not exist"):
        PipelineSpec(env_config="configs/no_such.yaml").validate()


def test_unknown_yaml_keys_are_refused_by_name(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("detector: mediapipe\nfrobnicator: yes\n")
    with pytest.raises(SpecError, match="frobnicator"):
        PipelineSpec.load(p)


def test_a_missing_spec_file_says_so(tmp_path):
    with pytest.raises(SpecError, match="no pipeline spec"):
        PipelineSpec.load(tmp_path / "absent.yaml")


def test_the_shipped_specs_load_and_validate():
    from pathlib import Path
    for p in sorted(Path("pipelines").glob("*.yaml")):
        PipelineSpec.load(p).validate()


# -------------------------------------------------------------- the cache key
def test_swapping_the_controller_does_not_invalidate_the_reference():
    """The whole point: comparing controllers must not re-run perception."""
    a, b = PipelineSpec(), PipelineSpec(controller="open_loop")
    assert a.stage_hash("retargeter", CLIP) == b.stage_hash("retargeter", CLIP)
    assert a.stage_hash("controller", CLIP) != b.stage_hash("controller", CLIP)


def test_swapping_the_estimator_invalidates_everything_downstream():
    a, b = PipelineSpec(), PipelineSpec(estimator="smpl")
    assert a.stage_hash("estimator", CLIP) != b.stage_hash("estimator", CLIP)
    assert a.stage_hash("retargeter", CLIP) != b.stage_hash("retargeter", CLIP)
    assert a.stage_hash("detector", CLIP) == b.stage_hash("detector", CLIP)


def test_stage_options_are_part_of_the_key():
    a = PipelineSpec(estimator="mediapipe")
    b = PipelineSpec(estimator={"name": "mediapipe", "pose_bundle": "pose_heavy"})
    assert a.stage_hash("estimator", CLIP) != b.stage_hash("estimator", CLIP)


def test_a_different_clip_is_a_different_artifact():
    s = PipelineSpec()
    assert s.stage_hash("retargeter", "clips/a.mp4") != \
        s.stage_hash("retargeter", "clips/b.mp4")


def test_config_files_are_keyed_by_content_not_path(tmp_path):
    """Editing a config must invalidate the cache; renaming it need not matter."""
    import shutil
    original = tmp_path / "one.yaml"
    shutil.copy("configs/upper_body.yaml", original)
    same = tmp_path / "two.yaml"
    shutil.copy("configs/upper_body.yaml", same)
    edited = tmp_path / "three.yaml"
    edited.write_text(original.read_text() + "\n# a change\n")

    def h(path):
        return PipelineSpec(env_config=str(path)).stage_hash("controller", CLIP)

    assert h(original) == h(same), "same content must hash the same"
    assert h(original) != h(edited), "edited content must hash differently"


def test_every_stage_has_a_hash_and_they_differ():
    s = PipelineSpec()
    hashes = [s.stage_hash(st, CLIP) for st in STAGES]
    assert len(set(hashes)) == len(STAGES)


# ----------------------------------------------------------------- the record
def _fake_inputs():
    role = {"episodes": 12, "survival_rate": 0.92, "survival_at_horizon": 0.833,
            "dof_err_mean_rad": 0.163, "link_err_m": 0.111,
            "foot_slip_mps": 0.263, "target_clip_frac": 0.051,
            "time_to_fall_s": 1.12, "episode_len_s": 4.45}
    nominal = {"results": {"reference": dict(role, survival_at_horizon=None),
                           "open_loop": dict(role, survival_at_horizon=0.0),
                           "humanoid_gpt": role}}
    motion_meta = {
        "model_hash": "b0aee9448b002f9e",
        "config": {"control_fps": 50.0, "max_root_yaw_rate": 60.0},
        "diagnostics": {"ik_residual_p95": 0.1464, "n_collision_frames": 266,
                        "n_foot_distance_fixed": 170, "n_not_converged": 0},
        "source": {"frontend": "mediapipe", "frontend_version": "1.0.1",
                   "coverage": 0.9889, "units": "metres_hip_centred"},
    }
    fidelity = {"retarget": (0.729, 0.0, 12), "policy": (0.594, 0.08, 12),
                "end_to_end": (0.550, 0.077, 12),
                "episode_steps": (223.0, 12.0, 12)}
    return nominal, motion_meta, fidelity


def _record(**over):
    nominal, motion_meta, fidelity = _fake_inputs()
    kw = dict(source="clips/a.mp4",
              artifacts={"clip": "clips/a.mp4", "motion": "motion/a.npz"},
              motion_meta=motion_meta,
              controller_info={"name": "humanoid_gpt", "class": "X"},
              eval_nominal=nominal, eval_random=None, fidelity=fidelity,
              seeds=12)
    kw.update(over)
    return build_record(PipelineSpec(), **kw)


def test_the_record_carries_the_attributed_chain():
    m = _record()["metrics"]
    assert m["source_to_reference"]["mean"] == pytest.approx(0.729)
    assert m["reference_to_policy"]["mean"] == pytest.approx(0.594)
    assert m["end_to_end"]["mean"] == pytest.approx(0.550)


def test_the_record_carries_stability_and_the_two_baselines():
    m = _record()["metrics"]
    assert m["stability"]["nominal"]["survival_at_horizon"] == pytest.approx(0.833)
    assert m["baselines"]["open_loop"]["survival_at_horizon"] == pytest.approx(0.0)
    assert m["baselines"]["reference"] is not None


def test_the_record_carries_the_provenance_the_motion_already_had():
    r = _record()
    assert r["components"]["estimator"]["frontend"] == "mediapipe"
    assert r["components"]["estimator"]["frontend_version"] == "1.0.1"
    assert r["components"]["retargeter"]["diagnostics"]["n_collision_frames"] == 266
    assert r["components"]["model_hash"] == "b0aee9448b002f9e"
    assert r["reference"]["ik_residual_p95"] == pytest.approx(0.1464)


def test_the_record_names_every_component_and_its_stage_hash():
    r = _record()
    for stage in STAGES:
        assert r["spec"][stage]["name"]
        assert r["stage_hashes"][stage]
    assert r["schema"] == SCHEMA


def test_the_randomised_pass_is_recorded_when_run():
    nominal, _, _ = _fake_inputs()
    randomised = {"results": {"humanoid_gpt":
                              dict(nominal["results"]["humanoid_gpt"],
                                   survival_at_horizon=0.5)}}
    m = _record(eval_random=randomised)["metrics"]
    assert m["stability"]["randomised"]["survival_at_horizon"] == pytest.approx(0.5)


def test_the_record_is_json_serialisable():
    json.dumps(_record(), default=str)


def test_the_summary_shows_the_chain_and_both_baselines():
    text = summarise(_record())
    for want in ("source -> reference", "reference -> policy", "end to end",
                 "nominal", "open_loop", "humanoid_gpt"):
        assert want in text, want
    assert "0.550" in text


def test_the_summary_does_not_rank_anything():
    """Reporting the vector was a decision; a winner line would undo it."""
    text = summarise(_record()).lower()
    for banned in ("best", "winner", "score:", "rank"):
        assert banned not in text, banned


# ---------------------------------------------------------------- comparison
def test_compare_puts_records_side_by_side():
    a = _record()
    b = build_record(PipelineSpec(controller="open_loop"),
                     source="clips/a.mp4", artifacts={},
                     motion_meta=_fake_inputs()[1],
                     controller_info={"name": "open_loop"},
                     eval_nominal=_fake_inputs()[0], eval_random=None,
                     fidelity={"retarget": (0.729, 0.0, 12),
                               "policy": (0.494, 0.14, 12),
                               "end_to_end": (0.376, 0.20, 12),
                               "episode_steps": (70.0, 9.0, 12)},
                     seeds=12)
    from g1dance.pipeline.record import compare
    text = compare([("zeroshot", a), ("baseline", b)])
    assert "zeroshot" in text and "baseline" in text
    assert "humanoid_gpt" in text and "open_loop" in text
    assert "0.550" in text and "0.376" in text
    assert "end to end" in text


def test_compare_of_nothing_says_so():
    from g1dance.pipeline.record import compare
    assert "no records" in compare([])


def test_compare_names_no_winner():
    """Ranking was explicitly out of scope; this keeps it out."""
    from g1dance.pipeline.record import compare
    text = compare([("a", _record())]).lower()
    for banned in ("best", "winner", "rank", "score:", "✓", "*"):
        assert banned not in text, banned


def test_compare_shows_the_upstream_rows_that_make_it_controlled():
    """A comparison is only readable if you can see what was held fixed."""
    from g1dance.pipeline.record import compare
    text = compare([("a", _record())])
    for row in ("detector", "estimator", "retargeter", "controller",
                "coverage", "ik residual p95", "collision frames"):
        assert row in text, row


def test_compare_shows_the_option_that_distinguishes_two_stages():
    """A table that hides what changed is worse than no table."""
    from g1dance.pipeline.record import compare
    a = _record()
    b = build_record(
        PipelineSpec(estimator={"name": "mediapipe", "pose_bundle": "pose_heavy"}),
        source="clips/a.mp4", artifacts={}, motion_meta=_fake_inputs()[1],
        controller_info={"name": "humanoid_gpt"},
        eval_nominal=_fake_inputs()[0], eval_random=None,
        fidelity=_fake_inputs()[2], seeds=12)
    text = compare([("full", a), ("heavy", b)])
    assert "pose_bundle=pose_heavy" in text


def test_compare_columns_do_not_run_together():
    """Two long values in adjacent columns must stay separated."""
    from g1dance.pipeline.record import compare
    text = compare([("a", _record()), ("b", _record())])
    for line in text.splitlines():
        assert "humanoid_gpthumanoid_gpt" not in line


@pytest.mark.parametrize("spec, stage", [
    (dict(estimator="smplest_x"), "hand_estimator"),
    (dict(estimator="smplest_x", hand_estimator="none", retargeter="builtin"),
     None),
    (dict(estimator="mediapipe", retargeter="gmr"), "retargeter"),
    (dict(estimator="gvhmr", retargeter="gmr"), None),
])
def test_an_impossible_combination_is_refused_before_anything_runs(spec, stage):
    s = PipelineSpec(**spec)
    if stage is None:
        s.validate()
    else:
        with pytest.raises(SpecError, match=stage):
            s.validate()


def test_every_shipped_component_declares_what_it_provides_or_needs():
    """An undeclared one is let through validation, so none may ship that way."""
    from g1dance.motion import hands, retargeters
    from g1dance.motion.frontends import PROVIDES, available_frontends
    assert set(PROVIDES) == set(available_frontends())
    assert set(hands.NEEDS) == set(hands.available_hand_estimators())
    assert set(retargeters.NEEDS) == set(retargeters.available_retargeters())


def test_a_change_to_a_stages_code_invalidates_it_and_only_downstream(monkeypatch):
    """Fixing a bug in the hand stage must not re-run the detector."""
    import g1dance.pipeline.spec as S
    s = PipelineSpec()
    before = {st: s.stage_hash(st, CLIP) for st in STAGES}
    real = S.code_digest
    monkeypatch.setattr(S, "code_digest",
                        lambda st: "changed" if st == "hand_estimator" else real(st))
    after = {st: s.stage_hash(st, CLIP) for st in STAGES}
    assert after["detector"] == before["detector"]
    assert after["estimator"] == before["estimator"]
    for st in ("hand_estimator", "retargeter", "controller"):
        assert after[st] != before[st]


def test_every_stage_names_code_that_exists():
    from pathlib import Path
    import g1dance
    root = Path(g1dance.__file__).parent
    from g1dance.pipeline.spec import STAGE_CODE
    assert set(STAGE_CODE) == set(STAGES)
    for paths in STAGE_CODE.values():
        for rel in paths:
            assert (root / rel).exists(), rel
