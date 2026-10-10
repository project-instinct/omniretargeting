"""Regression checks for the reviewed input and solver failure boundaries."""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from omniretargeting import MotionData, MotionFrame, OmniRetargeter, load_robot_config
from omniretargeting.data_sources.base import DataSource
from omniretargeting.data_sources.registry import create_data_source
from omniretargeting.retargeting import _solve_qp_clarabel
from omniretargeting.utils import create_flat_terrain, estimate_body_height

REPO_ROOT = Path(__file__).resolve().parents[1]
NAMES = ["Pelvis", "L_Hip", "R_Hip", "Spine1"]
POSITIONS = np.array([[0, 0, 1], [0, -0.1, 0.9], [0, 0.1, 0.9], [0, 0, 1.2]])


@pytest.fixture
def g1_kwargs(tmp_path):
    cfg = load_robot_config(REPO_ROOT / "robot_models/unitree_g1/unitree_g1.json")
    terrain = tmp_path / "terrain.obj"
    create_flat_terrain().export(terrain)
    return dict(
        robot_urdf_path=cfg["urdf_path"],
        terrain_mesh_path=terrain,
        joint_mapping={
            name: cfg["joint_mapping"][name if name != "Spine1" else "Spine2"]
            for name in NAMES
        },
        source_target_names=NAMES,
        base_orientation=dict(zip(["pelvis", "left_hip", "right_hip", "spine"], NAMES)),
        retargeting={"penetration_resolver": "xyz_nudge"},
    )


@pytest.fixture
def retargeter(g1_kwargs):
    return OmniRetargeter(**g1_kwargs)


def test_qp_structure_error_is_raised():
    with pytest.raises(RuntimeError, match="CLARABEL") as error:
        _solve_qp_clarabel(sp.eye(3), np.zeros(2), -np.ones(2), np.ones(2))
    assert error.value.__cause__ is not None
    assert "incompatible dimensions" in str(error.value.__cause__)


def test_backend_exception_has_frame_context_and_does_not_advance(
    retargeter, monkeypatch
):
    import omniretargeting.retargeting as engine

    state = retargeter.create_stream_state()
    before = state.q_init.copy()

    def fail_backend(*args, **kwargs):
        raise ValueError("backend construction failed")

    monkeypatch.setattr(engine.clarabel, "DefaultSolver", fail_backend)
    with pytest.raises(RuntimeError, match="frame 0.*backend construction failed"):
        retargeter.retarget_frame(
            MotionFrame(POSITIONS.copy(), target_names=NAMES), state
        )
    assert state.frame_idx == 0 and state.q_last is None
    np.testing.assert_array_equal(state.q_init, before)
    assert state.last_estimated_quat is None


@pytest.mark.parametrize("resolver", ["xyz_nudge", "hard_constraint"])
def test_exhausted_numerical_recovery_raises(retargeter, monkeypatch, resolver):
    import omniretargeting.retargeting as engine

    retargeter.retargeting_config["penetration_resolver"] = resolver
    monkeypatch.setattr(engine, "_solve_qp_clarabel", lambda *a, **k: (None, False))
    state = retargeter.create_stream_state()
    with pytest.raises(RuntimeError, match="frame 0.*qp_solver_failed"):
        retargeter.retarget_frame(
            MotionFrame(POSITIONS.copy(), target_names=NAMES), state
        )
    assert state.frame_idx == 0 and state.q_last is None
    assert state.retargeter.hard_penetration_constraint == (
        resolver == "hard_constraint"
    )
    with pytest.raises(RuntimeError, match="frame 0"):
        retargeter.retarget_motion(
            MotionData(POSITIONS[None].copy(), target_names=NAMES),
            visualize_trajectory=False,
        )


def test_expected_constraint_recovery_still_returns_a_success(retargeter, monkeypatch):
    from omniretargeting.retargeting import GenericInteractionRetargeter

    retargeter.retargeting_config["penetration_resolver"] = "hard_constraint"
    original = GenericInteractionRetargeter.retarget_frame

    def reject_constraints(self, positions, seed, **kwargs):
        if self.hard_penetration_constraint:
            self.last_solve_diagnostics = {
                "success": False,
                "failure_reason": "test_constraint",
            }
            return seed.copy()
        return original(self, positions, seed, **kwargs)

    monkeypatch.setattr(
        GenericInteractionRetargeter, "retarget_frame", reject_constraints
    )
    state = retargeter.create_stream_state()
    result = retargeter.retarget_frame(
        MotionFrame(POSITIONS.copy(), target_names=NAMES), state
    )
    assert state.frame_idx == 1 and state.retargeter.last_solve_diagnostics["success"]
    assert state.retargeter.hard_penetration_constraint
    assert retargeter._frame_fallback_count == 1
    assert np.isfinite(result).all()


@pytest.mark.parametrize("entry", ["frame", "stream", "motion"])
@pytest.mark.parametrize("mismatch", ["names", "count"])
def test_source_identity_is_checked_before_solving(
    retargeter, monkeypatch, entry, mismatch
):
    positions = POSITIONS.copy()
    names = ["Pelvis", "R_Hip", "L_Hip", "Spine1"]
    if mismatch == "count":
        positions = positions[:3]
        names = None

    def unexpected_estimation(*args):
        pytest.fail("source mismatch reached orientation estimation")

    monkeypatch.setattr(
        retargeter, "_estimate_base_orientation_from_joints", unexpected_estimation
    )
    with pytest.raises(ValueError, match="target"):
        if entry == "motion":
            retargeter.retarget_motion(
                MotionData(positions[None], target_names=names),
                visualize_trajectory=False,
            )
        elif entry == "stream":
            list(
                retargeter.retarget_stream([MotionFrame(positions, target_names=names)])
            )
        else:
            retargeter.retarget_frame(
                MotionFrame(positions, target_names=names),
                retargeter.create_stream_state(),
            )


@pytest.mark.parametrize(
    "mapping_value", ["missing_robot_link", {"robot_link": "missing_robot_link"}]
)
def test_invalid_mapping_does_not_remove_source_targets(g1_kwargs, mapping_value):
    g1_kwargs["joint_mapping"]["L_Hip"] = mapping_value
    with pytest.raises(ValueError, match="L_Hip.*missing_robot_link"):
        OmniRetargeter(**g1_kwargs)


def test_explicit_foot_name_is_not_replaced_by_inference(retargeter):
    with pytest.raises(ValueError, match="left.*typo_left_foot"):
        retargeter._resolve_foot_body_ids({"body_names": {"left": "typo_left_foot"}})
    assert all(value >= 0 for value in retargeter._resolve_foot_body_ids({}).values())


def test_unexpected_body_lookup_error_propagates(retargeter, monkeypatch):
    import mujoco

    def failed_lookup(*args):
        raise RuntimeError("unexpected lookup failure")

    monkeypatch.setattr(mujoco, "mj_name2id", failed_lookup)
    with pytest.raises(RuntimeError, match="unexpected lookup failure"):
        retargeter._resolve_foot_body_ids(
            {"body_names": {"left": "left_ankle_roll_link"}}
        )


def test_enabled_stabilization_with_no_feet_is_an_error(retargeter, monkeypatch):
    retargeter.retargeting_config["foot_stabilization"] = {"enabled": True}
    monkeypatch.setattr(retargeter, "_search_body_id_by_keywords", lambda side: -1)
    with pytest.raises(ValueError, match="stabilization.*foot"):
        retargeter._apply_foot_stabilization(
            retargeter.create_stream_state().q_init[None], retargeter.terrain_mesh
        )


@pytest.mark.parametrize(
    "orientation",
    [np.zeros(4), np.array([2.0, 0, 0, 0]), np.full(4, np.nan), np.ones((1, 4))],
)
@pytest.mark.parametrize("kind", ["frame", "motion"])
def test_invalid_root_orientations_raise_at_construction(kind, orientation):
    with pytest.raises(ValueError, match="root_orientation"):
        if kind == "frame":
            MotionFrame(POSITIONS.copy(), root_orientation=orientation)
        else:
            MotionData(POSITIONS[None].copy(), root_orientations=orientation[None])


@pytest.mark.parametrize(
    "translation",
    [np.array([np.nan, 0, 0]), np.array([0, np.inf, 0]), np.zeros((1, 3))],
)
@pytest.mark.parametrize("kind", ["frame", "motion"])
def test_invalid_root_translations_raise_at_construction(kind, translation):
    with pytest.raises(ValueError, match="root_translation"):
        if kind == "frame":
            MotionFrame(POSITIONS.copy(), root_translation=translation)
        else:
            MotionData(POSITIONS[None].copy(), root_translations=translation[None])


@pytest.mark.parametrize("field", ["source_height", "human_height", "framerate"])
@pytest.mark.parametrize("value", [-1.0, 0.0, np.nan, np.inf])
def test_invalid_motion_scalars_raise_at_construction(field, value):
    with pytest.raises(ValueError, match=field):
        MotionData(POSITIONS[None].copy(), **{field: value})


def test_conflicting_height_aliases_raise():
    with pytest.raises(ValueError, match="source_height.*human_height"):
        MotionData(POSITIONS[None].copy(), source_height=1.5, human_height=2.0)
    motion = MotionData(POSITIONS[None].copy(), human_height=1.8)
    assert motion.source_height == motion.human_height == 1.8


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_orientation", np.array([1.0, 0, 0, 0])),
        ("root_translation", np.zeros(3)),
        ("object_points", np.zeros((2, 3))),
    ],
)
def test_dense_optional_fields_cannot_be_discarded(field, value):
    class Source(DataSource):
        def iter_frames(self):
            yield MotionFrame(POSITIONS.copy(), **{field: value})
            yield MotionFrame(POSITIONS.copy())

    with pytest.raises(ValueError, match=field):
        Source().load()


def test_sparse_missing_contacts_and_dense_all_missing_remain_valid():
    class Source(DataSource):
        def iter_frames(self):
            yield MotionFrame(POSITIONS.copy(), contacts=[])
            yield MotionFrame(POSITIONS.copy())

    motion = Source().load()
    assert motion.root_orientations is None and motion.object_points is None
    assert motion.contact_trajectory == [[], None]


@pytest.mark.parametrize(
    "options",
    [
        {"contact_edge_weigth": 10},
        {"solver_max_iter": {"first_fram": 5}},
        {"foot_stabilization": {"enable": True}},
        {"foot_stabilization": {"body_names": {"lef": "left_ankle_roll_link"}}},
        {"bone_direction": {"enable": True}},
        {"penetration_correction": {"joint_weigth": 1.0}},
        {"joint_regularization_boost": {"defaut": 1.0}},
    ],
)
def test_unknown_solver_controls_raise_instead_of_defaulting(retargeter, options):
    retargeter.retargeting_config.update(options)
    with pytest.raises(ValueError, match="Unknown.*option"):
        retargeter.create_stream_state()


@pytest.mark.parametrize("container", ["source", "nested", "runtime"])
def test_unknown_omomo_controls_raise_before_loading(container):
    options = {"body_position_mdoe": "rest_offsets"}
    source, runtime = {}, {}
    if container == "source":
        source = options
    elif container == "nested":
        source = {"adapter_options": options}
    else:
        runtime = options
    with pytest.raises(ValueError, match="Unknown.*body_position_mdoe"):
        create_data_source("omomo", "unused.p", source, runtime)


def test_omomo_supported_aliases_and_profile_fields_are_preserved(monkeypatch):
    from omniretargeting.data_sources import omomo

    captured = {}
    monkeypatch.setattr(
        omomo, "OmomoDataSource", lambda **kwargs: captured.update(kwargs)
    )
    create_data_source(
        "omomo",
        "unused.p",
        {
            "name": "omomo",
            "type": "omomo",
            "target_mapping": {"Pelvis": "pelvis"},
            "base_orientation": {},
            "metadata": {"note": "test"},
            "height_estimation": {},
            "default_pose_on_robot": "T-Pose",
            "adapter_options": {"n_object_samples": 8},
            "target_names": NAMES,
        },
        {"target_names_override": ["custom"]},
    )
    assert captured["n_object_samples"] == 8 and captured["target_names"] == ["custom"]


def test_automatic_scaling_requires_measured_source_height(retargeter):
    motion = MotionData(POSITIONS[None].copy(), target_names=NAMES)
    with pytest.raises(ValueError, match="source_height"):
        retargeter.retarget_motion(
            motion, enable_scene_scaling=True, visualize_trajectory=False
        )
    scale, result = retargeter.retarget_motion(motion, visualize_trajectory=False)
    assert scale == 1 and np.isfinite(result).all()


def test_unavailable_height_landmarks_do_not_become_an_assumed_height():
    assert (
        estimate_body_height(POSITIONS[None], NAMES, head_joint="missing_head") is None
    )


@pytest.mark.parametrize("wrapper", ["generic", "smplx"])
@pytest.mark.parametrize("scaling", [False, True])
def test_array_wrappers_use_explicit_scaling_and_height(g1_kwargs, wrapper, scaling):
    from omniretargeting.retargeting import retarget_source_to_robot
    from omniretargeting.data_sources.smplx import retarget_smplx_to_robot

    kwargs = {key: value for key, value in g1_kwargs.items() if key != "retargeting"}
    if wrapper == "smplx":
        kwargs["smplx_joint_names"] = kwargs.pop("source_target_names")
        fn = retarget_smplx_to_robot
    else:
        fn = retarget_source_to_robot
    scale, result = fn(
        POSITIONS[None].copy(),
        source_height=1.8,
        enable_scene_scaling=scaling,
        **kwargs,
    )
    robot = OmniRetargeter(**g1_kwargs)
    assert scale == pytest.approx(robot.robot_height / 1.8 if scaling else 1.0)
    assert np.isfinite(result).all()
    with pytest.raises(ValueError, match="source_height"):
        fn(POSITIONS[None].copy(), enable_scene_scaling=True, **kwargs)


def test_loading_preserves_and_checks_frame_target_identity():
    class Source(DataSource):
        def iter_frames(self):
            yield MotionFrame(POSITIONS.copy(), target_names=NAMES)
            yield MotionFrame(POSITIONS.copy(), target_names=NAMES)

    assert Source().load().target_names == NAMES

    class ReorderedSource(Source):
        def iter_frames(self):
            yield MotionFrame(POSITIONS.copy(), target_names=NAMES)
            yield MotionFrame(POSITIONS.copy(), target_names=list(reversed(NAMES)))

    with pytest.raises(ValueError, match="target_names"):
        ReorderedSource().load()


def test_loading_does_not_hide_conflicting_height_aliases():
    class Source(DataSource):
        source_height = 1.5
        human_height = 2.0

        def iter_frames(self):
            yield MotionFrame(POSITIONS.copy())

    with pytest.raises(ValueError, match="source_height.*human_height"):
        Source().load()


@pytest.mark.parametrize(
    "declaration",
    [
        ["Pelvis", "R_Hip", "L_Hip", "Spine1"],
        ["Pelvis", "L_Hip", "R_Hip"],
    ],
)
@pytest.mark.parametrize("frame_names", [None, NAMES])
def test_stream_checks_datasource_identity_before_orientation(
    retargeter, monkeypatch, declaration, frame_names
):
    class Source(DataSource):
        target_names = declaration

        def iter_frames(self):
            yield MotionFrame(POSITIONS.copy(), target_names=frame_names)

    def unexpected_estimation(*args):
        pytest.fail("DataSource identity mismatch reached orientation estimation")

    monkeypatch.setattr(
        retargeter, "_estimate_base_orientation_from_joints", unexpected_estimation
    )
    with pytest.raises(ValueError, match="target"):
        list(retargeter.retarget_stream(Source()))
    with pytest.raises(ValueError, match="target"):
        retargeter.retarget_motion(Source(), visualize_trajectory=False)


@pytest.mark.parametrize("declaration", [None, NAMES])
def test_stream_keeps_lazy_and_positional_datasources_supported(
    retargeter, declaration
):
    class Source(DataSource):
        produced = 0

        def iter_frames(self):
            # Some adapters expose their target order only after iteration begins.
            self.target_names = declaration
            for _ in range(2):
                self.produced += 1
                yield MotionFrame(POSITIONS.copy())

        def load(self):
            pytest.fail("streaming loaded the complete source")

    source = Source()
    stream = retargeter.retarget_stream(source)
    assert source.produced == 0
    result = next(stream)
    assert np.isfinite(result).all()
    assert source.produced == 1
    stream.close()


@pytest.mark.parametrize(
    "resolver",
    [
        "hard_constraint",
        "xyz_nudge",
        "hard_constraint_slack",
    ],
)
@pytest.mark.parametrize("boundary", ["constructor", "stream_state"])
def test_unknown_slack_controls_raise_for_every_resolver(g1_kwargs, resolver, boundary):
    options = {
        "penetration_resolver": resolver,
        "penetration_slack": {"hard_boud": 0.1},
    }
    with pytest.raises(ValueError, match="Unknown.*hard_boud"):
        if boundary == "constructor":
            g1_kwargs["retargeting"].update(options)
            OmniRetargeter(**g1_kwargs)
        else:
            retargeter = OmniRetargeter(**g1_kwargs)
            retargeter.retargeting_config.update(options)
            retargeter.create_stream_state()


@pytest.mark.parametrize(
    "resolver",
    [
        "hard_constraint",
        "xyz_nudge",
        "hard_constraint_slack",
    ],
)
@pytest.mark.parametrize("options", [[], False, "not a dictionary"])
def test_supplied_slack_type_is_checked_when_inactive(g1_kwargs, resolver, options):
    g1_kwargs["retargeting"].update(
        penetration_resolver=resolver,
        penetration_slack=options,
    )
    with pytest.raises(ValueError, match="penetration_slack.*dictionary"):
        OmniRetargeter(**g1_kwargs)


@pytest.mark.parametrize(
    "resolver",
    [
        "hard_constraint",
        "xyz_nudge",
        "hard_constraint_slack",
    ],
)
def test_valid_slack_settings_only_activate_in_slack_mode(retargeter, resolver):
    options = {"soft_tolerance": 0.002, "hard_bound": 0.2, "slack_penalty": 1e4}
    retargeter.retargeting_config.update(
        penetration_resolver=resolver,
        penetration_slack=options,
    )
    state = retargeter.create_stream_state()
    assert state.retargeter.penetration_slack_enabled == (
        resolver == "hard_constraint_slack"
    )
    assert retargeter.retargeting_config["penetration_slack"] == options

    # Valid inactive settings remain available when the fallback selects slack.
    retargeter.retargeting_config["penetration_resolver"] = "hard_constraint_slack"
    state = retargeter.create_stream_state()
    assert state.retargeter.penetration_slack_enabled
    assert state.retargeter.penetration_hard_bound == 0.2
