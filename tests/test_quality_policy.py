import numpy as np

from omniretargeting.quality_policy import (
    checkpoint_key,
    classify,
    evaluate,
    pair_statistics,
    reset_states,
)


def test_pair_statistics_is_strict_and_reports_interval_runs():
    depth = np.array([0.0, 0.0021, 0.0021, 0.0, 0.0051, 0.010, 0.0])
    stats = pair_statistics(depth, 30.0)
    two = stats["thresholds_m"]["0.002"]
    five = stats["thresholds_m"]["0.005"]
    ten = stats["thresholds_m"]["0.01"]
    assert two["runs"] == [[1, 2], [4, 5]]
    assert two["run_count"] == 2
    assert two["longest_run_frames"] == 2
    assert two["longest_run_start"] == 1
    assert two["frames"] == [1, 2, 4, 5]
    assert five["frames"] == [4, 5]
    assert ten["frames"] == []
    assert stats["max_depth_m"] == 0.010
    assert stats["worst_frame"] == 5


def test_unknown_frames_provide_threshold_upper_bound():
    depth = np.array([0.0, 0.006, 0.0, 0.0, 0.0, 0.006])
    unknown = np.array([False, False, True, True, True, False])
    stats = pair_statistics(depth, 30.0, unknown)
    five = stats["thresholds_m"]["0.005"]
    assert stats["unknown_frame_count"] == 3
    assert five["frames"] == [1, 5]
    assert five["runs"] == [[1, 1], [5, 5]]
    assert five["frame_fraction"] == 2 / 6
    assert five["upper"]["runs"] == [[1, 5]]
    assert five["upper"]["longest_run_frames"] == 5
    assert five["upper"]["frame_fraction"] == 5 / 6


def test_support_residual_is_verified_every_over_threshold_frame():
    depth = np.full(31, 0.003)
    stats = pair_statistics(depth, 30.0)
    supported = classify(
        stats,
        lateral_or_self=False,
        support_frames=np.ones(31, dtype=bool),
        depth=depth,
    )
    one_lateral_frame = np.ones(31, dtype=bool)
    one_lateral_frame[7] = False
    unsupported = classify(
        stats,
        lateral_or_self=False,
        support_frames=one_lateral_frame,
        depth=depth,
    )
    assert supported["classes"] == ["support_residual"]
    assert unsupported["classes"] == ["persistent"]


def test_events_use_threshold_epsilon_and_five_frame_halo():
    depth = np.zeros(20)
    depth[10] = 0.011
    stats = pair_statistics(depth, 30.0)
    decision = classify(stats, lateral_or_self=False, depth=depth)
    assert decision["classes"] == ["deep"]
    assert len(decision["events"]) == 1
    event = decision["events"][0]
    assert event["eps_m"] == 0.010
    assert event["reason"] == "deep"
    assert event["frames"].tolist() == [False] * 5 + [True] * 11 + [False] * 4


def test_classify_can_use_statistics_without_resupplying_depth():
    depth = np.array([0.0, 0.004])
    decision = classify(pair_statistics(depth, 30.0), lateral_or_self=False)
    assert decision["classes"] == ["persistent"]


def test_lateral_persistent_event_targets_two_millimetre_frames():
    depth = np.zeros(40)
    depth[10:21] = 0.003
    decision = classify(
        pair_statistics(depth, 30.0),
        lateral_or_self=True,
        depth=depth,
    )
    assert decision["classes"] == ["persistent"]
    event = decision["events"][0]
    assert event["eps_m"] == 0.002
    assert event["reason"] == "persistent_2mm"
    assert event["frames"].tolist() == [False] * 5 + [True] * 21 + [False] * 14


def test_reset_states_priority_and_lateral_threshold():
    states = reset_states(
        static=np.array([0.006, 0.0, 0.0, 0.0, 0.0011]),
        moving=np.zeros(5),
        self_depth=np.zeros(5),
        lateral=np.array([False, False, False, True, True]),
        unknown=np.array([True, True, False, False, False]),
    )
    assert states.tolist() == [
        "invalid",
        "unknown",
        "candidate",
        "candidate",
        "invalid",
    ]


def test_checkpoint_key_orders_total_deep_excess_and_velocity_first():
    two_pairs = evaluate(
        {
            "a": {"depth": np.array([0.011]), "kind": "static"},
            "b": {"depth": np.array([0.011]), "kind": "moving"},
        },
        30.0,
    )
    one_huge = evaluate(
        {"a": {"depth": np.array([0.080]), "kind": "static"}},
        30.0,
    )
    common = dict(
        velocity_excess=0.0,
        chatter_excess=0.0,
        support_excess=0.0,
        control_excess=0.0,
        posture_excess=0.0,
        base_cost=0.0,
    )
    assert checkpoint_key(two_pairs, **common) < checkpoint_key(one_huge, **common)
    assert checkpoint_key(
        one_huge,
        velocity_excess=1.0,
        chatter_excess=99.0,
        support_excess=99.0,
        control_excess=99.0,
        posture_excess=99.0,
        base_cost=99.0,
    ) > checkpoint_key(
        one_huge,
        velocity_excess=0.0,
        chatter_excess=99.0,
        support_excess=99.0,
        control_excess=99.0,
        posture_excess=99.0,
        base_cost=99.0,
    )
