"""Net wrench feasibility, ambiguous support sets and complete centroidal motion."""

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from omniretargeting.centroidal_demand import estimate_centroidal_demand
from omniretargeting.contact_mechanics import (
    ContactCandidate,
    WrenchCone,
    explain_contacts,
)


def point(name, location=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), **kw):
    return ContactCandidate(
        name,
        np.array([location]),
        np.array([location]),
        np.array(normal),
        friction=kw.pop("friction", 0.5),
        reconstruction_bound_m=kw.pop("reconstruction_bound_m", 0.0),
        **kw,
    )


def test_optional_does_not_mean_no_support_and_flight_is_retained():
    contacts = [point("left"), point("right")]
    report = explain_contacts(contacts, [0, 0, 100, 0, 0, 0])
    assert report["roles"] == {"left": "optional_contact", "right": "optional_contact"}
    assert report["minimal_feasible_sets"] == [["left"], ["right"]]
    assert report["flight"]["status"] == "infeasible"
    flight = explain_contacts(contacts, np.zeros(6))
    assert flight["minimal_feasible_sets"] == [[]]
    assert not flight["force_distribution_inferred"]


def test_hand_is_necessary_only_when_its_removal_is_infeasible():
    contacts = [
        point("foot", friction=0.1),
        point("hand", normal=(1, 0, 0), friction=0),
    ]
    report = explain_contacts(contacts, [60, 0, 100, 0, 0, 0])
    assert set(report["roles"].values()) == {"necessary_support_candidate"}
    assert report["minimal_feasible_sets"] == [["foot", "hand"]]
    impossible = explain_contacts(contacts, [-60, 0, 100, 0, 0, 0])
    assert set(impossible["roles"].values()) == {"unexplained"}


def test_geometry_cannot_be_overruled_by_dynamics():
    hand = ContactCandidate(
        "hand",
        np.array([[0.05, 0, 0]]),
        np.zeros((1, 3)),
        np.array([1, 0, 0]),
        friction=0.5,
        reconstruction_bound_m=0.02,
    )
    report = explain_contacts([hand], [10, 0, 0, 0, 0, 0])
    assert report["roles"]["hand"] == "geometry_rejected"
    assert report["full"]["status"] == "infeasible"


def test_sliding_does_not_get_arbitrary_static_friction():
    contact = point(
        "sliding", mode="sliding", sliding_velocity=np.array([1.0, 0.0, 0.0])
    )
    cone = WrenchCone([contact])
    assert cone.check([-50, 0, 100, 0, 0, 0])["nominal_feasible"]
    assert not cone.check([50, 0, 100, 0, 0, 0])["nominal_feasible"]
    assert not cone.check([0, 0, 100, 0, 0, 0])["nominal_feasible"]


def test_uncertainty_is_not_forced_into_a_definite_support_role():
    cone = WrenchCone([point("foot")])
    assert (
        cone.check([51, 0, 100, 0, 0, 0], [2, 0, 0, 0, 0, 0])["status"] == "uncertain"
    )
    assert (
        cone.check([60, 0, 100, 0, 0, 0], [2, 0, 0, 0, 0, 0])["status"] == "infeasible"
    )


@pytest.mark.parametrize("seed", range(4))
def test_cone_conversion_matches_independent_ray_feasibility(seed):
    rng = np.random.default_rng(seed)
    contacts = [point(str(i), rng.normal(size=3), rng.normal(size=3)) for i in range(4)]
    rays = np.concatenate([contact.rays(np.zeros(3), 1.0) for contact in contacts])
    cone = WrenchCone(contacts)
    for wrench in [rng.uniform(size=len(rays)) @ rays, *rng.normal(size=(12, 6))]:
        reference = linprog(
            np.zeros(len(rays)),
            A_eq=rays.T,
            b_eq=wrench,
            bounds=(0, None),
            method="highs",
        )
        assert cone.check(wrench)["nominal_feasible"] == reference.success


def test_rigid_coordinates_and_moment_reference_do_not_change_feasibility():
    contact = point("foot", location=(0.2, -0.1, 0))
    wrench = np.r_[[10, 5, 100], np.cross(contact.surface_points[0], [10, 5, 100])]
    rotation = Rotation.from_rotvec([0.8, 0.1, -0.3]).as_matrix()
    shift = np.array([1.0, -2.0, 0.3])
    changed = ContactCandidate(
        "foot",
        contact.observed_points @ rotation.T + shift,
        contact.surface_points @ rotation.T + shift,
        contact.normals @ rotation.T,
        contact.friction,
        0.0,
    )
    # Origin translates with the world, so only rotation is needed here.
    transformed = np.r_[rotation @ wrench[:3], rotation @ wrench[3:]]
    assert WrenchCone([changed], origin=shift).check(transformed)["nominal_feasible"]
    about_zero = transformed.copy()
    about_zero[3:] += np.cross(shift, transformed[:3])
    assert WrenchCone([changed]).check(about_zero)["nominal_feasible"]
    assert not WrenchCone([changed]).check(transformed)["nominal_feasible"]


def segments(frames=61):
    centers = np.zeros((frames, 2, 3))
    centers[:, 0, 0] = -0.2
    centers[:, 1, 0] = 0.4
    rotations = np.tile(np.eye(3), (frames, 2, 1, 1))
    return centers, rotations, np.array([2.0, 1.0]), np.tile(np.eye(3) * 0.1, (2, 1, 1))


def test_com_uses_all_masses_and_apex_can_be_explained_by_flight():
    centers, rotations, masses, inertias = segments()
    times = np.arange(len(centers)) / 30 - 1
    centers[:, :, 2] = (2 - 0.5 * 9.81 * times**2)[:, None]
    before = centers.copy()
    demand = estimate_centroidal_demand(
        centers, rotations, masses, inertias, framerate=30
    )
    np.testing.assert_allclose(demand["com"][:, 0], 0, atol=1e-12)
    np.testing.assert_allclose(demand["wrench"], 0, atol=1e-9)
    np.testing.assert_array_equal(centers, before)


def test_segment_spin_changes_momentum_even_with_stationary_com():
    centers, rotations, masses, inertias = segments()
    times = np.arange(len(centers)) / 30
    rotations[:, 0] = Rotation.from_rotvec(
        np.c_[np.zeros((len(times), 2)), 0.5 * times**2]
    ).as_matrix()
    demand = estimate_centroidal_demand(
        centers, rotations, masses, inertias, framerate=30
    )
    np.testing.assert_allclose(demand["torque_com"][8:-8, 2], 0.1, atol=1e-6)
    np.testing.assert_allclose(demand["wrench"][:, 2], masses.sum() * 9.81, atol=1e-8)
