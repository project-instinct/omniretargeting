"""Tests for the source and robot sole frame contract."""

import numpy as np
import pytest

from omniretargeting.foot_frames import (
    calibrate_sole_rotation,
    orthonormalize,
    parse_body_to_sole,
    world_sole_rotations,
)


def _slab_foot(length=0.26, width=0.10, thickness=0.05, n=12):
    """A box-shaped foot lying flat: long in +x, wide in y, thin in z."""
    xs = np.linspace(0.0, length, n)
    ys = np.linspace(-width / 2, width / 2, n)
    zs = np.linspace(0.0, thickness, 3)
    grid = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    return grid


class TestSoleCalibration:
    def test_flat_foot_yields_axis_aligned_sole_frame(self):
        verts = _slab_foot()
        # Ankle above the heel end, toe joint above the far end.
        joints = np.zeros((12, 3))
        joints[7] = [0.0, 0.0, 0.08]
        joints[10] = [0.20, 0.0, 0.02]
        mask = np.ones(len(verts), dtype=bool)

        cal = calibrate_sole_rotation("left", verts, joints, mask, 7, 10)

        assert cal.rotation.shape == (3, 3)
        assert np.isclose(np.linalg.det(cal.rotation), 1.0)
        np.testing.assert_allclose(cal.rotation @ cal.rotation.T, np.eye(3), atol=1e-9)
        # Sole normal points away from the ground, toward the ankle.
        np.testing.assert_allclose(cal.rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-6)
        # Forward points at the toe.
        np.testing.assert_allclose(cal.rotation[:, 0], [1.0, 0.0, 0.0], atol=1e-6)

    def test_sole_normal_flips_to_follow_the_ankle(self):
        verts = _slab_foot()
        joints = np.zeros((12, 3))
        joints[7] = [0.0, 0.0, -0.08]  # ankle below: sole normal must flip
        joints[10] = [0.20, 0.0, -0.02]
        mask = np.ones(len(verts), dtype=bool)

        cal = calibrate_sole_rotation("left", verts, joints, mask, 7, 10)
        np.testing.assert_allclose(cal.rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-6)

    def test_toe_out_splay_is_captured(self):
        # A toe joint offset in y must show up as yaw in the calibration, which
        # is exactly the offset that makes the raw joint frame unusable.
        verts = _slab_foot()
        joints = np.zeros((12, 3))
        joints[7] = [0.0, 0.0, 0.08]
        joints[10] = [0.20, 0.08, 0.02]
        mask = np.ones(len(verts), dtype=bool)

        cal = calibrate_sole_rotation("left", verts, joints, mask, 7, 10)
        yaw = np.degrees(np.arctan2(cal.rotation[1, 0], cal.rotation[0, 0]))
        assert yaw == pytest.approx(np.degrees(np.arctan2(0.08, 0.20)), abs=1e-3)

    def test_instep_mass_does_not_tilt_the_sole_normal(self):
        """The whole-volume principal axis is not the sole normal.

        A real foot carries most of its mass above the heel, which tilts the
        smallest principal axis away from the ground plane; measured on SMPL-X
        the tilt is about fifteen degrees.  The calibration must recover the
        plane the foot actually stands on, not the volume's thin axis.
        """
        sole = _slab_foot(width=0.14, thickness=0.015, n=14)
        # A wedge of instep piled over the heel half, tilting the volume PCA.
        heel = sole[sole[:, 0] < 0.13].copy()
        instep = np.repeat(heel, 2, axis=0)
        instep[:, 2] += np.tile(np.linspace(0.02, 0.05, 2), len(heel))
        verts = np.vstack([sole, instep])

        joints = np.zeros((12, 3))
        joints[7] = [0.06, 0.0, 0.09]
        joints[10] = [0.20, 0.0, 0.015]
        mask = np.ones(len(verts), dtype=bool)

        raw = np.linalg.svd(verts - verts.mean(axis=0), full_matrices=False)[2][-1]
        raw = raw if raw[2] > 0 else -raw
        raw_tilt = np.degrees(np.arccos(np.clip(raw @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0)))

        cal = calibrate_sole_rotation("left", verts, joints, mask, 7, 10)
        refined_tilt = np.degrees(
            np.arccos(np.clip(cal.rotation[:, 2] @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
        )

        assert raw_tilt > 5.0, "fixture does not reproduce the instep bias it is meant to test"
        assert refined_tilt < raw_tilt / 2.0
        assert refined_tilt < 3.0

    def test_isotropic_volume_is_rejected(self):
        # A blob has no identifiable sole plane; guessing one would silently
        # produce a meaningless orientation target.
        rng = np.random.default_rng(0)
        blob = rng.normal(size=(400, 3))
        joints = np.zeros((12, 3))
        joints[7] = [0.0, 0.0, 1.0]
        joints[10] = [1.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="isotropic"):
            calibrate_sole_rotation("left", blob, joints, np.ones(len(blob), bool), 7, 10)

    def test_too_few_vertices_is_rejected(self):
        joints = np.zeros((12, 3))
        joints[7] = [0.0, 0.0, 1.0]
        joints[10] = [1.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="at least 4"):
            calibrate_sole_rotation("left", np.zeros((3, 3)), joints, np.ones(3, bool), 7, 10)

    def test_toe_parallel_to_sole_normal_is_rejected(self):
        verts = _slab_foot()
        joints = np.zeros((12, 3))
        joints[7] = [0.0, 0.0, 0.08]
        joints[10] = [0.0, 0.0, 0.20]  # straight up: no forward direction
        with pytest.raises(ValueError, match="parallel"):
            calibrate_sole_rotation("left", verts, joints, np.ones(len(verts), bool), 7, 10)


class TestWorldTransport:
    def test_identity_pose_returns_the_calibration(self):
        verts = _slab_foot()
        joints = np.zeros((12, 3))
        joints[7] = [0.0, 0.0, 0.08]
        joints[10] = [0.20, 0.0, 0.02]
        cal = calibrate_sole_rotation("left", verts, joints, np.ones(len(verts), bool), 7, 10)

        quats = np.zeros((4, 12, 4))
        quats[:, :, 0] = 1.0  # identity, wxyz
        world = world_sole_rotations(quats, cal)
        assert world.shape == (4, 3, 3)
        for t in range(4):
            np.testing.assert_allclose(world[t], cal.rotation, atol=1e-9)

    def test_joint_rotation_is_applied_on_the_left(self):
        from scipy.spatial.transform import Rotation

        verts = _slab_foot()
        joints = np.zeros((12, 3))
        joints[7] = [0.0, 0.0, 0.08]
        joints[10] = [0.20, 0.0, 0.02]
        cal = calibrate_sole_rotation("left", verts, joints, np.ones(len(verts), bool), 7, 10)

        spin = Rotation.from_euler("y", 30.0, degrees=True)
        quats = np.zeros((1, 12, 4))
        quats[0, :, 0] = 1.0
        quats[0, cal.toe_index] = spin.as_quat(scalar_first=True)

        world = world_sole_rotations(quats, cal)
        np.testing.assert_allclose(world[0], spin.as_matrix() @ cal.rotation, atol=1e-9)

    def test_wrong_shape_is_rejected(self):
        verts = _slab_foot()
        joints = np.zeros((12, 3))
        joints[7] = [0.0, 0.0, 0.08]
        joints[10] = [0.20, 0.0, 0.02]
        cal = calibrate_sole_rotation("left", verts, joints, np.ones(len(verts), bool), 7, 10)
        with pytest.raises(ValueError, match="shape"):
            world_sole_rotations(np.zeros((4, 4)), cal)

    def test_missing_toe_joint_is_rejected(self):
        verts = _slab_foot()
        joints = np.zeros((12, 3))
        joints[7] = [0.0, 0.0, 0.08]
        joints[10] = [0.20, 0.0, 0.02]
        cal = calibrate_sole_rotation("left", verts, joints, np.ones(len(verts), bool), 7, 10)
        with pytest.raises(ValueError, match="out of range"):
            world_sole_rotations(np.zeros((4, 8, 4)), cal)


class TestRobotTransform:
    def test_default_is_identity(self):
        np.testing.assert_allclose(parse_body_to_sole(None, "left"), np.eye(3))

    def test_valid_rotation_is_accepted(self):
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_euler("z", 12.0, degrees=True).as_matrix()
        np.testing.assert_allclose(parse_body_to_sole(rot.tolist(), "left"), rot, atol=1e-12)

    def test_non_orthonormal_is_rejected(self):
        with pytest.raises(ValueError, match="orthonormal"):
            parse_body_to_sole([[2, 0, 0], [0, 1, 0], [0, 0, 1]], "left")

    def test_reflection_is_rejected(self):
        with pytest.raises(ValueError, match="determinant"):
            parse_body_to_sole([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], "left")

    def test_wrong_shape_is_rejected(self):
        with pytest.raises(ValueError, match="3x3"):
            parse_body_to_sole([[1, 0], [0, 1]], "left")


class TestOrthonormalize:
    def test_drifted_rotation_is_repaired(self):
        from scipy.spatial.transform import Rotation

        exact = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
        drifted = exact + 1e-3 * np.ones((3, 3))
        repaired = orthonormalize(drifted)
        np.testing.assert_allclose(repaired @ repaired.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(repaired), 1.0)

    def test_reflection_is_mapped_to_a_proper_rotation(self):
        repaired = orthonormalize(np.diag([1.0, 1.0, -1.0]))
        assert np.isclose(np.linalg.det(repaired), 1.0)


class TestSoleDerivedFromCollisionGeoms:
    """The robot-side sole read off the foot's own collision geoms.

    A hand-measured offset per robot does not transfer: across the humanoids in
    this repository the sole plane sits between 3.5 cm and 19 cm below the foot
    body origin, and as a fraction of robot height that spans 2.8 to 11 percent,
    so neither a constant nor a height-normalised constant works.
    """

    @staticmethod
    def _slab(length=0.20, width=0.08, thickness=0.02, drop=0.05, n=9):
        """A flat sole plate hanging below the body origin, like a real foot."""
        xs = np.linspace(-0.05, length - 0.05, n)
        ys = np.linspace(-width / 2, width / 2, n)
        zs = np.array([-drop - thickness, -drop])
        return np.array([[x, y, z] for x in xs for y in ys for z in zs])

    def test_reference_point_lands_on_the_lowest_surface(self):
        from omniretargeting.foot_frames import derive_sole_from_points

        pts = self._slab()
        sole = derive_sole_from_points(pts)
        assert sole.reference_point[2] == pytest.approx(pts[:, 2].min(), abs=1e-9)
        np.testing.assert_allclose(sole.normal, [0.0, 0.0, 1.0], atol=1e-6)

    def test_extents_recover_the_plate_dimensions(self):
        from omniretargeting.foot_frames import derive_sole_from_points

        sole = derive_sole_from_points(self._slab(length=0.20, width=0.08))
        assert sole.extent_forward == pytest.approx(0.20, abs=0.01)
        assert sole.extent_lateral == pytest.approx(0.08, abs=0.01)

    def test_normal_points_away_from_the_ground_toward_the_ankle(self):
        from omniretargeting.foot_frames import derive_sole_from_points

        # Mirror the plate above the origin: the normal must flip with it.
        pts = self._slab()
        pts[:, 2] *= -1.0
        sole = derive_sole_from_points(pts)
        np.testing.assert_allclose(sole.normal, [0.0, 0.0, -1.0], atol=1e-6)

    def test_tilted_sole_is_recovered_by_refinement(self):
        """A first normal a few degrees off truncates the detected sole.

        On one robot here the unrefined pass reported a 7 cm sole for a 19 cm
        foot, because a tilted band only catches the lowest patch.
        """
        from scipy.spatial.transform import Rotation

        from omniretargeting.foot_frames import derive_sole_from_points

        tilt = Rotation.from_euler("y", 8.0, degrees=True).as_matrix()
        pts = self._slab(length=0.20) @ tilt.T
        sole = derive_sole_from_points(pts)
        assert sole.extent_forward == pytest.approx(0.20, abs=0.02)
        recovered = np.degrees(
            np.arccos(np.clip(sole.normal @ (tilt @ np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
        )
        assert recovered < 1.5

    def test_blobby_geometry_is_rejected_rather_than_guessed(self):
        from omniretargeting.foot_frames import derive_sole_from_points

        rng = np.random.default_rng(4)
        blob = rng.normal(size=(300, 3)) * 0.05
        with pytest.raises(ValueError, match="no distinguishable sole plane"):
            derive_sole_from_points(blob)

    def test_too_few_points_is_rejected(self):
        from omniretargeting.foot_frames import derive_sole_from_points

        with pytest.raises(ValueError, match="at least 4"):
            derive_sole_from_points(np.zeros((3, 3)))
