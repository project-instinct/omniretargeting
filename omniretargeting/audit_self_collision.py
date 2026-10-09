"""Audit-only compiled pair restoration and sampled structural-overlap evidence.

The production SelfCollisionQuery and its numerical behavior remain unchanged.
Unknown model samples never certify structural overlap in this audit.
"""
import mujoco
import numpy as np
from .geometry_query import GeometryQueryError
from .self_collision import (SelfCollisionQuery, STRUCTURAL_SEED, STRUCTURAL_SAMPLES,
                             _random_quaternion)


class AuditSelfCollisionQuery(SelfCollisionQuery):
    def use_model_pairs(self):
        """Use only MJCF pair filters for independent audits.

        Keep legacy acquisition's default structural exclusion unchanged. The
        returned body-name list is provenance, never used as collision IDs.
        """
        restored = [list(pair) for pair in self.structural_pairs]
        self.pairs = self.model_pairs.copy()
        self.pair_ids = self.pairs[:, 0] * self.model.ngeom + self.pairs[:, 1]
        self.structural_pairs = []
        self._segment_cache.clear()
        return restored

    def _exclude_structural_overlaps(self):
        """Drop pairs that interpenetrate in every sampled configuration.

        As in MoveIt's collision-matrix setup ("always in collision"), such
        overlaps belong to the model's collision geometry (e.g. a knee shell
        reaching into the ankle), not to a motion. Samples: the default pose
        and fixed-seed configurations of every joint within its limits (free
        and ball joints at random poses). Pairs that separate in any sample
        are kept.
        """
        self.model_pairs = self.pairs.copy()
        self.structural_evidence = {}
        model = self.model
        rng = np.random.default_rng(STRUCTURAL_SEED)
        joint_types = mujoco.mjtJoint
        overlapping = None
        sampled_depths = {}
        failed_samples = []
        for sample in range(STRUCTURAL_SAMPLES + 1):
            q = model.qpos0.copy()
            for joint in range(model.njnt) if sample else ():
                address, kind = model.jnt_qposadr[joint], model.jnt_type[joint]
                if kind == joint_types.mjJNT_FREE:
                    q[address:address + 3] = rng.uniform(-1.0, 1.0, 3)
                    q[address + 3:address + 7] = _random_quaternion(rng)
                elif kind == joint_types.mjJNT_BALL:
                    q[address:address + 4] = _random_quaternion(rng)
                elif model.jnt_limited[joint]:
                    q[address] = rng.uniform(*model.jnt_range[joint])
                elif kind == joint_types.mjJNT_HINGE:
                    q[address] = rng.uniform(-np.pi, np.pi)
            try:
                rows = self.query(q, distance=0.0)
            except GeometryQueryError:
                failed_samples.append(sample)
                continue
            current = {row["pair_index"] for row in rows if row["clearance"] < 0}
            for row in rows:
                sampled_depths.setdefault(row["pair_index"], []).append(max(0., -float(row["clearance"])))
            overlapping = current if overlapping is None else overlapping & current
            if not overlapping:
                return
        # Unknown samples are not evidence of a motion-independent overlap.
        if failed_samples:
            return
        keep = np.ones(len(self.pairs), bool)
        keep[sorted(overlapping or ())] = False
        for index in sorted(overlapping or ()):
            pair = tuple(map(int, self.pairs[index]))
            self.structural_evidence[pair] = dict(
                method="default_pose_and_fixed_seed_joint_limit_samples", seed=STRUCTURAL_SEED,
                samples=STRUCTURAL_SAMPLES + 1, overlapping_samples=len(sampled_depths[index]),
                unknown_samples=0, sample_depths_m=sampled_depths[index],
                minimum_sample_depth_m=min(sampled_depths[index]),
                scope="all sampled configurations; not an analytical proof for all joint angles")
        self.structural_pairs = [
            [model.body(int(model.geom_bodyid[a])).name, model.body(int(model.geom_bodyid[b])).name]
            for a, b in self.pairs[~keep]]
        self.pairs, self.pair_ids = self.pairs[keep], self.pair_ids[keep]
