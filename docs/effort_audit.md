# Joint effort audit

`scripts/review_joint_smoothness.py` uses the saved production MJCF for mass,
inertia, armature and passive forces. `--limits-urdf` selects the original URDF
velocity/effort limits. G1 mesh MJCFs default to the adjacent
`g1_29dof_popsicle.urdf` **for limits only**. Both hashes and missing limits are
reported, including the separate compiled collision semantic hash; a saved
collision-model semantic hash mismatch or MJCF/URDF effort mismatch
is an error. A missing limit cannot pass. Motor-limit tolerances are numerical
roundoff (1e-6 in the relevant units), independent of frame rate.

The old robot-only `mj_inverse` screen included soft self-contact and joint-limit
forces, but no ground/seat reactions. Its gravity load on the torso-rooted waist
and artificial separation loads were not actuator-effort measurements. See
[MuJoCo's equations of motion](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework).

The current instantaneous test computes
`d = M*qacc + bias - passive + sign(qvel)*frictionloss` and solves
`d = S.T*tau + J.T*f`. Free/ball coordinates have no fictitious actuation. The
contact forces must exactly balance their generalized demand; a least-squares
residual cannot be credited to an invisible root motor. The force split
minimizes the largest `abs(tau)/URDF_effort`, since an arbitrary minimum-norm
split exceeding a limit does not prove that every possible split exceeds it.
Reported joint peaks belong to an instantaneous minimax witness, not measured
or uniquely determined physical forces. Overload proves at least one motor
must exceed its limit on that frame; the overloaded joint assignment need not
be unique.

The audit queries every active robot collision geom against the original static
and moving triangles. The 20 mm collision shell includes all convex-hull
vertices, saved material points, and exact support extrema toward source and
nearest-face normals. Only points within 5 mm of an actual finite triangle can
carry a reaction. Each original normal/plane is tested separately: clipping a
wall does not erase a sole's floor reaction. Exact convex support extents reject
deep plane intrusion only when their projections lie on the finite original
surface; a thin slab's opposite face cannot hide the depth or create support
from the shell's crossing ring. Like the scene audit, sampled shell
coverage is an explicit model assumption, not an exact collision certificate.
Original face normals use the surface's known free side; open/unoriented faces
need nearby, aligned source evidence or remain unknown. Tangential drift from a
source correspondence is a separate contact-quality defect and does not make
a real floor contact disappear. Source support labels diagnose lost contacts,
but do not prescribe forces or exclude other geometrically attained contacts.
Only static geometry or objects proved still and grounded supply world reactions.
The existing stillness/transitive-grounding algorithm uses the audit's 5 mm
distance, rather than source detection's wider evidence radius; the separate
15 mm scenario repeats grounding at 15 mm. Pass `--prepared` and `--objects` to test
object grounding against original geometry. Contacts on other moving objects
have unknown loads: no object mass is inferred from an arbitrary visual mesh.

Friction is an explicit assumed coefficient (0.6), not measured material data.
Dense patches use an exact generator reduction. For the same body and force
ray, `J(p).T*f` is affine in `p cross f`; every interior moment is a convex
combination of extreme moments. Keeping their columns preserves the complete
nonnegative force cone and minimax optimum. Different normals are never merged,
degenerate hulls retain all columns, and coefficients expand back to the
original points for inspection. This changes no geometry gate or effort limit.
Eight-ray inner and outer polygons bracket the circular Coulomb cone. Inner
feasibility supplies a valid force witness and an outer optimum above the
limits proves overload on attained contacts. Polygon ambiguity is resolved by
a circular-cone SOCP, checking its primal balance/cone residual and dual lower
bound. Granting every source correspondence remains an optimistic diagnostic
field, separate from the primary actual-geometry classification.

An uncertified circular primal solve also tries a six-variable base separator.
Every contact must satisfy the exact dual-cone inequality
`g_normal >= mu*norm(g_tangent)`, and `separator*demand` must remain negative
even after the allowed 1e-6 base residual. Only that checked certificate proves
unsupported motion; failed numerical optimization alone does not. A positive
search margin strengthens certificate verification without loosening the
geometry gate, friction, base balance or motor limits.

`feasible` and `overload` concern joint effort with exact floating-base balance.
`infeasible_unsupported` is a definite failure within the stated geometry and
friction model: either no attained reaction exists, or the attained support
cones cannot balance the required base wrench. Its `reason`, actual pair gaps
and closest-cone wrench residual distinguish those cases. Residuals never count
as fictitious actuation. A ballistic unsupported pose can still be feasible if
its free-base demand vanishes. `unknown_object_load` requires an actual contact
with a nonfixed object whose mass/inertia are missing; a source-only object
label is insufficient. Missing triangle geometry, unresolved near-contact
normals and missing limits retain distinct unknown reasons. Unresolved numerical
certificates are separate `audit_error_*` checker failures, not missing-data
unknowns. None of these failures, unknowns or unaudited endpoints pass
`all_individual_budgets_pass`; physical replay is still separate acceptance.

`tolerance_15mm` repeats the force test with an explicitly relaxed ±15 mm
geometry gate, including shallow penetration. It is sensitivity analysis only,
never a threshold change or a pass. `original_triangle_pairs` records the body,
surface, signed gap, correspondence error and source support evidence.
`reference_frame_diagnostics` allows inherited/unreached support to be separated
from support attained in the fixed reference and lost in the final output.

Velocity and acceleration use raw centered three-frame differences in the
middle frame's quaternion tangent space. No filtering removes spikes to make
them pass. The first/last frames lack centered acceleration evidence. A
bounded scalar position error `epsilon` can produce `4*epsilon*fps**2`
acceleration error; the audit reports this sensitivity without assuming a
measured source noise level.

The fixed reference is a robot fit to the source, not the human's physical
effort. The audit also substitutes its derivatives at the final configuration
and contacts to isolate added inertial demand. A final overload that disappears
in that feasible counterfactual is attributed to added derivatives. If the
complete reference state is feasible but its derivatives overload at the final
pose and actual contacts, the audit identifies added pose/contact geometry.
If the reference only admits inner-cone source-contact forces but its derivatives overload
at the final pose, added pose/contact leverage is implicated, conditional on
the source-contact assumption. The other cases remain inherited source-fit or
geometry-confounded. Overloaded frames also test outer cones at μ=0.3 and 0.9.
A human-source impossibility claim
requires reliable source inertials and force/acceleration evidence.

`scripts/benchmark_fast.py` passes the saved preparation and objects to the
audit. Audits remain after solving; this change adds no production solver
constraint, smoothing, retiming or repair stage.

Frozen-output reproduction (all artifacts stay ignored under `out/`):

```sh
flock /tmp/omniretargeting_heavy.lock env \
  PYTHONPATH=/home/duanxin/Xyk/omni_opt_effort \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/review_effort_baseline.py \
  --baseline /home/duanxin/Xyk/omni_fix_self/out/baseline_1d4e2de \
  --output out/opt_effort/contact_followup/recomputed
```

`scripts/review_effort_object_support.py` explains actual nonfixed-object
contacts using the original stillness and grounding evidence; run it under the
same lock with `--baseline`, `--recomputed`, and `--output`. The JSON records
moving versus still-but-ungrounded frames, representative vertex speed and
distance, and the separate 15 mm grounding outcome. Vertex distance is evidence
for that grounding method, not an exact solid-contact or dynamics certificate.
`scripts/review_effort_support_report.py` renders per-clip classifications, old
unknown migrations, representative failures and a support-loss CSV from the
old and new audits. Its `--previous` points to the preserved preceding audit,
`--recomputed` to the new directory, and `--report` to the ignored Markdown
artifact. A checked base separator can also refine older numerical rows with
`scripts/refine_effort_certificates.py`; existing feasible/overload witnesses
and motions are preserved.
`scripts/review_effort_unknown_geometry.py` inspects representative unknown
frames under the lock and records the body/surface, unsigned gap, distance and
alignment of the nearest source direction evidence, and mesh/free-side flags.
Geometry and object-load deficiencies may overlap; primary status counts are
mutually exclusive, while diagnostic cause counts must not be added together.
