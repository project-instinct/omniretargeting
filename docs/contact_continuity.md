# Contact continuity and surface distance

The source trajectory, original object transforms, scene scale, and interaction
locations remain inputs. The implementation has no clip names, sitting frame
ranges, hand-specific reach translations, or piecewise root settings.

`contact_correspondence.coherent_sample_indices` finds a minimum-cost path through
actual skin-to-scene projections. It balances squared gap against changes in
projected position after subtracting the skin sample's own motion. The latter
penalizes changing material identity solely because of nearest-point rank noise;
it does not penalize the source's deliberate fast slide. Both costs have squared
metre units. `contact_sample_switch_weight` is an exposed dimensionless ratio
(default 1, equal positional uncertainty; 0 recovers independent nearest points).
Candidates remain bounded by the existing contact-distance configuration.
Moving objects are compared in their original material coordinates. No source
point or target is interpolated off the actual surface by this correspondence.

Robot-side contact offsets are held consistently within a contact run. Approach
and departure interpolate both the world target and the robot material point;
otherwise a continuous target can still induce an instantaneous change from a
hand centre to a fingertip. `slice_contact_task` retains the full-timeline approach
references when a caller crops a window, even when its contact lies beyond the
window. Full-clip review solves one trajectory instead of joining independently
configured root trajectories.

Sliding still follows a time-varying scene point. Its default tangent precision
now matches normal precision (`contact_tangent_sigma_m=0.0015`); constraining only
the infinite surface plane can leave a hand far outside a finite chair patch.
The root remains a regularized optimization variable, with the existing source
path offset and time-normalized derivative penalties. Root and joint smoothness
are solved together with contacts, rather than applied as a final unconstrained
filter. All three root coordinates and orientation have temporal penalties from
the first iteration, including the initial pose phase. Absolute smoothness is
not promised for incompatible observations.
During source-pose acquisition, joint derivatives help suppress solver chatter.
After that reference is fixed, joint derivative penalties apply to corrections
from it, so source-shaped fast motion is not itself penalized as chatter.
Fast source motion can still be reduced while acquiring that reference or by
robot reachability and competing contact constraints; compare actual motion
against the unmodified source before accepting a high-dynamic clip.

The source root denotes its mapped robot body, which may differ from the model's
floating body. Cold initialization aligns that body's FK origin to the immutable
source point; position tracking and free-target Jacobians use the same origin.
A chest-rooted model therefore does not place its chest at the human pelvis.

The core first fits a kinematic source reference without scene or self-collision
objectives. In the fast backend, this fit uses the adapter's permanent world
tracking semantics and morphology-relative free targets. Temporary contact
roles and moving-object anchors do not determine the reference height. Source
positions, scene registration and object transforms remain unchanged.
Calibrated source body rotations, when available, initialize a
continuous scalar-joint IK branch before this coupled solve. Previous-frame
initialization disambiguates axial twist; it does not filter or retime motion.
`SourceShapeGuard` fits source axes before endpoint costs, allowing
the existing uniform angular envelope for secondary temporal/endpoint fitting.
Source-reference line search first reduces a problematic frame's step locally;
one frame at a curved angular boundary must not strand the rest of the clip.
The full temporal objective and per-frame source priority are still checked
afterwards. This does not relax the fixed bounds used after reference acquisition.
An imperfect source fit is reported without aborting the pipeline. Subsequent
optimization continues inside the fixed reference's cumulative angle envelope;
out-of-bounds proposals backtrack. Neither the reference nor its budget is moved
to make an incompatible contact pass. These relative bounds cannot by themselves
certify that an initially poor reference resembles the source.
Source observations, scene registration and physical scale remain unchanged.

The fast solver freezes normal-error allowances for evidenced support and
unexplained holding contacts before contact acquisition. A reduced full-vector
error cannot justify increased separation, including for sticking contacts.
The same source-reference joint and root-correction continuity bounds remain
active through acquisition and repair. Executed root translation and world
angular increments also keep each coordinate's first, second and third
difference bounds; a lower aggregate RMS cannot hide one noisy axis. Banded directions that stall against
these bounds use constrained cones and minimum-change nonlinear restoration;
the original reference and allowances do not move. This still does not certify
initial source fitting or absolute root derivatives: the independent final
`budgets`, initialization audit and `skipped_tasks` must be reviewed.

Dense surface coverage is checked per frame, source region, welded robot body,
environment identity and contact mode. If retained samples cannot represent an
observed endpoint contact, its primary constraint stays active. Source contact
timing and world targets are unchanged, including through exported replay data.

When geometry is present, the core then acquires world contacts with continuous
root motion directly from that reference. It no longer inserts a relaxed
clearance pass with contact roles removed: that pass could detach a foot and
spend the fixed posture budget before contact acquisition started.
That trajectory supplies immutable endpoint error budgets to the
geometry/posture stage (`endpoint_regression_m`, default 1 mm). These are conic
constraints with nonlinear FK acceptance checks, not another weighted cost that
can trade hand detachment for reduced penetration. Unresolved intersections are
reported when the geometry and protected endpoints are incompatible.

Contact priorities apply independently in each frame in both the conic solve
and nonlinear acceptance. Scene clearance remains a secondary cost, not an
extra contact role that forbids a continuous correction because a neighbouring
free frame has slightly more penetration. Backtracking scales both required
priority progress and allowed regression by the step fraction, so small feasible
steps are not erased by a fixed progress deadband.

At curved posture/continuity boundaries, a minimum-change conic restoration
tries to bring the proposal into the unchanged nonlinear feasible set before
shrinking further. It retains the trial's contact-priority budgets, the original
step box, fixed reference and exact nonlinear acceptance checks. Its bounded
numerical retries introduce no new physical/user tuning parameter.

The conic subproblems batch local norm constraints into sparse matrix maps,
including adjacent-frame slip bounds, and retain the banded least-squares
quadratic without adding one auxiliary variable per residual. Each sequence
endpoint cone is expressed in units of its own budget; this exact scaling does
not relax the tolerance. The QDLDL sparse factorization avoids machine-dependent
automatic multithreaded solver selection. Nonlinear FK rejects contact-invalid
trial steps before the expensive original-mesh query, with identical contact
acceptance conditions. Dense independent conic comparisons cover multiple coordinate scales.

The source-shaped reference freezes posture and temporal budgets across contact
acquisition and later stages, separately from attained endpoint budgets. The geometry
stage may not increase any scalar joint's per-frame first, second, or third
differences; improvements in one joint cannot compensate chatter in another.
It also preserves the per-frame norms of first/second
differences of root position and quaternion corrections relative to the source.
These bounds are measured from the acquired trajectory; the only additive slack is
roundoff tolerance. Sparse cones couple neighboring frames, and actual nonlinear
quaternion/FK checks accept the update. Thus collision reduction cannot purchase
new chatter by outweighing a smoothness penalty. If these constraints prevent a
repair, the intersection remains reported. Source timing and rapid source turns
are preserved; this guard never filters the output.

`SequencePoseGuard` bounds only the root and slide-joint translation relative to
the acquired robot reference, using the sequence's metric trust budget. Scalar
joint changes, whole-body orientation changes and source bone directions have no
angular limit; they are reported against the fixed reference and the immutable
source, which is never rebased, so repeated passes cannot hide an accumulated twist.
The final review also checks the source skeleton and reports posture fidelity.

For large sparse trajectories, `sequence_solver.linear_solver="sparse_qp"`
optionally solves the same bounded linear least-squares step through a diagonally
equilibrated quadratic program. Its tolerances and iteration limit are exposed
as `linear_solver_tolerance` and `linear_solver_iterations`. It accepts only a
converged QP and otherwise falls back to the existing rectangular LSMR solver;
the nonlinear objective and backtracking remain identical. Tests compare it to
independent dense bounded least squares over differently scaled coordinates.
The default remains `lsmr`. See the [OSQP Python interface](https://osqp.org/docs/interfaces/python.html).

`_sequence.proximity.SurfaceProximityTasks` transfers the closest observed
skin-to-scene normal gap to the corresponding **collision geometry's extent**.
A positive observed gap remains a gap; it does not become a support label. This
prevents a learned point on the far side of a pelvis primitive from representing
the contact-facing body surface. Every observed tangent plane compares the
source skin's support extent with the robot's support extent. This retains
simultaneous directions such as a seat and its back, without making a far-side
skin sample the closest-surface target. Rows are weighted smoothly by the
existing observation distance, sigma and weight, and normalized by the sample
count and the number of geometries in the anatomical part.
Duplicating a sample does not strengthen its objective. The model's own sphere,
box, capsule, cylinder, ellipsoid or compiled mesh defines the extent.

Prescribed moving meshes now participate in the sequence solver's clearance
objective as well as in contact detection. They are not moved to help the robot.
`scene_enclosed_free_space` explicitly distinguishes a room's navigable interior
from a solid obstacle. An oriented closed room and its oppositely oriented
interior obstacle shells form one winding boundary. Contact normals use that
free side even when noisy source skin crosses the boundary. Derived normals
come from triangle winding, independently of stale file/mesh normal caches.

Default containment uses winding numbers, with exact winding resolving ambiguous
fast-winding results. Optional `scene_closed_sign_method="pseudonormal"` caches
angle-weighted feature normals for explicitly watertight, consistently oriented,
non-self-intersecting boundaries. Open or self-intersecting scans do not satisfy
that contract. The review separately audits original triangles with winding
containment. See the [libigl signed-distance documentation](https://libigl.github.io/tutorial/)
for the geometric assumptions. Unreliable collision queries reject trial steps;
they are never silently treated as free space.

Winding-based sequence queries batch static collision shells across frames in
`_sequence/scene_cache.py`. Every original point is retained and results are
cached only for the exact frame pose; changed poses invalidate their entries.
The point-count limit bounds temporary memory, not distance accuracy or sampling
density. This amortizes libigl's per-call winding preprocessing on large rooms.
Unlimited offline trajectory validation uses `scene_probe_batch.py` to batch the
same centre-culling and sample queries, including interpolated poses. Original
narrow-phase witnesses are retained. Explicit query-budget runs keep sequential
accounting. Tests compare all fields and query-point counts with individual probes.
`_sequence/moving_cache.py` applies the same batching to original moving-object
components in their material coordinates, retaining all witnesses, original
object poses and scales. Reverse collision queries and Jacobians remain exact
for their sampled geometry; pose changes invalidate cached results.
The cross-clip review falls back from pseudonormals to winding when a mesh has
degenerate triangles, keeping the original mesh and recording the fallback.

Validation keeps endpoint world errors, contact vector/normal errors, root
offset derivatives, root orientation derivatives, joint derivatives, and scene
penetration separate. Sampled penetration is a lower bound, not a complete
continuous-time collision-free certificate. Generalization tests cover different
models/frame rates, rapid slides, rigid coordinates, sample duplication, window
boundaries, primitive shapes and geometry conventions. One real clip alone does
not establish broad dataset generalization.

For a reproducible cold-start cross-clip review, run
`scripts/trumans_review_inputs.py` without `--baseline`, then
`scripts/trumans_crossclip_review.py --prepared <clip>/prepared.npy`.
This captures the model reference initialization and saves each stage separately.
`scripts/review_crossclip_summary.py --root <experiment>` requires every selected
clip to finish, verifies identical configuration hashes, and reports contact,
pose, continuity and original-mesh penetration independently. The selection
manifest is fixed before solving; action labels are used only in the review,
never by the optimizer. `scripts/review_render.py` and the recording-only mode
of `scripts/review_webviser.py` visualize the saved motion in its original scene.
