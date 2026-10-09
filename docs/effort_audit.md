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

Contact candidates are immutable source-observed dense collision-geom material
correspondences, including optional geometric contacts. A source `support`
label alone neither proves that the robot attained the patch nor fixes its force.
The attained set requires the full distance to the original projected surface
point to be at most 5 mm. This conservatively excludes correspondence drift;
failure can mean missing contact evidence and is not a torque violation.
It does not extend a finite source surface into an infinite support plane.
Only static geometry or objects proved still and grounded by the existing source
support rule supply world reactions. Pass `--prepared` and `--objects` to test
object grounding against original geometry. Contacts on other moving objects
have unknown loads: no object mass is inferred from an arbitrary visual mesh.

Friction is an explicit assumed coefficient (0.6), not measured material data.
Eight-ray inner and outer polygons bracket the circular Coulomb cone. Inner
feasibility supplies a valid force witness. Before reporting unavoidable
overload, the outer polygon and every source-correspondence patch are granted,
including unattained ones. Otherwise the frame is unknown. This optimistic
diagnostic cannot certify contact, support, or physical execution. Unknown
object loads, missing geometry/limits, infeasible base balance and unaudited
endpoints are not passes. `all_individual_budgets_pass` remains false when any
of these are present; physical replay remains a separate acceptance step.

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
reference admits inner-cone source-contact forces but its derivatives overload
at the final pose, added pose/contact leverage is implicated, conditional on
the source-contact assumption. The other cases remain inherited source-fit or
geometry-confounded. Overloaded frames also test outer cones at μ=0.3 and 0.9.
A human-source impossibility claim
requires reliable source inertials and force/acceleration evidence.

`scripts/benchmark_fast.py` passes the saved preparation and objects to the
audit. Audits remain after solving; this change adds no production solver
constraint, smoothing, retiming or repair stage.
