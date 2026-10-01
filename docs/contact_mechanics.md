# Independent net-wrench contact explanations

`contact_mechanics.py` is an independent source-motion checker. It does not add
contact-force variables to retargeting or claim to recover load percentages.
`centroidal_demand.py` accepts an explicitly supplied segment inertia model:
segment COM trajectories, segment rotations, masses and COM inertias. Joint
origins (including the pelvis) are not substitutes for segment COMs.

With all quantities in world axes, its demand is

\[
F=m(\ddot c-g),\qquad \tau_O=\dot L_c+(c-O)\times F.
\]

`L_c` includes both segment spin and orbital angular momentum. Local cubic fits
estimate derivatives on a copy of the observations. Disagreement between two
window lengths is a noise diagnostic, **not** a statistical confidence bound.
Impacts and sequence edges require separate interpretation; single-frame
derivative peaks are not proof of support.

Each `ContactCandidate` requires observed skin points, proposed actual surface
points, normals, friction, and an explicit allowed reconstruction discrepancy in
metres. A proposal outside that bound is geometrically rejected. Sticking uses
an inscribed friction polygon; sliding uses friction opposing the observed
material-relative tangent velocity. Patch moment capability comes from the
supplied patch points, not from an invented torsional moment.

`WrenchCone` converts wrench rays to homogeneous inequalities `A @ wrench <= 0`
using an SVD span reduction and convex-hull facets through the origin. This
handles rank-deficient contacts and the empty/flight cone. Its explicit fixed
moment origin must match the demand. Conversion is cached for exactly repeated
geometry; moving points/normals produce a different cache entry. This conversion
can be expensive. It does not remove the cost of constructing a cone.

An optional six-component uncertainty box gives conservative statuses:

* `feasible`: every wrench in the box satisfies every inequality;
* `infeasible`: at least one inequality excludes the entire box;
* `uncertain`: neither statement is proved. Facet-wise overlap alone is not
  incorrectly treated as a proof that the uncertainty box intersects the cone.

`explain_contacts` compares combinations including no contact and retains all
minimal feasible alternatives. Removing a contact changes its role to
`necessary_support_candidate` only if the complete set is feasible and its
removal is infeasible. `optional_contact` does not mean absent or unloaded.
Redundant left/right supports are therefore not both discarded. The number of
candidates is explicitly bounded because enumeration is exponential.

`contact_support.explain_source_support` carries per-frame explanations into
`retarget_motion(source_support_evidence=...)`. Candidate names map to arbitrary
source region/surface pairs: abdomen, back, forearm, knee and previously unseen
regions use exactly the same path. The union of minimal feasible alternatives
retains redundant support candidates. Optional contacts keep their interaction
targets; unexplained frames retain conservative geometric protection. Sliding
and moving-object contacts can receive support priority. No body-part allowlist
or robot-reachability evidence enters the source checker.

`contact_temporal.explain_support_timeline` retains every minimum-change path
through the inclusion-minimal feasible sets. Forward/backward dynamic programming
preserves tied support explanations and ambiguous handoff times. Mechanics is a
hard admissibility condition, so a required one-frame push or flight interval
survives without a minimum-duration threshold. Unexplained frames split the
inference rather than filling gaps with invented support. This discrete
continuity prior changes priority among feasible explanations, not their measured
physical feasibility, and introduces no per-body or per-clip tuning parameter.
It does not claim that the least-switching explanation is the true load sharing.

Necessary support proposals can include stable `ContactCandidate.sample_indices`
into the original whole-body source skin. `support_observation` checks these
identities against the immutable observed points before activating the matching
surface tasks. The reconstruction bound has already been checked by mechanics;
the actual scene points become the robot targets while the raw skin vectors are
saved separately. Optional explanations alone do not create a gap contact.
Missing sample correspondence is explicitly reported, and conflicting targets
for the same source sample are rejected. Candidate geometry is copied when
creating the evidence, so later caller mutations cannot change its targets.

Shared whole-body temporal penalties and nonlinear contact guards constrain
robot adjustments separately from the source hypothesis sequence.

`object_control.observe_object_control` separately accepts source material points,
actual surface points, controller identities, object transforms and an explicit
metric observation uncertainty. Coherent motion of two independent controllers
with the same object can preserve the interaction even when human support is
already explained by other contacts. It never upgrades mere proximity, stationary
hands, a one-frame glimpse, a gap beyond the error budget, or relative sliding.
Stationary holding remains undecided; co-motion does not prove causal control or
force closure. Use `retarget_motion(source_control_evidence=...)` to carry this
evidence into the same role table. Support always wins over control. Unexplained
geometric holds remain conservatively protected until mechanics can explain them.

The TRUMANS review does not supply a
validated human segment inertia model, so its contact labels are geometric
observations, not claimed force-feasibility results. Robot torque/actuator limits
are also outside this checker. Source explanations and robot reachability remain
separate questions.

The formulation follows [Caron et al., gravito-inertial wrench cones](https://scaron.info/papers/journal/caron-tro-2016.pdf)
and the [wrench friction cone derivation](https://scaron.info/robotics/wrench-friction-cones.html).
Tests compare the inequality representation against an independent ray-based
linear feasibility program, plus flight, redundant supports, directed sliding,
coordinate changes, segment spin, geometric gaps and uncertainty cases.
