# Contact/task output contract (v1)

The CLI adds the `contact_task_*` arrays below to its existing NPZ. `qpos`
continues to contain the complete `q_ref(t)` and every existing output key
keeps its meaning. `retarget_motion()` still returns `(terrain_scale, qpos)`.
Python callers request `retargeter.get_contact_task_output()` after solving;
the returned `ContactTaskOutput` supplies `export()` and `frame(t)`.

All scalar task arrays have shape `(T, K)`; vectors have `(T, K, 3)`.
`T` is exactly the saved qpos timeline. `K` is the Cartesian product of the
candidate body parts and surface registry, sorted by body-part name, then
surface ID. Slots exist throughout the clip, including inactive frames.
Separate slots preserve simultaneous contacts with different objects.

| NPZ key (`contact_task_` prefix) | Shape | Meaning |
| --- | --- | --- |
| `schema_version` | scalar | Integer `1`; reject unsupported versions. |
| `body_parts` | `(K,)` | Candidate names; repeats once per surface. |
| `slot_scene_object_ids` | `(K,)` | Permanent slot association with the surface registry. |
| `scene_object_names` | `(S,)` | Names indexed by surface ID; static mesh is `scene`, followed by supplied objects. Without a static mesh, objects start at zero. |
| `timestamps_s` | `(T,)` | Clip-relative seconds, `arange(T) / framerate`. |
| `observed` | `(T, K)` | At least one valid material observation for this part/surface in this frame. |
| `active` | `(T, K)` | Source contact is temporally accepted and has valid target geometry. |
| `contact_types` | `(T, K)` | `none`, `support`, `control`, `touch`, `slide`. Support/control take precedence over motion type. |
| `contact_modes` | `(T, K)` | `none`, `touch`, `sticking`, `sliding`, `unknown`. Orthogonal to role: a support may slide. |
| `roles` | `(T, K)` | `free`, `support`, `hold`, `control`, `follow`; follows existing evidence priority. |
| `importance_weights` | `(T, K)` | Advisory downstream weight in `[0, 1]`, independent of confidence and solver costs. Defaults: support `1`, hold `.75`, control `.5`, follow `.25`, inactive `0`. |
| `confidence` | `(T, K)` | Existing geometric confidence in `[0, 1]`; not a force/support probability. |
| `target_positions_w` | `(T, K, 3)` | Metres in world coordinates. Representative source projection onto the observed surface; endpoint-only fallback uses its existing contact anchor. |
| `surface_normals_w` | `(T, K, 3)` | Unit free-side surface direction in world coordinates. |
| `scene_object_ids` | `(T, K)` | Active surface ID; `-1` when inactive. IDs index this contract's registry, not the legacy `scene_object_names` array. |
| `source_sample_indices` | `(T, K)` | Representative source material index; `-1` for endpoint-only/empty slots. |

Inactive tasks use type/mode `none`, role `free`, zero confidence/weight,
`-1` object/sample IDs, and NaN target/normal vectors. `frame(t)` represents
missing geometry and object names as `None`, so records are strict JSON-ready.
`observed=False` means unknown, not verified absence of contact. A far-away
valid observation can have `observed=True, active=False`. The detector only
stores the nearest surface per sample; unobserved slots do not imply clearance
from all other objects. No geometry/evidence means inactive unknown slots,
never contact guessed from joint centres.

Every observed source region becomes a candidate without a hand/foot
whitelist. The defaults also reserve `left_hand`, `right_hand`, `left_foot`,
`right_foot`, `left_knee`, `right_knee`, `left_forearm`, `right_forearm`,
`pelvis`, and `abdomen`. Profiles and material markers may supply additional
names. Missing parts remain unknown. SMPL-X's existing coarse region labels
(`left_shin`, `lower_torso`, etc.) are preserved; a knee is not inferred from
every shin sample. Provide labelled material samples for precise knee/abdomen
contacts, or explicitly declare a region grouping when that approximation is
appropriate:

```json
{
  "retargeting": {
    "contact_task_output": {
      "candidate_parts": ["left_knee", "right_knee", "abdomen"],
      "body_part_regions": {"abdomen": ["lower_torso"]},
      "importance_weights": {"follow": 0.2}
    }
  }
}
```

`candidate_parts` replaces the reserved defaults; observed regions, declared
effectors, markers and group names are still included. Groups only read the
declared source labels and never change solver segmentation. A profile's
explicit `source_contact_part` provides a default group for that effector.

For each part/surface pair, the highest-priority active role selects a
representative; within that role, highest confidence wins, with ties broken
by sample index. Targets/normals come from that actual sample, avoiding
averages inside an object. They are semantic source goals, not certificates
that the final robot has achieved contact. Existing surface/endpoint arrays
still describe the solver's detailed material anchors and correspondences.

Support is emitted only with existing mechanical evidence, control only with
existing object-motion evidence. Unexplained mechanics retain `hold`; a
sticking contact without mechanics is a geometric hold, not support. These
weights do not change solver priorities or introduce a new optimizer.

The getter reuses the source table when available. If contact solving was
disabled, it can observe stored adapter material samples in the export layer;
it never attaches that table to the solver or modifies source/object motion.
Use `get_contact_task_output(source_markers=...)` to supply fallback material
markers when no source table or dense adapter samples exist. Arrays are
owned, read-only numeric/string buffers and
NPZ is readable with `np.load(path, allow_pickle=False)`.

```python
with np.load("motion.npz", allow_pickle=False) as motion:
    assert int(motion["contact_task_schema_version"]) == 1
    q_ref = motion["qpos"]
    active = motion["contact_task_active"][t]
    parts = motion["contact_task_body_parts"][active]
    targets = motion["contact_task_target_positions_w"][t, active]
```
