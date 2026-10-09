# Documentation

These documents are working documentation, not an archive. They describe the
current contract and the non-obvious reason behind it; code and tests remain
the source of truth for details.

- [`retargeting_invariants.md`](retargeting_invariants.md) — current solver
  legacy production mechanisms, mesh collision contract, deadline, and pair/reset audits.
- [`code_layout.md`](code_layout.md) — where entry points and responsibilities
  live.
- [`ADDING_SOURCE_ADAPTERS.md`](ADDING_SOURCE_ADAPTERS.md) — source-neutral
  loader contract and registration.
- [`smplx_frame_ids.md`](smplx_frame_ids.md) — SMPL-X frame numbering modes.
- [`contact_mechanics.md`](contact_mechanics.md) — contact evidence and
  uncertainty semantics.
- [`contact_task_output.md`](contact_task_output.md) — versioned per-frame
  body-part tasks alongside complete qpos, including inactive/unknown slots.
- [`fast_speed.md`](fast_speed.md) — where fast-solver wall time goes, the
  exact accelerations available, and directions already measured to fail.

When changing documentation:

Before adding a sentence, ask whether its absence would cause an agent or
contributor to make a wrong decision; if not, omit it.

1. Keep a page below 200 lines and delete anything an agent can safely learn
   from code or tests.
2. Do not add dated experiment logs, finished implementation plans, paper
   summaries, changelogs, or copies of report output. Git history is the
   archive.
3. Describe current behavior and its failure modes; do not present an
   implementation detail as a permanent requirement.
4. Replace or delete adjacent documentation in the same change that alters the
   behavior it describes.
5. Prefer one authoritative page over parallel histories of the same mechanism.
