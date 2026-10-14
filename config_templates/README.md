# Source Configuration Templates

A source config is a small YAML/JSON file that names one motion and the dataset
facts needed to load it. Solver policy is not configured here: it comes from the
shared defaults (`omniretargeting/retargeting_defaults.py`) and the robot profile.

## Quick start

```bash
# Solve with the production fast solver and run the standard audits once.
.venv/bin/python scripts/retarget_and_audit.py \
  --source config_templates/omomo_template.yaml --out /path/to/run_dir

# Solve only.
.venv/bin/python -m omniretargeting.main \
  --source-config config_templates/omomo_template.yaml --output output.npz
```

Both entry points default to `robot_models/unitree_g1/unitree_g1_mesh.json`
(the collision-model contract) and the `fast` solver. Common motion files need
no config at all: pass `--source-motion clip.pkl` (or `--source clip.npz` to the
one-command entry); `.pkl` is read as `parc_ms` and `.npz` as `smplx`. Use
`--source-type` for ambiguous formats such as BVH and `--smplx-model-dir` for
raw SMPL-X.

## Keys

| Key | Adapters | Meaning |
| - | - | - |
| `type` | all (required) | adapter name: `smplx`, `omomo`, `trumans`, `parc_ms`, `lafan1`, `nokov`, `bones_seed` |
| `motion` | all (required) | motion file (TRUMANS: the `Data_release` directory) |
| `terrain` | all | static scene mesh; flat ground when omitted |
| `model_directory` | `smplx`, `omomo`, `trumans` | SMPL-X model files (`SMPLX_{gender}.pkl`) |
| `gender`, `betas` | `smplx` | body model gender (default `neutral`) and shape (default: from the motion file) |
| `target_names_override` | `smplx`, `omomo` | source joint names when the file uses a non-default skeleton |
| `sequence_index`, `n_object_samples` | `omomo` | sequence in the `.p` file; object surface samples |
| `framerate` | `omomo`, `trumans` | dataset frame rate (default 30); other adapters read it from the file |
| `data_root` | `omomo`, `trumans` | dataset root (OMOMO default: two levels above `motion`) |
| `segment`, `frame_start`, `frame_stop` | `trumans` | recording segment and frame window |
| `start_frame` | `lafan1`, `nokov`, `bones_seed` | skip leading BVH frames |
| `source_denoise` | `smplx`, `omomo`, `trumans` | `off` (default) or `auto`, same as `--source-denoise` |
| `joint_mapping`, `base_orientation` | all | per-source override of the robot profile |

Relative paths (`motion`, `terrain`, `model_directory`, `data_root`) are
resolved against the directory of the config file; `~` and `${VAR}` are
expanded.

## Templates

| File | Source |
| - | - |
| `smplx_template.yaml` | SMPL-X / AMASS `.npz` (raw parameters or joint positions) |
| `omomo_template.yaml`, `omomo_floorlamp_example.yaml` | OMOMO human-object sequences |
| `lafan1_template.yaml` | LAFAN1 BVH (use `robot_models/unitree_g1/unitree_g1_lafan1.json`) |
| `nokov_template.yaml` | Nokov BVH (pass `--mapping` for its bone names) |
| `bones_seed_template.yaml` | Bones-Seed SOMA BVH |

## Outputs

- `--output`: retargeted motion (`*_retargeted.npz`, with contact tasks and
  `collision_model_sha256`).
- `--save-video`: optional offscreen render (`MUJOCO_GL=egl` when headless).
- The one-command entry also writes the audit inputs and reports into `--out`.

Scenes and objects keep `scale=1.0`. `--output-scaled-terrain` enables terrain
scaling and is only for explicit experiments.

## Adding a data source

1. Add an adapter in `omniretargeting/data_sources/` and register it with
   `register_data_source()`.
2. Read its keys from the source config (see `create_omomo_data_source`).
3. Add a template here and a row to the table above.
