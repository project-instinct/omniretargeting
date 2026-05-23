# Source Configuration Templates

This directory contains template YAML files for different motion data sources supported by OmniRetargeting.

## Overview

Instead of passing multiple command-line arguments, you create a YAML configuration file that specifies:
- The data source type (OMOMO, SMPL-X, etc.)
- Path to the motion file
- All source-specific parameters

## Available Templates

### 1. OMOMO (Object Manipulation)
**File:** `omomo_template.yaml`

For OMOMO dataset sequences with object manipulation. Includes object mesh loading and 6D pose tracking.

**Key parameters:**
- `type: omomo`
- `motion`: Path to .p sequence file
- `sequence_index`: Which sequence to load
- `data_root`: OMOMO dataset root directory
- `n_object_samples`: Number of points to sample from object mesh

**Example:**
```yaml
type: omomo
motion: /localhdd/Datasets/OMOMO/data/test_diffusion_manip_seq_joints24.p
sequence_index: 318
data_root: /localhdd/Datasets/OMOMO
n_object_samples: 100
```

### 2. SMPL-X (Parametric Body Model)
**File:** `smplx_template.yaml`

For SMPL-X motion data with 22 body joints. Supports both raw parameters and processed joint positions.

**Key parameters:**
- `type: smplx`
- `motion`: Path to .npz or .npy file
- `model_directory`: Path to SMPL-X model files
- `gender`: "neutral", "male", or "female"
- `betas`: Shape parameters (10 values)

**Example:**
```yaml
type: smplx
motion: /path/to/motion.npz
model_directory: /localhdd/Datasets/smplx
gender: neutral
betas: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

## Usage

### Basic Usage
```bash
python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --source-config config_templates/omomo_template.yaml \
  --output output.npz
```

### With Video and Scaled Outputs
```bash
python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --source-config my_config.yaml \
  --output output.npz \
  --save-video output.mp4 \
  --output-scaled-terrain scaled_terrain.obj \
  --scaled-objects scaled_objects/
```

### Output Options

- `--output`: Required. Path to save retargeted motion (.npz)
- `--save-video`: Optional. Path to save visualization video (.mp4)
- `--output-scaled-terrain`: Optional. Path to save scaled terrain mesh (.obj)
- `--scaled-objects`: Optional. Directory to save scaled object meshes and pose trajectories

When `--scaled-objects` is specified, the following files are created:
- `{object_name}.obj`: Scaled object mesh
- `{object_name}_poses.json`: Per-frame object poses (translation, rotation, scale)

## Creating Your Own Config

1. Copy the appropriate template file
2. Update the paths and parameters for your data
3. Save with a descriptive name (e.g., `my_floorlamp_motion.yaml`)
4. Use with `--source-config` flag

**Example workflow:**
```bash
# Copy template
cp config_templates/omomo_template.yaml my_motion.yaml

# Edit with your paths
vim my_motion.yaml

# Run retargeting
python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --source-config my_motion.yaml \
  --output results/output.npz \
  --save-video results/output.mp4 \
  --scaled-objects results/objects/
```

## Common Parameters

All source configs support these optional parameters:

- `framerate`: Override motion framerate (default: auto-detected or 30.0)

## Adding New Data Sources

To add support for a new data source:

1. Create a new adapter in `omniretargeting/data_sources/`
2. Register it with `register_data_source()`
3. Create a template YAML in this directory
4. Document the parameters in the template file

See `omniretargeting/data_sources/omomo.py` for a reference implementation.

## Notes

- YAML files use standard YAML syntax
- Paths can be absolute or relative to the current working directory
- Comments start with `#`
- Lists use YAML array syntax: `[item1, item2]` or multi-line format
- All templates include detailed inline documentation
- `--source-config` is the recommended interface for new workflows
- Legacy flags such as `--motion`, `--model-dir`, and `--source-options` still work, but they are deprecated
