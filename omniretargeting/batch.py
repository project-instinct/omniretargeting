"""
Batch processing script for OmniRetargeting.

Scans a folder of motion files, writes per-motion source configs,
and executes main.py for each file with optional fixed scaling and video export.

Usage:
    python -m omniretargeting.batch \
        --source-folder /path/to/motions \
        --source-type smplx \
        --robot-config robot_models/unitree_g1/unitree_g1.json \
        --output-dir /tmp/batch_output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omniretargeting.utils.batch_processing import (
    _output_exists,
    _summarize,
    detect_resources,
    export_shared_scaled_terrain,
    get_activation_prefix,
    log_repo_status,
    process_batch,
    scan_source_folder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniRetargeting Batch Processor")
    parser.add_argument("--source-folder", required=True,
                        help="Folder containing motion files to process")
    parser.add_argument("--source-type", required=True,
                        help="Type of motion data source (e.g. smplx, lafan1, nokov, omomo, bones_seed)")
    parser.add_argument("--robot-config", required=True,
                        help="Path to robot configuration JSON file")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save batch outputs")
    parser.add_argument("--terrain", default=None,
                        help="Path to terrain mesh file applied to all motions")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Maximum parallel workers (default: auto-detect)")
    parser.add_argument("--framerate", type=float, default=None,
                        help="Override framerate for all motions")
    parser.add_argument("--source-options", default=None,
                        help="JSON string of extra source options for all motions")
    parser.add_argument("--skip-test-job", action="store_true",
                        help="Skip the initial probe job and process all files directly")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Per-file timeout in seconds (default: 3600)")
    parser.add_argument("--reserved-memory-ratio", type=float, default=0.4,
                        help="Fraction of total memory to reserve (default: 0.4). "
                             "Workers are sized from (1 - ratio) * total_memory / per_job_memory.")
    parser.add_argument("--recursive", action="store_true",
                        help="Scan source folder recursively for motion files")
    parser.add_argument("--exclude-suffix", default=None,
                        help="Exclude files ending with this suffix (e.g. '_M.bvh' for mirrored files)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip motions whose retargeted output already exists")
    parser.add_argument("--video", action="store_true",
                        help="Render a video for each motion (offscreen; requires imageio[ffmpeg])")
    parser.add_argument("--progress", action="store_true",
                        help="Show a progress bar for each motion's frame retargeting")
    parser.add_argument("--output-framerate", type=float, default=None,
                        help="Resample motion to this framerate before retargeting (e.g. 30 to downsample 120fps data)")
    parser.add_argument("--scale-factor", type=float, default=None,
                        help="Force one source-to-robot scale factor for every motion "
                             "(instead of per-motion height estimates), so all motions and "
                             "their terrain stay consistent. The scaled terrain is saved "
                             "once under OUTPUT_DIR/terrain/.")

    args = parser.parse_args()

    source_folder = Path(args.source_folder)
    if not source_folder.is_dir():
        print(f"Error: source folder not found: {args.source_folder}")
        sys.exit(1)

    robot_config_path = str(Path(args.robot_config).expanduser())
    if not Path(robot_config_path).exists():
        print(f"Error: robot config not found: {args.robot_config}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.recursive and output_dir.resolve().is_relative_to(source_folder.resolve()):
        raise ValueError(
            f"--output-dir ({args.output_dir}) must not be inside "
            f"--source-folder ({args.source_folder}) when --recursive is used"
        )

    # Environment detection
    resources = detect_resources()
    print(f"CPU cores: {resources['cpu_count']}")
    if resources["memory_gb"] is not None:
        print(f"Total memory: {resources['memory_gb']:.1f} GB")

    max_workers = args.max_workers
    if max_workers:
        print(f"Max workers override: {max_workers}")
    else:
        print("Max workers: auto (determined from test job memory)")

    activation_prefix = get_activation_prefix()
    if activation_prefix:
        print(f"Environment activation: detected")
    else:
        print("Warning: no conda/virtualenv detected, using bare Python")

    # Log repository status before processing
    log_repo_status(output_dir, args.robot_config)

    # Scan for motion files
    motion_files = scan_source_folder(source_folder, args.source_type, recursive=args.recursive, exclude_suffix=args.exclude_suffix)
    if not motion_files:
        print(f"Error: no motion files found in {args.source_folder} for type '{args.source_type}'")
        sys.exit(1)
    print(f"Found {len(motion_files)} motion file(s)")

    # Resume: skip motions whose retargeted output already exists
    if args.resume:
        completed = [f for f in motion_files if _output_exists(f, output_dir)]
        if completed:
            print(f"Resume: skipping {len(completed)} already-processed motion(s)")
        completed_set = set(completed)
        motion_files = [f for f in motion_files if f not in completed_set]
        if not motion_files:
            print("All motions already processed. Nothing to do.")
            sys.exit(0)

    # Build shared extra options
    extra_options: dict = {}
    if args.source_options:
        extra_options.update(json.loads(args.source_options))

    # With a uniform scale factor the scaled terrain is a single shared
    # artifact: export it once here instead of once per motion.
    if args.scale_factor is not None and args.terrain:
        export_shared_scaled_terrain(args.terrain, args.scale_factor, output_dir)

    # Process batch
    results = process_batch(
        motion_files=motion_files,
        source_type=args.source_type,
        robot_config_path=robot_config_path,
        output_dir=output_dir,
        activation_prefix=activation_prefix,
        terrain_path=args.terrain,
        extra_options=extra_options,
        max_workers=args.max_workers,
        framerate=args.framerate,
        skip_test_job=args.skip_test_job,
        timeout=args.timeout,
        reserved_memory_ratio=args.reserved_memory_ratio,
        source_folder=source_folder if args.recursive else None,
        save_video=args.video,
        output_framerate=args.output_framerate,
        scale_factor=args.scale_factor,
        progress=args.progress,
    )

    failed = _summarize(results)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
