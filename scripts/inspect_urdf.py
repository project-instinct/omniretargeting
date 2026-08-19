#!/usr/bin/env python3
"""Inspect URDF body names for joint mapping."""

import mujoco
import sys

urdf_path = "/home/leo/NutstoreFiles/2Projects/isaacSim/instinctlab/source/instinctlab/instinctlab/assets/resources/unitree_g1/g1_29dof.urdf"

try:
    model = mujoco.MjModel.from_xml_path(urdf_path)
    
    print(f"Total bodies: {model.nbody}")
    print("\nBody names:")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  {i:3d}: {name}")
    
    print(f"\nTotal joints: {model.njnt}")
    print("\nJoint names:")
    for i in range(model.njnt):
        name = model.joint(i).name
        print(f"  {i:3d}: {name}")
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

