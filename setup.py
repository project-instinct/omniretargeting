from setuptools import find_packages, setup  # type: ignore[import-untyped]

setup(
    name="omniretargeting",
    version="0.1.0",
    description="Generic motion retargeting for any humanoid URDF and terrain mesh.",
    author="OmniRetargeting Team",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        # Core dependencies from holosoma_retargeting
        "numpy==2.3.5",
        "torch",
        "tqdm",
        "scipy",
        "matplotlib",
        "trimesh",
        "smplx",
        "jinja2",
        "mujoco",
        "viser",
        "mjviser @ git+https://github.com/mujocolab/mjviser@4ba054b9bdaf4f7604d6fa7a8b8c2f917daf43c2",
        "robot_descriptions",
        "yourdfpy",
        "cvxpy",
        "libigl",
        "tyro",
        # Additional dependencies for generic mesh processing
        "open3d",
        "pyvista",
    ],
    entry_points={
        "console_scripts": [
            "omniretargeting-mjviser=omniretargeting.mjviser_viewer:main",
        ],
    },
)
