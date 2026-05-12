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
        "tqdm",
        "scipy",
        "matplotlib",
        "trimesh",
        "jinja2",
        # Due to mujoco `strippath` default value in URDF parsing updated, `strippath=false` by default.
        "mujoco>=3.7.0",
        "viser",
        "robot_descriptions",
        "yourdfpy",
        "cvxpy",
        "libigl",
        "tyro",
        "imageio[ffmpeg]",
        # Additional dependencies for generic mesh processing
        "open3d",
        "pyvista",
    ],
    extras_require={
        "smplx": ["smplx", "torch"],
        "all": ["smplx", "torch"],
        "dev": ["mypy", "ruff", "pytest", "black"],
        "test": ["pytest", "pytest-cov"],
    },
)
