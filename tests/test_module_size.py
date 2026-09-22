"""Keep maintained Python modules small enough to navigate and review."""

from pathlib import Path


def test_python_modules_stay_within_1000_lines():
    root = Path(__file__).resolve().parents[1]
    oversized = []
    for directory in ("omniretargeting", "scripts", "tests"):
        for path in sorted((root / directory).rglob("*.py")):
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count > 1000:
                oversized.append(f"{path.relative_to(root)}: {line_count} lines")
    assert not oversized, "Split modules by responsibility:\n" + "\n".join(oversized)
