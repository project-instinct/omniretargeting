"""Deterministic acceptance JSON with wall-clock diagnostics in a sidecar."""
import json


def write_report(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    stable = {key: value for key, value in report.items() if key != 'timing'}
    output.write_text(json.dumps(stable, indent=2, sort_keys=True, allow_nan=False) + '\n')
    output.with_suffix('.timing.json').write_text(
        json.dumps(report['timing'], indent=2, sort_keys=True, allow_nan=False) + '\n')
