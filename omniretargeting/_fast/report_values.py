"""Convert numerical history values into JSON-compatible values."""
import numpy as np


def report_value(value):
    """JSON history form; proposal objects retain tuple-keyed dictionaries."""
    if isinstance(value, dict):
        return {str(k) if isinstance(k, tuple) else k: report_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [report_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
