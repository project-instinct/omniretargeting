"""Replay input capture with importable native-handle reconstruction."""

import ctypes
from collections import OrderedDict
import pickle


def _native_function(library_type, path, name, argtypes, restype):
    library = library_type(path)
    function = getattr(library, name)
    function.argtypes, function.restype = argtypes, restype
    return function


class InputPickler(pickle.Pickler):
    def reducer_override(self, value):
        if isinstance(value, ctypes.CDLL):
            return type(value), (value._name,)
        if isinstance(value, ctypes._CFuncPtr):
            library = next((v for v in (value._objects or {}).values()
                            if isinstance(v, ctypes.CDLL)), None)
            if library is not None and hasattr(value, "__name__"):
                return _native_function, (type(library), library._name, value.__name__,
                                          value.argtypes, value.restype)
        return NotImplemented


def capture_inputs(path, value):
    """Exclude lazily rebuilt native handles, preserving all numerical state."""
    seen, removed = set(), []

    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif type(item).__module__.startswith("omniretargeting") and hasattr(item, "__dict__"):
            for name, child in list(vars(item).items()):
                kin = type(item).__name__ == "TrajectoryKinematics"
                if kin and name == "cache":
                    removed.append((item, name, child))
                    item.cache = OrderedDict()
                elif (name.startswith("_native")
                      or name in ("_self_batch", "_window_views", "_window_frame_cache", "_gpu_frames")
                      or (kin and name in ("xpos", "xmat", "geom_xpos", "geom_xmat", "jacp", "jacr"))):
                    removed.append((item, name, child))
                    delattr(item, name)
                else:
                    visit(child)
    try:
        visit(value)
        with path.open("wb") as handle:
            InputPickler(handle, protocol=5).dump(value)
    finally:
        for owner, name, child in removed:
            setattr(owner, name, child)

