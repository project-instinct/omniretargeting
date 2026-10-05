"""Shared physical defaults, independent of solver stage and body labels."""

# Posture has no angular limit: it is a soft preference for the source-shaped
# reference, measured on landmark relations (``pose_relations``) and reported
# against the original motion.

# Source-relative chatter is measured on the high-frequency content of a
# signal, not on the magnitude of its derivative. A slowly varying clearance
# correction is therefore allowed even where the source-shaped reference is
# nearly stationary.
CHATTER_WINDOW = 7
# Chatter is the content of a signal below this Gaussian scale (frames).
CHATTER_SCALE_FRAMES = 2.0
CHATTER_RMS_FRACTION = 0.1
CHATTER_RMS_ALLOWANCE = 0.5
CHATTER_ABS_AMPLITUDE = 2e-3
CHATTER_EXCESS_TOLERANCE = 1e-3


def high_frequency(signal):
    """High-frequency content of a per-frame scalar/vector signal."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    signal = np.asarray(signal, dtype=float)
    if not signal.shape[0]:
        return np.zeros_like(signal)
    return signal - gaussian_filter1d(
        signal, CHATTER_SCALE_FRAMES, axis=0, mode="nearest"
    )


def chatter_budget(reference):
    """Per-coordinate high-frequency amplitude allowed by the source signal.

    The allowance follows the source locally and also receives a fraction of
    its whole-clip RMS plus a small absolute numerical/motion floor. Thus a
    fast source remains fast, while a smooth low-frequency robot correction is
    not mistaken for chatter.
    """
    import numpy as np
    from scipy.ndimage import maximum_filter1d

    content = np.abs(high_frequency(reference))
    if not content.shape[0]:
        return content
    local = maximum_filter1d(content, CHATTER_WINDOW, axis=0, mode="nearest")
    rms = np.sqrt(np.mean(content**2, axis=0))
    return local + CHATTER_RMS_ALLOWANCE * rms + CHATTER_ABS_AMPLITUDE


def chatter_excess(reference, candidate):
    """Positive excess high-frequency amplitude of ``candidate`` over source."""
    import numpy as np

    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    if reference.shape != candidate.shape:
        raise ValueError("Chatter reference and candidate must have the same shape")
    if not reference.shape[0]:
        return np.zeros(reference.shape[:-1] if reference.ndim > 1 else reference.shape)
    return np.abs(high_frequency(candidate - reference)) - chatter_budget(reference)


def windowed_budget(magnitudes):
    """Legacy derivative-magnitude budget for explicit regression experiments."""
    import numpy as np
    from scipy.ndimage import maximum_filter1d

    magnitudes = np.abs(np.asarray(magnitudes, dtype=float))
    if not magnitudes.shape[0]:
        return magnitudes
    local = maximum_filter1d(magnitudes, CHATTER_WINDOW, axis=0, mode="nearest")
    return local + CHATTER_RMS_FRACTION * np.sqrt(np.mean(magnitudes**2, axis=0))
