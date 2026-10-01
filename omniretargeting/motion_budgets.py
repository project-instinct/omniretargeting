"""Shared physical defaults, independent of solver stage and body labels."""

# Posture has no angular limit: it is a soft preference for the source-shaped
# reference, measured on landmark relations (``pose_relations``) and reported
# against the original motion.

# Per-joint temporal budgets on executed motion (velocity, acceleration, jerk per
# frame): at every stencil, the largest magnitude of the fixed reference's own
# derivative within CHATTER_WINDOW stencils around it, plus CHATTER_RMS_FRACTION
# of that coordinate's RMS derivative. The same relaxed envelope screens added
# chatter; a stencil-exact budget left no room where the reference is smooth.
CHATTER_WINDOW = 7
CHATTER_RMS_FRACTION = 0.1
# Chatter is the content of a signal below this Gaussian scale (frames).
CHATTER_SCALE_FRAMES = 2.0


def windowed_budget(magnitudes):
    """Budget of derivative magnitudes (stencils, ...) of one reference signal."""
    import numpy as np
    from scipy.ndimage import maximum_filter1d

    magnitudes = np.abs(np.asarray(magnitudes, dtype=float))
    if not magnitudes.shape[0]:
        return magnitudes
    local = maximum_filter1d(magnitudes, CHATTER_WINDOW, axis=0, mode="nearest")
    return local + CHATTER_RMS_FRACTION * np.sqrt(np.mean(magnitudes**2, axis=0))
