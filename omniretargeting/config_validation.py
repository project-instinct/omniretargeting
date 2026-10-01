"""Reject misspelled and retired algorithm options at configuration boundaries."""

from collections.abc import Mapping


def known_options(config, defaults, scope, *, extra=()):
    if config is None:
        return
    if not isinstance(config, Mapping):
        raise TypeError(f"{scope} configuration must be a mapping")
    unknown = set(config) - set(defaults) - set(extra)
    if unknown:
        raise ValueError(f"Unknown {scope} options: {sorted(unknown)}")
