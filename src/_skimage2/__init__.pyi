_submodules = [
    "data",
    "feature",
    "filters",
    "morphology",
    "util",
]

__all__ = _submodules + ["__version__", "ExperimentalAPIWarning"]  # noqa: F822

from . import (
    data,
    feature,
    filters,
    morphology,
    util,
)
