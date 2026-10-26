class EdgeBenchError(Exception):
    """An actionable, user-facing failure."""


class BackendUnavailable(EdgeBenchError):
    """Dependency, device, or native runtime unavailable."""
