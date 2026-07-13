"""Configuration helpers for gated RMSNorm approximation sweeps."""

DEFAULT_CANDIDATES = ("31:4", "24:4", "15:4", "31:3", "24:3", "15:3")


def parse_candidate(spec: str) -> tuple[int, int]:
    degree_text, separator, iterations_text = spec.partition(":")
    if not separator:
        raise ValueError(f"candidate must be DEGREE:ITERATIONS, got {spec!r}")
    degree = int(degree_text)
    iterations = int(iterations_text)
    if degree < 1 or iterations < 1:
        raise ValueError("candidate degree and iterations must be positive")
    return degree, iterations
