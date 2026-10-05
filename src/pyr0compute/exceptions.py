"""Exceptions raised by pyR0compute."""


class R0ComputeError(ValueError):
    """Base class for all pyR0compute errors."""


class ModelSpecificationError(R0ComputeError):
    """The model (equations, variables, infected compartments...) is ill-defined."""


class DiseaseFreeEquilibriumError(R0ComputeError):
    """The disease-free equilibrium could not be determined unambiguously."""


class NextGenerationError(R0ComputeError):
    """The next-generation matrix or its spectral radius could not be computed."""
