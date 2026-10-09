"""pyR0compute: symbolic computation of the basic reproduction number R0.

Implements the next-generation matrix method of van den Driessche & Watmough
(2002) for compartmental ODE models. Write the model, say which compartments
are infected, and every other symbol is treated as a parameter::

    from pyr0compute import R0Model

    model = R0Model('''
        dS/dt = Lambda - beta*S*I - mu*S
        dI/dt = beta*S*I - (gamma + mu)*I
        dR/dt = gamma*I - mu*R
    ''', infected=["I"])

    model.R0          # Lambda*beta/(mu*(gamma + mu))
"""

from .compat import GeneralEpidemiologicalModel
from .exceptions import (
    DiseaseFreeEquilibriumError,
    ModelSpecificationError,
    NextGenerationError,
    R0ComputeError,
)
from .global_sensitivity import GlobalSensitivityResult
from .model import DFESymbol, R0Model
from .parsing import parse_expression

__version__ = "0.1.0"

__all__ = [
    "R0Model",
    "DFESymbol",
    "GlobalSensitivityResult",
    "GeneralEpidemiologicalModel",
    "parse_expression",
    "R0ComputeError",
    "ModelSpecificationError",
    "DiseaseFreeEquilibriumError",
    "NextGenerationError",
    "__version__",
]
