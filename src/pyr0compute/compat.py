"""Backward-compatible interface of the original notebook."""

from __future__ import annotations

from typing import Optional, Sequence

import sympy as sp

from .model import R0Model

__all__ = ["GeneralEpidemiologicalModel"]


class GeneralEpidemiologicalModel(R0Model):
    """Interface of the original ``pyR0compute_examples.ipynb`` notebook.

    Existing code keeps working, with two differences: the equations no longer
    need to list the infected compartments first, and ``parameters`` is
    optional (it is detected automatically). New code should use
    :class:`R0Model`.

    Parameters
    ----------
    variables, parameters, equations, infected_indices
        As in the original notebook; ``infected_indices`` are positions in
        ``variables``.
    new_infection_terms
        Optional list aligned with ``variables`` (zeros for uninfected).
    equilibrium_point
        Optional disease-free equilibrium aligned with ``variables``.
    total_population
        Accepted for compatibility; not needed.
    verbose
        Print the step-by-step report when :meth:`calculate_R0` is called
        (default ``True``, like the original).
    """

    def __init__(
        self,
        variables: Sequence[sp.Symbol],
        parameters: Optional[Sequence[sp.Symbol]],
        equations: Sequence[sp.Expr],
        infected_indices: Sequence[int],
        new_infection_terms: Optional[Sequence[sp.Expr]] = None,
        equilibrium_point: Optional[Sequence[sp.Expr]] = None,
        total_population=None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        dfe = None
        if equilibrium_point is not None:
            dfe = dict(zip(variables, list(equilibrium_point)))
        self._legacy_verbose = verbose
        super().__init__(
            equations=list(equations),
            infected=list(infected_indices),
            variables=list(variables),
            parameters=list(parameters) if parameters else None,
            new_infections=list(new_infection_terms) if new_infection_terms is not None else None,
            dfe=dfe,
            **kwargs,
        )

    def identify_F_V_terms(self):
        """Return the vectors of new-infection and transition terms."""
        F = sp.Matrix([self.new_infections[v] for v in self.infected])
        V = sp.Matrix([self.transitions[v] for v in self.infected])
        return F, V

    def calculate_next_generation_matrices(self):
        """Return ``(F, V)`` evaluated at the disease-free equilibrium."""
        return self.F, self.V

    def calculate_R0(self):
        """Return R0 (printing the report if ``verbose``)."""
        R0 = self.R0
        if self._legacy_verbose:
            print(self.report())
        return R0
