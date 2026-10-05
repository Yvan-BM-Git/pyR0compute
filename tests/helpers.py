import numpy as np
import sympy as sp


def assert_same(a, b):
    """Symbolic equality up to simplification."""
    diff = sp.simplify(sp.sympify(a) - sp.sympify(b))
    is_zero = diff.is_zero_matrix if isinstance(diff, sp.MatrixBase) else diff == 0
    assert is_zero, f"{a} != {b} (difference {diff})"


def spectral_radius(K, values):
    """Numerical spectral radius of a symbolic matrix."""
    subs = {s: values[s.name] for s in K.free_symbols}
    return float(np.max(np.abs(np.linalg.eigvals(np.array(K.subs(subs).evalf(), dtype=complex)))))


def evaluate(expr, values):
    return float(sp.N(expr.subs({s: values[s.name] for s in expr.free_symbols})))
