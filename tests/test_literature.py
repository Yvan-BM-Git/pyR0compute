"""Models from the literature and from the original notebook."""

import warnings

import pytest
import sympy as sp

from helpers import assert_same
from pyr0compute import R0Model
from pyr0compute import parse_expression as P


def test_van_den_driessche_watmough_treatment_model(no_warnings):
    """Section 4.1 of van den Driessche & Watmough (2002), Math. Biosci. 180:29-48."""
    model = R0Model("""
        N = S + E + I + T
        dE/dt = beta1*S*I/N + beta2*T*I/N - (d + nu + r1)*E + p*r2*I
        dI/dt = nu*E - (d + r2)*I
        dS/dt = b - d*S - beta1*S*I/N
        dT/dt = -d*T + r1*E + q*r2*I - beta2*T*I/N
    """, infected=["E", "I"])
    beta1, nu, d, r1, r2, p = sp.symbols("beta1 nu d r1 r2 p")
    assert_same(model.R0, beta1 * nu / ((d + nu + r1) * (d + r2) - nu * p * r2))
    # p*r2*I (treatment failure) is a transition, not a new infection
    assert_same(model.new_infections[sp.Symbol("E")], P("(beta1*S*I + beta2*T*I)/(S + E + I + T)"))


def test_within_host_tiv(no_warnings):
    """Target cell - infected cell - virus model (Ciupe & Heffernan 2017)."""
    model = R0Model("""
        dT/dt = s - d*T - beta*T*V
        dI/dt = beta*T*V - delta*I
        dV/dt = p*I - c*V
    """, infected=["I", "V"])
    s, d, beta, delta, p, c = sp.symbols("s d beta delta p c")
    assert_same(model.R0, beta * p * s / (c * d * delta))


def test_within_host_eivt_with_immune_response(no_warnings):
    """Example 2 of the original notebook (epithelial cells and immune response)."""
    model = R0Model("""
        dI/dt = k*E*V - a*I - beta*T*I
        dV/dt = v*I - u*V - t*T*V
        dE/dt = l - d*E - k*E*V
        dT/dt = b - c*T + g*I*T
    """, infected=["I", "V"])
    k, a, beta, v, u, t, l, d, b, c = sp.symbols("k a beta v u t l d b c")
    assert_same(model.R0, c**2 * k * l * v / (d * (a * c + b * beta) * (b * t + c * u)))


def test_host_vector_with_carrier_immunity(no_warnings):
    """Page-13 example of the notebook (I, V, S, M): automatic F matches the manual one."""
    model = R0Model("""
        dI/dt = beta_s*S*V - (b + gamma)*I
        dV/dt = beta_m*M*I - c*V
        dS/dt = b - b*S + gamma*I - beta_s*S*V
        dM/dt = c - c*M - beta_m*M*I
    """, infected=["I", "V"])
    beta_s, beta_m, b, gamma, c = sp.symbols("beta_s beta_m b gamma c")
    assert_same(model.R0**2, beta_s * beta_m / (c * (b + gamma)))


def test_two_strains_with_superinfection():
    """Page-12 example: I1 I2 nu is a transfer between infected classes, not F."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = R0Model("""
            dI1/dt = beta1*I1*S - (b + gamma1)*I1 + nu*I1*I2
            dI2/dt = beta2*I2*S - (b + gamma2)*I2 - nu*I1*I2
            dS/dt = b - b*S + gamma1*I1 + gamma2*I2 - (beta1*I1 + beta2*I2)*S
        """, infected=["I1", "I2"])
        R0 = model.R0
    I1, I2 = sp.symbols("I1 I2")
    assert_same(model.new_infections[I1], P("beta1*I1*S"))
    assert_same(model.new_infections[I2], P("beta2*I2*S"))
    beta1, beta2, b, gamma1, gamma2 = sp.symbols("beta1 beta2 b gamma1 gamma2")
    expected = {beta1 / (b + gamma1), beta2 / (b + gamma2)}
    assert {sp.simplify(e) for e in model.eigenvalues} == {sp.simplify(e) for e in expected}
    # which strain dominates depends on the parameters: R0 = max of both
    assert any("dominant eigenvalue" in str(w.message) for w in caught)
    assert isinstance(R0, sp.Max)
    assert {sp.simplify(a) for a in R0.args} == {sp.simplify(e) for e in expected}
    values = {"beta1": 0.5, "beta2": 0.9, "b": 0.1, "gamma1": 0.1, "gamma2": 0.2, "nu": 0.3}
    assert model.R0_numeric(values) == pytest.approx(max(0.5 / 0.2, 0.9 / 0.3))
