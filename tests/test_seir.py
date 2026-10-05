"""SEIR models, sensitivity indices and the legacy interface."""

import pytest
import sympy as sp

from helpers import assert_same
from pyr0compute import GeneralEpidemiologicalModel, R0Model
from pyr0compute import parse_expression as P

L, beta, sigma, gamma, mu, N, p = sp.symbols("Lambda beta sigma gamma mu N p")
SEIR_R0 = beta * L * sigma / (mu * (sigma + mu) * (gamma + mu))

SEIR_TEXT = """
dS/dt = Lambda - beta*S*I - mu*S
dE/dt = beta*S*I - (sigma + mu)*E
dI/dt = sigma*E - (gamma + mu)*I
dR/dt = gamma*I - mu*R
"""


def test_seir_vital_dynamics(no_warnings):
    model = R0Model(SEIR_TEXT, infected=["E", "I"])
    assert_same(model.R0, SEIR_R0)
    E, I = sp.symbols("E I")
    # only S -> E is a new infection; E -> I is a transition
    assert_same(model.new_infections[E], P("beta*S*I"))
    assert model.new_infections[I] == 0
    assert_same(model.transitions[I], P("(gamma + mu)*I - sigma*E"))


def test_seir_matrices(no_warnings):
    model = R0Model(SEIR_TEXT, infected=["E", "I"])
    assert_same(model.F - sp.Matrix([[0, beta * L / mu], [0, 0]]), sp.zeros(2))
    assert_same(model.V - sp.Matrix([[sigma + mu, 0], [-sigma, gamma + mu]]), sp.zeros(2))


def test_seir_infected_order_is_irrelevant(no_warnings):
    a = R0Model(SEIR_TEXT, infected=["E", "I"]).R0
    b = R0Model(SEIR_TEXT, infected=["I", "E"]).R0
    assert_same(a, b)


def test_seir_frequency_dependent(no_warnings):
    model = R0Model("""
        dS/dt = mu*N - beta*S*I/N - mu*S
        dE/dt = beta*S*I/N - (sigma + mu)*E
        dI/dt = sigma*E - (gamma + mu)*I
        dR/dt = gamma*I - mu*R
    """, infected=["E", "I"])
    assert_same(model.R0, beta * sigma / ((sigma + mu) * (gamma + mu)))


def test_seir_vaccination_proportion(no_warnings):
    """(1 - p) factors are kept together; the original code split them into F and V."""
    model = R0Model("""
        dS/dt = (1 - p)*Lambda - beta*S*I - mu*S
        dV/dt = p*Lambda - mu*V
        dE/dt = beta*S*I - (sigma + mu)*E
        dI/dt = sigma*E - (gamma + mu)*I
    """, infected=["E", "I"])
    assert_same(model.R0, (1 - p) * SEIR_R0)


def test_seir_sensitivity_indices(no_warnings):
    model = R0Model(SEIR_TEXT, infected=["E", "I"])
    idx = model.sensitivity_indices()
    assert_same(idx[beta], 1)
    assert_same(idx[L], 1)
    assert_same(idx[sigma], mu / (sigma + mu))
    assert_same(idx[gamma], -gamma / (gamma + mu))
    values = {"Lambda": 10, "beta": 0.001, "sigma": 0.2, "gamma": 0.1, "mu": 0.02}
    numeric = model.sensitivity_indices(values)
    assert numeric[sigma] == pytest.approx(0.02 / 0.22)


def test_seir_numeric_matches_symbolic(no_warnings):
    model = R0Model(SEIR_TEXT, infected=["E", "I"])
    values = {"Lambda": 10, "beta": 0.001, "sigma": 0.2, "gamma": 0.1, "mu": 0.02}
    expected = 10 * 0.001 * 0.2 / (0.02 * 0.22 * 0.12)
    assert model.R0_numeric(values) == pytest.approx(expected)
    assert float(model.R0.subs({sp.Symbol(k): v for k, v in values.items()})) == pytest.approx(expected)


def test_legacy_interface_still_works(capsys):
    """Code written for the original notebook keeps running unchanged."""
    E, I, S, R = sp.symbols("E I S R")
    b, s, g, m, lam = sp.symbols("beta sigma gamma mu Lambda", positive=True)
    model = GeneralEpidemiologicalModel(
        [E, I, S, R],
        [b, s, g, m, lam],
        [b * S * I - (s + m) * E, s * E - (g + m) * I, lam - b * S * I - m * S, g * I - m * R],
        [0, 1],
    )
    R0 = model.calculate_R0()
    assert_same(R0, lam * b * s / (m * (s + m) * (g + m)))
    assert "R0 =" in capsys.readouterr().out


def test_legacy_interface_with_manual_terms(no_warnings):
    I, S, R = sp.symbols("I S R")
    b, g, n = sp.symbols("beta gamma N", positive=True)
    model = GeneralEpidemiologicalModel(
        [I, S, R], [b, g, n],
        [b * S * I / n - g * I, -b * S * I / n, g * I], [0],
        new_infection_terms=[b * S * I / n, 0, 0],
        equilibrium_point=[0, n, 0],
        verbose=False,
    )
    assert_same(model.calculate_R0(), b / g)
