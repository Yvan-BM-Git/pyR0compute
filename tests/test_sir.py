"""SIR models: the three input styles, parameter detection and the DFE."""

import pytest
import sympy as sp

from helpers import assert_same
from pyr0compute import DiseaseFreeEquilibriumError, R0Model
from pyr0compute import parse_expression as P

SIR_TEXT = """
dS/dt = Lambda - beta*S*I - mu*S
dI/dt = beta*S*I - (gamma + mu)*I
dR/dt = gamma*I - mu*R
"""
L, beta, gamma, mu = sp.symbols("Lambda beta gamma mu")
SIR_R0 = beta * L / (mu * (gamma + mu))


def test_sir_vital_dynamics_text(no_warnings):
    model = R0Model(SIR_TEXT, infected=["I"])
    assert_same(model.R0, SIR_R0)
    assert [str(p) for p in model.parameters] == ["Lambda", "beta", "gamma", "mu"]
    assert model.dfe == {sp.Symbol("S"): L / mu, sp.Symbol("I"): 0, sp.Symbol("R"): 0}


def test_sir_parameters_are_detected_automatically(no_warnings):
    """Only the equations and the infected compartment are given."""
    S, I, R = sp.symbols("S I R")
    model = R0Model({S: L - beta * S * I - mu * S,
                     I: beta * S * I - (gamma + mu) * I,
                     R: gamma * I - mu * R}, infected=[I])
    assert set(model.parameters) == {L, beta, gamma, mu}
    assert set(model.variables) == {S, I, R}
    assert_same(model.R0, SIR_R0)
    # results are expressed with the user's own symbols
    assert model.R0.free_symbols <= {L, beta, gamma, mu}


def test_three_input_styles_agree(no_warnings):
    S, I, R = sp.symbols("S I R")
    rhs = [L - beta * S * I - mu * S, beta * S * I - (gamma + mu) * I, gamma * I - mu * R]
    from_list = R0Model(rhs, infected=[I], variables=[S, I, R]).R0
    from_dict_strings = R0Model(
        {"S": "Lambda - beta*S*I - mu*S", "I": "beta*S*I - (gamma+mu)*I", "R": "gamma*I - mu*R"},
        infected="I",
    ).R0
    from_text = R0Model(SIR_TEXT, infected=["I"]).R0
    assert_same(from_list, from_dict_strings)
    assert_same(from_list, from_text)


def test_equation_order_does_not_matter(no_warnings):
    """The original notebook required infected equations first; not anymore."""
    S, I, R = sp.symbols("S I R")
    eqs = {S: L - beta * S * I - mu * S, I: beta * S * I - (gamma + mu) * I, R: gamma * I - mu * R}
    infected_first = R0Model({I: eqs[I], S: eqs[S], R: eqs[R]}, infected=[I]).R0
    natural_order = R0Model(eqs, infected=[I]).R0
    by_index = R0Model(list(eqs.values()), infected=[1], variables=[S, I, R]).R0
    assert_same(infected_first, natural_order)
    assert_same(by_index, natural_order)


def test_frequency_dependent_sir_closed_population():
    """Classic SIR without births: the DFE needs S = N (the original code returned 0)."""
    eqs = {"S": "-beta*S*I/N", "I": "beta*S*I/N - gamma*I", "R": "gamma*I"}
    with pytest.raises(DiseaseFreeEquilibriumError, match="dfe"):
        R0Model(eqs, infected=["I"])
    model = R0Model(eqs, infected=["I"], dfe={"S": "N"})
    assert_same(model.R0, beta / gamma)
    assert_same(model.new_infections[sp.Symbol("I")], P("beta*S*I/N"))


def test_frequency_dependent_sir_with_auxiliary_total(no_warnings):
    """N = S + I + R is substituted; with births the DFE is found automatically."""
    model = R0Model("""
        N = S + I + R
        dS/dt = Lambda - beta*S*I/N - mu*S
        dI/dt = beta*S*I/N - (gamma + mu)*I
        dR/dt = gamma*I - mu*R
    """, infected=["I"])
    assert_same(model.R0, beta / (gamma + mu))
    assert "N" not in {p.name for p in model.parameters}


def test_sympy_reserved_names_are_plain_symbols(no_warnings):
    """I, S, E, N, beta, gamma and lambda are ordinary names in a model."""
    model = R0Model("""
        dS/dt = lambda - beta*S*I/N - mu*S
        dE/dt = beta*S*I/N - (sigma + mu)*E
        dI/dt = sigma*E - (gamma + mu)*I
    """, infected=["E", "I"], dfe=None)
    names = {s.name for s in model.R0.free_symbols}
    assert names == {"lambda", "beta", "N", "sigma", "gamma", "mu"}
    assert not model.R0.has(sp.I)  # not the imaginary unit


def test_nonlinear_incidence_with_caret(no_warnings):
    """Saturated incidence written with ^ (Capasso-Serio type)."""
    model = R0Model({"S": "Lambda - beta*S*I/(1 + alpha*I^2) - mu*S",
                     "I": "beta*S*I/(1 + alpha*I^2) - (gamma+mu)*I"}, infected=["I"])
    assert_same(model.R0, SIR_R0)
    assert "alpha" in {p.name for p in model.parameters}
