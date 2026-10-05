"""Input validation and informative errors."""

import pytest
import sympy as sp

from pyr0compute import (
    DiseaseFreeEquilibriumError,
    ModelSpecificationError,
    NextGenerationError,
    R0Model,
)

SIR = {"S": "Lambda - beta*S*I - mu*S", "I": "beta*S*I - (gamma+mu)*I", "R": "gamma*I - mu*R"}


def test_unknown_infected_compartment():
    with pytest.raises(ModelSpecificationError, match="not a state variable"):
        R0Model(SIR, infected=["X"])


def test_no_new_infections_is_an_error_not_zero():
    """The original code silently returned R0 = 0 in this situation."""
    with pytest.raises(ModelSpecificationError, match="No new-infection terms"):
        R0Model(SIR, infected=["R"])


def test_ambiguous_sign_asks_for_manual_terms():
    eqs = {"S": "Lambda - (a - b)*S*I - mu*S", "I": "(a - b)*S*I - (gamma+mu)*I"}
    with pytest.raises(ModelSpecificationError, match="new_infections"):
        R0Model(eqs, infected=["I"])
    model = R0Model(eqs, infected=["I"], new_infections={"I": "(a - b)*S*I"})
    a, b, L, gamma, mu = sp.symbols("a b Lambda gamma mu")
    assert sp.simplify(model.R0 - (a - b) * L / (mu * (gamma + mu))) == 0


def test_declared_parameters_are_checked():
    with pytest.warns(UserWarning, match="not declared"):
        R0Model(SIR, infected=["I"], parameters=["beta", "gamma", "mu"])
    with pytest.warns(UserWarning, match="not used"):
        R0Model(SIR, infected=["I"], parameters=["beta", "gamma", "mu", "Lambda", "kappa"])


def test_list_equations_need_variables():
    with pytest.raises(ModelSpecificationError, match="variables"):
        R0Model(["-beta*S*I", "beta*S*I - gamma*I"], infected=[1])


def test_dfe_cannot_have_infection():
    with pytest.raises(DiseaseFreeEquilibriumError, match="must be 0"):
        R0Model(SIR, infected=["I"], dfe={"I": 1})


def test_wrong_dfe_is_rejected():
    with pytest.raises(DiseaseFreeEquilibriumError, match="do not satisfy"):
        R0Model(SIR, infected=["I"], dfe={"S": "Lambda", "R": 0})


def test_infected_inflow_without_infection_is_not_a_dfe():
    eqs = {"S": "Lambda - beta*S*I - mu*S", "I": "theta + beta*S*I - (gamma+mu)*I"}
    with pytest.raises(DiseaseFreeEquilibriumError, match="not an equilibrium"):
        R0Model(eqs, infected=["I"])


def test_singular_transition_matrix():
    eqs = {"S": "-beta*S*I", "I": "beta*S*I"}
    with pytest.raises(NextGenerationError, match="singular"):
        R0Model(eqs, infected=["I"], dfe={"S": "N"})


def test_text_model_syntax_error():
    with pytest.raises(ModelSpecificationError, match="Line 2"):
        R0Model("dS/dt = -beta*S*I\nthis is not an equation\n", infected=["S"])


def test_missing_numeric_values():
    model = R0Model(SIR, infected=["I"])
    with pytest.raises(ModelSpecificationError, match="Missing values"):
        model.R0_numeric({"beta": 0.1})


def test_report_contains_all_steps():
    report = R0Model(SIR, infected=["I"]).report()
    for section in ["Parameters", "New-infection terms", "Disease-free equilibrium",
                    "F =", "V =", "K = F V^-1", "R0 ="]:
        assert section in report


def test_report_latex_document_and_markdown():
    model = R0Model(SIR, infected=["I"])
    doc = model.report_latex()
    for piece in ["\\begin{align*}", "\\mathcal{F}_{I}", "\\begin{bmatrix}", "\\mathcal{R}_0"]:
        assert piece in doc
    full = model.report_latex(standalone=True, mat_str="pmatrix")
    assert full.startswith("\\documentclass") and full.rstrip().endswith("\\end{document}")
    assert "\\begin{pmatrix}" in full
    md = model.report_latex("markdown")
    assert md.startswith("## ") and "$$\n\\begin{aligned}" in md and "| Equation |" in md
    with pytest.raises(ValueError):
        model.report_latex("html")
