"""Vector-borne models: Ross-Macdonald and the host-vector model of the notebook."""

import warnings

import pytest
import sympy as sp

from helpers import assert_same, evaluate, spectral_radius
from pyr0compute import R0Model
from pyr0compute import parse_expression as P

ROSS_MACDONALD = """
dSh/dt = Lh - a*bh*Sh*Iv/Nh - muh*Sh
dIh/dt = a*bh*Sh*Iv/Nh - (gamma + muh)*Ih
dRh/dt = gamma*Ih - muh*Rh
dSv/dt = Lv - a*bv*Sv*Ih/Nh - muv*Sv
dIv/dt = a*bv*Sv*Ih/Nh - muv*Iv
"""
RM_VALUES = {"Lh": 100, "Lv": 5000, "Nh": 5000, "a": 0.3, "bh": 0.5, "bv": 0.4,
             "gamma": 0.1, "muh": 0.02, "muv": 0.1}


def test_ross_macdonald_r0_is_the_positive_root(no_warnings):
    model = R0Model(ROSS_MACDONALD, infected=["Ih", "Iv"])
    a, bh, bv, Lh, Lv, Nh, muh, muv, gamma = sp.symbols("a bh bv Lh Lv Nh muh muv gamma")
    R0_squared = a**2 * bh * bv * (Lh / muh) * (Lv / muv) / (Nh**2 * muv * (gamma + muh))
    assert_same(model.R0**2, R0_squared)
    assert evaluate(model.R0, RM_VALUES) > 0  # Perron root, not -sqrt(...)
    assert sorted(evaluate(e, RM_VALUES) for e in model.eigenvalues)[0] < 0


def test_ross_macdonald_numeric_agrees(no_warnings):
    model = R0Model(ROSS_MACDONALD, infected=["Ih", "Iv"])
    numeric = model.R0_numeric(RM_VALUES)
    assert numeric == pytest.approx(evaluate(model.R0, RM_VALUES))
    assert numeric == pytest.approx(spectral_radius(model.K, RM_VALUES))


def test_ross_macdonald_terms_detected_automatically(no_warnings):
    model = R0Model(ROSS_MACDONALD, infected=["Ih", "Iv"])
    Ih, Iv = sp.symbols("Ih Iv")
    assert_same(model.new_infections[Ih], P("a*bh*Sh*Iv/Nh"))
    assert_same(model.new_infections[Iv], P("a*bv*Sv*Ih/Nh"))


# Example 4 of the original notebook: SEIR host + SEI vector with human-human
# transmission and logistic vector recruitment. The notebook needed manual F
# terms and a hand-written DFE (with Mv = 0, which is inconsistent with Sv > 0).
HOST_VECTOR = """
Nh = Sh + Eh + Ih + Rh
dSh/dt = Deltah - delta*alphahv*Sh*Iv/Nh - beta*alphahh*Sh*Ih/Nh - (theta + muh)*Sh
dEh/dt = delta*alphahv*Sh*Iv/Nh + beta*alphahh*Sh*Ih/Nh - (sigma + muh)*Eh
dIh/dt = sigma*Eh - (gamma + muh)*Ih
dRh/dt = gamma*Ih - muh*Rh + theta*Sh
dMv/dt = phiv*Mv*(1 - Mv/kappa) - (omega + mum)*Mv
dSv/dt = omega*Mv - delta*alphavh*Sv*Ih/Nh - (muv + zeta)*Sv
dEv/dt = delta*alphavh*Sv*Ih/Nh - (psi + muv + zeta)*Ev
dIv/dt = psi*Ev - (muv + zeta)*Iv
"""
HV_INFECTED = ["Eh", "Ih", "Ev", "Iv"]
HV_VALUES = {"Deltah": 10, "delta": 0.5, "alphahv": 0.3, "alphahh": 0.05, "alphavh": 0.4,
             "beta": 0.2, "sigma": 0.2, "gamma": 0.14, "theta": 0.01, "muh": 0.0004,
             "mum": 0.05, "muv": 0.07, "zeta": 0.01, "psi": 0.1, "kappa": 1e4,
             "phiv": 0.5, "omega": 0.1}


@pytest.fixture(scope="module")
def host_vector():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = R0Model(HOST_VECTOR, infected=HV_INFECTED)
    return model, caught


def test_host_vector_new_infections_automatic(host_vector):
    model, _ = host_vector
    Eh, Ih, Ev, Iv = sp.symbols("Eh Ih Ev Iv")
    Nh = P("Sh + Eh + Ih + Rh")
    S_h, S_v = sp.symbols("Sh Sv")
    d, ahv, ahh, avh, b = sp.symbols("delta alphahv alphahh alphavh beta")
    assert_same(model.new_infections[Eh], d * ahv * S_h * Iv / Nh + b * ahh * S_h * Ih / Nh)
    assert_same(model.new_infections[Ev], d * avh * S_v * Ih / Nh)
    assert model.new_infections[Ih] == 0 and model.new_infections[Iv] == 0


def test_host_vector_chooses_dfe_with_vectors(host_vector):
    model, caught = host_vector
    assert any("disease-free equilibria" in str(w.message) for w in caught)
    assert len(model.dfe_candidates) == 2
    Mv, Sv, Sh, Rh = sp.symbols("Mv Sv Sh Rh")
    kappa, phiv, omega, mum, muv, zeta = sp.symbols("kappa phiv omega mum muv zeta")
    Deltah, theta, muh = sp.symbols("Deltah theta muh")
    assert_same(model.dfe[Mv], kappa * (1 - (omega + mum) / phiv))
    assert_same(model.dfe[Sv], omega * model.dfe[Mv] / (muv + zeta))
    assert_same(model.dfe[Sh], Deltah / (theta + muh))
    assert_same(model.dfe[Rh], theta * Deltah / (muh * (theta + muh)))


def test_host_vector_matches_manual_specification(host_vector):
    model, _ = host_vector
    manual = R0Model(HOST_VECTOR, infected=HV_INFECTED, dfe=model.dfe,
                     new_infections={
                         "Eh": "delta*alphahv*Sh*Iv/Nh + beta*alphahh*Sh*Ih/Nh",
                         "Ev": "delta*alphavh*Sv*Ih/Nh",
                     })
    assert evaluate(model.R0, HV_VALUES) == pytest.approx(evaluate(manual.R0, HV_VALUES))


def test_host_vector_r0_is_spectral_radius(host_vector):
    model, _ = host_vector
    expected = spectral_radius(model.K, HV_VALUES)
    assert evaluate(model.R0, HV_VALUES) == pytest.approx(expected)
    assert model.R0_numeric(HV_VALUES) == pytest.approx(expected)
