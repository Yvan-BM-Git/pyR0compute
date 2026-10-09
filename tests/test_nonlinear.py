"""Models with a nonlinear disease-free equilibrium (block-by-block solve).

References
----------
L. Cuesta-Herrera, L. Pastenes, F. Cordova-Lepe, A. D. Arencibia (2025).
Mathematical modeling of the immune response mediated by human T-helper
lymphocytes in viral diseases. Math. Biosci. Eng. 22(11), 2807-2825.
doi:10.3934/mbe.2025103
"""

import numpy as np
import pytest
import sympy as sp

from helpers import assert_same
from pyr0compute import DFESymbol, R0Model

# ------------------------------------------------------------------ MBE 2025
TH_MODEL = """
    dE/dt = lambda - d*E - kappa*E*V
    dI/dt = kappa*E*V - alpha*I - beta*Th*I
    dV/dt = nu*I - mu*V - tau*Th*V
    dTh/dt = b - c*Th + gamma*I*Th
"""

TABLE1 = dict(lambda_=5e5, d=0.1, kappa=1e-7, alpha=0.1, beta=1e-7, nu=995,
              mu=0.5, tau=1e-7, b=5e4, c=0.1, gamma=1e-7)


def _values(model, table):
    names = {s.name: s for s in model.parameters}
    return {names["lambda" if k == "lambda_" else k]: v for k, v in table.items()}


def test_mbe2025_th_model_formula(no_warnings):
    """Eq. (2.4): R0* = (kappa lambda nu / d) / ((alpha + beta b/c)(mu + tau b/c))."""
    model = R0Model(TH_MODEL, infected=["I", "V"])
    p = {s.name: s for s in model.parameters}
    lam, d, kappa, alpha, beta, nu, mu, tau, b, c = (
        p[n] for n in ["lambda", "d", "kappa", "alpha", "beta", "nu", "mu", "tau", "b", "c"])
    expected = (kappa * lam * nu / d) / ((alpha + beta * b / c) * (mu + tau * b / c))
    assert_same(model.R0, expected)
    # a simple DFE is written out in full: no X* symbols
    assert model.dfe_definitions == {}
    assert_same(model.R0_compact, model.R0)
    dfe = {v.name: val for v, val in model.dfe.items()}
    assert_same(dfe["E"], lam / d)
    assert_same(dfe["Th"], b / c)
    # rank-one next-generation matrix (proof of Theorem 1)
    assert model.eigenvalues.count(0) == 1


def test_mbe2025_th_model_figure3_values(no_warnings):
    """R0* reported in Figure 3a (6.0303e3) and Figure 3c (19.8920)."""
    model = R0Model(TH_MODEL, infected=["I", "V"])
    assert model.R0_numeric(_values(model, TABLE1)) == pytest.approx(6030.3, rel=1e-4)
    fig3c = dict(TABLE1, beta=1.1e-7, tau=1e-4, d=0.2, lambda_=1e6, kappa=1e-8, b=2e4)
    assert model.R0_numeric(_values(model, fig3c)) == pytest.approx(19.8920, rel=1e-4)


def test_mbe2025_th_model_dfe_stability(no_warnings):
    """Uninfected block of J(x0): eigenvalues -d and -c (proof of Theorem 1)."""
    model = R0Model(TH_MODEL, infected=["I", "V"])
    d, c = sp.symbols("d c")
    out = model.dfe_stability(n_samples=20)
    assert sorted(map(str, out["eigenvalues"])) == sorted(map(str, [-d, -c]))
    assert out["feasible"] == out["stable"] == 20
    assert model.dfe_stability(_values(model, TABLE1))["stable"]


# ---------------------------------------------- immune response with Th/Tc/B/Treg
IMMUNE_MODEL = """
    dE/dt = g_E - (d_E + tau_E*V/(V + z_VE))*E
    dI/dt = tau_E*V*E/(V + z_VE) - (d_I + tau_I*C/(C + z_CI))*I
    dV/dt = nu*I - (d_V + tau_V*B/(B + z_BV))*V
    dH/dt = g_H - (d_H - tau_H*I/(I + z_IH) + rho_H*R/(R + z_RH))*H
    dC/dt = g_C - (d_C - tau_C*H/(H + z_HC) + rho_C*R/(R + z_RC))*C
    dB/dt = g_B - (d_B - tau_B*H/(H + z_HB) + rho_B*R/(R + z_RB))*B
    dR/dt = g_R - (d_R - tau_R*H/(H + z_HR) + rho_R)*R
"""

IMMUNE_VALUES = dict(
    g_E=1.0, d_E=0.1, tau_E=2.0, z_VE=1.0, d_I=0.5, tau_I=0.5, z_CI=1.0,
    nu=10.0, d_V=1.0, tau_V=0.5, z_BV=1.0,
    g_H=1.0, d_H=1.0, tau_H=0.5, rho_H=0.3, z_IH=1.0, z_RH=1.0,
    g_C=1.0, d_C=1.0, tau_C=0.5, rho_C=0.3, z_HC=1.0, z_RC=1.0,
    g_B=1.0, d_B=1.0, tau_B=0.5, rho_B=0.3, z_HB=1.0, z_RB=1.0,
    g_R=1.0, d_R=1.0, tau_R=0.5, rho_R=0.3, z_HR=1.0,
)


@pytest.fixture(scope="module")
def immune():
    return R0Model(IMMUNE_MODEL, infected=["I", "V"])


def _integrated_dfe(values):
    """Independent check: integrate the uninfected subsystem (I = V = 0) to steady state."""
    integrate = pytest.importorskip("scipy.integrate")
    v = values

    def rhs(_t, y):
        E, H, C, B, R = y
        return [
            v["g_E"] - v["d_E"] * E,
            v["g_H"] - (v["d_H"] + v["rho_H"] * R / (R + v["z_RH"])) * H,
            v["g_C"] - (v["d_C"] - v["tau_C"] * H / (H + v["z_HC"]) + v["rho_C"] * R / (R + v["z_RC"])) * C,
            v["g_B"] - (v["d_B"] - v["tau_B"] * H / (H + v["z_HB"]) + v["rho_B"] * R / (R + v["z_RB"])) * B,
            v["g_R"] - (v["d_R"] - v["tau_R"] * H / (H + v["z_HR"]) + v["rho_R"]) * R,
        ]

    sol = integrate.solve_ivp(rhs, (0, 400), [1, 1, 1, 1, 1], rtol=1e-11, atol=1e-13)
    return dict(zip("EHCBR", sol.y[:, -1]))


def _closed_form_R0(values, dfe):
    """R0 = g_E nu tau_E / (d_E z_VE k_I k_V) with the immune-modulated clearances."""
    v = values
    k_I = v["d_I"] + v["tau_I"] * dfe["C"] / (dfe["C"] + v["z_CI"])
    k_V = v["d_V"] + v["tau_V"] * dfe["B"] / (dfe["B"] + v["z_BV"])
    return v["g_E"] * v["nu"] * v["tau_E"] / (v["d_E"] * v["z_VE"] * k_I * k_V)


def test_immune_model_compact_R0(immune):
    p = {s.name: s for s in immune.parameters}
    Cs, Bs = DFESymbol("C_star", positive=True), DFESymbol("B_star", positive=True)
    k_I = p["d_I"] + p["tau_I"] * Cs / (Cs + p["z_CI"])
    k_V = p["d_V"] + p["tau_V"] * Bs / (Bs + p["z_BV"])
    expected = p["g_E"] * p["nu"] * p["tau_E"] / (p["d_E"] * p["z_VE"] * k_I * k_V)
    assert_same(immune.R0_compact, expected)
    # E* = g_E/d_E is written out; the others are found block by block: [H, R] -> C, B
    assert_same(immune.dfe_compact[sp.Symbol("E")], p["g_E"] / p["d_E"])
    names = [s.name for s in immune.dfe_definitions]
    assert set(names[:2]) == {"H_star", "R_star"} and set(names[2:]) == {"C_star", "B_star"}
    assert not immune.dfe_is_implicit
    assert "\\frac" in immune.latex() and "C^{*}" in immune.latex()


def test_immune_model_numeric_agreement(immune):
    dfe_ref = _integrated_dfe(IMMUNE_VALUES)
    dfe = {v.name: val for v, val in immune.dfe_numeric(IMMUNE_VALUES).items()}
    for name in "EHCBR":
        assert dfe[name] == pytest.approx(dfe_ref[name], rel=1e-7)
    R0_ref = _closed_form_R0(IMMUNE_VALUES, dfe_ref)
    assert immune.R0_numeric(IMMUNE_VALUES) == pytest.approx(R0_ref, rel=1e-7)
    # the explicit R0 (DFE values substituted, nested radicals) gives the same number
    explicit = immune.R0.subs({s: IMMUNE_VALUES[s.name] for s in immune.R0.free_symbols})
    assert complex(sp.N(explicit)).real == pytest.approx(R0_ref, rel=1e-7)
    assert immune.dfe_stability(IMMUNE_VALUES)["stable"]


@pytest.mark.parametrize("tau_E", [0.005, 0.01, 2.0])
def test_immune_model_threshold(immune, tau_E):
    """R0 > 1 exactly when the DFE of the full model is unstable (Theorem 2 of vdD-W)."""
    values = dict(IMMUNE_VALUES, tau_E=tau_E)
    R0 = immune.R0_numeric(values)
    dfe = {v: val for v, val in immune.dfe_numeric(values).items()}
    f = sp.Matrix([immune.equations[v] for v in immune.variables])
    J = f.jacobian(list(immune.variables)).subs(dfe)
    J = J.subs({s: values[s.name] for s in J.free_symbols})
    abscissa = float(np.max(np.linalg.eigvals(np.array(J.evalf(), dtype=float)).real))
    assert abs(R0 - 1) > 0.02  # away from the threshold
    assert (R0 > 1) == (abscissa > 0)


def test_immune_model_chain_evaluation(immune):
    """R0 and dR0/dp evaluated through the X* chain match direct computations."""
    from pyr0compute.global_sensitivity import _R0Evaluator

    ev = _R0Evaluator(immune)
    cols = [np.array([IMMUNE_VALUES[name]]) for name in ev.names]
    assert ev(cols)[0] == pytest.approx(immune.R0_numeric(IMMUNE_VALUES), rel=1e-10)
    for name in ["tau_E", "g_H", "rho_R", "tau_C"]:  # direct and through H*, R*, C*, B*
        i = ev.names.index(name)
        h = 1e-6 * IMMUNE_VALUES[name]
        up = immune.R0_numeric(dict(IMMUNE_VALUES, **{name: IMMUNE_VALUES[name] + h}))
        down = immune.R0_numeric(dict(IMMUNE_VALUES, **{name: IMMUNE_VALUES[name] - h}))
        exact = complex(np.asarray(ev._derivative(cols, i)).ravel()[0]).real
        assert exact == pytest.approx((up - down) / (2 * h), rel=1e-5)


def test_immune_model_global_sensitivity(immune):
    pytest.importorskip("scipy")
    result = immune.prcc(baseline=IMMUNE_VALUES, spread=0.2, n=200, seed=1)
    prcc = {name: row["PRCC"] for name, row in result.as_dict().items()}
    assert np.all(np.isfinite(list(prcc.values())))
    assert prcc["tau_E"] > 0 and prcc["d_E"] < 0


# ------------------------------------------------------------ implicit DFE
QUINTIC = """
    dS/dt = Lambda - mu*S - k*S^5 - beta*S*I
    dI/dt = beta*S*I - (gamma + mu)*I
"""


def test_dfe_without_closed_form_is_kept_implicit(no_warnings):
    pytest.importorskip("scipy")
    model = R0Model(QUINTIC, infected=["I"])
    assert model.dfe_is_implicit
    S_star = DFESymbol("S_star", positive=True)
    beta, gamma, mu = sp.symbols("beta gamma mu")
    assert_same(model.R0_compact, beta * S_star / (gamma + mu))
    assert model.dfe_definitions == {S_star: None}

    values = dict(Lambda=2.0, mu=0.5, k=0.1, beta=1.5, gamma=0.3)
    S = model.dfe_numeric(values)[sp.Symbol("S")]
    assert 2.0 - 0.5 * S - 0.1 * S**5 == pytest.approx(0, abs=1e-10)
    assert model.R0_numeric(values) == pytest.approx(1.5 * S / 0.8, rel=1e-10)
    out = model.dfe_stability(values)
    assert out["stable"] and out["eigenvalues"][0].real == pytest.approx(-0.5 - 0.5 * S**4)


def test_implicit_dfe_global_sensitivity(no_warnings):
    pytest.importorskip("scipy")
    model = R0Model(QUINTIC, infected=["I"])
    values = dict(Lambda=2.0, mu=0.5, k=0.1, beta=1.5, gamma=0.3)
    result = model.prcc(baseline=values, spread=0.2, n=60, seed=0, check_monotonicity=False)
    prcc = {name: row["PRCC"] for name, row in result.as_dict().items()}
    assert prcc["beta"] > 0.5 and prcc["k"] < 0
