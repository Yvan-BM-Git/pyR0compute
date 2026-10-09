"""Time limit for the symbolic solve of the disease-free equilibrium (dfe_timeout)."""

import signal
import time
import warnings

import numpy as np
import pytest
import sympy as sp

from helpers import assert_same
from pyr0compute import DFESymbol, ModelSpecificationError, R0Model
from pyr0compute import _solve

# Within-host model with an immune regulatory loop: interferon F activates NK
# cells N (Hill function of order 2), NK cells activate macrophages M, and
# macrophages produce interferon. Setting I = V = 0, {N, M, F} is one coupled
# block for which sympy.solve does not finish (hours).
LOOP = """
    dT/dt = s - d_T*T - beta*T*V
    dI/dt = beta*T*V - (delta + k_N*N)*I
    dV/dt = p*I - (c + k_M*M)*V
    dN/dt = a_N - d_N*N + h_N*F^2/(F^2 + K_F^2)
    dM/dt = a_M - d_M*M + h_M*N/(N + K_N)
    dF/dt = a_F - d_F*F + h_F*M/(M + K_M)
"""

VALUES = dict(s=10, d_T=0.1, beta=0.002, delta=0.5, k_N=0.2, p=20, c=3, k_M=0.3,
              a_N=1, d_N=0.5, h_N=2, K_F=1, a_M=1, d_M=0.4, h_M=1.5, K_N=1,
              a_F=0.5, d_F=1, h_F=2, K_M=1)

TIMEOUT = 3


@pytest.fixture(scope="module")
def loop():
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = R0Model(LOOP, infected=["I", "V"], dfe_timeout=TIMEOUT)
    model._build_seconds = time.perf_counter() - start
    model._warnings = [str(w.message) for w in caught]
    return model


def _immune_dfe(values):
    """Independent check: integrate the uninfected subsystem (I = V = 0) to steady state."""
    integrate = pytest.importorskip("scipy.integrate")
    v = values

    def rhs(_t, y):
        T, N, M, F = y
        return [
            v["s"] - v["d_T"] * T,
            v["a_N"] - v["d_N"] * N + v["h_N"] * F**2 / (F**2 + v["K_F"] ** 2),
            v["a_M"] - v["d_M"] * M + v["h_M"] * N / (N + v["K_N"]),
            v["a_F"] - v["d_F"] * F + v["h_F"] * M / (M + v["K_M"]),
        ]

    sol = integrate.solve_ivp(rhs, (0, 2000), [1, 1, 1, 1], method="LSODA", rtol=1e-11, atol=1e-13)
    return dict(zip("TNMF", sol.y[:, -1]))


def test_loop_block_stops_at_the_time_limit(loop):
    # stops shortly after the limit instead of running indefinitely
    assert loop._build_seconds < TIMEOUT + 15
    assert any("dfe_timeout=3" in w for w in loop._warnings)
    blocks = {tuple(sorted(b["variables"])): b["status"] for b in loop.dfe_blocks}
    assert blocks == {("T",): "closed form", ("F", "M", "N"): "time limit"}
    assert loop.dfe_is_implicit
    assert set(map(str, loop.dfe_definitions)) == {"N_star", "M_star", "F_star"}
    assert all(v is None for v in loop.dfe_definitions.values())
    assert "not found within dfe_timeout = 3 s" in loop.report()


def test_loop_compact_R0(loop):
    p = {s.name: s for s in loop.parameters}
    Ns, Ms = DFESymbol("N_star", positive=True), DFESymbol("M_star", positive=True)
    expected = p["beta"] * p["p"] * p["s"] / (
        p["d_T"] * (p["delta"] + p["k_N"] * Ns) * (p["c"] + p["k_M"] * Ms))
    assert_same(loop.R0_compact, expected)


def test_loop_numeric_dfe_and_R0(loop):
    ref = _immune_dfe(VALUES)
    dfe = {v.name: val for v, val in loop.dfe_numeric(VALUES).items()}
    for name in "TNMF":
        assert dfe[name] == pytest.approx(ref[name], rel=1e-8)
    v = VALUES
    R0_ref = v["beta"] * v["p"] * ref["T"] / ((v["delta"] + v["k_N"] * ref["N"]) * (v["c"] + v["k_M"] * ref["M"]))
    assert loop.R0_numeric(VALUES) == pytest.approx(R0_ref, rel=1e-8)
    assert loop.dfe_stability(VALUES)["stable"]


@pytest.mark.parametrize("beta", [0.002, 0.01])
def test_loop_threshold(loop, beta):
    """R0 < 1 exactly when the DFE of the full 6-equation model is stable."""
    values = dict(VALUES, beta=beta)
    R0 = loop.R0_numeric(values)
    dfe = loop.dfe_numeric(values)
    f = sp.Matrix([loop.equations[x] for x in loop.variables])
    J = f.jacobian(list(loop.variables)).subs(dfe)
    J = J.subs({s: values[s.name] for s in J.free_symbols})
    abscissa = float(np.max(np.linalg.eigvals(np.array(J.evalf(), dtype=float)).real))
    assert abs(R0 - 1) > 0.05
    assert (R0 > 1) == (abscissa > 0)


def test_loop_global_sensitivity(loop):
    pytest.importorskip("scipy")
    result = loop.prcc(baseline=VALUES, spread=0.2, n=200, seed=0, check_monotonicity=False)
    prcc = {name: row["PRCC"] for name, row in result.as_dict().items()}
    assert np.all(np.isfinite(list(prcc.values())))
    assert prcc["beta"] > 0.5 and prcc["c"] < 0 and prcc["d_T"] < 0


def test_easy_models_are_solved_in_closed_form(no_warnings):
    model = R0Model("""
        dS/dt = Lambda - beta*S*I - mu*S
        dE/dt = beta*S*I - (sigma + mu)*E
        dI/dt = sigma*E - (gamma + mu)*I
        dR/dt = gamma*I - mu*R
    """, infected=["E", "I"])
    assert all(b["status"] == "closed form" for b in model.dfe_blocks)
    assert not model.dfe_is_implicit


def test_timeout_none_and_validation(no_warnings):
    model = R0Model("dS/dt = Lambda - mu*S - beta*S*I\ndI/dt = beta*S*I - g*I",
                    infected=["I"], dfe_timeout=None)
    assert_same(model.R0, sp.Symbol("Lambda") * sp.Symbol("beta") / (sp.Symbol("mu") * sp.Symbol("g")))
    for bad in (0, -1):
        with pytest.raises(ModelSpecificationError, match="dfe_timeout"):
            R0Model("dS/dt = L - m*S - b*S*I\ndI/dt = b*S*I - g*I", infected=["I"], dfe_timeout=bad)


def test_worker_solves_and_times_out():
    x, y, a = sp.symbols("x y a")
    status, sols, _ = _solve.solve_with_time_limit([x**2 + y - a, x - y], [x, y], 30)
    assert status == "ok" and len(sols) == 2
    # the {N, M, F} block of the loop model: the worker is stopped at the limit
    N, M, F = sp.symbols("N M F")
    pars = {n: sp.Symbol(n, positive=True) for n in
            "a_N d_N h_N K_F a_M d_M h_M K_N a_F d_F h_F K_M".split()}
    system = [sp.numer(sp.together(sp.sympify(e, locals={**pars, "N": N, "M": M, "F": F}))) for e in [
        "a_N - d_N*N + h_N*F**2/(F**2 + K_F**2)",
        "a_M - d_M*M + h_M*N/(N + K_N)",
        "a_F - d_F*F + h_F*M/(M + K_M)"]]
    start = time.perf_counter()
    status, sols, seconds = _solve.solve_with_time_limit(system, [N, M, F], 1)
    assert status == "timeout" and sols is None
    assert time.perf_counter() - start < 10


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="needs SIGALRM")
@pytest.mark.filterwarnings("ignore:The disease-free values")
def test_fallback_without_worker(monkeypatch):
    """If the worker process cannot start, the limit is enforced with SIGALRM in-process."""
    def broken(*args, **kwargs):
        raise OSError("no interpreter")

    monkeypatch.setattr(_solve, "_solve_in_worker", broken)
    monkeypatch.setattr(_solve, "_WORKER_UNAVAILABLE", False)
    start = time.perf_counter()
    with pytest.warns(RuntimeWarning, match="separate Python process"):
        model = R0Model(LOOP, infected=["I", "V"], dfe_timeout=2)
    assert time.perf_counter() - start < 15
    assert model.dfe_is_implicit
    assert model.R0_numeric(VALUES) == pytest.approx(0.5448843161497, rel=1e-8)
