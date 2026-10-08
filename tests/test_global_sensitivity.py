"""Global sensitivity analysis: LHS-PRCC and Sobol indices."""

import warnings

import numpy as np
import pytest
import sympy as sp

pytest.importorskip("scipy")

from pyr0compute import GlobalSensitivityResult, ModelSpecificationError, R0Model  # noqa: E402
from pyr0compute.global_sensitivity import _R0Evaluator, _prcc, make_distribution  # noqa: E402

SEIR = """
dS/dt = Lambda - beta*S*I - mu*S
dE/dt = beta*S*I - (sigma + mu)*E
dI/dt = sigma*E - (gamma + mu)*I
dR/dt = gamma*I - mu*R
"""
SEIR_BASE = {"Lambda": 10, "beta": 0.002, "mu": 0.02, "sigma": 0.2, "gamma": 0.1}

ROSS_MACDONALD = """
dSh/dt = Lh - a*bh*Sh*Iv/Nh - muh*Sh
dIh/dt = a*bh*Sh*Iv/Nh - (gamma + muh)*Ih
dRh/dt = gamma*Ih - muh*Rh
dSv/dt = Lv - a*bv*Sv*Ih/Nh - muv*Sv
dIv/dt = a*bv*Sv*Ih/Nh - muv*Iv
"""


@pytest.fixture(scope="module")
def seir():
    return R0Model(SEIR, infected=["E", "I"])


@pytest.fixture(scope="module")
def ross_macdonald():
    return R0Model(ROSS_MACDONALD, infected=["Ih", "Iv"])


# ------------------------------------------------------------ distributions
def test_make_distribution_forms():
    import scipy.stats as st

    u = make_distribution((1, 3))
    assert u.ppf(0) == pytest.approx(1) and u.ppf(1) == pytest.approx(3)
    lu = make_distribution(("loguniform", 0.01, 1))
    assert lu.ppf(0.5) == pytest.approx(0.1)  # geometric mean
    assert make_distribution(("normal", 2, 0.5)).mean() == pytest.approx(2)
    tn = make_distribution(("truncnormal", 0.02, 0.05, 0, 1))
    assert tn.ppf(0) == pytest.approx(0) and tn.ppf(1) == pytest.approx(1)
    tri = make_distribution(("triangular", 0, 1, 4))
    assert tri.ppf(0) == pytest.approx(0) and tri.ppf(1) == pytest.approx(4)
    frozen = st.gamma(2, scale=0.1)
    assert make_distribution(frozen) is frozen


@pytest.mark.parametrize("spec", [
    ("cauchy", 0, 1), ("uniform", 1), (3, 1), ("loguniform", 0, 1), ("normal", 0, -1),
    ("triangular", 0, 5, 1), "abc",
])
def test_make_distribution_rejects_bad_specs(spec):
    with pytest.raises(ModelSpecificationError):
        make_distribution(spec)


# ---------------------------------------------------------------- evaluator
def test_evaluator_matches_numeric_spectral_radius(ross_macdonald):
    ev = _R0Evaluator(ross_macdonald)
    rng = np.random.default_rng(3)
    n = 50
    cols = [10 ** rng.uniform(-2, 1, n) for _ in ev.names]
    y = ev(cols)
    for i in range(0, n, 10):
        values = {name: c[i] for name, c in zip(ev.names, cols)}
        assert y[i] == pytest.approx(ross_macdonald.R0_numeric(values), rel=1e-9)


def test_evaluator_with_max_of_strains():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = R0Model("""
            dI1/dt = beta1*I1*S - (b + gamma1)*I1 + nu*I1*I2
            dI2/dt = beta2*I2*S - (b + gamma2)*I2 - nu*I1*I2
            dS/dt = b - b*S + gamma1*I1 + gamma2*I2 - (beta1*I1 + beta2*I2)*S
        """, infected=["I1", "I2"])
        assert isinstance(model.R0, sp.Max)
    ev = _R0Evaluator(model)
    rng = np.random.default_rng(0)
    cols = [rng.uniform(0.1, 1, 20) for _ in ev.names]
    v = dict(zip(ev.names, cols))
    expected = np.maximum(v["beta1"] / (v["b"] + v["gamma1"]), v["beta2"] / (v["b"] + v["gamma2"]))
    np.testing.assert_allclose(ev(cols), expected)


def test_numeric_fallback_without_closed_form(seir, monkeypatch):
    """Without a closed-form R0 the spectral radius of K is used, with the same values."""
    symbolic = seir.prcc(baseline=SEIR_BASE, spread=0.3, n=300, seed=4)
    from pyr0compute.exceptions import NextGenerationError

    def no_closed_form(self):
        raise NextGenerationError("no closed form")

    monkeypatch.setattr(R0Model, "R0", property(no_closed_form))
    numeric = seir.prcc(baseline=SEIR_BASE, spread=0.3, n=300, seed=4)
    np.testing.assert_allclose(numeric.output, symbolic.output, rtol=1e-9)
    np.testing.assert_allclose(numeric.indices["PRCC"], symbolic.indices["PRCC"], rtol=1e-9)
    assert any("no closed form" in n for n in numeric.notes)
    assert all(v is None for v in numeric.monotonic.values())


# --------------------------------------------------------------------- PRCC
def test_prcc_formula_matches_residual_regression():
    """PRCC via the inverse correlation matrix = correlation of rank residuals."""
    from scipy.stats import rankdata

    rng = np.random.default_rng(1)
    X = rng.uniform(size=(400, 4))
    y = np.exp(2 * X[:, 0]) - X[:, 1] ** 3 + 0.3 * X[:, 2] + 0.1 * rng.normal(size=400)
    R = rankdata(np.column_stack([X, y]), axis=0)
    expected = []
    for i in range(4):
        others = np.column_stack([np.ones(400), np.delete(R[:, :4], i, axis=1)])
        rx = R[:, i] - others @ np.linalg.lstsq(others, R[:, i], rcond=None)[0]
        ry = R[:, 4] - others @ np.linalg.lstsq(others, R[:, 4], rcond=None)[0]
        expected.append(np.corrcoef(rx, ry)[0, 1])
    np.testing.assert_allclose(_prcc(X, y), expected, atol=1e-10)


def test_prcc_signs_follow_the_derivatives(seir, no_warnings):
    res = seir.prcc(baseline=SEIR_BASE, spread=0.3, n=2000, seed=1)
    assert isinstance(res, GlobalSensitivityResult)
    r = {k: v["PRCC"] for k, v in res.as_dict().items()}
    assert r["beta"] > 0.8 and r["Lambda"] > 0.8 and r["sigma"] > 0
    assert r["mu"] < -0.8 and r["gamma"] < -0.8
    assert all(res[p]["p_value"] < 0.001 for p in ("beta", "Lambda", "mu", "gamma"))
    assert all(res.monotonic.values())
    assert set(res.ranking()[:4]) == {"beta", "Lambda", "mu", "gamma"}
    assert res.samples.shape == (2000, 5) and res.output.shape == (2000,)


def test_prcc_is_reproducible_with_seed(seir):
    a = seir.prcc(baseline=SEIR_BASE, spread=0.2, n=200, seed=7)
    b = seir.prcc(baseline=SEIR_BASE, spread=0.2, n=200, seed=7)
    np.testing.assert_array_equal(a.indices["PRCC"], b.indices["PRCC"])


def test_irrelevant_parameter_has_small_prcc():
    """A parameter that does not affect R0 is not even sampled."""
    model = R0Model("""
        dS/dt = Lambda - beta*S*I - mu*S
        dI/dt = beta*S*I - (gamma + mu)*I
        dR/dt = gamma*I - (mu + omega)*R
    """, infected=["I"])
    with pytest.warns(UserWarning, match="do not appear in R0"):
        res = model.prcc({"beta": (1, 2), "omega": (0, 1)},
                         fixed={"Lambda": 1, "mu": 0.1, "gamma": 0.2}, n=100, seed=0)
    assert [p.name for p in res.parameters] == ["beta"]
    assert res["beta"]["PRCC"] == pytest.approx(1.0)


def test_non_monotone_parameter_is_flagged():
    model = R0Model("""
        dS/dt = Lambda - beta*a*(1 - a)*S*I - mu*S
        dI/dt = beta*a*(1 - a)*S*I - (gamma + mu)*I
    """, infected=["I"])
    with pytest.warns(UserWarning, match="not monotone"):
        res = model.prcc({"a": (0.01, 0.99), "beta": (1, 2)},
                         fixed={"Lambda": 1, "mu": 0.1, "gamma": 0.2}, n=1000, seed=0)
    assert res.monotonic == {"a": False, "beta": True}
    # R0 depends strongly on a, but symmetrically: PRCC misses it, Sobol does not
    assert abs(res["a"]["PRCC"]) < 0.15
    sob = model.sobol_indices({"a": (0.01, 0.99), "beta": (1, 2)},
                              fixed={"Lambda": 1, "mu": 0.1, "gamma": 0.2},
                              n=2 ** 12, seed=0, confidence_level=None)
    assert sob["a"]["ST"] > 0.5


def test_fixed_and_baseline_semantics(seir):
    res = seir.prcc({"beta": (0.001, 0.003)}, baseline=SEIR_BASE, n=100, seed=0)
    assert [p.name for p in res.parameters] == ["beta"]
    assert res.fixed == {k: float(v) for k, v in SEIR_BASE.items() if k != "beta"}
    res = seir.prcc(baseline=SEIR_BASE, spread=0.1, fixed={"sigma": 0.2}, n=100, seed=0)
    assert "sigma" not in [p.name for p in res.parameters]
    assert res.distributions["mu"].startswith("uniform [0.018, 0.022")


@pytest.mark.parametrize("kwargs, match", [
    ({"distributions": {"beta": (1, 2)}}, "No distribution or value"),
    ({"baseline": SEIR_BASE, "spread": 1.5}, "spread"),
    ({"baseline": SEIR_BASE}, "At least one parameter must vary"),
    ({"distributions": {"zeta": (1, 2)}, "baseline": SEIR_BASE, "spread": 0.1}, "not a symbol"),
    ({"distributions": {"beta": (1, 2)}, "fixed": {"beta": 1}, "baseline": SEIR_BASE},
     "both a distribution and a fixed value"),
    ({"baseline": SEIR_BASE, "spread": 0.1, "n": 5}, "too small"),
])
def test_prcc_input_errors(seir, kwargs, match):
    with pytest.raises(ModelSpecificationError, match=match):
        seir.prcc(**kwargs)


def test_invalid_samples_are_dropped_with_warning(ross_macdonald):
    dists = {"a": ("normal", 0.3, 0.3)}
    fixed = {"bh": 0.5, "bv": 0.4, "Lh": 100, "Nh": 5000, "gamma": 0.1, "muh": 0.02,
             "muv": 0.1, "Lv": -5000}  # sqrt of a negative number for every sample
    with pytest.raises(ModelSpecificationError, match="Too few valid samples"):
        with pytest.warns(UserWarning, match="non-finite"):
            ross_macdonald.prcc(dists, fixed=fixed, n=50, seed=0)


# -------------------------------------------------------------------- Sobol
def test_sobol_matches_analytic_indices_for_product_form(ross_macdonald):
    """log R0 = sum a_i log p_i + c, so S1 = ST = a_i^2 Var(log p_i) / sum_j a_j^2 Var(log p_j).

    With gamma and muh fixed, the Ross-Macdonald R0 is a monomial with exponents
    a = 1 for a, 1/2 for bh, bv, Lh, Lv and -1 for muv and Nh, which
    also checks the square-root convention.
    """
    ranges = {"a": (0.1, 0.5), "bh": (0.1, 0.9), "bv": (0.1, 0.9), "Lh": (50, 200),
              "Lv": (1000, 10000), "muv": (0.05, 0.2), "Nh": (2000, 8000)}
    exponents = {"a": 1, "bh": 0.5, "bv": 0.5, "Lh": 0.5, "Lv": 0.5, "muv": -1, "Nh": -1}
    weights = {p: exponents[p] ** 2 * np.log(hi / lo) ** 2 / 12 for p, (lo, hi) in ranges.items()}
    total = sum(weights.values())

    res = ross_macdonald.sobol_indices(
        {p: ("loguniform",) + r for p, r in ranges.items()},
        fixed={"gamma": 0.1, "muh": 0.02}, n=2 ** 13, seed=2, log_output=True)
    assert res.output_name == "log R0"
    assert res.n_evaluations == 2 ** 13 * (7 + 2)
    for p in ranges:
        exact = weights[p] / total
        assert res[p]["S1"] == pytest.approx(exact, abs=0.03)
        assert res[p]["ST"] == pytest.approx(exact, abs=0.03)
        assert res[p]["ST_low"] <= exact + 0.01 and exact - 0.01 <= res[p]["ST_high"]
    assert sum(res.indices["S1"]) == pytest.approx(1, abs=0.05)


def test_sobol_detects_interactions(seir):
    res = seir.sobol_indices(baseline=SEIR_BASE, spread=0.9, n=2 ** 12, seed=1,
                             confidence_level=None)
    assert "S1_low" not in res.indices
    interaction = res.indices["ST"] - res.indices["S1"]
    # R0 is a product: wide ranges create interactions between the factors
    assert interaction.max() > 0.05
    assert all(res.indices["ST"] >= res.indices["S1"] - 0.02)


@pytest.mark.parametrize("n", [1000, 1])
def test_sobol_needs_power_of_two(seir, n):
    with pytest.raises(ModelSpecificationError, match="power of 2"):
        seir.sobol_indices(baseline=SEIR_BASE, spread=0.2, n=n)


def test_sobol_rejects_inadmissible_values(seir):
    with pytest.raises(ModelSpecificationError, match="non-finite log R0"):
        seir.sobol_indices({"mu": ("normal", 0.02, 0.05)}, baseline=SEIR_BASE,
                           n=256, seed=0, log_output=True)


def test_global_sensitivity_dispatch(seir):
    a = seir.global_sensitivity(method="prcc", baseline=SEIR_BASE, spread=0.2, n=100, seed=0)
    assert a.method == "LHS-PRCC"
    b = seir.global_sensitivity(method="sobol", baseline=SEIR_BASE, spread=0.2, n=256, seed=0,
                                confidence_level=None)
    assert b.method == "Sobol"
    with pytest.raises(ValueError):
        seir.global_sensitivity(method="morris")


# -------------------------------------------------------------------- output
def test_result_output_formats(seir):
    res = seir.prcc(baseline=SEIR_BASE, spread=0.3, n=500, seed=0)
    tex = res.to_latex()
    assert "\\begin{tabular}" in tex and "\\mu" in tex and "<0.001" in tex
    md = res.to_latex("markdown")
    assert md.count("\n|") == 1 + 1 + 5  # header, rule, five parameters
    assert "LHS-PRCC" in repr(res) and "monotonic" in repr(res)
    with pytest.raises(KeyError):
        res["zeta"]
    with pytest.raises(ValueError):
        res.to_latex("html")

    sob = seir.sobol_indices(baseline=SEIR_BASE, spread=0.3, n=512, seed=0, n_resamples=99)
    assert "IC 95\\% $S_1$" in sob.to_latex()
    assert "[" in repr(sob)


def test_result_dataframe_and_plot(seir):
    pd = pytest.importorskip("pandas")
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    res = seir.prcc(baseline=SEIR_BASE, spread=0.3, n=500, seed=0)
    df = res.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == res.ranking()
    assert {"PRCC", "p_value", "monotonic", "distribution"} <= set(df.columns)
    ax = res.plot()
    assert len(ax.patches) == 5
    sob = seir.sobol_indices(baseline=SEIR_BASE, spread=0.3, n=512, seed=0, n_resamples=99)
    ax = sob.plot()
    assert len(ax.patches) == 10
    import matplotlib.pyplot as plt
    plt.close("all")
