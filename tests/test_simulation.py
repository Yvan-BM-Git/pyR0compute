"""simulate(): numerical solution of the model, random parameters and plots."""

import numpy as np
import pytest

from pyr0compute import ModelSpecificationError, R0Model, SimulationResult

pytest.importorskip("scipy")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SIR = """
    dS/dt = Lambda - beta*S*I - mu*S
    dI/dt = beta*S*I - (gamma + mu)*I
    dR/dt = gamma*I - mu*R
"""
BASE = dict(Lambda=10, mu=0.1, gamma=0.5)


@pytest.fixture(scope="module")
def sir():
    return R0Model(SIR, infected=["I"])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_R0_below_one_dies_out(sir):
    res = sir.simulate((0, 300), values=dict(BASE, beta=0.002))
    assert isinstance(res, SimulationResult) and len(res) == 1
    assert res.R0 == pytest.approx(10 * 0.002 / (0.1 * 0.6))
    assert res.R0 < 1
    # starts at the DFE with a small perturbation of the infected compartment
    assert res.initial == pytest.approx({"S": 100.0, "I": 1.0, "R": 0.0})
    assert res.final()["I"] < 1e-8
    assert res.final()["S"] == pytest.approx(100.0, rel=1e-6)


def test_R0_above_one_reaches_the_endemic_equilibrium(sir):
    beta = 0.01
    res = sir.simulate((0, 400), values=dict(BASE, beta=beta))
    assert res.R0 > 1
    S_end = (BASE["gamma"] + BASE["mu"]) / beta
    I_end = BASE["Lambda"] / (BASE["gamma"] + BASE["mu"]) - BASE["mu"] / beta
    assert res.final()["S"] == pytest.approx(S_end, rel=1e-5)
    assert res.final()["I"] == pytest.approx(I_end, rel=1e-5)
    assert res["I"].shape == res.t.shape


def test_random_parameters_are_reproducible_and_fixed_ones_respected(sir):
    a = sir.simulate((0, 20), values={"Lambda": 10}, seed=3, n_runs=4)
    b = sir.simulate((0, 20), values={"Lambda": 10}, seed=3, n_runs=4)
    assert [r.parameters for r in a.runs] == [r.parameters for r in b.runs]
    assert all(r.parameters["Lambda"] == 10 for r in a.runs)
    for r in a.runs:
        for name in ("beta", "gamma", "mu"):
            assert 0.01 <= r.parameters[name] <= 1.0           # default_range
    c = sir.simulate((0, 20), values=BASE, ranges={"beta": (0.001, 0.002)}, sampling="uniform",
                     n_runs=5, seed=0)
    assert all(0.001 <= r.parameters["beta"] <= 0.002 for r in c.runs)
    R0 = c.R0_values
    assert np.allclose(R0, [10 * r.parameters["beta"] / 0.06 for r in c.runs])


@pytest.mark.parametrize("R0_range, check", [((1, None), lambda r: r > 1), ((None, 1), lambda r: r < 1),
                                             ((2, 3), lambda r: 2 <= r <= 3)])
def test_R0_range(sir, R0_range, check):
    res = sir.simulate((0, 10), values=BASE, ranges={"beta": (0.001, 0.05)}, R0_range=R0_range,
                       n_runs=6, seed=1)
    assert all(check(r) for r in res.R0_values)


def test_R0_range_errors(sir):
    with pytest.raises(ModelSpecificationError, match="outside R0_range"):
        sir.simulate((0, 10), values=dict(BASE, beta=0.002), R0_range=(1, None))
    with pytest.raises(ModelSpecificationError, match="No parameter draw"):
        sir.simulate((0, 10), values=BASE, ranges={"beta": (0.001, 0.002)}, R0_range=(5, None),
                     max_tries=50)


def test_initial_conditions_and_validation(sir):
    res = sir.simulate((0, 5), values=dict(BASE, beta=0.01), initial={"I": 5, "S": 50})
    assert res.initial == {"S": 50.0, "I": 5.0, "R": 0.0}
    assert res["S"][0] == pytest.approx(50)
    with pytest.raises(ModelSpecificationError, match="not a parameter"):
        sir.simulate(values={"delta": 1})
    with pytest.raises(ModelSpecificationError, match="not a state variable"):
        sir.simulate(values=BASE, initial={"X": 1})
    with pytest.raises(ValueError):
        sir.simulate((5, 1))
    with pytest.raises(KeyError):
        res["X"]


def test_within_host_model_and_tables():
    tiv = R0Model("""
        dT/dt = s - d*T - beta*T*V
        dI/dt = beta*T*V - delta*I
        dV/dt = p*I - c*V
    """, infected=["I", "V"])
    values = dict(s=10, d=0.1, beta=1e-3, delta=1, p=50, c=5)        # R0 = 1
    up = tiv.simulate((0, 300), values=dict(values, p=100))
    down = tiv.simulate((0, 300), values=dict(values, p=25))
    assert up.R0 == pytest.approx(2) and down.R0 == pytest.approx(0.5)
    assert up.final()["V"] > 1 and down.final()["V"] < 1e-6
    df = up.to_dataframe()
    assert list(df.columns) == ["run", "t", "T", "I", "V", "R0"]
    assert down.summary().loc[0, "R0"] == pytest.approx(0.5)


def test_plot_options_and_saving(sir, tmp_path):
    one = sir.simulate((0, 100), values=dict(BASE, beta=0.01))
    fig, ax = one.plot(["S", "I"], title="SIR, R0 = {R0}", xlabel="días", ylabel="individuos",
                       labels={"S": "susceptibles", "I": "infectados"},
                       colors=["#2a78d6", "#e34948"], linestyles={"I": "--"}, linewidth=2.5,
                       figsize=(7, 3), logy=True, ylim=(1e-2, 200), show_dfe=True,
                       save=tmp_path / "out" / "sir.png")
    assert (tmp_path / "out" / "sir.png").exists()
    assert ax.get_title(loc="left") == "SIR, R0 = 1.67"
    assert ax.get_xlabel() == "días" and ax.get_ylabel() == "individuos"
    assert ax.get_yscale() == "log"
    colors = {matplotlib.colors.to_hex(line.get_color()) for line in ax.get_lines()}
    assert {"#2a78d6", "#e34948"} <= colors
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["susceptibles", "infectados"]
    assert tuple(fig.get_size_inches()) == (7, 3)

    ens = sir.simulate((0, 100), values=BASE, ranges={"beta": (0.001, 0.02)}, n_runs=12, seed=2)
    fig, axes = ens.plot(["S", "I"], subplots=True, ncols=2, titles={"I": "Infectados"})
    assert len(axes) == 2 and axes[1].get_title(loc="left") == "Infectados"
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert set(texts) == {r"$\mathcal{R}_0 < 1$", r"$\mathcal{R}_0 > 1$"}
    fig, ax = ens.plot("I", color_by_R0=False, legend=False)
    assert ax.get_legend() is None
    fig, ax = ens.plot_phase("S", "I", title="Plano de fase", save=tmp_path / "fase.png")
    assert (tmp_path / "fase.png").exists() and ax.get_title(loc="left") == "Plano de fase"
    assert "DFE" in [t.get_text() for t in ax.get_legend().get_texts()]
    # draw two simulations on the same axes
    fig, ax = plt.subplots()
    one.plot("I", ax=ax, colors="#e34948", labels="sin vacuna", legend=False)
    sir.simulate((0, 100), values=dict(BASE, beta=0.007)).plot("I", ax=ax, labels="con vacuna", legend=False)
    assert len(ax.get_lines()) == 2
    assert [t.get_text() for t in ax.legend().get_texts()] == ["sin vacuna", "con vacuna"]


def test_implicit_dfe_model_can_be_simulated():
    import warnings

    from test_time_limit import LOOP, VALUES
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loop = R0Model(LOOP, infected=["I", "V"], dfe_timeout=2)
    low = loop.simulate((0, 400), values=VALUES)                       # R0 = 0.54
    high = loop.simulate((0, 400), values=dict(VALUES, beta=0.01))     # R0 = 2.72
    assert low.R0 < 1 < high.R0
    assert low.final()["V"] < 1e-6 and high.final()["V"] > 1
    assert low.initial["N"] == pytest.approx(5.314824, rel=1e-6)     # DFE of the implicit block
