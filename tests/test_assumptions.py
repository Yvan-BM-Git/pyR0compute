"""check_assumptions(): assumptions (A1)-(A5) of van den Driessche & Watmough (2002).

Each violating model breaks exactly one assumption, so the test also checks
that the other four are still reported as holding.
"""

import shutil
import subprocess
import warnings

import pytest
import sympy as sp

from pyr0compute import AssumptionsReport, ModelSpecificationError, R0Model

SEIR = """
    dS/dt = Lambda - beta*S*I - mu*S
    dE/dt = beta*S*I - (sigma + mu)*E
    dI/dt = sigma*E - (gamma + mu)*I
    dR/dt = gamma*I - mu*R
"""

TH_MODEL = """
    dE/dt = lambda - d*E - kappa*E*V
    dI/dt = kappa*E*V - alpha*I - beta*Th*I
    dV/dt = nu*I - mu*V - tau*Th*V
    dTh/dt = b - c*Th + gamma*I*Th
"""

# (A1): new infections given by hand with a factor (1 - k I) that becomes negative
A1_MODEL = """
    dS/dt = Lambda - beta*S*I - mu*S
    dI/dt = beta*S*I*(1 - k*I) - (gamma + mu)*I
"""
# (A2): constant harvesting h of susceptibles, also when S = 0
A2_MODEL = """
    dS/dt = Lambda - mu*S - h - beta*S*I
    dI/dt = beta*S*I - (gamma + mu)*I
"""
# (A4): relapse of recovered individuals R into I, with R declared uninfected
A4_MODEL = """
    dS/dt = Lambda - beta*S*I - mu*S
    dI/dt = beta*S*I - (gamma + mu)*I + eps*R
    dR/dt = gamma*I - (mu + eps)*R
"""
# (A5): strong Allee effect, the DFE S = K is unstable without infection
ALLEE_MODEL = """
    dS/dt = r*S*(S/K - 1) - beta*S*I
    dI/dt = beta*S*I - g*I
"""
# (A5): infected cells proliferate (a I) and it is counted in V: V = g - a
PROLIF_MODEL = """
    dS/dt = L - mu*S - beta*S*I
    dI/dt = beta*S*I + a*I - g*I
"""
PROLIF_VALUES = dict(L=1, mu=0.1, beta=0.5, a=0.5, g=0.2)


def _others_hold(report, failing):
    for key, result in report.results.items():
        if key != failing:
            assert result.status == "holds", (key, result)


@pytest.mark.parametrize("text, infected", [(SEIR, ["E", "I"]), (TH_MODEL, ["I", "V"])])
def test_classical_models_hold_symbolically(no_warnings, text, infected):
    report = R0Model(text, infected=infected).check_assumptions()
    assert isinstance(report, AssumptionsReport)
    assert report.holds
    assert {r.basis for r in report} <= {"symbolic", "construction"}
    assert all(res.status == "holds" for res in report.lemma1.values())


def test_th_model_J4_eigenvalues(no_warnings):
    """J4 = diag(d, c), as in the proof of Theorem 1 of Cuesta-Herrera et al. (2025)."""
    report = R0Model(TH_MODEL, infected=["I", "V"]).check_assumptions()
    tex = report.to_latex("markdown")
    assert "$J_4$: $d > 0$, $c > 0$" in tex


def test_A1_negative_new_infections():
    model = R0Model(A1_MODEL, infected=["I"], new_infections={"I": "beta*S*I*(1 - k*I)"})
    report = model.check_assumptions()
    assert report["A1"].status == "fails"
    _others_hold(report, "A1")
    failing = [it for it in report["A1"].items if it.decision.status == "fails"]
    witness = failing[0].decision.witness
    k, I = (next(s for s in witness if s.name == n) for n in ("k", "I"))
    assert witness[k] * witness[I] > 1  # 1 - k I < 0 at the counterexample
    # at given values it is decided there too
    at = model.check_assumptions(values=dict(Lambda=1, beta=0.5, mu=0.1, gamma=0.2, k=0.1))
    assert at["A1"].status == "fails" and at["A1"].basis == "values"
    # the automatic decomposition puts beta*k*S*I**2 in V^-: (A1) holds
    auto = R0Model(A1_MODEL, infected=["I"]).check_assumptions()
    assert auto.holds


def test_A2_constant_harvesting():
    report = R0Model(A2_MODEL, infected=["I"]).check_assumptions()
    assert report["A2"].status == "fails" and report["A2"].basis == "symbolic"
    _others_hold(report, "A2")
    assert "= h$" in report.to_latex("markdown")


def test_A3_is_enforced_by_construction():
    report = R0Model(SEIR, infected=["E", "I"]).check_assumptions()
    assert report["A3"].status == "holds" and report["A3"].basis == "construction"
    with pytest.raises(ModelSpecificationError, match="uninfected"):
        R0Model(SEIR, infected=["E", "I"], new_infections=["beta*S*I", "beta*S*I", 0, 0])


def test_A4_relapse_into_infection():
    report = R0Model(A4_MODEL, infected=["I"]).check_assumptions()
    assert report["A4"].status == "fails" and report["A4"].basis == "symbolic"
    _others_hold(report, "A4")
    # Lemma 1: dV_I/dR(x0) = -eps is not zero, so the infected block is not closed
    assert report.lemma1["block"].status == "fails"
    # declaring R infected (latent infection) restores the assumptions
    fixed = R0Model(A4_MODEL, infected=["I", "R"]).check_assumptions()
    assert fixed.holds


def test_A5_unstable_DFE_and_alternative():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # two DFEs found
        model = R0Model(ALLEE_MODEL, infected=["I"])
    report = model.check_assumptions()
    assert report["A5"].status == "fails" and report["A5"].basis == "symbolic"
    _others_hold(report, "A5")
    alternatives = {tuple(cand.values()): d.status for _, cand, d in report._alternatives}
    K = sp.Symbol("K")
    assert alternatives == {(K,): "fails", (0,): "holds"}
    assert "dfe_candidates" in report.to_latex("markdown")


def test_A5_V_not_an_M_matrix_depends_on_parameters():
    model = R0Model(PROLIF_MODEL, infected=["I"])
    report = model.check_assumptions()
    assert report["A5"].status == "undecided" and report["A5"].basis == "sampling"
    _others_hold(report, "A5")
    bad = model.check_assumptions(values=PROLIF_VALUES)          # a = 0.5 > g = 0.2
    assert bad["A5"].status == "fails" and bad["A5"].basis == "values"
    assert model.R0_numeric(PROLIF_VALUES) > 0 and float(model.R0.subs(
        {s: PROLIF_VALUES[s.name] for s in model.R0.free_symbols})) < 0  # R0 formula is negative
    good = model.check_assumptions(values=dict(PROLIF_VALUES, a=0.1))
    assert good.holds
    # counting proliferation as new infections: V = g, (A5) holds for all parameters
    other = R0Model(PROLIF_MODEL, infected=["I"], new_infections={"I": "beta*S*I + a*I"})
    assert other.check_assumptions().holds


def test_large_models_with_values():
    from test_nonlinear import IMMUNE_MODEL, IMMUNE_VALUES
    from test_time_limit import LOOP, VALUES

    immune = R0Model(IMMUNE_MODEL, infected=["I", "V"])
    assert immune.check_assumptions()["A5"].status == "undecided"   # stable in samples, not proved
    assert immune.check_assumptions(values=IMMUNE_VALUES).holds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dfe_timeout reached: implicit DFE
        loop = R0Model(LOOP, infected=["I", "V"], dfe_timeout=3)
    assert loop.check_assumptions(values=VALUES).holds


def test_report_formats_and_errors(no_warnings):
    model = R0Model(SEIR, infected=["E", "I"])
    report = model.check_assumptions(language="es")
    md = report.to_latex("markdown")
    assert "**cumple**" in md and "| (A5) |" in md
    assert report._repr_markdown_() == md
    en = report.to_latex("markdown", language="en")
    assert "**holds**" in en
    doc = report.to_latex(standalone=True)
    assert doc.startswith("\\documentclass") and "\\begin{tabular}" in doc
    assert "(A1) holds" in repr(report)
    assert report.status == {"A1": "holds", "A2": "holds", "A3": "holds", "A4": "holds", "A5": "holds"}
    names = [str(x) for x, *_ in report.decomposition]
    assert names == ["S", "E", "I", "R"]
    with pytest.raises(ValueError):
        model.check_assumptions(language="fr")
    with pytest.raises(ValueError):
        report.to_latex(style="html")
    with pytest.raises(ModelSpecificationError, match="Missing values"):
        model.check_assumptions(values={"beta": 1})
    bad = R0Model(A2_MODEL, infected=["I"]).check_assumptions(language="es").to_latex("markdown")
    assert "**no cumple**" in bad and "No se cumple (A2)" in bad


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="needs pdflatex")
def test_latex_report_compiles(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models = [R0Model(SEIR, infected=["E", "I"]), R0Model(A4_MODEL, infected=["I"]),
                  R0Model(ALLEE_MODEL, infected=["I"])]
    for k, model in enumerate(models):
        for lang in ("en", "es"):
            path = tmp_path / f"report_{k}_{lang}.tex"
            path.write_text(model.check_assumptions(language=lang).to_latex(standalone=True))
            run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", path.name],
                                 cwd=tmp_path, capture_output=True, text=True, timeout=120)
            assert run.returncode == 0, run.stdout[-2000:]
