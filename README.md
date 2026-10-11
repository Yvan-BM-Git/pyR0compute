# pyR0compute

Symbolic computation of the basic reproduction number $R_0$ of compartmental ODE models with the next-generation matrix method (van den Driessche and Watmough, 2002).

Write the model, name the infected compartments and pyR0compute does the rest: every symbol that is not a state variable is treated as a parameter, the new-infection terms $\mathcal{F}$ and transition terms $\mathcal{V}$ are identified, the disease-free equilibrium (DFE) is solved and $R_0 = \rho(FV^{-1})$ is returned as a SymPy expression.

[![PyPI](https://img.shields.io/pypi/v/pyR0compute)](https://pypi.org/project/pyR0compute/)
[![Python](https://img.shields.io/pypi/pyversions/pyR0compute)](https://pypi.org/project/pyR0compute/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Yvan-BM-Git/pyR0compute/blob/main/LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yvan-BM-Git/pyR0compute/blob/main/examples/01_examples.ipynb)
[![DOI](https://zenodo.org/badge/1033480469.svg)](https://doi.org/10.5281/zenodo.23298227)

*Versión en español: [README_es.md](README_es.md).*

## Installation

```bash
pip install pyR0compute
```

In a Jupyter or Colab notebook use `%pip install pyR0compute`. The development version can be installed from GitHub with `pip install git+https://github.com/Yvan-BM-Git/pyR0compute`.

Requires Python ≥ 3.9. `pip install pyR0compute` also installs its dependencies: SymPy, NumPy, SciPy (global sensitivity, numerical DFE and simulations), matplotlib (plots) and pandas (result tables).

## Quick start

```python
from pyr0compute import R0Model

model = R0Model("""
    dS/dt = Lambda - beta*S*I - mu*S
    dE/dt = beta*S*I - (sigma + mu)*E
    dI/dt = sigma*E - (gamma + mu)*I
    dR/dt = gamma*I - mu*R
""", infected=["E", "I"])

model.parameters   # (Lambda, beta, gamma, mu, sigma)  <- detected automatically
model.dfe          # {S: Lambda/mu, E: 0, I: 0, R: 0}
model.R0           # Lambda*beta*sigma/(mu*(gamma + mu)*(mu + sigma))
```

The equations can be written in any order; the infected compartments do not need to come first.

### Three ways to enter a model

```python
# 1. Text: one line "dX/dt = ..." (or "X' = ...") per variable.
#    Auxiliary lines such as "N = S + I + R" are substituted.
R0Model("""
    N = S + I + R
    dS/dt = Lambda - beta*S*I/N - mu*S
    dI/dt = beta*S*I/N - (gamma + mu)*I
    dR/dt = gamma*I - mu*R
""", infected=["I"])

# 2. A dictionary {variable: right-hand side} (text or SymPy expressions)
R0Model({"S": "Lambda - beta*S*I - mu*S",
         "I": "beta*S*I - (gamma + mu)*I"}, infected=["I"])

# 3. SymPy symbols
import sympy as sp
S, I = sp.symbols("S I")
beta, gamma, mu, Lam = sp.symbols("beta gamma mu Lambda")
R0Model({S: Lam - beta*S*I - mu*S, I: beta*S*I - (gamma + mu)*I}, infected=[I])
```

Use subscripts in the names to get clean LaTeX: `mu_h`, `alpha_hv` and `Delta_h` are written $\mu_h$, $\alpha_{hv}$ and $\Delta_h$. Names that are special in SymPy (`I`, `S`, `E`, `N`, `beta`, `gamma`, `Lambda`, even `lambda`) are ordinary symbols inside text models, and `^` means power.

### What you get

| Attribute / method | Content |
|---|---|
| `R0` | Basic reproduction number (spectral radius of $FV^{-1}$) |
| `parameters` | Automatically detected parameters |
| `new_infections`, `transitions` | Terms $\mathcal{F}_i$ and $\mathcal{V}_i$ of each infected compartment |
| `dfe`, `dfe_candidates` | Disease-free equilibrium used, and all non-negative equilibria found |
| `F`, `V`, `K` | Jacobians at the DFE and next-generation matrix $K = FV^{-1}$ |
| `next_generation_matrix_small` | $K$ restricted to the compartments that receive new infections |
| `eigenvalues` | Eigenvalues of $K$ |
| `R0_compact`, `dfe_compact`, `dfe_definitions` | $R_0$ and the DFE written with the values $X^*$ that have no short closed form, and their definitions in solving order (see *Systems with a nonlinear DFE*) |
| `R0_numeric(values)`, `dfe_numeric(values)` | Numerical spectral radius and DFE (for large models, or a DFE without closed form) |
| `dfe_blocks`, `dfe_is_implicit` | How each block of the DFE was solved (closed form, no closed form or `dfe_timeout` time limit) and how long it took |
| `simulate(t_span, values=None, initial=None, n_runs=1, ...)` | Numerical simulation of the ODE with fixed or random parameters and editable plots (see *Simulation*) |
| `check_assumptions(values=None, language="en")` | Checks assumptions (A1)-(A5) of van den Driessche and Watmough (2002) and returns a LaTeX report (see *Assumptions of van den Driessche and Watmough*) |
| `dfe_stability(values=None)` | Condition (A5) of van den Driessche and Watmough: stability of the DFE in the absence of infection |
| `sensitivity_indices(values=None)` | Normalized (local) sensitivity indices $\Upsilon_p = \frac{\partial R_0}{\partial p}\frac{p}{R_0}$ |
| `prcc(distributions, n, ...)` | Global sensitivity by Latin hypercube sampling and partial rank correlation coefficients (LHS-PRCC) |
| `sobol_indices(distributions, n, ...)` | First-order $S_1$ and total $S_T$ Sobol indices, with bootstrap confidence intervals |
| `report()` | Step-by-step description of the whole computation |
| `report_latex(style, standalone, mat_str)` | The same report in LaTeX (`"document"`, compilable with `standalone=True`) or in Markdown for notebooks (`"markdown"`) |
| `latex()` | LaTeX code of $R_0$ |

### Global sensitivity

The index $\Upsilon_p$ is local: it depends on the point where it is evaluated and captures neither parameter uncertainty nor interactions. `prcc` and `sobol_indices` explore the whole parameter space. Since $R_0$ is in closed form, it is evaluated vectorized with NumPy and $10^5$ evaluations take seconds; if the eigenvalues have no closed form, the numerical spectral radius of $K$ is used.

```python
dists = {
    "beta":  ("loguniform", 1e-3, 4e-3),   # log p uniform
    "mu":    ("truncnormal", 0.02, 0.005, 0.005, 0.05),
    "gamma": (0.05, 0.2),                  # uniform
}
res = model.prcc(dists, fixed={"Lambda": 10, "sigma": 0.2}, n=2000, seed=1)
res = model.prcc(baseline={"Lambda": 10, "beta": 0.002, "mu": 0.02, "sigma": 0.2, "gamma": 0.1},
                 spread=0.3, n=2000, seed=1)       # all uniform within ±30 %
sob = model.sobol_indices(dists, fixed={"Lambda": 10, "sigma": 0.2}, n=4096, seed=1)

res.to_dataframe()           # table sorted by influence (requires pandas)
res.to_latex()               # LaTeX table ("markdown" for notebooks)
sob.plot()                   # S1 and ST with intervals (requires matplotlib)
```

* **LHS-PRCC** (Marino et al., 2008) measures the strength and sign of a **monotone** relation. The library checks monotonicity with the sign of $\partial R_0/\partial p$ in each sample and warns when it does not hold.
* **Sobol** (Saltelli et al., 2010) decomposes the variance of $R_0$; $S_T - S_1$ quantifies interactions. It uses $n \cdot (k + 2)$ evaluations, with $n$ a power of 2, and assumes independent parameters.
* With `log_output=True`, $\log R_0$ is analysed. If $R_0 = c\prod p_i^{a_i}$ and the parameters are independent and log-uniform, $S_{1,i} = S_{T,i} = a_i^2\,\mathrm{Var}(\log p_i) / \sum_j a_j^2\,\mathrm{Var}(\log p_j)$, which links the local index $\Upsilon_{p_i} = a_i$ with the global one. This result is used as a validation test.

Distributions: `(low, high)` or `("uniform", low, high)`, `("loguniform", low, high)`, `("normal", mean, sd)`, `("truncnormal", mean, sd, low, high)`, `("triangular", low, mode, high)` or any frozen `scipy.stats` distribution.

### Systems with a nonlinear DFE

In within-host models with an immune response, the disease-free equilibrium can be a coupled nonlinear system (for example, helper $H$ and regulatory $R$ T lymphocytes that regulate each other, and cells $C$, $B$ activated by both). Solving it at once with `sympy.solve` may never finish. The library:

1. With the infected compartments at zero, splits the equations of the uninfected compartments into the strongly connected components of their dependency graph and solves them in topological order, for example $[E] \to [H, R] \to [C] \to [B]$.
2. Values already solved enter the following blocks as symbols $X^*$, not as their explicit expressions. Short values without radicals (such as $E^* = g_E/d_E$) are written out in full.
3. The Jacobians $F$ and $V$ are evaluated at this compact DFE, so $R_0$ is simplified while it is still small. `model.R0_compact` returns it in that form and `model.R0` with the explicit values substituted (without simplification).
4. A block without closed form (for example the root of a quintic) stays implicit: `dfe_numeric` and `R0_numeric` solve it numerically (requires SciPy).
5. Each coupled block, or block of degree greater than 2, is solved in a separate process with a time limit, `dfe_timeout` (20 s by default). If SymPy does not finish in time, the process is stopped, the block stays implicit and a warning is shown: the model is still built and $R_0$ is returned in compact and numerical form. `model.dfe_blocks` reports how each block was solved and how long it took; `dfe_timeout=None` removes the limit.
6. Global sensitivity evaluates $R_0$ as a chain ($X^*$ in order and then $R_0$), with derivatives by the chain rule, instead of the explicit expression.

```python
model = R0Model('''
    dE/dt = g_E - (d_E + tau_E*V/(V + z_VE))*E
    dI/dt = tau_E*V*E/(V + z_VE) - (d_I + tau_I*C/(C + z_CI))*I
    dV/dt = nu*I - (d_V + tau_V*B/(B + z_BV))*V
    dH/dt = g_H - (d_H - tau_H*I/(I + z_IH) + rho_H*R/(R + z_RH))*H
    dC/dt = g_C - (d_C - tau_C*H/(H + z_HC) + rho_C*R/(R + z_RC))*C
    dB/dt = g_B - (d_B - tau_B*H/(H + z_HB) + rho_B*R/(R + z_RB))*B
    dR/dt = g_R - (d_R - tau_R*H/(H + z_HR) + rho_R)*R
''', infected=["I", "V"])

model.R0_compact       # g_E*nu*tau_E*(B_star + z_BV)*(C_star + z_CI)/(d_E*z_VE*(...)*(...))
model.dfe_definitions  # {H_star: ..., R_star: ..., C_star: ..., B_star: ...}
model.latex()          # uses X^{*} when the DFE has compact values
model.dfe_stability(values)   # {"eigenvalues": ..., "stable": True, "dfe": {...}}
```

The symbols $X^*$ are called `X_star` in text (so that `S_star*beta` is not read as a power) and are written $X^{*}$ in LaTeX. The notebook `examples/03_nonlinear_systems.ipynb` develops this model in full.

Condition (A5) of van den Driessche and Watmough requires the DFE to be stable in the absence of infection; without it $R_0$ is not a threshold. `dfe_stability()` checks it at a point (`values=...`) or on random parameter samples, and returns the symbolic eigenvalues when the uninfected block is triangular.

### Assumptions of van den Driessche and Watmough

Theorem 2 of van den Driessche and Watmough (2002) guarantees that $R_0 = \rho(FV^{-1})$ is a threshold (the DFE is locally asymptotically stable if $R_0 < 1$ and unstable if $R_0 > 1$) only if assumptions (A1)-(A5) hold: non-negative flows, no outflow from an empty compartment, no new infections in uninfected compartments, invariant infection-free subspace, and a DFE that is stable in the absence of new infections. `check_assumptions()` decides each one as *holds*, *fails* or *undecided*, with its basis: by construction, symbolic proof for every positive parameter, at the values given with `values=`, or random samples (a violation in every sample is *fails*; an assumption that could not be proved is *undecided*).

```python
report = model.check_assumptions()      # language="es" for Spanish
report.holds                     # True if the five hold
report.status                    # {"A1": "holds", ..., "A5": "undecided"}
report["A5"].basis               # "symbolic", "values", "sampling" or "construction"
report.to_latex(standalone=True) # compilable LaTeX report; "markdown" for notebooks
model.check_assumptions(values={...})   # decides at a point what could not be proved
```

The report includes the decomposition $\mathcal{F}_i$, $\mathcal{V}_i^+$, $\mathcal{V}_i^-$ used, the details of each assumption with counterexamples when it fails, the consequences of Lemma 1 ($F \ge 0$, $V$ with Z sign pattern and a nonsingular M-matrix) and, if the chosen DFE is not stable, the other DFEs found that are. In Jupyter, evaluating the report shows it in Markdown. The notebook `examples/05_vdw_assumptions.ipynb` presents a model that satisfies the five assumptions and models that violate each one.

### Simulation

`simulate()` integrates the ODE (SciPy) and `plot()` / `plot_phase()` plot the result (matplotlib). The parameters given in `values` are fixed and the others are drawn at random (log-uniform in `default_range`, or in `ranges={...}`). By default the uninfected compartments start at the DFE and the infected ones at a small perturbation, which is the situation described by $R_0$.

```python
res = model.simulate((0, 200), values={"Lambda": 10, "mu": 0.1, "gamma": 0.5, "beta": 0.01})
res.R0, res.final()                          # R0 of the simulation and final state
res.plot(["S", "I"], title="SIR with R0 = {R0}", xlabel="days", labels={"I": "infected"},
         colors={"I": "#e34948"}, linestyles={"I": "--"}, logy=True, figsize=(7, 3.5),
         show_dfe=True, save="outputs/sir.png")

ens = model.simulate((0, 200), values={"Lambda": 10, "mu": 0.1, "gamma": 0.5},
                     ranges={"beta": (0.001, 0.02)}, n_runs=40, seed=1)   # random beta
ens.plot(["S", "I"], subplots=True)          # each simulation colored by R0 < 1 or R0 > 1
ens.plot_phase("S", "I")                      # phase plane with the DFE
model.simulate(..., R0_range=(1, None))       # only draws with R0 > 1
ens.summary(); ens.to_dataframe()             # parameters and R0 of each simulation; long-format data
```

`plot()` accepts `variables`, `title` (`"{R0}"` is replaced by its value), `titles` per panel, `xlabel`, `ylabel`, `labels`, `colors` and `linestyles` (dictionary, list or a single value), `linewidth`, `alpha`, `legend`, `legend_kw`, `figsize`, `subplots`, `ncols`, `logy`, `logx`, `xlim`, `ylim`, `grid`, `color_by_R0`, `show_dfe`, `ax` (to compare scenarios on the same axes), `save` and `dpi`. The notebook `examples/06_simulations.ipynb` shows cases with $R_0 < 1$ and $R_0 > 1$ in epidemiological and within-host models.

### When the automatic choices need help

* **Closed populations** (no births), for example the classical SIR model: the DFE is not unique, so you must give it with `dfe={"S": "N"}`. The error message states which value is missing.
* **Several DFEs** (for example, vector populations with logistic growth): the one with the most nonzero compartments is used and a warning is shown. You can choose another one with `dfe={...}`; all of them are in `model.dfe_candidates`.
* **Custom decompositions**: the $\mathcal{F}$/$\mathcal{V}$ split is not unique. You can replace the automatic one with `new_infections={"E": "beta*S*I/N"}`.

### How new infections are detected

In the equation of an infected compartment, a positive term is a new infection when it involves an infected compartment and also (i) is lost from an uninfected compartment (a transfer such as $S \to E$), or (ii) involves an uninfected compartment and is not a transfer between infected compartments. Progression ($E \to I$), treatment failure, superinfection between strains, deaths and recoveries go to $\mathcal{V}$. `model.report()` shows the classification of each term and its justification.

## Validation

The test suite (`pytest`) reproduces known results: SIR (with and without vital dynamics, with mass action and with frequency-dependent incidence), SEIR, a vaccination model, Ross-Macdonald, a host-vector SEIR/SEI model with human-to-human transmission and logistic vectors, the treatment model of van den Driessche and Watmough (2002, §4.1), within-host models of target cells, infected cells and virus, a two-strain model with superinfection, the helper T lymphocyte model of Cuesta-Herrera et al. (2025, Eq. 2.4 and the values of Figure 3) and a 7-equation immune model with a nonlinear DFE, whose DFE and $R_0$ are checked against numerical integration of the system and whose threshold is checked against the stability of the DFE of the full system. `check_assumptions()` is tested with models that satisfy the five assumptions and with models that violate each one: negative new infections (A1), constant harvest (A2), new infections assigned to an uninfected compartment (A3, rejected by construction), relapse from a compartment declared uninfected (A4), and a DFE made unstable by an Allee effect or a matrix $V$ that is not an M-matrix (A5); the LaTeX reports are compiled with `pdflatex`. The Sobol indices are checked against their analytical value in a product-form model and the PRCC against its definition by regression of residuals. The symbolic results are also checked against the numerical spectral radius.

The notebook `examples/07_validation.ipynb` validates $R_0$ against 15 published cases (see *Examples*).

## Previous interface

Code written for the original notebook still works:

```python
from pyr0compute import GeneralEpidemiologicalModel
model = GeneralEpidemiologicalModel(variables, parameters, equations, infected_indices)
model.calculate_R0()
```

## Examples

| Notebook | Content |
|---|---|
| [`01_examples.ipynb`](examples/01_examples.ipynb) | SEIR, SIR, Ross-Macdonald, a host-vector model, a within-host model, the treatment model of van den Driessche and Watmough, and two strains with superinfection |
| [`02_global_sensitivity.ipynb`](examples/02_global_sensitivity.ipynb) | Global sensitivity (LHS-PRCC and Sobol) and its relation with the local index |
| [`03_nonlinear_systems.ipynb`](examples/03_nonlinear_systems.ipynb) | $R_0$ in systems with a nonlinear DFE: the model of Cuesta-Herrera et al. (2025), a 7-equation immune model and a DFE without closed form |
| [`04_dfe_time_limit.ipynb`](examples/04_dfe_time_limit.ipynb) | The `dfe_timeout` time limit, with an immune regulation loop (interferon, NK cells and macrophages) without a timely symbolic solution |
| [`05_vdw_assumptions.ipynb`](examples/05_vdw_assumptions.ipynb) | `check_assumptions()` with a model that satisfies (A1)-(A5) and models that violate each one |
| [`06_simulations.ipynb`](examples/06_simulations.ipynb) | `simulate()`: SIR with $R_0 < 1$ and $R_0 > 1$, random and fixed parameters, ensembles colored by $R_0$, phase plane, transcritical bifurcation, vaccination, within-host models, the T-helper model and Ross-Macdonald |
| [`07_validation.ipynb`](examples/07_validation.ipynb) | Validation of $R_0$ against 15 published cases: van den Driessche and Watmough (2002), Ciupe and Heffernan (2017) and Cuesta-Herrera et al. (2024, 2025); the table is saved in `examples/outputs/07_validation.csv` |
| [`08_paper_examples.ipynb`](examples/08_paper_examples.ipynb) | The usage examples of the paper (SEIR and T-helper models) and its Figures 2 and 3 |

All notebooks run in Google Colab and save their figures in `examples/outputs/`. The Spanish versions are in `examples/spanish/`, with their figures in `examples/spanish/outputs/`.

## Citation

If you use pyR0compute in your research, cite it with its Zenodo DOI, which gathers all versions and resolves to the latest one:

> Baldera-Moreno, Y. *pyR0compute: A Python library for automated symbolic computation of the basic reproduction number via the next-generation matrix*. Zenodo. https://doi.org/10.5281/zenodo.23298227

Each GitHub release is archived in Zenodo with its own DOI (version 0.3.1 is https://doi.org/10.5281/zenodo.23298228). The `CITATION.cff` file and GitHub's *Cite this repository* button give the citation in APA and BibTeX. Please also cite:

* van den Driessche, P., & Watmough, J. (2002). Reproduction numbers and sub-threshold endemic equilibria for compartmental models of disease transmission. *Mathematical Biosciences*, 180(1-2), 29-48.
* Diekmann, O., Heesterbeek, J. A. P., & Roberts, M. G. (2010). The construction of next-generation matrices for compartmental epidemic models. *Journal of the Royal Society Interface*, 7(47), 873-885.

For global sensitivity analysis:

* Marino, S., Hogue, I. B., Ray, C. J., & Kirschner, D. E. (2008). A methodology for performing global uncertainty and sensitivity analysis in systems biology. *Journal of Theoretical Biology*, 254(1), 178-196.
* Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010). Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Computer Physics Communications*, 181(2), 259-270.

## License

MIT, see `LICENSE`.
