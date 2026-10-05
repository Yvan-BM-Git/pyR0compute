# pyR0compute

Symbolic computation of the basic reproduction number $R_0$ of compartmental ODE models by the next-generation matrix method (van den Driessche & Watmough, 2002).

Write the model, say which compartments are infected, and pyR0compute does the rest: every symbol that is not a state variable is treated as a parameter, the new-infection terms $\mathcal{F}$ and transitions $\mathcal{V}$ are identified, the disease-free equilibrium (DFE) is solved, and $R_0 = \rho(FV^{-1})$ is returned as a SymPy expression.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yvan-BM-Git/pyR0compute/blob/main/examples/pyR0compute_examples.ipynb)

## Installation

```bash
pip install pyR0compute
```

Until the first release is on PyPI, install it from GitHub:

```bash
pip install git+https://github.com/Yvan-BM-Git/pyR0compute
```

Requires Python ≥ 3.9, SymPy and NumPy.

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

The equations can be written in any order; there is no need to list the infected compartments first.

### Three ways to enter a model

```python
# 1. Text: one "dX/dt = ..." (or "X' = ...") line per variable.
#    Auxiliary lines such as "N = S + I + R" are substituted.
R0Model("""
    N = S + I + R
    dS/dt = Lambda - beta*S*I/N - mu*S
    dI/dt = beta*S*I/N - (gamma + mu)*I
    dR/dt = gamma*I - mu*R
""", infected=["I"])

# 2. A dict {variable: right-hand side} (strings or SymPy expressions)
R0Model({"S": "Lambda - beta*S*I - mu*S",
         "I": "beta*S*I - (gamma + mu)*I"}, infected=["I"])

# 3. SymPy symbols
import sympy as sp
S, I = sp.symbols("S I")
beta, gamma, mu, Lam = sp.symbols("beta gamma mu Lambda")
R0Model({S: Lam - beta*S*I - mu*S, I: beta*S*I - (gamma + mu)*I}, infected=[I])
```

Use subscripts in names for clean LaTeX: `mu_h`, `alpha_hv`, `Delta_h` are written $\mu_h$, $\alpha_{hv}$, $\Delta_h$. Names that are special in SymPy (`I`, `S`, `E`, `N`, `beta`, `gamma`, `Lambda`, even `lambda`) are plain symbols inside text models, and `^` means a power.

### What you get

| Attribute / method | Content |
|---|---|
| `R0` | Basic reproduction number (spectral radius of $FV^{-1}$) |
| `parameters` | Parameters detected automatically |
| `new_infections`, `transitions` | Terms $\mathcal{F}_i$ and $\mathcal{V}_i$ of each infected compartment |
| `dfe`, `dfe_candidates` | Disease-free equilibrium used, and all non-negative ones found |
| `F`, `V`, `K` | Jacobians at the DFE and next-generation matrix $K = FV^{-1}$ |
| `next_generation_matrix_small` | $K$ restricted to compartments receiving new infections |
| `eigenvalues` | Eigenvalues of $K$ |
| `R0_numeric(values)` | Numerical spectral radius (for large models without a closed form) |
| `sensitivity_indices(values=None)` | Normalized forward sensitivity indices $\Upsilon_p = \frac{\partial R_0}{\partial p}\frac{p}{R_0}$ |
| `report()` | Step-by-step description of the whole computation |
| `report_latex(style, standalone, mat_str)` | The same report in LaTeX (`"document"`, compilable with `standalone=True`) or Markdown for notebooks (`"markdown"`) |
| `latex()` | LaTeX code of $R_0$ |

### When the automatic choices need help

* **Closed populations** (no births), e.g. the classic SIR: the DFE is not unique, so give it, `dfe={"S": "N"}`. The error message says which value is missing.
* **Several DFEs** (e.g. logistic vector populations): the one with the most non-zero compartments is used and a warning is shown. Choose another with `dfe={...}`; all are in `model.dfe_candidates`.
* **Custom decompositions**: the split $\mathcal{F}$/$\mathcal{V}$ is not unique. Override the automatic one with `new_infections={"E": "beta*S*I/N"}`.

### How new infections are detected

In the equation of an infected compartment, a positive term is a new infection when it involves an infected compartment and either (i) it is lost from an uninfected compartment (a transfer such as $S \to E$), or (ii) it involves an uninfected compartment and is not a transfer between infected compartments. Progression ($E \to I$), treatment failure, superinfection between strains, deaths and recoveries go to $\mathcal{V}$. `model.report()` shows the classification of every term and the reason.

## Validation

The test suite (`pytest`) reproduces known results: SIR (with and without vital dynamics, mass-action and frequency-dependent), SEIR, a vaccination model, Ross-Macdonald, a host-vector SEIR/SEI model with human-to-human transmission and logistic vectors, the treatment model of van den Driessche & Watmough (2002, §4.1), within-host target cell-infected cell-virus models and a two-strain model with superinfection. Symbolic results are also checked against the numerical spectral radius.

## Previous interface

Code written for the original notebook keeps working:

```python
from pyr0compute import GeneralEpidemiologicalModel
model = GeneralEpidemiologicalModel(variables, parameters, equations, infected_indices)
model.calculate_R0()
```

## Citation

If you use pyR0compute in your research, please cite it (see `CITATION.cff`) together with:

* van den Driessche, P., & Watmough, J. (2002). Reproduction numbers and sub-threshold endemic equilibria for compartmental models of disease transmission. *Mathematical Biosciences*, 180(1-2), 29-48.
* Diekmann, O., Heesterbeek, J. A. P., & Roberts, M. G. (2010). The construction of next-generation matrices for compartmental epidemic models. *Journal of the Royal Society Interface*, 7(47), 873-885.

## Resumen en español

pyR0compute calcula simbólicamente el número básico de reproducción $R_0$ mediante el método de la matriz de próxima generación. Basta escribir el sistema de EDO e indicar los compartimentos infectados: todos los demás símbolos se reconocen automáticamente como parámetros, se identifican los términos de nuevas infecciones, se calcula el equilibrio libre de enfermedad y se obtiene $R_0$. El notebook `examples/pyR0compute_examples.ipynb` contiene ejemplos listos para Google Colab.

## License

MIT, see `LICENSE`.
