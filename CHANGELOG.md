# Changelog

## 0.2.0 (2026-10-09)

### Added
- Models with a nonlinear disease-free equilibrium (e.g. within-host models with coupled immune responses):
  - The DFE is solved block by block: strongly connected components of the dependency graph of the uninfected equations, in topological order. Values already found enter later blocks as symbols `X_star` (printed `X^{*}` in LaTeX), which keeps every block small. A 7-equation immune-response model that did not finish in 5 minutes is now built in about 10 s.
  - `R0Model.R0_compact`, `dfe_compact`, `dfe_definitions` and `next_generation_matrix_compact`: results written with the DFE values that have no short closed form. `R0` keeps the explicit values substituted. `report()`, `report_latex()` and `latex()` show the compact form when it exists.
  - DFE blocks without a closed form (e.g. the root of a quintic) are kept implicit: `dfe_numeric(values)` and `R0_numeric(values)` solve them numerically (SciPy); `dfe_is_implicit` flags them.
  - `R0Model.dfe_stability(values=None)`: condition (A5) of van den Driessche & Watmough (2002), at a point or over random parameter samples, with symbolic eigenvalues when the uninfected block is triangular.
  - Global sensitivity evaluates R0 in chain (the `X_star` in solving order, then R0) and differentiates by the chain rule, instead of the large explicit expression (PRCC on the 7-equation model: 60 s to 5 s).
  - `DFESymbol` exported.
- Tests reproducing Cuesta-Herrera et al. (2025), Math. Biosci. Eng. 22(11):2807-2825 (Eq. 2.4 and the R0* of Figure 3), and checking the 7-equation model against numerical integration and the stability of the full DFE.
- Example notebook `examples/03_sistemas_no_lineales.ipynb` (with an Open in Colab badge).
- Global sensitivity analysis of R0 (optional dependency: `pip install "pyR0compute[global]"`, SciPy >= 1.11):
  - `R0Model.prcc()`: Latin hypercube sampling with partial rank correlation coefficients and their p-values (Marino et al. 2008). Monotonicity of R0 in each parameter is checked with its symbolic derivative at every sample.
  - `R0Model.sobol_indices()`: first-order and total Sobol indices with bootstrap confidence intervals (Saltelli et al. 2010 estimators of `scipy.stats.sobol_indices`); `log_output=True` analyses log R0.
  - `R0Model.global_sensitivity(method=...)` dispatcher and `GlobalSensitivityResult` with `as_dict`, `ranking`, `to_dataframe`, `to_latex` (document or Markdown) and `plot`.
  - Distributions given as `(low, high)`, `("loguniform", ...)`, `("normal", ...)`, `("truncnormal", ...)`, `("triangular", ...)` or frozen `scipy.stats` objects; or `baseline=` with `spread=`; `fixed=` parameters.
  - Vectorized evaluation of the closed-form R0 (including `Max(...)` for competing strains), with fallback to the numerical spectral radius of K.
- Example notebook `examples/02_sensibilidad_global.ipynb`.

## 0.1.0 (2026-10-05)

First packaged release. The code of the original notebook was turned into an installable library.

### Added
- `R0Model`: models can be entered as text (`dS/dt = ...`), as a dict or as SymPy symbols.
- Automatic detection of parameters: every symbol that is not a state variable.
- Equations in any order; infected compartments given by name, symbol or index.
- Safe parsing of names that are special in SymPy (`I`, `S`, `E`, `N`, `beta`, `gamma`, `lambda`) and `^` for powers.
- Auxiliary definitions in text models (`N = S + I + R`).
- Partial user-given disease-free equilibrium (`dfe={"S": "N"}`), list of all non-negative DFEs (`dfe_candidates`).
- `report_latex()`: step-by-step report in LaTeX (document or Markdown for notebooks).
- `R0_numeric`, `sensitivity_indices`, `report`, `latex`, `eigenvalues`, `next_generation_matrix_small`.
- Informative exceptions (`ModelSpecificationError`, `DiseaseFreeEquilibriumError`, `NextGenerationError`).
- Test suite (SIR, SEIR, vector-borne, literature models) and continuous integration.

### Fixed
- Terms with a division (e.g. `beta*S*I/N`) were never recognised as new infections, so R0 was silently returned as 0. The sign test `term.has(-1)` matched the exponent of `1/N`.
- Coefficients such as `(1 - p)*beta*S*I` were split into a new-infection part and a spurious transition part, giving a wrong R0.
- For vector-borne models the negative eigenvalue `-sqrt(...)` could be returned as R0; the Perron root is now used.
- When the dominant eigenvalue depends on the parameters (competing strains), R0 is returned as `Max(...)` instead of an arbitrary branch.
- Undetermined DFE values were replaced by unexplained symbols such as `S_eq`; an error now says which value must be given.
- Several DFEs: the first solution returned by SymPy was taken silently.

### Changed
- `GeneralEpidemiologicalModel` keeps the original interface; `parameters` is now optional and the infected equations no longer need to come first.
- Output is silent by default (`verbose=True` or `report()` for the step-by-step description).
