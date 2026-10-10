# Changelog

## 0.3.0 (2026-10-10)

### Added
- `R0Model.check_assumptions(values=None, n_samples=100, seed=0, language="en")`: checks assumptions (A1)-(A5) of van den Driessche & Watmough (2002), under which Theorem 2 makes R0 a threshold. Each assumption is reported as `"holds"`, `"fails"` or `"undecided"` with its basis: by construction (A3), symbolic proof for all positive parameters, at the given `values`, or random samples (a violation in every parameter sample is `"fails"`; a property that was not proved is `"undecided"`, with the counterexample when there is one).
  - (A1) non-negativity of `F_i`, `V_i^+`, `V_i^-` (canonical split of `V_i` into its negative and positive terms); (A2) `V_i^-` at `x_i = 0`; (A4) `F_i` and `V_i^+` on the whole disease-free subspace `X_s`, not only at `x0`; (A5) positive stability of both `V` and `J_4` (block-triangular structure, trace and determinant, Z-matrix minors, or eigenvalues at the given values or samples).
  - Consequences of Lemma 1: `F >= 0`, Z sign pattern of `V` (non-singular M-matrix), and the zero block `dV_i/dx_j(x0) = 0` for infected `i` and uninfected `j`.
  - When the DFE used is not stable, the stability of the other disease-free equilibria found (`dfe_candidates`) is reported.
- `AssumptionsReport` with `holds`, `status`, `as_dict`, `decomposition`, `to_latex(style="document" | "markdown", standalone, language="en" | "es")` (cumple / no cumple / no se pudo decidir) and Markdown display in Jupyter; `AssumptionResult`.
- Tests with models that violate each assumption; the LaTeX reports are compiled with `pdflatex` when it is available.
- Example notebook `examples/05_supuestos_vdw.ipynb` (with an Open in Colab badge).
- `R0Model.simulate(t_span, values=None, initial=None, n_points, n_runs, ranges, default_range, sampling, R0_range, perturbation, method, rtol, atol, seed)`: numerical solution of the ODE model (SciPy `solve_ivp`). Parameters in `values` are fixed and the others drawn at random (log-uniform by default); `R0_range` keeps only draws with R0 in a range; by default the uninfected compartments start at the DFE of each run and the infected ones at a small perturbation. Works with implicit DFEs.
- `SimulationResult` with the R0, parameters, initial condition and DFE of every run, `final`, `summary`, `to_dataframe`, and the plots `plot` (selected variables, title with `{R0}`, axis labels, labels, colors, line styles, line width, legend, figure size, panels per variable, log scales, limits, runs coloured by R0 < 1 or R0 > 1, DFE lines, drawing on existing axes, saving) and `plot_phase` (phase plane with start points and the DFE).
- Optional dependency group `plot` (SciPy and matplotlib).
- Example notebook `examples/06_simulaciones.ipynb`; all example notebooks are stored executed, with their figures in `examples/outputs/`.

## 0.2.1 (2026-10-09)

### Added
- `dfe_timeout` (seconds, default 20): each coupled DFE block, or single equation of degree > 2, is solved symbolically in a separate Python process that is stopped at the limit. A block not solved in time is kept implicit (`X_star`) and computed numerically, with a warning, instead of running indefinitely. `dfe_timeout=None` restores the previous behaviour (no limit, same process). If the worker process cannot be started, the limit is enforced in-process with `SIGALRM` where available.
- `R0Model.dfe_blocks`: status (`"closed form"`, `"no closed form"`, `"time limit"`) and solving time of each block; `report()` says which values were not found within the limit.
- Example notebook `examples/04_limite_de_tiempo_dfe.ipynb` (with an Open in Colab badge): a within-host model with an interferon, NK cell and macrophage loop, whose DFE block did not finish with 0.2.0.

### Changed
- The vector field of an implicit DFE block is compiled once instead of at every evaluation (R0_numeric is about 3 ms per call on the example; PRCC with n = 1000 takes seconds).

### Fixed
- Implicit DFE blocks were rejected when the numerical integration had already reached the equilibrium (`fsolve` reports no progress, ier = 5); the root is now accepted by its residual.

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
