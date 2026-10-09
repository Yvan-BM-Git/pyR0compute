"""Global sensitivity analysis of R0: LHS-PRCC and Sobol indices.

The normalized forward sensitivity index of :meth:`R0Model.sensitivity_indices`
is local: it is evaluated at one parameter point and says nothing about
parameter uncertainty or interactions. The two methods here sample the whole
parameter space instead.

* **LHS-PRCC** (Marino et al. 2008): parameters are drawn by Latin hypercube
  sampling and, for each parameter, the partial rank correlation coefficient
  between it and R0 is computed after removing the linear effect of the
  ranks of all other parameters. It measures the strength and sign of a
  *monotone* relation, so monotonicity is checked with the symbolic
  derivatives of R0 at every sample.
* **Sobol indices** (Sobol' 2001; Saltelli et al. 2010): variance-based. The
  first-order index ``S1`` is the fraction of Var(R0) explained by one
  parameter alone; the total index ``ST`` adds every interaction it takes part
  in, so ``ST - S1`` quantifies interactions.

Because R0 is available in closed form it is lambdified and evaluated
vectorially, so 10^5 - 10^6 evaluations take seconds. Models whose eigenvalues
have no closed form fall back to the numerical spectral radius of ``K = F V^-1``.

References
----------
S. Marino, I. B. Hogue, C. J. Ray, D. E. Kirschner (2008). A methodology for
performing global uncertainty and sensitivity analysis in systems biology.
J. Theor. Biol. 254, 178-196.

I. M. Sobol' (2001). Global sensitivity indices for nonlinear mathematical
models and their Monte Carlo estimates. Math. Comput. Simul. 55, 271-280.

A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto, S. Tarantola
(2010). Variance based sensitivity analysis of model output. Design and
estimator for the total sensitivity index. Comput. Phys. Commun. 181, 259-270.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from .exceptions import ModelSpecificationError, NextGenerationError, R0ComputeError

__all__ = ["GlobalSensitivityResult", "make_distribution", "prcc", "sobol"]


# ---------------------------------------------------------------- scipy
def _scipy_stats(min_version: Tuple[int, int] = (1, 7)):
    try:
        import scipy
        import scipy.stats as stats
        from scipy.stats import qmc  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Global sensitivity analysis needs SciPy. Install it with "
            "pip install 'pyR0compute[global]' (or pip install scipy)."
        ) from exc
    version = tuple(int(p) for p in scipy.__version__.split(".")[:2] if p.isdigit())
    if version < min_version:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"This method needs SciPy >= {'.'.join(map(str, min_version))}; "
            f"found {scipy.__version__}."
        )
    return stats


def _rng_kwarg(func, seed):
    """SciPy renamed ``seed``/``random_state`` to ``rng`` (SciPy 1.15)."""
    params = inspect.signature(func).parameters
    for name in ("rng", "seed", "random_state"):
        if name in params:
            return {name: seed}
    return {}


# --------------------------------------------------------- distributions
_DIST_NAMES = ("uniform", "loguniform", "normal", "truncnormal", "triangular")


def make_distribution(spec):
    """Turn a parameter specification into a frozen SciPy distribution.

    Accepted forms:

    * ``(low, high)``: uniform on ``[low, high]``;
    * ``("uniform", low, high)``;
    * ``("loguniform", low, high)``: ``log p`` uniform, for parameters
      spanning orders of magnitude (needs ``low > 0``);
    * ``("normal", mean, sd)``;
    * ``("truncnormal", mean, sd, low, high)``: normal truncated to
      ``[low, high]``, e.g. to keep a rate positive;
    * ``("triangular", low, mode, high)``;
    * any frozen ``scipy.stats`` distribution (anything with a ``ppf`` method).
    """
    stats = _scipy_stats()
    if hasattr(spec, "ppf"):
        return spec
    if not isinstance(spec, (tuple, list)) or not spec:
        raise ModelSpecificationError(f"Invalid distribution specification: {spec!r}.")
    if isinstance(spec[0], str):
        name, args = spec[0].lower(), [float(a) for a in spec[1:]]
    else:
        name, args = "uniform", [float(a) for a in spec]

    expected = {"uniform": 2, "loguniform": 2, "normal": 2, "truncnormal": 4, "triangular": 3}
    if name not in expected:
        raise ModelSpecificationError(
            f"Unknown distribution '{spec[0]}'. Use one of {list(_DIST_NAMES)} "
            f"or a frozen scipy.stats distribution."
        )
    if len(args) != expected[name]:
        raise ModelSpecificationError(
            f"'{name}' needs {expected[name]} numbers, got {len(args)}: {spec!r}."
        )

    if name in ("uniform", "loguniform"):
        low, high = args
        if not low < high:
            raise ModelSpecificationError(f"Need low < high in {spec!r}.")
        if name == "uniform":
            return stats.uniform(loc=low, scale=high - low)
        if low <= 0:
            raise ModelSpecificationError(f"'loguniform' needs low > 0 in {spec!r}.")
        return stats.loguniform(low, high)
    if name == "normal":
        mean, sd = args
        if sd <= 0:
            raise ModelSpecificationError(f"Need sd > 0 in {spec!r}.")
        return stats.norm(loc=mean, scale=sd)
    if name == "truncnormal":
        mean, sd, low, high = args
        if sd <= 0 or not low < high:
            raise ModelSpecificationError(f"Need sd > 0 and low < high in {spec!r}.")
        return stats.truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)
    low, mode, high = args
    if not low <= mode <= high or not low < high:
        raise ModelSpecificationError(f"Need low <= mode <= high and low < high in {spec!r}.")
    return stats.triang((mode - low) / (high - low), loc=low, scale=high - low)


def _describe(dist) -> str:
    name = getattr(getattr(dist, "dist", None), "name", type(dist).__name__)
    low, high = dist.ppf(0.0), dist.ppf(1.0)
    if np.isfinite(low) and np.isfinite(high):
        return f"{name} [{low:.4g}, {high:.4g}]"
    return f"{name} (mean {float(dist.mean()):.4g}, sd {float(dist.std()):.4g})"


# -------------------------------------------------------------- evaluator
class _R0Evaluator:
    """Vectorized numerical evaluation of R0 (and its derivatives)."""

    def __init__(self, model):
        self.model = model
        self.implicit = bool(getattr(model, "dfe_is_implicit", False))
        try:
            expr = model.R0
        except NextGenerationError:
            expr = None
        self.symbolic = expr is not None and not self.implicit
        if self.implicit:
            # part of the DFE has no closed form: R0 is computed sample by sample
            self.expr = None
            self.parameters = tuple(sorted(model.parameters, key=lambda s: s.name))
        elif self.symbolic:
            self.expr = expr
            self.parameters = tuple(sorted(expr.free_symbols, key=lambda s: s.name))
            # DFE values X* without a short closed form: evaluate the compact R0
            # after the X* in solving order (chain rule for derivatives) instead
            # of the large explicit expression
            self._chain = None
            stars = getattr(model, "dfe_definitions", {})
            if stars:
                self._chain = [(s, e) for s, e in stars.items()]
                self._compact = model.R0_compact
                args = self.parameters + tuple(s for s, _ in self._chain)
                self._args = args
                self._defs = [sp.lambdify(args, e, "numpy") for _, e in self._chain]
                self._f = sp.lambdify(args, self._compact, "numpy")
            else:
                self._f = sp.lambdify(self.parameters, expr, "numpy")
        else:
            K = model.next_generation_matrix
            self.expr = None
            self.parameters = tuple(sorted(K.free_symbols, key=lambda s: s.name))
            # one function per entry: a lambdified Matrix mixes scalars and arrays
            self._K = [[sp.lambdify(self.parameters, K[i, j], "numpy") for j in range(K.shape[1])]
                       for i in range(K.shape[0])]
            self._shape = K.shape

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.parameters]

    def __call__(self, columns: Sequence[np.ndarray]) -> np.ndarray:
        n = len(columns[0]) if columns else 1
        if self.implicit:
            out = np.full(n, np.nan)
            for k in range(n):
                point = {p.name: float(np.broadcast_to(c, (n,))[k])
                         for p, c in zip(self.parameters, columns)}
                try:
                    out[k] = self.model.R0_numeric(point)
                except R0ComputeError:
                    pass
            return out
        with np.errstate(all="ignore"):
            if self.symbolic:
                out = np.asarray(self._f(*self._arguments(columns)), dtype=complex)
                out = np.broadcast_to(out, (n,))
                bad = np.abs(out.imag) > 1e-9 * np.maximum(1.0, np.abs(out.real))
                return np.where(bad, np.nan, out.real).astype(float)
            m = self._shape[0]
            K = np.empty((n, m, m))
            for i in range(m):
                for j in range(m):
                    K[:, i, j] = np.broadcast_to(np.asarray(self._K[i][j](*columns), dtype=float), (n,))
            out = np.full(n, np.nan)
            ok = np.all(np.isfinite(K), axis=(1, 2))
            if ok.any():
                out[ok] = np.max(np.abs(np.linalg.eigvals(K[ok])), axis=1)
            return out

    def _arguments(self, columns):
        """Parameter columns, followed by the X* values when R0 is evaluated in chain."""
        if not self.symbolic or self._chain is None:
            return list(columns)
        n = len(columns[0]) if columns else 1
        args = [np.asarray(c, dtype=complex) for c in columns]
        star_vals = [np.zeros(n, dtype=complex) for _ in self._chain]
        for k, f in enumerate(self._defs):
            star_vals[k] = np.broadcast_to(np.asarray(f(*args, *star_vals), dtype=complex), (n,))
        return args + star_vals

    def _derivative(self, columns, i):
        """dR0/dp_i at the sample, by the chain rule through the X* when needed."""
        p = self.parameters[i]
        if self._chain is None:
            d = sp.lambdify(self.parameters, sp.diff(self.expr, p), "numpy")
            return d(*columns)
        args = self._arguments(columns)

        def total(expr, d_stars):
            # d expr / dp = partial in p + sum over X* of (partial in X*) * dX*/dp
            value = sp.lambdify(self._args, sp.diff(expr, p), "numpy")(*args)
            for (s, _), ds in zip(self._chain, d_stars):
                if s in expr.free_symbols:
                    value = value + sp.lambdify(self._args, sp.diff(expr, s), "numpy")(*args) * ds
            return value

        d_stars = []
        for _, e in self._chain:
            d_stars.append(total(e, d_stars))
        return total(self._compact, d_stars)

    def derivative_signs(self, columns, idx: Sequence[int]) -> Dict[int, Optional[bool]]:
        """For each parameter index: True if dR0/dp keeps one sign on the sample."""
        out: Dict[int, Optional[bool]] = {}
        if not self.symbolic:
            return {i: None for i in idx}
        n = len(columns[0])
        for i in idx:
            with np.errstate(all="ignore"):
                vals = np.broadcast_to(np.real(np.asarray(self._derivative(columns, i), dtype=complex)), (n,))
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                out[i] = None
                continue
            tol = 1e-12 * max(1.0, float(np.max(np.abs(vals))))
            out[i] = not (np.any(vals > tol) and np.any(vals < -tol))
        return out


def _resolve(model, evaluator, distributions, baseline, spread, fixed):
    """Split the parameters of R0 into varying (with distribution) and fixed."""
    distributions = dict(distributions or {})
    baseline = dict(baseline or {})
    fixed = dict(fixed or {})
    if spread is not None and not 0 < spread < 1:
        raise ModelSpecificationError("spread must be in (0, 1), e.g. 0.2 for +/-20%.")

    def name_of(key):
        name = key.name if isinstance(key, sp.Symbol) else str(key)
        if name not in model.symbols:
            raise ModelSpecificationError(f"'{name}' is not a symbol of this model.")
        return name

    distributions = {name_of(k): v for k, v in distributions.items()}
    baseline = {name_of(k): float(v) for k, v in baseline.items()}
    fixed = {name_of(k): float(v) for k, v in fixed.items()}

    names = evaluator.names
    unused = sorted(set(distributions) - set(names))
    if unused:
        warnings.warn(f"These parameters do not appear in R0 and are ignored: {unused}", stacklevel=4)
    both = sorted(set(distributions) & set(fixed))
    if both:
        raise ModelSpecificationError(f"Parameters given both a distribution and a fixed value: {both}")

    varying: Dict[str, Any] = {}
    constants: Dict[str, float] = {}
    missing = []
    for name in names:
        if name in distributions:
            varying[name] = make_distribution(distributions[name])
        elif name in fixed:
            constants[name] = fixed[name]
        elif name in baseline and spread is not None:
            b = baseline[name]
            low, high = sorted((b * (1 - spread), b * (1 + spread)))
            if low == high:
                constants[name] = b
            else:
                varying[name] = make_distribution((low, high))
        elif name in baseline:
            constants[name] = baseline[name]
        else:
            missing.append(name)
    if missing:
        raise ModelSpecificationError(
            f"No distribution or value for the parameters {missing}. Give them in "
            f"distributions={{...}}, fixed={{...}}, or baseline={{...}} with spread=..."
        )
    if not varying:
        raise ModelSpecificationError("At least one parameter must vary.")
    return varying, constants


def _columns(evaluator, varying, constants, X):
    """Argument list for the evaluator from a sample matrix X (n x k)."""
    order = list(varying)
    n = X.shape[0]
    return [X[:, order.index(name)] if name in varying else np.full(n, constants[name])
            for name in evaluator.names]


# ---------------------------------------------------------------- result
@dataclass
class GlobalSensitivityResult:
    """Result of :func:`prcc` or :func:`sobol`.

    ``indices`` maps a column name (``"PRCC"``, ``"p_value"``, ``"S1"``,
    ``"ST"``, confidence limits...) to an array aligned with ``parameters``.
    """

    method: str
    parameters: Tuple[sp.Symbol, ...]
    indices: Dict[str, np.ndarray]
    n: int
    n_evaluations: int
    distributions: Dict[str, str]
    fixed: Dict[str, float] = field(default_factory=dict)
    output_name: str = "R0"
    samples: Optional[np.ndarray] = None
    output: Optional[np.ndarray] = None
    monotonic: Optional[Dict[str, Optional[bool]]] = None
    confidence_level: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def main_index(self) -> str:
        return "PRCC" if self.method == "LHS-PRCC" else "ST"

    def __getitem__(self, parameter) -> Dict[str, float]:
        name = parameter.name if isinstance(parameter, sp.Symbol) else str(parameter)
        names = [p.name for p in self.parameters]
        if name not in names:
            raise KeyError(name)
        i = names.index(name)
        return {k: float(v[i]) for k, v in self.indices.items()}

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        """``{parameter name: {column: value}}``."""
        return {p.name: self[p] for p in self.parameters}

    def ranking(self) -> List[str]:
        """Parameter names, most influential first (|PRCC| or ST)."""
        values = np.abs(self.indices[self.main_index])
        return [self.parameters[i].name for i in np.argsort(-values, kind="stable")]

    def to_dataframe(self):
        """Results as a pandas DataFrame indexed by parameter (needs pandas)."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("to_dataframe() needs pandas: pip install pandas") from exc
        df = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in self.indices.items()},
                          index=[p.name for p in self.parameters])
        df.index.name = "parameter"
        if self.monotonic is not None:
            df["monotonic"] = [self.monotonic.get(p.name) for p in self.parameters]
        df["distribution"] = [self.distributions[p.name] for p in self.parameters]
        return df.loc[self.ranking()]

    def _columns_to_show(self):
        if self.method == "LHS-PRCC":
            return [("PRCC", "PRCC"), ("p_value", "p")]
        cols = [("S1", "S_1"), ("ST", "S_T")]
        if "S1_low" in self.indices:
            cols = [("S1", "S_1"), ("S1_ci", "IC S_1"), ("ST", "S_T"), ("ST_ci", "IC S_T")]
        return cols

    def _cell(self, key, i, digits):
        if key.endswith("_ci"):
            base = key[:-3]
            lo, hi = self.indices[f"{base}_low"][i], self.indices[f"{base}_high"][i]
            return f"[{lo:.{digits}f}, {hi:.{digits}f}]"
        v = float(self.indices[key][i])
        if key == "p_value":
            return "<0.001" if v < 0.001 else f"{v:.3f}"
        return f"{v:.{digits}f}"

    def to_latex(self, style: str = "document", digits: int = 3) -> str:
        """Table of results in LaTeX (``"document"``) or Markdown (``"markdown"``)."""
        if style not in ("document", "markdown"):
            raise ValueError("style must be 'document' or 'markdown'.")
        cols = self._columns_to_show()
        names = [p.name for p in self.parameters]
        order = [names.index(n) for n in self.ranking()]
        md = style == "markdown"

        def head(label):
            if label.startswith("IC "):
                return f"IC {(self.confidence_level or 0.95):.0%} ${label[3:]}$".replace("%", "\\%" if not md else "%")
            return label if label == "PRCC" else f"${label}$"

        caption = (f"{self.method}, $n = {self.n}$ "
                   f"({self.n_evaluations} evaluations of ${self._output_tex()}$)")
        if md:
            rows = ["| Parameter | Distribution | " + " | ".join(head(l) for _, l in cols) + " |",
                    "|---|---|" + "---:|" * len(cols)]
            for i in order:
                cells = [self._cell(k, i, digits) for k, _ in cols]
                rows.append(f"| ${sp.latex(self.parameters[i])}$ | {self.distributions[names[i]]} | "
                            + " | ".join(cells) + " |")
            return caption + "\n\n" + "\n".join(rows)
        rows = ["\\begin{table}[ht]", "\\centering",
                f"\\caption{{{caption}}}",
                "\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\hline",
                "Parameter & Distribution & " + " & ".join(head(l) for _, l in cols) + " \\\\",
                "\\hline"]
        for i in order:
            cells = [self._cell(k, i, digits) for k, _ in cols]
            rows.append(f"${sp.latex(self.parameters[i])}$ & {self.distributions[names[i]]} & "
                        + " & ".join(cells) + " \\\\")
        rows += ["\\hline", "\\end{tabular}", "\\end{table}"]
        return "\n".join(rows)

    def _output_tex(self):
        return "\\log \\mathcal{R}_0" if self.output_name == "log R0" else "\\mathcal{R}_0"

    def plot(self, ax=None):
        """Horizontal bar chart (tornado for PRCC, S1 vs ST for Sobol). Needs matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("plot() needs matplotlib: pip install matplotlib") from exc
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 0.45 * len(self.parameters) + 1.2))
        names = [p.name for p in self.parameters]
        order = [names.index(n) for n in self.ranking()][::-1]  # largest on top
        labels = [f"${sp.latex(self.parameters[i])}$" for i in order]
        y = np.arange(len(order))
        blue, red, orange = "#2a78d6", "#e34948", "#eb6834"
        if self.method == "LHS-PRCC":
            vals = self.indices["PRCC"][order]
            ax.barh(y, vals, height=0.55, color=[blue if v >= 0 else red for v in vals])
            ax.axvline(0, color="#52514e", linewidth=0.8)
            ax.set_xlim(-1.12, 1.12)  # room for the significance marks
            ax.set_xticks(np.linspace(-1, 1, 5))
            ax.set_xlabel(f"PRCC with ${self._output_tex()}$")
            for yi, i, v in zip(y, order, vals):
                if self.indices["p_value"][i] < 0.05:
                    ax.text(v + (0.03 if v >= 0 else -0.03), yi, "*", va="center",
                            ha="left" if v >= 0 else "right", color="#0b0b0b")
        else:
            h = 0.36
            for key, color, dy, label in (("S1", blue, h / 2, "$S_1$ (first order)"),
                                          ("ST", orange, -h / 2, "$S_T$ (total)")):
                vals = self.indices[key][order]
                err = None
                if f"{key}_low" in self.indices:
                    lo, hi = self.indices[f"{key}_low"][order], self.indices[f"{key}_high"][order]
                    err = np.vstack([np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)])
                ax.barh(y + dy, vals, height=h * 0.9, color=color, label=label,
                        xerr=err, error_kw={"elinewidth": 0.8, "ecolor": "#52514e", "capsize": 2})
            ax.set_xlim(0, max(1.0, float(np.nanmax(self.indices["ST"])) * 1.05))
            ax.set_xlabel(f"Sobol index of ${self._output_tex()}$")
            ax.legend(frameon=False, loc="lower right")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.grid(axis="x", color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{self.method} (n = {self.n})", loc="left", fontsize=11)
        return ax

    def __repr__(self) -> str:
        cols = self._columns_to_show()
        names = [p.name for p in self.parameters]
        width = max(9, max(len(n) for n in names) + 2)
        lines = [f"GlobalSensitivityResult: {self.method}, output {self.output_name}, "
                 f"n = {self.n} ({self.n_evaluations} evaluations)"]
        header = "parameter".ljust(width) + "".join(l.replace("_", "").rjust(20) for _, l in cols)
        if self.monotonic is not None:
            header += "  monotonic"
        lines.append(header)
        for name in self.ranking():
            i = names.index(name)
            row = name.ljust(width) + "".join(self._cell(k, i, 4).rjust(20) for k, _ in cols)
            if self.monotonic is not None:
                row += f"  {self.monotonic.get(name)}"
            lines.append(row)
        if self.fixed:
            lines.append(f"fixed: {self.fixed}")
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


# ------------------------------------------------------------------ PRCC
def _prcc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Partial rank correlation coefficients of each column of X with y."""
    stats = _scipy_stats()
    R = stats.rankdata(np.column_stack([X, y]), axis=0)
    C = np.corrcoef(R, rowvar=False)
    if X.shape[1] == 1:  # nothing to partial out: Spearman correlation
        return np.array([np.clip(C[0, 1], -1.0, 1.0)])
    P = np.linalg.pinv(C)
    k = X.shape[1]
    r = -P[:k, k] / np.sqrt(P[np.arange(k), np.arange(k)] * P[k, k])
    return np.clip(r, -1.0, 1.0)


def prcc(model, distributions=None, *, n: int = 1000, baseline=None, spread=None,
         fixed=None, seed: Optional[int] = None, check_monotonicity: bool = True,
         keep_samples: bool = True) -> GlobalSensitivityResult:
    """Latin hypercube sampling with partial rank correlation coefficients.

    Parameters
    ----------
    model
        An :class:`R0Model`.
    distributions
        ``{parameter: spec}``; see :func:`make_distribution` for the specs.
    n
        Number of Latin hypercube samples (= evaluations of R0).
    baseline, spread
        With ``spread=0.2``, every parameter of ``baseline`` without an explicit
        distribution varies uniformly in ``baseline*(1 +/- 0.2)``. Without
        ``spread``, the parameters of ``baseline`` are kept fixed.
    fixed
        ``{parameter: value}`` held constant.
    seed
        Seed of the Latin hypercube.
    check_monotonicity
        Check, with the symbolic derivatives of R0, that R0 is monotone in each
        parameter over the sample. PRCC is only meaningful for monotone
        relations; non-monotone parameters trigger a warning.
    """
    stats = _scipy_stats()
    evaluator = _R0Evaluator(model)
    varying, constants = _resolve(model, evaluator, distributions, baseline, spread, fixed)
    names = list(varying)
    k = len(names)
    if n < k + 3:
        raise ModelSpecificationError(f"n = {n} is too small for {k} varying parameters (need n >= {k + 3}).")

    sampler = stats.qmc.LatinHypercube(d=k, **_rng_kwarg(stats.qmc.LatinHypercube, seed))
    U = sampler.random(n)
    X = np.column_stack([varying[name].ppf(U[:, j]) for j, name in enumerate(names)])
    cols = _columns(evaluator, varying, constants, X)
    y = evaluator(cols)

    notes = []
    ok = np.isfinite(y)
    if not ok.all():
        notes.append(f"{int((~ok).sum())} of {n} samples gave a non-finite R0 and were dropped.")
        warnings.warn(notes[-1] + " Check that the distributions only give admissible "
                      "(e.g. positive) parameter values.", stacklevel=3)
        X, y = X[ok], y[ok]
        cols = [c[ok] for c in cols]
    m = len(y)
    if m < k + 3:
        raise ModelSpecificationError("Too few valid samples to compute PRCC.")

    r = _prcc(X, y)
    df = m - k - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(df / np.clip(1.0 - r ** 2, 1e-300, None))
    p = 2.0 * stats.t.sf(np.abs(t), df)

    monotonic = None
    if check_monotonicity:
        sym_index = {s.name: i for i, s in enumerate(evaluator.parameters)}
        signs = evaluator.derivative_signs(cols, [sym_index[nm] for nm in names])
        monotonic = {nm: signs[sym_index[nm]] for nm in names}
        bad = [nm for nm, v in monotonic.items() if v is False]
        if bad:
            notes.append(f"R0 is not monotone in {bad} over the sample; their PRCC can "
                         "underestimate their influence. Use Sobol indices for them.")
            warnings.warn(notes[-1], stacklevel=3)
    if not evaluator.symbolic:
        notes.append("R0 has no closed form; the numerical spectral radius of K was used.")

    symbols = {s.name: s for s in evaluator.parameters}
    return GlobalSensitivityResult(
        method="LHS-PRCC",
        parameters=tuple(symbols[nm] for nm in names),
        indices={"PRCC": r, "t": t, "p_value": p},
        n=n,
        n_evaluations=n,
        distributions={nm: _describe(varying[nm]) for nm in names},
        fixed=constants,
        samples=X if keep_samples else None,
        output=y if keep_samples else None,
        monotonic=monotonic,
        notes=notes,
    )


# ----------------------------------------------------------------- Sobol
def sobol(model, distributions=None, *, n: int = 4096, baseline=None, spread=None,
          fixed=None, seed: Optional[int] = None, log_output: bool = False,
          confidence_level: Optional[float] = 0.95, n_resamples: int = 999,
          ) -> GlobalSensitivityResult:
    """Variance-based Sobol indices of R0 (first order ``S1`` and total ``ST``).

    Uses the Saltelli (2010) estimators of :func:`scipy.stats.sobol_indices`
    (SciPy >= 1.11) with a scrambled Sobol' sequence: ``n * (k + 2)`` evaluations
    of R0 for ``k`` varying parameters. ``n`` must be a power of 2.

    Parameters
    ----------
    model, distributions, baseline, spread, fixed, seed
        As in :func:`prcc`.
    n
        Base sample size (power of 2).
    log_output
        Analyse ``log R0`` instead of ``R0``. For R0 of product form
        ``c * prod p_i^a_i`` with independent log-uniform parameters,
        ``log R0`` is additive and ``S1 = ST = a_i^2 Var(log p_i) / sum_j ...``.
    confidence_level
        Bootstrap confidence intervals of the indices (``None`` to skip).
    n_resamples
        Number of bootstrap resamples.

    Notes
    -----
    The indices assume independent parameters. Small negative estimates of
    ``S1`` for non-influential parameters are Monte Carlo noise; increase ``n``.
    """
    stats = _scipy_stats(min_version=(1, 11))
    evaluator = _R0Evaluator(model)
    varying, constants = _resolve(model, evaluator, distributions, baseline, spread, fixed)
    names = list(varying)
    k = len(names)
    if n < 2 or n & (n - 1):
        raise ModelSpecificationError(f"n must be a power of 2 (e.g. 1024, 4096); got {n}.")

    def func(x):
        y = evaluator(_columns(evaluator, varying, constants, np.asarray(x).T))
        if log_output:
            with np.errstate(all="ignore"):
                y = np.where(y > 0, np.log(y), np.nan)
        if not np.all(np.isfinite(y)):
            bad = int((~np.isfinite(y)).sum())
            raise ModelSpecificationError(
                f"{bad} samples gave a non-finite {'log ' if log_output else ''}R0. Sobol indices "
                "need valid values everywhere: restrict the distributions to admissible "
                "(e.g. positive) values, for instance with 'truncnormal'."
            )
        return y[None, :]

    res = stats.sobol_indices(func=func, n=n, dists=[varying[nm] for nm in names],
                              **_rng_kwarg(stats.sobol_indices, seed))
    S1 = np.asarray(res.first_order, dtype=float).reshape(-1)[:k]
    ST = np.asarray(res.total_order, dtype=float).reshape(-1)[:k]
    indices = {"S1": S1, "ST": ST}
    if confidence_level is not None:
        boot = res.bootstrap(confidence_level=confidence_level, n_resamples=n_resamples)
        for key, ci in (("S1", boot.first_order.confidence_interval),
                        ("ST", boot.total_order.confidence_interval)):
            indices[f"{key}_low"] = np.asarray(ci.low, dtype=float).reshape(-1)[:k]
            indices[f"{key}_high"] = np.asarray(ci.high, dtype=float).reshape(-1)[:k]

    notes = []
    if not evaluator.symbolic:
        notes.append("R0 has no closed form; the numerical spectral radius of K was used.")
    total = float(np.sum(S1))
    if total < 0.9:
        notes.append(f"sum(S1) = {total:.3f}: about {1 - total:.0%} of the variance comes from "
                     "parameter interactions (see ST - S1).")

    symbols = {s.name: s for s in evaluator.parameters}
    return GlobalSensitivityResult(
        method="Sobol",
        parameters=tuple(symbols[nm] for nm in names),
        indices=indices,
        n=n,
        n_evaluations=n * (k + 2),
        distributions={nm: _describe(varying[nm]) for nm in names},
        fixed=constants,
        output_name="log R0" if log_output else "R0",
        confidence_level=confidence_level,
        notes=notes,
    )
