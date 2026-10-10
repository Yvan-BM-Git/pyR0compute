"""Numerical simulation of the ODE model and plots of the trajectories.

:func:`simulate` integrates the model with :func:`scipy.integrate.solve_ivp`.
Parameters given with ``values=`` are fixed; every other parameter is drawn
at random (log-uniform on ``default_range`` or on its own range in
``ranges``), so a model can be explored before its parameters are known.
Several runs (``n_runs``) give an ensemble, and ``R0_range`` keeps only
draws whose R0 lies in a range (e.g. ``(1, None)`` for R0 > 1). By default
the uninfected compartments start at the disease-free equilibrium of each
run and the infected ones at a small perturbation, which is the situation
R0 describes: the infection either dies out (R0 < 1) or invades (R0 > 1).

:class:`SimulationResult` keeps the runs with their parameters, initial
conditions and R0, and draws them with :meth:`SimulationResult.plot` (time
series of selected variables) and :meth:`SimulationResult.plot_phase`
(phase plane).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import sympy as sp

from .exceptions import ModelSpecificationError, R0ComputeError

__all__ = ["Run", "SimulationResult", "simulate", "PALETTE"]

#: Categorical colours (fixed order) used for the variables.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_BELOW, _ABOVE = "#2a78d6", "#eb6834"   # R0 < 1, R0 > 1
_GRID = "#e5e4e0"
_LINESTYLES = ["-", "--", ":", "-."]


@dataclass
class Run:
    """One simulation: parameter values, initial condition, R0 and trajectory."""

    parameters: Dict[str, float]
    initial: Dict[str, float]
    y: np.ndarray                      # shape (number of variables, number of times)
    R0: Optional[float]
    dfe: Optional[Dict[str, float]]
    success: bool = True
    message: str = ""

    def __repr__(self) -> str:
        r0 = "None" if self.R0 is None else f"{self.R0:.4g}"
        return f"Run(R0={r0}, success={self.success})"


def _require_scipy():
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("simulate() needs SciPy: pip install scipy") from exc
    return solve_ivp


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Plots need matplotlib: pip install matplotlib") from exc
    return plt


def simulate(model, t_span: Tuple[float, float] = (0.0, 100.0),
             values: Optional[Mapping] = None, initial: Optional[Mapping] = None,
             n_points: int = 500, n_runs: int = 1,
             ranges: Optional[Mapping] = None, default_range: Tuple[float, float] = (0.01, 1.0),
             sampling: str = "loguniform", R0_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
             perturbation: float = 0.01, method: str = "LSODA", rtol: float = 1e-8, atol: float = 1e-10,
             seed: Optional[int] = None, max_tries: int = 2000) -> "SimulationResult":
    """Integrate the model.

    Parameters
    ----------
    t_span
        ``(t0, t1)`` time interval.
    values
        Fixed parameter values ``{name: value}``; may be partial.
    initial
        Initial values ``{variable: value}``; may be partial. By default the
        uninfected compartments start at the disease-free equilibrium of the
        run and every infected compartment at ``perturbation`` times the total
        of the uninfected ones (or ``perturbation`` if that total is 0).
    n_points
        Number of output times.
    n_runs
        Number of runs; each draws its own random parameters.
    ranges
        ``{name: (low, high)}`` range of a random parameter.
    default_range
        Range of the random parameters not in ``ranges``.
    sampling
        ``"loguniform"`` (default) or ``"uniform"``.
    R0_range
        ``(low, high)``, either may be None: keep only draws with R0 in that
        range, e.g. ``(None, 1)`` for R0 < 1 and ``(1, None)`` for R0 > 1.
    method, rtol, atol
        Passed to :func:`scipy.integrate.solve_ivp`.
    seed
        Seed of the random parameters.
    """
    solve_ivp = _require_scipy()
    if sampling not in ("loguniform", "uniform"):
        raise ValueError("sampling must be 'loguniform' or 'uniform'.")
    if n_runs < 1 or n_points < 2:
        raise ValueError("n_runs must be >= 1 and n_points >= 2.")
    t0, t1 = map(float, t_span)
    if not t1 > t0:
        raise ValueError("t_span must be (t0, t1) with t1 > t0.")

    variables = list(model.variables)
    var_names = [v.name for v in variables]
    params = sorted(model.parameters, key=lambda s: s.name)
    par_names = [p.name for p in params]

    fixed = {}
    for key, val in (values or {}).items():
        name = key.name if isinstance(key, sp.Symbol) else str(key)
        if name not in par_names:
            raise ModelSpecificationError(f"'{name}' is not a parameter of this model. Parameters: {par_names}")
        fixed[name] = float(val)
    bounds = {}
    for key, rng_ in (ranges or {}).items():
        name = key.name if isinstance(key, sp.Symbol) else str(key)
        if name not in par_names:
            raise ModelSpecificationError(f"'{name}' is not a parameter of this model.")
        low, high = map(float, rng_)
        if not 0 <= low < high or (sampling == "loguniform" and low <= 0):
            raise ValueError(f"Invalid range for {name}: {rng_}.")
        bounds[name] = (low, high)
    init_given = {}
    for key, val in (initial or {}).items():
        name = key.name if isinstance(key, sp.Symbol) else str(key)
        if name not in var_names:
            raise ModelSpecificationError(f"'{name}' is not a state variable. Variables: {var_names}")
        init_given[name] = float(val)

    rhs_expr = [model.equations[v] for v in variables]
    f = sp.lambdify(variables + params, rhs_expr, "numpy")
    rng = np.random.default_rng(seed)
    t_eval = np.linspace(t0, t1, n_points)
    random_names = [n for n in par_names if n not in fixed]

    def draw() -> Dict[str, float]:
        p = dict(fixed)
        for name in random_names:
            low, high = bounds.get(name, default_range)
            p[name] = float(np.exp(rng.uniform(np.log(low), np.log(high)))) if sampling == "loguniform" \
                else float(rng.uniform(low, high))
        return p

    def r0_and_dfe(p):
        try:
            r0 = model.R0_numeric(p)
        except (R0ComputeError, ImportError, ValueError, ZeroDivisionError, TypeError):
            r0 = None
        try:
            dfe = {k.name: float(v) for k, v in model.dfe_numeric(p).items()}
        except (R0ComputeError, ImportError, ValueError, ZeroDivisionError, TypeError):
            dfe = None
        return r0, dfe

    def in_range(r0):
        if R0_range is None:
            return True
        if r0 is None:
            return False
        low, high = R0_range
        return (low is None or r0 >= low) and (high is None or r0 <= high)

    runs: List[Run] = []
    infected = {v.name for v in model.infected}
    for _ in range(n_runs):
        tries = max_tries if (R0_range is not None and random_names) else 1
        for _attempt in range(tries):
            p = draw()
            r0, dfe = r0_and_dfe(p)
            if in_range(r0):
                break
        if not in_range(r0):
            if random_names:
                raise ModelSpecificationError(
                    f"No parameter draw with R0 in {R0_range} was found in {max_tries} tries; "
                    f"widen the ranges or fix some parameters with values=.")
            raise ModelSpecificationError(f"R0 = {r0} at the given values is outside R0_range = {R0_range}.")

        y0 = {}
        if len(init_given) < len(var_names):
            if dfe is None:
                missing = [n for n in var_names if n not in init_given]
                raise ModelSpecificationError(
                    f"The disease-free equilibrium could not be evaluated, so give the initial values of "
                    f"{missing} with initial={{...}}.")
            total = sum(max(0.0, dfe[n]) for n in var_names if n not in infected)
            seed_size = perturbation * (total if total > 0 else 1.0)
            for n in var_names:
                y0[n] = seed_size if n in infected else max(0.0, dfe[n])
        y0.update(init_given)
        pvals = [p[n] for n in par_names]

        def rhs(_t, y, pvals=pvals):
            return np.asarray(f(*y, *pvals), dtype=float).ravel()

        with np.errstate(all="ignore"):
            sol = solve_ivp(rhs, (t0, t1), [y0[n] for n in var_names], t_eval=t_eval,
                            method=method, rtol=rtol, atol=atol)
        y = sol.y if sol.success else np.full((len(var_names), n_points), np.nan)
        if sol.success and y.shape[1] < n_points:  # pragma: no cover - solver stopped early
            y = np.hstack([y, np.full((len(var_names), n_points - y.shape[1]), np.nan)])
        runs.append(Run(p, y0, y, r0, dfe, bool(sol.success), str(sol.message)))
    return SimulationResult(model, t_eval, var_names, runs)


class SimulationResult:
    """Runs of :func:`simulate`.

    ``result.t`` are the times and ``result.runs`` the :class:`Run` objects;
    for a single run, ``result.y``, ``result.parameters``, ``result.R0`` and
    ``result.initial`` refer to it, and ``result["I"]`` is the trajectory of
    ``I``.
    """

    def __init__(self, model, t, variables, runs):
        self.model = model
        self.t = t
        self.variables = list(variables)
        self.runs: List[Run] = runs

    # ------------------------------------------------------------ access
    def _single(self):
        return self.runs[0]

    @property
    def y(self) -> np.ndarray:
        return self._single().y

    @property
    def parameters(self) -> Dict[str, float]:
        return self._single().parameters

    @property
    def initial(self) -> Dict[str, float]:
        return self._single().initial

    @property
    def R0(self) -> Optional[float]:
        return self._single().R0

    @property
    def R0_values(self) -> np.ndarray:
        """R0 of every run (NaN where it could not be computed)."""
        return np.array([np.nan if r.R0 is None else r.R0 for r in self.runs])

    def __getitem__(self, variable) -> np.ndarray:
        name = variable.name if isinstance(variable, sp.Symbol) else str(variable)
        if name not in self.variables:
            raise KeyError(f"'{name}' is not a state variable. Variables: {self.variables}")
        idx = self.variables.index(name)
        if len(self.runs) == 1:
            return self.runs[0].y[idx]
        return np.array([r.y[idx] for r in self.runs])

    def final(self, run: int = 0) -> Dict[str, float]:
        """State at the last time of a run."""
        return {n: float(self.runs[run].y[i, -1]) for i, n in enumerate(self.variables)}

    def to_dataframe(self):
        """Long table with columns ``run``, ``t``, the variables and ``R0`` (needs pandas)."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("to_dataframe() needs pandas: pip install pandas") from exc
        frames = []
        for k, r in enumerate(self.runs):
            df = pd.DataFrame(r.y.T, columns=self.variables)
            df.insert(0, "t", self.t)
            df.insert(0, "run", k)
            df["R0"] = r.R0
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def summary(self):
        """One row per run: R0 and the parameter values (pandas DataFrame if available)."""
        rows = [{"run": k, "R0": r.R0, **r.parameters} for k, r in enumerate(self.runs)]
        try:
            import pandas as pd
            return pd.DataFrame(rows).set_index("run")
        except ImportError:  # pragma: no cover
            return rows

    def __len__(self) -> int:
        return len(self.runs)

    def __repr__(self) -> str:
        r0 = self.R0_values
        if len(self.runs) == 1:
            txt = "None" if np.isnan(r0[0]) else f"{r0[0]:.4g}"
            return f"SimulationResult(1 run, R0 = {txt}, t in [{self.t[0]:g}, {self.t[-1]:g}])"
        ok = r0[~np.isnan(r0)]
        return (f"SimulationResult({len(self.runs)} runs, R0 < 1 in {int((ok < 1).sum())}, "
                f"R0 > 1 in {int((ok > 1).sum())}, t in [{self.t[0]:g}, {self.t[-1]:g}])")

    # ------------------------------------------------------------ helpers
    def _names(self, variables):
        if variables is None:
            return list(self.variables)
        if isinstance(variables, (str, sp.Symbol)):
            variables = [variables]
        names = [v.name if isinstance(v, sp.Symbol) else str(v) for v in variables]
        bad = [n for n in names if n not in self.variables]
        if bad:
            raise KeyError(f"Not state variables: {bad}. Variables: {self.variables}")
        return names

    @staticmethod
    def _per_variable(option, names, default):
        """Option given as a dict {variable: value}, a list (in order) or a single value."""
        if option is None:
            return {n: default(k) for k, n in enumerate(names)}
        if isinstance(option, Mapping):
            return {n: option.get(n, default(k)) for k, n in enumerate(names)}
        if isinstance(option, (list, tuple)):
            return {n: option[k % len(option)] for k, n in enumerate(names)}
        return {n: option for n in names}

    def _title(self, title):
        if title is None:
            if len(self.runs) == 1 and self.R0 is not None:
                return rf"$\mathcal{{R}}_0 = {self.R0:.3g}$"
            return None
        if "{R0}" in title:
            r0 = "?" if self.R0 is None else f"{self.R0:.3g}"
            title = title.replace("{R0}", r0)
        return title

    @staticmethod
    def _style(ax, grid, logy, logx, xlim, ylim):
        if grid:
            ax.grid(True, color=_GRID, linewidth=0.6)
            ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if logy:
            ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    @staticmethod
    def _save(fig, save, dpi):
        if save:
            folder = os.path.dirname(str(save))
            if folder:
                os.makedirs(folder, exist_ok=True)
            fig.savefig(save, dpi=dpi, bbox_inches="tight")

    # --------------------------------------------------------------- plots
    def plot(self, variables=None, ax=None, *, subplots: bool = False, ncols: int = 2,
             figsize: Optional[Tuple[float, float]] = None, title: Optional[str] = None,
             titles: Optional[Mapping] = None, xlabel: str = "$t$", ylabel: Optional[str] = None,
             labels=None, colors=None, linestyles=None, linewidth: float = 1.8,
             alpha: Optional[float] = None, legend: Union[bool, str] = True, legend_kw: Optional[dict] = None,
             logy: bool = False, logx: bool = False, xlim=None, ylim=None, grid: bool = True,
             color_by_R0: Optional[bool] = None, show_dfe: bool = False,
             save: Optional[str] = None, dpi: int = 200):
        """Time series of the selected variables.

        Parameters
        ----------
        variables
            Variables to draw (names or symbols); all by default.
        ax
            Existing matplotlib axes to draw on (single panel), e.g. to compare
            several simulations in one figure.
        subplots, ncols
            One panel per variable, in ``ncols`` columns.
        figsize
            Figure size in inches.
        title
            Figure title; ``"{R0}"`` is replaced by the value of R0. By default
            a single run is titled with its R0.
        titles
            ``{variable: title}`` of each panel when ``subplots=True``.
        xlabel, ylabel
            Axis labels (LaTeX between ``$...$`` is allowed).
        labels, colors, linestyles
            Per variable: a dict ``{variable: value}``, a list in the order of
            ``variables`` or a single value. Labels default to the LaTeX name.
        linewidth, alpha
            Line width and opacity (runs of an ensemble default to 0.6).
        legend, legend_kw
            Show the legend (``True``, ``False`` or a matplotlib location such
            as ``"upper right"``) and extra arguments for ``ax.legend``.
        logy, logx, xlim, ylim, grid
            Axis scales, limits and grid.
        color_by_R0
            Colour each run by R0 < 1 or R0 > 1 (default for ensembles); the
            variables are then told apart by line style.
        show_dfe
            Dashed horizontal line at the disease-free value of each variable.
        save, dpi
            File to save the figure to (folders are created).

        Returns ``(fig, axes)``.
        """
        plt = _require_matplotlib()
        names = self._names(variables)
        ensemble = len(self.runs) > 1
        if color_by_R0 is None:
            color_by_R0 = ensemble and not np.all(np.isnan(self.R0_values))
        labels_ = self._per_variable(labels, names, lambda k: f"${sp.latex(sp.Symbol(names[k]))}$")
        colors_ = self._per_variable(colors, names, lambda k: PALETTE[k % len(PALETTE)])
        vary_style = color_by_R0 and not subplots and len(names) > 1
        styles_ = self._per_variable(linestyles, names,
                                     lambda k: _LINESTYLES[k % len(_LINESTYLES)] if vary_style else "-")
        alpha_ = alpha if alpha is not None else (0.6 if ensemble else 1.0)

        if subplots:
            if ax is not None:
                raise ValueError("ax cannot be used with subplots=True.")
            n = len(names)
            cols = max(1, min(ncols, n))
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=figsize or (4.2 * cols, 3.0 * rows),
                                     sharex=True, squeeze=False)
            axes = axes.ravel()
            for extra in axes[n:]:
                extra.set_visible(False)
            panel = {nm: axes[k] for k, nm in enumerate(names)}
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize or (6.4, 4.0))
            else:
                fig = ax.figure
            axes = np.array([ax])
            panel = {nm: ax for nm in names}

        handles = {}
        for k_run, run in enumerate(self.runs):
            for nm in names:
                i = self.variables.index(nm)
                if color_by_R0 and run.R0 is not None:
                    above = run.R0 > 1
                    color = _ABOVE if above else _BELOW
                    key = ("R0", above, nm if not subplots and len(names) > 1 else None)
                    lab = (r"$\mathcal{R}_0 > 1$" if above else r"$\mathcal{R}_0 < 1$")
                    if key[2] is not None:
                        lab = f"{labels_[nm]}, {lab}"
                else:
                    color = colors_[nm]
                    key = ("var", nm)
                    lab = labels_[nm]
                # label only the first line of each series, so ax.legend() also works
                # when several simulations are drawn on the same axes
                line, = panel[nm].plot(self.t, run.y[i], color=color, linestyle=styles_[nm],
                                       linewidth=linewidth, alpha=alpha_,
                                       label=lab if key not in handles else "_nolegend_")
                handles.setdefault(key, (line, lab))
                if show_dfe and run.dfe is not None and nm in run.dfe and (k_run == 0 or not ensemble):
                    panel[nm].axhline(run.dfe[nm], color=color, linestyle=(0, (4, 3)), linewidth=1.0,
                                      alpha=0.8)

        for nm, a in ({"_": axes[0]} if not subplots else panel).items():
            self._style(a, grid, logy, logx, xlim, ylim)
        if subplots:
            for nm, a in panel.items():
                a.set_title((titles or {}).get(nm, labels_[nm]), loc="left", fontsize=11)
                a.set_ylabel(ylabel or "")
            for a in axes[-cols:] if len(axes) else []:
                a.set_xlabel(xlabel)
            for a in axes:
                if a.get_visible():
                    a.tick_params(labelbottom=True)
        else:
            axes[0].set_xlabel(xlabel)
            if ylabel is not None:
                axes[0].set_ylabel(ylabel)
            elif len(names) == 1:
                axes[0].set_ylabel(labels_[names[0]])

        # one series per panel needs no legend (the panel title names it)
        if subplots:
            show_legend = legend is not False and color_by_R0 and len(handles) >= 1
        else:
            show_legend = legend is not False and (len(handles) > 1 or (color_by_R0 and len(handles) >= 1))
        if show_legend:
            kw = dict(legend_kw or {})
            if isinstance(legend, str):
                kw.setdefault("loc", legend)
            kw.setdefault("frameon", True)
            kw.setdefault("framealpha", 0.9)
            kw.setdefault("edgecolor", "none")
            items = list(handles.values())
            if subplots and not isinstance(legend, str):
                # above the panels, so it never covers a trajectory
                kw.setdefault("loc", "upper right")
                kw.setdefault("ncol", len(items))
                kw.setdefault("bbox_to_anchor", (1.0, 1.0 + 0.02 / fig.get_size_inches()[1]))
                fig.legend([h for h, _ in items], [lab for _, lab in items], **kw)
            else:
                axes[0].legend([h for h, _ in items], [lab for _, lab in items], **kw)

        t = self._title(title)
        if t and not subplots:
            axes[0].set_title(t, loc="left", fontsize=12)
        if subplots and (show_legend and not isinstance(legend, str) or t):
            # reserve a fixed band (in inches) at the top for the title and the legend
            height = fig.get_size_inches()[1]
            fig.tight_layout(rect=(0, 0, 1, 1 - 0.42 / height))
            if t:
                fig.suptitle(t, x=0.01, y=1 - 0.06 / height, ha="left", va="top", fontsize=12)
        else:
            fig.tight_layout()
        self._save(fig, save, dpi)
        return fig, (axes if subplots else axes[0])

    def plot_phase(self, x, y, ax=None, *, figsize: Optional[Tuple[float, float]] = None,
                   title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                   color: Optional[str] = None, color_by_R0: Optional[bool] = None,
                   linewidth: float = 1.6, alpha: Optional[float] = None, mark_start: bool = True,
                   show_dfe: bool = True, legend: Union[bool, str] = True, logx: bool = False,
                   logy: bool = False, xlim=None, ylim=None, grid: bool = True,
                   save: Optional[str] = None, dpi: int = 200):
        """Phase plane ``y`` against ``x`` (every run), with start points and the DFE.

        Returns ``(fig, ax)``.
        """
        plt = _require_matplotlib()
        nx, ny = self._names([x, y])
        ix, iy = self.variables.index(nx), self.variables.index(ny)
        ensemble = len(self.runs) > 1
        if color_by_R0 is None:
            color_by_R0 = not np.all(np.isnan(self.R0_values)) and ensemble
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (5.2, 4.4))
        else:
            fig = ax.figure
        handles = {}
        for run in self.runs:
            if color_by_R0 and run.R0 is not None:
                above = run.R0 > 1
                c = _ABOVE if above else _BELOW
                key, lab = above, (r"$\mathcal{R}_0 > 1$" if above else r"$\mathcal{R}_0 < 1$")
            else:
                c, key, lab = color or PALETTE[0], "traj", None
            line, = ax.plot(run.y[ix], run.y[iy], color=c, linewidth=linewidth,
                            alpha=alpha if alpha is not None else (0.7 if ensemble else 1.0),
                            label=lab if (lab and key not in handles) else "_nolegend_")
            if lab:
                handles.setdefault(key, (line, lab))
            if mark_start:
                ax.plot(run.y[ix, 0], run.y[iy, 0], "o", color=c, markersize=5, markeredgecolor="white",
                        markeredgewidth=1.0)
        if show_dfe:
            pts = {(round(r.dfe[nx], 12), round(r.dfe[ny], 12)) for r in self.runs
                   if r.dfe is not None and nx in r.dfe and ny in r.dfe}
            first = True
            for px, py in pts:
                h, = ax.plot(px, py, marker="X", color="#0b0b0b", markersize=9, linestyle="none",
                             markeredgecolor="white", markeredgewidth=1.0)
                if first:
                    handles.setdefault("dfe", (h, "DFE"))
                    first = False
        self._style(ax, grid, logy, logx, xlim, ylim)
        ax.set_xlabel(xlabel or f"${sp.latex(sp.Symbol(nx))}$")
        ax.set_ylabel(ylabel or f"${sp.latex(sp.Symbol(ny))}$")
        if legend is not False and handles:
            kw = {"frameon": True, "framealpha": 0.9, "edgecolor": "none"}
            if isinstance(legend, str):
                kw["loc"] = legend
            ax.legend([h for h, _ in handles.values()], [lab for _, lab in handles.values()], **kw)
        t = self._title(title)
        if t:
            ax.set_title(t, loc="left", fontsize=12)
        fig.tight_layout()
        self._save(fig, save, dpi)
        return fig, ax
