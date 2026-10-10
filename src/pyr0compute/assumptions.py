"""Assumptions (A1)-(A5) of van den Driessche & Watmough (2002).

The model is written, for every compartment ``i = 1..n`` (infected first,
``i <= m``), as

    dx_i/dt = F_i(x) - V_i(x),      V_i = V_i^- - V_i^+,

where ``F_i`` is the appearance of new infections, ``V_i^+`` every other
inflow and ``V_i^-`` every outflow. Theorem 2 of the paper (R0 < 1: the DFE
is locally asymptotically stable; R0 > 1: it is unstable) holds under:

(A1) x >= 0  =>  F_i, V_i^+, V_i^- >= 0.
(A2) x_i = 0  =>  V_i^- = 0.
(A3) F_i = 0 for i > m.
(A4) x in X_s (all infected = 0)  =>  F_i = 0 and V_i^+ = 0 for i <= m.
(A5) With F = 0, all eigenvalues of Df(x0) have negative real parts;
     equivalently (Lemma 1) those of V and of J_4 have positive real parts.

:func:`check_assumptions` decides each one as ``"holds"``, ``"fails"`` or
``"undecided"``, recording the basis of the decision:

* ``"construction"``: guaranteed by how pyR0compute builds the model (A3);
* ``"symbolic"``: proved (or refuted) for every positive value of the
  parameters, with SymPy's sign assumptions;
* ``"values"``: decided at the parameter values given with ``values=``;
* ``"sampling"``: decided from random parameter samples. A violation found
  for every sampled parameter set is reported as ``"fails"``; a property
  that holds in all samples but was not proved, or that holds for some
  samples only, is ``"undecided"`` (pass ``values=`` to decide it at a point).

The split of ``V_i`` into ``V_i^+`` and ``V_i^-`` is not unique; the one used
puts every expanded term of ``V_i`` with a positive sign in ``V_i^-`` and
every negative one, with its sign changed, in ``V_i^+``.

Reference: P. van den Driessche, J. Watmough (2002). Mathematical Biosciences
180, 29-48.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import sympy as sp

from .exceptions import ModelSpecificationError, R0ComputeError

__all__ = ["AssumptionResult", "AssumptionsReport", "check_assumptions"]

HOLDS, FAILS, UNDECIDED = "holds", "fails", "undecided"
_SEVERITY = {HOLDS: 0, UNDECIDED: 1, FAILS: 2}
_BASIS_RANK = {"construction": 0, "symbolic": 1, "values": 2, "sampling": 3}


def _t(en: str, es: str) -> Dict[str, str]:
    return {"en": en, "es": es}


STATUS_TEXT = {
    HOLDS: _t("holds", "cumple"),
    FAILS: _t("fails", "no cumple"),
    UNDECIDED: _t("undecided", "no se pudo decidir"),
}
BASIS_TEXT = {
    "construction": _t("by construction", "por construcción"),
    "symbolic": _t("symbolic proof for all positive parameters",
                   "demostración simbólica para todo parámetro positivo"),
    "values": _t("at the given parameter values", "en los valores de los parámetros dados"),
    "sampling": _t("random parameter samples", "muestras aleatorias de los parámetros"),
}
STATEMENTS = {
    "A1": _t(r"If $x \ge 0$, then $\mathcal{F}_i,\ \mathcal{V}_i^+,\ \mathcal{V}_i^- \ge 0$ for $i = 1,\dots,n$.",
             r"Si $x \ge 0$, entonces $\mathcal{F}_i,\ \mathcal{V}_i^+,\ \mathcal{V}_i^- \ge 0$ para $i = 1,\dots,n$."),
    "A2": _t(r"If $x_i = 0$, then $\mathcal{V}_i^- = 0$.",
             r"Si $x_i = 0$, entonces $\mathcal{V}_i^- = 0$."),
    "A3": _t(r"$\mathcal{F}_i = 0$ for $i > m$ (uninfected compartments).",
             r"$\mathcal{F}_i = 0$ para $i > m$ (compartimentos no infectados)."),
    "A4": _t(r"If $x \in X_s$, then $\mathcal{F}_i(x) = 0$ and $\mathcal{V}_i^+(x) = 0$ for $i = 1,\dots,m$.",
             r"Si $x \in X_s$, entonces $\mathcal{F}_i(x) = 0$ y $\mathcal{V}_i^+(x) = 0$ para $i = 1,\dots,m$."),
    "A5": _t(r"If $\mathcal{F}$ is set to zero, all eigenvalues of $Df(x_0)$ have negative real parts; "
             r"equivalently, those of $V$ and of $J_4$ have positive real parts.",
             r"Con $\mathcal{F} = 0$, todos los valores propios de $Df(x_0)$ tienen parte real negativa; "
             r"equivalentemente, los de $V$ y los de $J_4$ tienen parte real positiva."),
}
SHORT = {
    "A1": _t("non-negative flows", "flujos no negativos"),
    "A2": _t("no outflow from an empty compartment", "sin salidas de un compartimento vacío"),
    "A3": _t("no new infections in uninfected compartments", "sin nuevas infecciones en no infectados"),
    "A4": _t("the disease-free subspace is invariant", "el subespacio libre de infección es invariante"),
    "A5": _t("the DFE is stable without new infections", "el DFE es estable sin nuevas infecciones"),
}


# ----------------------------------------------------------------- results
@dataclass
class _Decision:
    status: str
    basis: str
    note: Optional[Dict[str, str]] = None
    witness: Optional[Dict[sp.Symbol, float]] = None


@dataclass
class _Item:
    """One verified statement inside an assumption (e.g. V_S^- at S = 0 is 0)."""

    math: str                      # LaTeX of the statement checked
    decision: _Decision


@dataclass
class AssumptionResult:
    """Outcome of one assumption: ``status`` is ``"holds"``, ``"fails"`` or ``"undecided"``."""

    key: str
    status: str
    basis: str
    items: List[_Item] = field(default_factory=list)
    notes: List[Dict[str, str]] = field(default_factory=list)

    @property
    def holds(self) -> bool:
        return self.status == HOLDS

    def __repr__(self) -> str:
        return f"AssumptionResult({self.key}: {self.status}, {self.basis})"


def _combine(key: str, items: List[_Item], notes=None) -> AssumptionResult:
    if not items:
        return AssumptionResult(key, HOLDS, "symbolic", [], list(notes or []))
    worst = max(items, key=lambda it: (_SEVERITY[it.decision.status], _BASIS_RANK[it.decision.basis]))
    status = worst.decision.status
    if status == HOLDS:
        basis = max((it.decision.basis for it in items), key=lambda b: _BASIS_RANK[b])
    else:
        basis = worst.decision.basis
    return AssumptionResult(key, status, basis, items, list(notes or []))


# ----------------------------------------------------------------- checker
class _Checker:
    """Sign and zero decisions over x >= 0 and the parameters of a model."""

    N_X = 64       # state samples per parameter set
    N_PAR = 40     # parameter sets for the flow checks (A1, A2, A4)

    def __init__(self, model, values, n_samples: int, seed: int):
        self.m = model
        self.rng = np.random.default_rng(seed)
        self.x = list(model._x)
        self.params = sorted((p for p in model._to_orig if p not in model._x), key=lambda s: s.name)
        self.n_samples = n_samples
        self.vals: Optional[Dict[sp.Symbol, sp.Float]] = None
        if values is not None:
            given = model._values(values)
            missing = sorted(str(model._orig(p)) for p in self.params if p not in given)
            if missing:
                raise ModelSpecificationError(f"Missing values for parameters: {missing}")
            self.vals = {k: sp.Float(v) for k, v in given.items()}
            self.psets = np.array([[float(self.vals[p]) for p in self.params]])
        else:
            self.psets = 10 ** self.rng.uniform(-2, 1, (self.N_PAR, len(self.params)))
        self.xs = 10 ** self.rng.uniform(-3, 3, (self.N_X, len(self.x)))

    # --------------------------------------------------------- evaluation
    def evaluate(self, expr) -> np.ndarray:
        """Values of ``expr`` on (parameter set, state) samples, shape (P, X); NaN if undefined."""
        P, X = len(self.psets), len(self.xs)
        f = sp.lambdify(self.x + self.params, expr, "numpy")
        cols = [np.tile(self.xs[:, k], P) for k in range(len(self.x))]
        cols += [np.repeat(self.psets[:, k], X) for k in range(len(self.params))]
        with np.errstate(all="ignore"):
            try:
                out = np.asarray(f(*cols), dtype=complex)
            except (ZeroDivisionError, OverflowError, ValueError, TypeError):
                return np.full((P, X), np.nan)
        out = np.broadcast_to(out, (P * X,)).copy()
        bad = ~np.isfinite(out) | (np.abs(out.imag) > 1e-9 * np.maximum(1.0, np.abs(out.real)))
        real = out.real
        real[bad] = np.nan
        return real.reshape(P, X)

    def _witness(self, expr, p: int, k: int) -> Dict[sp.Symbol, float]:
        point = {}
        free = sp.sympify(expr).free_symbols
        for j, s in enumerate(self.x):
            if s in free:
                point[s] = float(self.xs[k, j])
        if self.vals is None:
            for j, s in enumerate(self.params):
                if s in free:
                    point[s] = float(self.psets[p, j])
        return point

    @staticmethod
    def _tol(vals: np.ndarray) -> float:
        finite = vals[np.isfinite(vals)]
        return 1e-10 * max(1.0, float(np.max(np.abs(finite)))) if finite.size else 0.0

    # ------------------------------------------------------------ decisions
    def nonneg(self, expr) -> _Decision:
        """Is ``expr >= 0`` for all x >= 0 (and all positive parameters, or at ``values``)?"""
        e = sp.sympify(expr)
        if e == 0 or e.is_nonnegative:
            return _Decision(HOLDS, "symbolic")
        if self.vals is not None:
            ev = e.xreplace(self.vals)
            if ev == 0 or ev.is_nonnegative:
                return _Decision(HOLDS, "values")
            vals = self.evaluate(e)
            neg = vals < -self._tol(vals)
            if neg.any():
                p, k = map(int, np.argwhere(neg)[0])
                return _Decision(FAILS, "values", witness=self._witness(e, p, k))
            return _Decision(UNDECIDED, "values", _t(
                f"not proved; no counterexample in {vals.size} state samples",
                f"no demostrado; sin contraejemplo en {vals.size} muestras de estados"))
        vals = self.evaluate(e)
        neg = vals < -self._tol(vals)
        if e.is_negative:
            p, k = (map(int, np.argwhere(neg)[0]) if neg.any() else (0, 0))
            return _Decision(FAILS, "symbolic", witness=self._witness(e, p, k))
        rows = neg.any(axis=1)
        if rows.any():
            p, k = map(int, np.argwhere(neg)[0])
            if rows.all():
                return _Decision(FAILS, "sampling", _t(
                    f"negative for some $x \\ge 0$ in all {len(rows)} parameter samples",
                    f"negativo para algún $x \\ge 0$ en las {len(rows)} muestras de parámetros"),
                    witness=self._witness(e, p, k))
            return _Decision(UNDECIDED, "sampling", _t(
                f"depends on the parameters: negative in {int(rows.sum())} of {len(rows)} samples",
                f"depende de los parámetros: negativo en {int(rows.sum())} de {len(rows)} muestras"),
                witness=self._witness(e, p, k))
        return _Decision(UNDECIDED, "sampling", _t(
            f"not proved; non-negative in all {vals.size} samples",
            f"no demostrado; no negativo en las {vals.size} muestras"))

    def zero(self, expr) -> _Decision:
        """Is ``expr = 0`` for all x >= 0 (and all positive parameters, or at ``values``)?"""
        e = sp.cancel(sp.together(sp.sympify(expr)))
        if e == 0:
            return _Decision(HOLDS, "symbolic")
        if self.vals is not None:
            ev = sp.cancel(e.xreplace(self.vals))
            if ev == 0:
                return _Decision(HOLDS, "values")
            vals = self.evaluate(e)
            nz = np.abs(vals) > self._tol(vals)
            if nz.any():
                p, k = map(int, np.argwhere(nz)[0])
                return _Decision(FAILS, "values", witness=self._witness(e, p, k))
            return _Decision(UNDECIDED, "values", _t(
                "not identically zero symbolically, but zero in all samples",
                "no es idénticamente cero simbólicamente, pero vale cero en todas las muestras"))
        vals = self.evaluate(e)
        nz = np.abs(vals) > self._tol(vals)
        rows = nz.any(axis=1)
        p, k = (map(int, np.argwhere(nz)[0]) if nz.any() else (0, 0))
        if e.is_positive or e.is_negative:
            return _Decision(FAILS, "symbolic", witness=self._witness(e, p, k))
        if rows.all():
            return _Decision(FAILS, "sampling", _t(
                f"non-zero for some $x$ in all {len(rows)} parameter samples",
                f"distinto de cero para algún $x$ en las {len(rows)} muestras de parámetros"),
                witness=self._witness(e, p, k))
        if rows.any():
            return _Decision(UNDECIDED, "sampling", _t(
                f"depends on the parameters: non-zero in {int(rows.sum())} of {len(rows)} samples",
                f"depende de los parámetros: distinto de cero en {int(rows.sum())} de {len(rows)} muestras"),
                witness=self._witness(e, p, k))
        return _Decision(UNDECIDED, "sampling", _t(
            "not identically zero symbolically, but zero in all samples",
            "no es idénticamente cero simbólicamente, pero vale cero en todas las muestras"))

    def sign(self, term) -> Tuple[Optional[int], str]:
        """Sign of a term on x > 0: (+1 / -1 / None, basis)."""
        if term.is_positive:
            return 1, "symbolic"
        if term.is_negative:
            return -1, "symbolic"
        if self.vals is not None:
            tv = term.xreplace(self.vals)
            if tv.is_positive:
                return 1, "values"
            if tv.is_negative:
                return -1, "values"
        vals = self.evaluate(term)
        finite = vals[np.isfinite(vals)]
        basis = "values" if self.vals is not None else "sampling"
        if finite.size and np.all(finite > 0):
            return 1, basis
        if finite.size and np.all(finite < 0):
            return -1, basis
        return None, basis

    # ----------------------------------------------------------- stability
    def _block_symbolic(self, B) -> Tuple[Optional[bool], str]:
        """Do all eigenvalues of the block B have positive real part? (True/False/None, LaTeX)."""
        n = B.shape[0]
        if n == 1:
            e = sp.simplify(B[0, 0])
            tex = sp.latex(self.m._orig(e))
            if e.is_positive:
                return True, rf"{tex} > 0"
            if e.is_nonpositive:
                return False, rf"{tex} \le 0"
            return None, tex
        if n == 2:
            tr, det = sp.simplify(B.trace()), sp.simplify(B.det())
            tex = (rf"\operatorname{{tr}} = {sp.latex(self.m._orig(tr))},\ "
                   rf"\det = {sp.latex(self.m._orig(det))}")
            if tr.is_positive and det.is_positive:
                return True, tex + r" > 0"
            if tr.is_nonpositive or det.is_nonpositive:
                return False, tex
            return None, tex
        offdiag = [B[i, j] for i in range(n) for j in range(n) if i != j]
        if all(sp.sympify(e).is_nonpositive or e == 0 for e in offdiag):
            # Z-matrix: positive stable iff all leading principal minors are positive
            minors = [sp.simplify(B.extract(list(range(k)), list(range(k))).det()) for k in range(1, n + 1)]
            if all(mn.is_positive for mn in minors):
                return True, r"\text{Z-matrix with positive leading principal minors}"
            if any(mn.is_nonpositive for mn in minors):
                return False, r"\text{Z-matrix with a non-positive leading principal minor}"
        return None, ""

    def stable_symbolic(self, M) -> Tuple[Optional[bool], List[str]]:
        n = M.shape[0]
        if n == 0:
            return True, []
        pattern = [[i != j and sp.simplify(M[i, j]) != 0 for j in range(n)] for i in range(n)]
        edges = [(i, j) for i in range(n) for j in range(n) if pattern[i][j]]
        comps = sp.utilities.iterables.strongly_connected_components((list(range(n)), edges))
        verdicts, texts = [], []
        for comp in comps:
            ok, tex = self._block_symbolic(M.extract(comp, comp))
            verdicts.append(ok)
            if tex:
                texts.append(tex)
        if any(v is False for v in verdicts):
            return False, texts
        if all(v is True for v in verdicts):
            return True, texts
        return None, texts

    def stable(self, M, point: Mapping[sp.Symbol, sp.Expr]) -> Tuple[_Decision, List[str], Optional[np.ndarray]]:
        """Do all eigenvalues of M (at the DFE ``point``) have positive real part?"""
        ok, texts = self.stable_symbolic(M)
        if ok is True:
            return _Decision(HOLDS, "symbolic"), texts, None
        if ok is False:
            return _Decision(FAILS, "symbolic"), texts, None
        if M.shape[0] == 0:
            return _Decision(HOLDS, "symbolic"), texts, None
        from .model import DFESymbol
        stars = sorted({s for e in list(M) + list(point.values()) for s in sp.sympify(e).free_symbols
                        if isinstance(s, DFESymbol)}, key=lambda s: s.name)
        args = self.params + stars
        fM = sp.lambdify(args, M, "numpy")
        fp = sp.lambdify(args, [point[x] for x in point] or [0], "numpy")

        def at(pvals: Dict[sp.Symbol, float]):
            try:
                star_vals = self.m._numeric_stars(pvals) if stars else {}
            except (R0ComputeError, ImportError, ValueError, ZeroDivisionError):
                return None, None
            a = [float(pvals[p]) for p in self.params] + [float(star_vals[s]) for s in stars]
            with np.errstate(all="ignore"):
                dfe = np.array(fp(*a), dtype=complex).ravel()
                Mn = np.array(fM(*a), dtype=complex)
            if not (np.all(np.isfinite(dfe)) and np.all(np.isfinite(Mn))) or np.any(dfe.real < -1e-9):
                return None, None
            eig = np.linalg.eigvals(Mn)
            scale = max(1.0, float(np.max(np.abs(eig))))
            return bool(np.all(eig.real > 1e-10 * scale)), eig

        if self.vals is not None:
            stable, eig = at(self.vals)
            if stable is None:
                return _Decision(UNDECIDED, "values", _t(
                    "the DFE is not defined or not non-negative at these values",
                    "el DFE no está definido o no es no negativo en estos valores")), texts, None
            return _Decision(HOLDS if stable else FAILS, "values"), texts, eig
        n_ok = n_bad = 0
        witness = None
        for _ in range(self.n_samples):
            pvals = dict(zip(self.params, 10 ** self.rng.uniform(-2, 1, len(self.params))))
            stable, _eig = at(pvals)
            if stable is None:
                continue
            if stable:
                n_ok += 1
            else:
                n_bad += 1
                if witness is None:
                    free = set().union(*(sp.sympify(e).free_symbols for e in M)) if len(M) else set()
                    witness = {p: v for p, v in pvals.items() if p in free}
        total = n_ok + n_bad
        if total == 0:
            return _Decision(UNDECIDED, "sampling", _t(
                "no parameter sample gave a non-negative DFE",
                "ninguna muestra de parámetros dio un DFE no negativo")), texts, None
        if n_bad == 0:
            return _Decision(UNDECIDED, "sampling", _t(
                f"not proved; stable in all {total} feasible samples (pass values=... to decide at a point)",
                f"no demostrado; estable en las {total} muestras factibles (use values=... para decidir en un punto)")
            ), texts, None
        if n_ok == 0:
            return _Decision(FAILS, "sampling", _t(
                f"unstable in all {total} feasible samples", f"inestable en las {total} muestras factibles"),
                witness=witness), texts, None
        return _Decision(UNDECIDED, "sampling", _t(
            f"depends on the parameters: unstable in {n_bad} of {total} feasible samples",
            f"depende de los parámetros: inestable en {n_bad} de {total} muestras factibles"),
            witness=witness), texts, None


# ------------------------------------------------------------ the checks
def _flows(model, chk: _Checker):
    """Canonical decomposition F_i, V_i^+, V_i^- of every compartment, and the A1 items for V."""
    rows, items = [], []
    for x in model._x:
        F = model._F_terms.get(x, sp.Integer(0))
        V = sp.expand(F - model._f[x])
        plus, minus = [], []
        for term in sp.Add.make_args(V):
            if term == 0:
                continue
            sgn, basis = chk.sign(term)
            if sgn == 1:
                minus.append(term)
            elif sgn == -1:
                plus.append(-term)
            else:
                minus.append(term)  # kept in V^- so the decomposition is complete; flagged below
                items.append(_Item(
                    rf"\text{{sign of }} {sp.latex(model._orig(term))} \text{{ in }} \mathcal{{V}}_{{{sp.latex(model._to_orig[x])}}}",
                    _Decision(UNDECIDED, basis, _t(
                        "the term changes sign, so it is neither an inflow nor an outflow",
                        "el término cambia de signo: no es ni entrada ni salida"))))
                continue
            if basis != "symbolic":
                items.append(_Item(
                    rf"\text{{sign of }} {sp.latex(model._orig(term))}",
                    _Decision(HOLDS if basis == "values" else UNDECIDED, basis, None if basis == "values" else _t(
                        "sign consistent in all samples but not proved",
                        "signo constante en todas las muestras, pero no demostrado"))))
        rows.append((x, F, sp.Add(*plus), sp.Add(*minus)))
    return rows, items


def check_assumptions(model, values: Optional[Mapping] = None, n_samples: int = 100, seed: int = 0,
                      language: str = "en") -> "AssumptionsReport":
    """Check assumptions (A1)-(A5) of van den Driessche & Watmough (2002).

    Parameters
    ----------
    model
        An :class:`~pyr0compute.R0Model`.
    values
        Optional parameter values ``{name: value}`` (all parameters). Without
        them, each assumption is proved for every positive value of the
        parameters when possible and otherwise explored with random samples;
        with them, what could not be proved is decided at that point.
    n_samples
        Parameter samples used for (A5) when it cannot be proved symbolically.
    seed
        Seed of the random samples.
    language
        ``"en"`` or ``"es"``: default language of the report.
    """
    if language not in ("en", "es"):
        raise ValueError("language must be 'en' or 'es'.")
    chk = _Checker(model, values, n_samples, seed)
    orig = model._orig
    name = lambda x: sp.latex(model._to_orig[x])  # noqa: E731
    rows, v_items = _flows(model, chk)
    results: Dict[str, AssumptionResult] = {}

    # (A1) non-negativity of F_i, V_i^+, V_i^-
    items = []
    for x, F, Vp, Vm in rows:
        if x in model._xi and F != 0:
            items.append(_Item(rf"\mathcal{{F}}_{{{name(x)}}} = {sp.latex(orig(F))} \ge 0", chk.nonneg(F)))
    items += v_items
    results["A1"] = _combine("A1", items, [_t(
        r"$\mathcal{V}_i^+$ and $\mathcal{V}_i^-$ collect the negative and positive terms of $\mathcal{V}_i$; "
        r"they are non-negative when every term has a definite sign.",
        r"$\mathcal{V}_i^+$ y $\mathcal{V}_i^-$ reúnen los términos negativos y positivos de $\mathcal{V}_i$; "
        r"son no negativos cuando cada término tiene signo definido.")])

    # (A2) no outflow from an empty compartment
    items = []
    for x, F, Vp, Vm in rows:
        at0 = sp.cancel(sp.together(Vm.xreplace({x: 0})))
        items.append(_Item(
            rf"\left.\mathcal{{V}}_{{{name(x)}}}^-\right\rvert_{{{name(x)} = 0}} = {sp.latex(orig(at0))}",
            chk.zero(at0)))
    results["A2"] = _combine("A2", items)

    # (A3) by construction
    results["A3"] = AssumptionResult("A3", HOLDS, "construction", [], [_t(
        r"pyR0compute defines $\mathcal{F}_i$ only for the infected compartments and rejects "
        r"new-infection terms for uninfected ones.",
        r"pyR0compute define $\mathcal{F}_i$ solo para los compartimentos infectados y rechaza "
        r"términos de nuevas infecciones en los no infectados.")])

    # (A4) the disease-free subspace X_s is invariant
    items = []
    on_xs = {x: sp.Integer(0) for x in model._xi}
    for x, F, Vp, Vm in rows:
        if x not in model._xi:
            continue
        Fs = sp.cancel(sp.together(F.xreplace(on_xs)))
        Vs = sp.cancel(sp.together(Vp.xreplace(on_xs)))
        items.append(_Item(rf"\left.\mathcal{{F}}_{{{name(x)}}}\right\rvert_{{X_s}} = {sp.latex(orig(Fs))}",
                           chk.zero(Fs)))
        items.append(_Item(rf"\left.\mathcal{{V}}_{{{name(x)}}}^+\right\rvert_{{X_s}} = {sp.latex(orig(Vs))}",
                           chk.zero(Vs)))
    results["A4"] = _combine("A4", items, [_t(
        r"$X_s = \{x \ge 0 : x_i = 0,\ i \le m\}$; the uninfected compartments are free, not fixed at $x_0$.",
        r"$X_s = \{x \ge 0 : x_i = 0,\ i \le m\}$; los no infectados quedan libres, no fijos en $x_0$.")])

    # (A5) stability without new infections: V and J4 positive stable
    point = {x: model._dfe_compact[x] for x in model._xu if model._dfe_compact.get(x) is not x}
    xu, J = model._uninfected_jacobian()
    J4 = -J
    dV, texts_V, eig_V = chk.stable(model._Vc, point)
    dJ, texts_J, eig_J = chk.stable(J4, point)
    items = [
        _Item(rf"V = {sp.latex(orig(model._Vc))}", dV),
        _Item(rf"J_4 = -\left.\frac{{\partial f_u}}{{\partial x_u}}\right\rvert_{{x_0}} = {sp.latex(orig(J4))}", dJ),
    ]
    notes = []
    for label, texts, eig in (("V", texts_V, eig_V), ("J_4", texts_J, eig_J)):
        if texts:
            notes.append(_t(rf"${label}$: " + ", ".join(f"${t}$" for t in texts),
                            rf"${label}$: " + ", ".join(f"${t}$" for t in texts)))
        if eig is not None:
            ev = ", ".join(_fmt_complex(z) for z in eig)
            notes.append(_t(rf"eigenvalues of ${label}$ at the given values: ${ev}$",
                            rf"valores propios de ${label}$ en los valores dados: ${ev}$"))
    results["A5"] = _combine("A5", items, notes)

    alternatives = []
    if dJ.status != HOLDS and len(model.dfe_candidates) > 1 and not model.dfe_is_implicit:
        alternatives = _alternative_dfes(model, chk)

    lemma = _lemma1(model, chk)
    return AssumptionsReport(model, results, rows, lemma, alternatives, values is not None, language)


def _alternative_dfes(model, chk: _Checker):
    """Stability of J_4 at every disease-free equilibrium found (A5 for each candidate)."""
    out = []
    for k, cand in enumerate(model.dfe_candidates):
        point = {model._to_pos[v]: model._pos(val) for v, val in cand.items()}
        xu = [x for x in model._xu if point.get(x) is not x]
        f = sp.Matrix([model._f[x] for x in xu])
        J4 = -f.jacobian(xu).xreplace(point)
        decision, _, _ = chk.stable(J4, {x: point[x] for x in xu})
        out.append((k, {v: val for v, val in cand.items() if v in model.uninfected}, decision))
    return out


def _lemma1(model, chk: _Checker):
    """Consequences of (A1)-(A5) in Lemma 1: F >= 0, V has the Z sign pattern, zero block of DV."""
    from .model import DFESymbol

    def no_stars(e):
        return not any(isinstance(s, DFESymbol) for s in sp.sympify(e).free_symbols)

    def symbolic_only(decision_fn, e):
        if no_stars(e):
            return decision_fn(e)
        if sp.sympify(e) == 0:
            return _Decision(HOLDS, "symbolic")
        return _Decision(UNDECIDED, "symbolic", _t(
            "depends on disease-free values without closed form", "depende de valores del DFE sin forma cerrada"))

    def nonneg_entry(e):
        e = sp.simplify(e)
        if e == 0 or e.is_nonnegative:
            return _Decision(HOLDS, "symbolic")
        return symbolic_only(chk.nonneg, e)

    def nonpos_entry(e):
        e = sp.simplify(e)
        if e == 0 or e.is_nonpositive:
            return _Decision(HOLDS, "symbolic")
        return symbolic_only(chk.nonneg, -e)

    def zero_entry(e):
        e = sp.simplify(e)
        if e == 0:
            return _Decision(HOLDS, "symbolic")
        return symbolic_only(chk.zero, e)

    F, V = model._Fc, model._Vc
    m = F.shape[0]
    F_items = [_Item(rf"F_{{{i + 1}{j + 1}}} = {sp.latex(model._orig(F[i, j]))} \ge 0", nonneg_entry(F[i, j]))
               for i in range(m) for j in range(m) if F[i, j] != 0]
    Z_items = [_Item(rf"V_{{{i + 1}{j + 1}}} = {sp.latex(model._orig(V[i, j]))} \le 0", nonpos_entry(V[i, j]))
               for i in range(m) for j in range(m) if i != j and V[i, j] != 0]
    block = []
    for x in model._xi:
        Vi = model._F_terms[x] - model._f[x]
        for y in model._xu:
            d = sp.diff(Vi, y).xreplace(model._dfe_compact)
            if sp.simplify(d) != 0:
                block.append(_Item(
                    rf"\frac{{\partial \mathcal{{V}}_{{{sp.latex(model._to_orig[x])}}}}}"
                    rf"{{\partial {sp.latex(model._to_orig[y])}}}(x_0) = {sp.latex(model._orig(d))}",
                    zero_entry(d)))
    return {
        "F": _combine("F", F_items),
        "Z": _combine("Z", Z_items),
        "block": _combine("block", block),
    }


def _fmt_complex(z: complex) -> str:
    if abs(z.imag) <= 1e-12 * max(1.0, abs(z)):
        return f"{z.real:.4g}"
    return f"{z.real:.4g} {'+' if z.imag >= 0 else '-'} {abs(z.imag):.4g}i"


# ----------------------------------------------------------------- report
class AssumptionsReport:
    """Result of :meth:`R0Model.check_assumptions`.

    ``report["A2"]`` gives one :class:`AssumptionResult`; ``report.holds`` is
    True when the five assumptions hold, so that Theorem 2 applies.
    :meth:`to_latex` writes the report as LaTeX (``style="document"``) or as
    Markdown with LaTeX math (``style="markdown"``); in Jupyter the report is
    displayed in Markdown.
    """

    def __init__(self, model, results, flows, lemma, alternatives, with_values, language):
        self.model = model
        self.results: Dict[str, AssumptionResult] = results
        self._flows = flows
        self.lemma1 = lemma
        self._alternatives = alternatives
        self._with_values = with_values
        self.language = language

    def __getitem__(self, key: str) -> AssumptionResult:
        return self.results[key.upper()]

    def __iter__(self):
        return iter(self.results.values())

    @property
    def holds(self) -> bool:
        """True if (A1)-(A5) all hold: R0 is then a threshold (Theorem 2)."""
        return all(r.status == HOLDS for r in self.results.values())

    @property
    def status(self) -> Dict[str, str]:
        """``{"A1": "holds", ...}``."""
        return {k: r.status for k, r in self.results.items()}

    def as_dict(self) -> Dict[str, Dict[str, str]]:
        return {k: {"status": r.status, "basis": r.basis} for k, r in self.results.items()}

    @property
    def decomposition(self) -> List[Tuple[sp.Symbol, sp.Expr, sp.Expr, sp.Expr]]:
        """``(compartment, F_i, V_i^+, V_i^-)`` for every compartment."""
        o = self.model._orig
        return [(self.model._to_orig[x], o(F), o(Vp), o(Vm)) for x, F, Vp, Vm in self._flows]

    def __repr__(self) -> str:
        lines = ["van den Driessche & Watmough (2002) assumptions:"]
        for k, r in self.results.items():
            lines.append(f"  ({k}) {r.status:<9} [{BASIS_TEXT[r.basis]['en']}]  {SHORT[k]['en']}")
        lines.append("Theorem 2 applies (R0 is a threshold)." if self.holds else
                     "Theorem 2 is not guaranteed; see to_latex() for the details.")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        return self.to_latex(style="markdown")

    # ---------------------------------------------------------------- LaTeX
    def to_latex(self, style: str = "document", standalone: bool = False,
                 language: Optional[str] = None) -> str:
        """The report in LaTeX.

        Parameters
        ----------
        style
            ``"document"``: LaTeX to paste into a paper or compile (needs
            ``amsmath``). ``"markdown"``: Markdown with ``$...$`` math, for
            ``display(Markdown(...))`` in Jupyter.
        standalone
            With ``style="document"``, a complete file that compiles as is.
        language
            ``"en"`` or ``"es"``; by default the one given to ``check_assumptions``.
        """
        if style not in ("document", "markdown"):
            raise ValueError("style must be 'document' or 'markdown'.")
        lang = language or self.language
        if lang not in ("en", "es"):
            raise ValueError("language must be 'en' or 'es'.")
        md = style == "markdown"
        T = lambda d: d[lang]  # noqa: E731
        es = lang == "es"
        o = self.model._orig

        def heading(text, level=2):
            if md:
                return "#" * (level + 1) + " " + text
            return ("\\section*{" if level == 1 else "\\subsection*{") + text + "}"

        def bold(text):
            return f"**{text}**" if md else f"\\textbf{{{text}}}"

        def display(math):
            return f"$$\n{math}\n$$" if md else f"\\[\n{math}\n\\]"

        def bullets(lines):
            if not lines:
                return ""
            if md:
                return "\n".join(f"* {ln}" for ln in lines)
            return "\\begin{itemize}\n" + "\n".join(f"  \\item {ln}" for ln in lines) + "\n\\end{itemize}"

        def status_word(status):
            return bold(T(STATUS_TEXT[status]))

        def witness_tex(w):
            if not w:
                return ""
            parts = ",\\ ".join(f"{sp.latex(o(s))} = {v:.4g}" for s, v in
                                sorted(w.items(), key=lambda kv: kv[0].name))
            return (" (contraejemplo: $" if es else " (counterexample: $") + parts + "$)"

        def item_line(it):
            d = it.decision
            line = f"${it.math}$: {status_word(d.status)}"
            if d.basis not in ("symbolic",):
                line += f", {T(BASIS_TEXT[d.basis])}"
            if d.note:
                line += f"; {T(d.note)}"
            line += witness_tex(d.witness)
            return line

        out = []
        out.append(heading("Supuestos de van den Driessche y Watmough (2002)" if es
                           else "Assumptions of van den Driessche and Watmough (2002)", level=1))
        names = ", ".join(sp.latex(v) for v in self.model.infected)
        out.append((f"Compartimentos infectados ($i \\le m$): ${names}$. " if es
                    else f"Infected compartments ($i \\le m$): ${names}$. ")
                   + ("Los supuestos se deciden en los valores de los parámetros dados." if es and self._with_values
                      else "Assumptions are decided at the given parameter values." if self._with_values
                      else "Sin valores de los parámetros: cada supuesto se demuestra para todo parámetro positivo "
                           "cuando es posible; si no, se explora con muestras aleatorias." if es
                      else "No parameter values given: each assumption is proved for every positive parameter "
                           "value when possible, and otherwise explored with random samples."))

        # summary table
        head = ("Supuesto", "Enunciado", "Resultado", "Fundamento") if es else \
               ("Assumption", "Statement", "Result", "Basis")
        if md:
            rows = [f"| {' | '.join(head)} |", "|:--:|---|:--:|---|"]
            for k, r in self.results.items():
                rows.append(f"| ({k}) | {T(SHORT[k])} | {status_word(r.status)} | {T(BASIS_TEXT[r.basis])} |")
            out.append("\n".join(rows))
        else:
            rows = ["\\begin{center}", "\\begin{tabular}{c p{0.34\\textwidth} c p{0.3\\textwidth}}", "\\hline",
                    " & ".join(head) + " \\\\", "\\hline"]
            for k, r in self.results.items():
                rows.append(f"({k}) & {T(SHORT[k])} & {status_word(r.status)} & {T(BASIS_TEXT[r.basis])} \\\\")
            rows += ["\\hline", "\\end{tabular}", "\\end{center}"]
            out.append("\n".join(rows))

        # decomposition
        out.append(heading("Descomposición utilizada" if es else "Decomposition used"))
        out.append((r"$\dot x_i = \mathcal{F}_i - \mathcal{V}_i$, con $\mathcal{V}_i = \mathcal{V}_i^- - \mathcal{V}_i^+$ "
                    r"($\mathcal{V}_i^+$: otras entradas; $\mathcal{V}_i^-$: salidas)." if es else
                    r"$\dot x_i = \mathcal{F}_i - \mathcal{V}_i$, with $\mathcal{V}_i = \mathcal{V}_i^- - \mathcal{V}_i^+$ "
                    r"($\mathcal{V}_i^+$: other inflows; $\mathcal{V}_i^-$: outflows)."))
        lines = []
        for x, F, Vp, Vm in self.decomposition:
            nm = sp.latex(x)
            lines.append(rf"\mathcal{{F}}_{{{nm}}} &= {sp.latex(F)}, & \mathcal{{V}}_{{{nm}}}^+ &= {sp.latex(Vp)}, "
                         rf"& \mathcal{{V}}_{{{nm}}}^- &= {sp.latex(Vm)}")
        body = " \\\\\n".join(lines)
        out.append(display("\\begin{aligned}\n" + body + "\n\\end{aligned}") if md
                   else "\\begin{align*}\n" + body + "\n\\end{align*}")

        # each assumption
        for k, r in self.results.items():
            out.append(heading(f"({k}) {T(SHORT[k])}"))
            out.append(T(STATEMENTS[k]))
            out.append(("Resultado: " if es else "Result: ") + status_word(r.status)
                       + f" ({T(BASIS_TEXT[r.basis])}).")
            shown = [it for it in r.items if it.decision.status != HOLDS or k in ("A2", "A4", "A5")]
            lines = [item_line(it) for it in shown] + [T(n) for n in r.notes]
            if lines:
                out.append(bullets(lines))
            if k == "A5" and self._alternatives:
                alt = []
                for idx, cand, d in self._alternatives:
                    pt = ",\\ ".join(f"{sp.latex(v)} = {sp.latex(val)}" for v, val in cand.items())
                    tag = (" (el utilizado)" if es else " (the one used)") if idx == 0 else ""
                    alt.append(f"DFE ${pt}$" + tag + f": $J_4$ {status_word(d.status)}"
                               + (f"; {T(d.note)}" if d.note else ""))
                out.append(("Otros equilibrios libres de infección encontrados (`dfe_candidates`); "
                            "se puede elegir uno con `dfe={...}`:" if es else
                            "Other disease-free equilibria found (`dfe_candidates`); one can be chosen "
                            "with `dfe={...}`:") if md else
                           ("Otros equilibrios libres de infección encontrados; se puede elegir uno con "
                            "\\texttt{dfe=\\{...\\}}:" if es else
                            "Other disease-free equilibria found; one can be chosen with \\texttt{dfe=\\{...\\}}:"))
                out.append(bullets(alt))

        # lemma 1
        out.append(heading("Consecuencias (Lema 1)" if es else "Consequences (Lemma 1)"))
        labels = {
            "F": _t(r"$F \ge 0$ (entrywise)", r"$F \ge 0$ (por entradas)"),
            "Z": _t(r"$V$ has the Z sign pattern ($V_{ij} \le 0$, $i \ne j$)",
                    r"$V$ tiene patrón de signos Z ($V_{ij} \le 0$, $i \ne j$)"),
            "block": _t(r"$\partial \mathcal{V}_i / \partial x_j (x_0) = 0$ for $i \le m < j$",
                        r"$\partial \mathcal{V}_i / \partial x_j (x_0) = 0$ para $i \le m < j$"),
        }
        lines = []
        for key in ("F", "Z", "block"):
            res = self.lemma1[key]
            lines.append(f"{T(labels[key])}: {status_word(res.status)}")
            lines += [item_line(it) for it in res.items if it.decision.status != HOLDS]
        out.append(bullets(lines))
        out.append(("Si $V$ tiene patrón Z y sus valores propios tienen parte real positiva, $V$ es una "
                    "M-matriz no singular y $V^{-1} \\ge 0$." if es else
                    "If $V$ has the Z pattern and its eigenvalues have positive real parts, $V$ is a "
                    "non-singular M-matrix and $V^{-1} \\ge 0$."))

        # conclusion
        out.append(heading("Conclusión" if es else "Conclusion"))
        failing = [k for k, r in self.results.items() if r.status == FAILS]
        open_ = [k for k, r in self.results.items() if r.status == UNDECIDED]
        if self.holds:
            out.append("Se cumplen (A1) a (A5). Por el Teorema 2, el DFE $x_0$ es localmente asintóticamente "
                       "estable si $\\mathcal{R}_0 < 1$ e inestable si $\\mathcal{R}_0 > 1$." if es else
                       "(A1) to (A5) hold. By Theorem 2, the DFE $x_0$ is locally asymptotically stable if "
                       "$\\mathcal{R}_0 < 1$ and unstable if $\\mathcal{R}_0 > 1$.")
        elif failing:
            lst = ", ".join(f"({k})" for k in failing)
            out.append(f"No se cumple {lst}: el Teorema 2 no se aplica y $\\mathcal{{R}}_0 = \\rho(FV^{{-1}})$ "
                       "puede no ser un umbral para el modelo tal como está escrito." if es else
                       f"{lst} fail: Theorem 2 does not apply, and $\\mathcal{{R}}_0 = \\rho(FV^{{-1}})$ need "
                       "not be a threshold for the model as written.")
        else:
            lst = ", ".join(f"({k})" for k in open_)
            out.append(f"No se pudo decidir {lst}. Si se cumplen, el Teorema 2 se aplica. Con "
                       "`values={...}` se deciden en un punto del espacio de parámetros." if es and md else
                       f"No se pudo decidir {lst}. Si se cumplen, el Teorema 2 se aplica. Con "
                       "\\texttt{values=\\{...\\}} se deciden en un punto del espacio de parámetros." if es else
                       f"{lst} could not be decided. If they hold, Theorem 2 applies. With `values={{...}}` "
                       "they are decided at one point of the parameter space." if md else
                       f"{lst} could not be decided. If they hold, Theorem 2 applies. With "
                       "\\texttt{values=\\{...\\}} they are decided at one point of the parameter space.")

        text = "\n\n".join(s for s in out if s)
        if standalone and not md:
            text = ("\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage{amsmath,amssymb}\n"
                    "\\usepackage[margin=2cm]{geometry}\n\\begin{document}\n\n"
                    f"{text}\n\n\\end{{document}}\n")
        return text
