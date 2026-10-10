"""Core of pyR0compute: the next-generation matrix method.

Notation follows van den Driessche & Watmough (2002). For the infected
compartments ``x_i`` the model is written as ``dx_i/dt = F_i(x) - V_i(x)``,
where ``F_i`` collects the appearance of *new infections* and ``V_i`` every
other transfer. With ``F = [dF_i/dx_j]`` and ``V = [dV_i/dx_j]`` evaluated at
the disease-free equilibrium (DFE), the basic reproduction number is the
spectral radius of the next-generation matrix ``K = F V^{-1}``.

Reference
---------
P. van den Driessche, J. Watmough (2002). Reproduction numbers and
sub-threshold endemic equilibria for compartmental models of disease
transmission. Mathematical Biosciences 180, 29-48.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import sympy as sp

from ._solve import solve_with_time_limit
from .exceptions import (
    DiseaseFreeEquilibriumError,
    ModelSpecificationError,
    NextGenerationError,
)
from .parsing import (
    ExprLike,
    SymbolLike,
    parse_expression,
    parse_text_model,
    to_symbol,
    unique_symbols,
)

EquationsLike = Union[str, Mapping[SymbolLike, ExprLike], Sequence[ExprLike]]

__all__ = ["R0Model", "DFESymbol"]


class DFESymbol(sp.Symbol):
    """Disease-free value ``X^*`` of a compartment ``X``.

    Named ``X_star`` (so that ``X_star*beta`` is not read as a power), shown
    as ``X*`` in pretty printing and as ``X^{*}`` in LaTeX. Its definition is
    in :attr:`R0Model.dfe_definitions`.
    """

    @property
    def compartment(self) -> sp.Symbol:
        return sp.Symbol(self.name[: -len("_star")])

    def _latex(self, printer):
        return printer._print(self.compartment) + "^{*}"

    def _pretty(self, printer):
        from sympy.printing.pretty.stringpict import prettyForm
        return prettyForm(*printer._print(self.compartment).right("*"))

    def _sympystr(self, printer):
        return self.name


class R0Model:
    """Basic reproduction number of a compartmental ODE model.

    Only the equations and the infected compartments are required. Every
    symbol that is not a state variable is treated as a parameter.

    Parameters
    ----------
    equations
        The ODE system, in any of these forms:

        * a dict ``{variable: right-hand side}`` (keys and values may be SymPy
          objects or strings);
        * a text block with one ``dX/dt = ...`` line per variable (auxiliary
          lines such as ``N = S + I + R`` are substituted);
        * a list of right-hand sides, together with ``variables``.
    infected
        Infected compartments, as symbols, names or indices into the
        variables. Their order does not matter and the equations do not need
        to be reordered.
    variables
        State variables, only needed when ``equations`` is a list.
    parameters
        Optional. Parameters are detected automatically; if given, the list is
        only used to warn about undeclared or unused parameters.
    new_infections
        Optional manual specification of the new-infection terms ``F_i``, as a
        dict ``{infected variable: term}`` or a list aligned with ``infected``
        (or with all variables). By default they are detected automatically.
    dfe
        Optional (possibly partial) disease-free equilibrium, e.g.
        ``{"S": "N"}``. Missing values are solved for.
    simplify
        Simplify intermediate matrices and the final expression.
    verbose
        Print a step-by-step report after the computation.
    seed
        Seed for the random parameter samples used to classify ambiguous
        signs and to identify the dominant eigenvalue.
    dfe_timeout
        Time limit, in seconds, for solving each coupled block of the
        disease-free equilibrium symbolically (default 20). A block that is
        not solved in time is kept implicit (``X_star`` without closed form)
        and computed numerically by :meth:`dfe_numeric` and
        :meth:`R0_numeric`; see :attr:`dfe_blocks`. ``None`` removes the limit.

    Examples
    --------
    >>> m = R0Model('''
    ...     dS/dt = Lambda - beta*S*I - mu*S
    ...     dI/dt = beta*S*I - (gamma + mu)*I
    ...     dR/dt = gamma*I - mu*R
    ... ''', infected=["I"])
    >>> m.R0
    Lambda*beta/(mu*(gamma + mu))
    """

    # ------------------------------------------------------------------ setup
    def __init__(
        self,
        equations: EquationsLike,
        infected: Sequence[Union[SymbolLike, int]],
        variables: Optional[Sequence[SymbolLike]] = None,
        parameters: Optional[Sequence[SymbolLike]] = None,
        new_infections: Optional[Union[Mapping[SymbolLike, ExprLike], Sequence[ExprLike]]] = None,
        dfe: Optional[Mapping[SymbolLike, ExprLike]] = None,
        simplify: bool = True,
        verbose: bool = False,
        seed: int = 0,
        dfe_timeout: Optional[float] = 20.0,
    ) -> None:
        if dfe_timeout is not None and not dfe_timeout > 0:
            raise ModelSpecificationError("dfe_timeout must be a positive number of seconds or None.")
        self.dfe_timeout = dfe_timeout
        self._block_log: List[Dict[str, object]] = []
        self._implicit_reason: Dict[sp.Symbol, str] = {}
        self._implicit_cache: Dict[int, tuple] = {}
        self._simplify_enabled = simplify
        self._rng = np.random.default_rng(seed)
        self._R0_cache: Optional[sp.Expr] = None
        self._eigs_cache: Optional[List[sp.Expr]] = None
        self._R0c_cache: Optional[sp.Expr] = None
        self._eigs_c_cache: Optional[List[sp.Expr]] = None
        self._stars: Dict[sp.Symbol, Optional[sp.Expr]] = {}
        self._star_full: Dict[sp.Symbol, sp.Expr] = {}
        self._implicit_blocks: List = []

        self.definitions: Dict[str, sp.Expr] = {}
        self.variables, self.equations = self._normalize_equations(equations, variables)
        self.infected = self._normalize_infected(infected)
        self.uninfected = tuple(v for v in self.variables if v not in self.infected)

        table = {v.name: v for v in self.variables}
        table.update(unique_symbols(self.equations.values()))
        table.update(self.definitions)  # e.g. N = S + I + R, usable in dfe/new_infections
        manual_F = self._normalize_new_infections(new_infections, table)
        user_dfe = self._normalize_dfe(dfe, table)

        extra = list(manual_F.values()) + list(user_dfe.values())
        self.symbols: Dict[str, sp.Symbol] = unique_symbols(list(self.equations.values()) + extra)
        self.symbols.update({v.name: v for v in self.variables})
        self.parameters = tuple(
            sorted((s for s in self.symbols.values() if s not in self.variables), key=lambda s: s.name)
        )
        self._check_declared_parameters(parameters)

        # Work internally with positive symbols so that SymPy can simplify
        # square roots and decide signs; results are mapped back at the end.
        self._to_pos = {s: sp.Symbol(s.name, positive=True) for s in self.symbols.values()}
        self._to_orig = {p: s for s, p in self._to_pos.items()}
        self._x = [self._to_pos[v] for v in self.variables]
        self._xi = [self._to_pos[v] for v in self.infected]
        self._xu = [self._to_pos[v] for v in self.uninfected]
        self._f = {self._to_pos[v]: self._pos(self.equations[v]) for v in self.variables}

        if manual_F:
            self._F_terms = {self._to_pos[v]: self._pos(manual_F.get(v, 0)) for v in self.infected}
            self._classification = []
            self._new_infections_mode = "manual"
        else:
            self._F_terms, self._classification = self._detect_new_infections()
            self._new_infections_mode = "automatic"
        if all(sp.simplify(t) == 0 for t in self._F_terms.values()):
            raise ModelSpecificationError(
                "No new-infection terms were found, so R0 would be 0. Check the list of "
                "infected compartments or pass new_infections={...} explicitly."
            )
        self._V_terms = {x: self._F_terms[x] - self._f[x] for x in self._xi}

        self._dfe, self.dfe_candidates = self._compute_dfe(user_dfe)
        self._build_matrices()

        if verbose:
            print(self.report())

    # ------------------------------------------------------- input handling
    def _normalize_equations(self, equations, variables):
        if isinstance(equations, str):
            if variables is not None:
                raise ModelSpecificationError(
                    "With a text model the variables are read from the 'dX/dt' lines; "
                    "do not pass 'variables'."
                )
            eqs, _, self.definitions = parse_text_model(equations)
            return tuple(eqs), eqs

        if isinstance(equations, Mapping):
            if variables is not None:
                raise ModelSpecificationError(
                    "When equations is a dict its keys are the variables; do not pass 'variables'."
                )
            keys = [to_symbol(k) for k in equations]
            table = {k.name: k for k in keys}
            eqs = {k: parse_expression(rhs, table) for k, rhs in zip(keys, equations.values())}
            return tuple(keys), eqs

        if isinstance(equations, (list, tuple, sp.MatrixBase)):
            rhs = list(equations)
            if variables is None:
                raise ModelSpecificationError(
                    "A list of equations needs 'variables' (or use a dict {variable: rhs})."
                )
            vars_ = [to_symbol(v) for v in variables]
            if len(vars_) != len(rhs):
                raise ModelSpecificationError(
                    f"{len(vars_)} variables but {len(rhs)} equations were given."
                )
            table = {v.name: v for v in vars_}
            return tuple(vars_), {v: parse_expression(e, table) for v, e in zip(vars_, rhs)}

        raise ModelSpecificationError(
            "equations must be a text block, a dict {variable: rhs} or a list of right-hand sides."
        )

    def _normalize_infected(self, infected):
        if isinstance(infected, (str, sp.Symbol, int)):
            infected = [infected]
        table = {v.name: v for v in self.variables}
        out: List[sp.Symbol] = []
        for item in infected:
            if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
                if not 0 <= item < len(self.variables):
                    raise ModelSpecificationError(f"Infected index {item} is out of range.")
                sym = self.variables[item]
            else:
                sym = to_symbol(item, table)
                if sym not in self.variables:
                    raise ModelSpecificationError(
                        f"Infected compartment '{sym}' is not a state variable. "
                        f"Variables: {[v.name for v in self.variables]}"
                    )
            if sym not in out:
                out.append(sym)
        if not out:
            raise ModelSpecificationError("At least one infected compartment is required.")
        return tuple(out)

    def _normalize_new_infections(self, new_infections, table):
        if new_infections is None:
            return {}
        if isinstance(new_infections, Mapping):
            out = {}
            for key, value in new_infections.items():
                sym = to_symbol(key, table)
                if sym not in self.infected:
                    raise ModelSpecificationError(
                        f"new_infections given for '{sym}', which is not an infected compartment."
                    )
                out[sym] = parse_expression(value, table)
            return out
        terms = list(new_infections)
        if len(terms) == len(self.infected):
            return {v: parse_expression(t, table) for v, t in zip(self.infected, terms)}
        if len(terms) == len(self.variables):
            out = {}
            for v, t in zip(self.variables, terms):
                expr = parse_expression(t, table)
                if v in self.infected:
                    out[v] = expr
                elif sp.simplify(expr) != 0:
                    raise ModelSpecificationError(
                        f"A non-zero new-infection term was given for the uninfected compartment '{v}'."
                    )
            return out
        raise ModelSpecificationError(
            "new_infections must be a dict, or a list with one entry per infected compartment "
            "(or per variable)."
        )

    def _normalize_dfe(self, dfe, table):
        if not dfe:
            return {}
        out = {}
        for key, value in dfe.items():
            sym = to_symbol(key, table)
            if sym not in self.variables:
                raise ModelSpecificationError(f"dfe given for '{sym}', which is not a state variable.")
            out[sym] = parse_expression(value, table)
        return out

    def _check_declared_parameters(self, declared):
        if declared is None:
            return
        declared = {to_symbol(p).name for p in declared}
        detected = {p.name for p in self.parameters}
        missing = sorted(detected - declared)
        unused = sorted(declared - detected)
        if missing:
            warnings.warn(
                f"These symbols were not declared as parameters but are treated as such: {missing}",
                stacklevel=3,
            )
        if unused:
            warnings.warn(f"Declared parameters not used by the model: {unused}", stacklevel=3)

    # --------------------------------------------------------------- helpers
    def _pos(self, expr):
        return sp.sympify(expr).xreplace(self._to_pos)

    def _orig(self, expr):
        return expr.xreplace(self._to_orig)

    def _simp(self, expr):
        if not self._simplify_enabled:
            return expr
        if isinstance(expr, sp.MatrixBase):
            return expr.applyfunc(self._simp)
        expr = sp.simplify(expr)
        # factored rational functions read like the formulas in papers:
        # mu*(gamma + mu)*(mu + sigma) rather than mu*(gamma*mu + gamma*sigma + ...)
        if expr.is_rational_function():
            expr = sp.factor(expr)
        return expr

    def _sign(self, coeff) -> Optional[int]:
        """+1 / -1 if the sign of ``coeff`` is clear, ``None`` otherwise."""
        coeff = sp.sympify(coeff)
        if coeff.is_positive:
            return 1
        if coeff.is_negative:
            return -1
        if coeff.is_zero:
            return 0
        syms = sorted(coeff.free_symbols, key=lambda s: s.name)
        f = sp.lambdify(syms, coeff, "numpy")
        values = []
        for _ in range(12):
            with np.errstate(all="ignore"):
                values.append(complex(f(*self._rng.uniform(0.01, 1.0, len(syms)))).real)
        if all(v > 0 for v in values):
            return 1
        if all(v < 0 for v in values):
            return -1
        return None

    def _groups(self, expr) -> Dict[sp.Expr, sp.Expr]:
        """Group the expanded terms of ``expr`` by their state-variable part."""
        groups: Dict[sp.Expr, sp.Expr] = {}
        for term in sp.Add.make_args(sp.expand(expr)):
            coeff, dep = term.as_independent(*self._x, as_Add=False)
            groups[dep] = groups.get(dep, 0) + coeff
        return {d: c for d, c in groups.items() if sp.simplify(c) != 0}

    # ------------------------------------------------- new-infection terms
    def _detect_new_infections(self):
        """Split each infected equation into new infections (F) and the rest (V).

        A positive term in the equation of an infected compartment is a new
        infection when it depends on an infected compartment and either it is
        lost from an uninfected compartment (a transfer such as S -> E), or it
        depends on an uninfected compartment and is not a transfer between
        infected compartments.
        """
        xi, xu = set(self._xi), set(self._xu)
        groups = {x: self._groups(self._f[x]) for x in self._x}

        def lost_from(dep, compartments, exclude=None):
            return [
                x for x in compartments
                if x is not exclude and dep in groups[x] and self._sign(groups[x][dep]) == -1
            ]

        F_terms: Dict[sp.Symbol, sp.Expr] = {}
        classification = []
        for x in self._xi:
            selected = []
            for dep, coeff in groups[x].items():
                term = coeff * dep
                vars_in = dep.free_symbols & set(self._x)
                if not vars_in & xi:
                    classification.append((x, term, "V", "does not involve infected compartments"))
                    continue
                from_uninf = lost_from(dep, self._xu)
                from_inf = lost_from(dep, self._xi, exclude=x)
                involves_uninf = bool(vars_in & xu)
                candidate = bool(from_uninf) or (involves_uninf and not from_inf)
                sign = self._sign(coeff)
                if sign is None and candidate:
                    raise ModelSpecificationError(
                        f"Cannot decide the sign of the term {self._orig(term)} in d{self._orig(x)}/dt. "
                        f"Specify the new-infection terms with new_infections={{...}}."
                    )
                if sign != 1:
                    classification.append((x, term, "V", "negative term"))
                elif from_uninf:
                    names = ", ".join(str(self._orig(s)) for s in from_uninf)
                    classification.append((x, term, "F", f"transfer from uninfected {names}"))
                    selected.append(term)
                elif from_inf:
                    names = ", ".join(str(self._orig(s)) for s in from_inf)
                    classification.append((x, term, "V", f"transfer from infected {names}"))
                elif involves_uninf:
                    classification.append((x, term, "F", "contact between infected and uninfected"))
                    selected.append(term)
                else:
                    classification.append((x, term, "V", "involves only infected compartments"))
            F_terms[x] = sp.Add(*selected)
        return F_terms, classification

    # ------------------------------------------------ disease-free equilibrium
    def _compute_dfe(self, user_dfe):
        """Disease-free equilibrium, solved block by block when possible.

        Setting the infected compartments to zero, the equations of the
        uninfected compartments are split into the strongly connected
        components of their dependency graph and solved in topological order
        (e.g. ``[E] -> [H, R] -> [C] -> [B]`` for an immune-response model).
        Values already found enter later blocks as symbols ``X^*`` rather than
        as their explicit expressions, which keeps every block small; values
        without a short closed form stay as ``X^*`` in the compact results
        (``R0_compact``, ``dfe_definitions``). A block with no closed form is
        kept implicit and solved numerically when values are given.
        Structures that do not fit this scheme fall back to a global solve.
        """
        known: Dict[sp.Symbol, sp.Expr] = {x: sp.Integer(0) for x in self._xi}
        for v, value in user_dfe.items():
            p = self._to_pos[v]
            value = self._pos(value)
            if p in known and sp.simplify(value) != 0:
                raise DiseaseFreeEquilibriumError(
                    f"At the disease-free equilibrium the infected compartment '{v}' must be 0."
                )
            known[p] = value

        blocks = self._dfe_blocks(known)
        if blocks is not None:
            return self._compute_dfe_blocks(known, *blocks)
        self._stars, self._star_full, self._implicit_blocks = {}, {}, []
        chosen, candidates = self._compute_dfe_global(known)
        self._dfe_compact = dict(chosen)
        return chosen, candidates

    def _star(self, x) -> sp.Symbol:
        star = DFESymbol(f"{x.name}_star", positive=True)
        if star.name in self.symbols:
            raise ModelSpecificationError(
                f"'{star.name}' is a symbol of the model, but it is the name used for the "
                f"disease-free value of {x.name}. Please rename it."
            )
        return star

    def _dfe_blocks(self, known):
        """Equations and solving order for the block method, or None to use the global solve."""
        unknown = [x for x in self._xu if x not in known]
        eqs: Dict[sp.Symbol, sp.Expr] = {}
        for x in self._xu:
            eq = sp.together(self._f[x].xreplace(known))
            if x in known:
                if sp.simplify(eq) != 0:
                    if not eq.free_symbols & set(unknown):
                        raise DiseaseFreeEquilibriumError(
                            f"The given dfe values do not satisfy d{self._orig(x)}/dt = 0 "
                            f"(residual {self._orig(eq)})."
                        )
                    return None  # an extra constraint on the unknowns: solve globally
                continue
            if sp.simplify(eq) == 0:
                continue  # undetermined at the DFE (checked when building F and V)
            if x not in eq.free_symbols:
                return None  # dX/dt does not determine X: no natural block structure
            eqs[x] = eq
        order = list(eqs)
        edges = [(x, y) for x in order for y in order if x is not y and y in eqs[x].free_symbols]
        components = sp.utilities.iterables.strongly_connected_components((order, edges))
        solved = set()
        for comp in components:  # dependencies must come first
            deps = {y for x in comp for y in eqs[x].free_symbols if y in eqs}
            if not deps <= solved | set(comp):
                return None
            solved |= set(comp)
        return eqs, components

    def _random_params(self, n: int):
        params = sorted((p for p in self._to_orig if p not in self._x), key=lambda s: s.name)
        return params, [self._rng.uniform(0.01, 1.0, len(params)) for _ in range(n)]

    def _plausible(self, values: Mapping[sp.Symbol, sp.Expr]) -> bool:
        """False if some value is clearly negative or non-real for every positive parameter set."""
        exprs = [sp.sympify(v) for v in values.values()]
        if any(e.is_negative or e.is_real is False for e in exprs):
            return False
        nontrivial = [e for e in exprs if e.free_symbols]
        if not nontrivial:
            return True
        params, samples = self._random_params(24)
        funcs = [sp.lambdify(params, e, "numpy") for e in nontrivial]
        for f in funcs:
            seen, ok = 0, False
            for point in samples:
                with np.errstate(all="ignore"):
                    try:
                        val = complex(f(*point))
                    except (ZeroDivisionError, OverflowError, ValueError):
                        continue
                if not np.isfinite(val):
                    continue
                seen += 1
                if val.real >= -1e-12 * max(1.0, abs(val)) and abs(val.imag) <= 1e-9 * max(1.0, abs(val)):
                    ok = True
                    break
            if seen and not ok:
                return False
        return True

    _INLINE_OPS = 40

    def _inline(self, value) -> bool:
        """Short, radical-free values in terms of parameters only are written out in full."""
        if value.free_symbols & set(self._star_full):
            return False
        if value.has(sp.RootOf, sp.CRootOf):
            return False
        if any(not p.exp.is_Integer for p in value.atoms(sp.Pow)):
            return False
        return sp.count_ops(value) <= self._INLINE_OPS

    @staticmethod
    def _easy_block(system, unknowns) -> bool:
        """One polynomial equation of degree <= 2: solved directly, without a time limit."""
        if len(system) != 1:
            return False
        try:
            return sp.Poly(system[0], unknowns[0]).degree() <= 2
        except sp.PolynomialError:
            return False

    def _solve_block(self, comp, eqs, compact):
        """Solutions of one block, as ``(status, solutions, seconds)``.

        ``status`` is ``"closed form"``, ``"no closed form"`` or ``"time limit"``;
        ``solutions`` is None unless the status is ``"closed form"``.
        """
        free = {x: sp.Symbol(f"_{x.name}_dfe") for x in comp}
        back = {d: x for x, d in free.items()}
        system = [sp.numer(sp.together(eqs[x].xreplace(compact))).xreplace(free) for x in comp]
        unknowns = list(free.values())
        status, raw, seconds = solve_with_time_limit(
            system, unknowns, self.dfe_timeout, in_process=self._easy_block(system, unknowns))
        if status == "timeout":
            return "time limit", None, seconds
        if status != "ok" or not raw:
            # SymPy found no closed form (e.g. a general quintic): keep it implicit,
            # it is solved numerically when values are given
            return "no closed form", None, seconds
        out = []
        for s in raw:
            if set(s) != set(free.values()):
                return "no closed form", None, seconds  # solved only partially
            sol = {back[d]: sp.sympify(v).xreplace(back) for d, v in s.items()}
            if any(v.has(sp.RootOf, sp.CRootOf) for v in sol.values()):
                return "no closed form", None, seconds
            out.append(sol)
        return "closed form", out, seconds

    def _compute_dfe_blocks(self, known, eqs, components):
        # state: (compact point, star definitions, star -> explicit value)
        states = [(dict(known), {}, {})]
        self._implicit_blocks = []
        self._block_log = []
        for comp in components:
            new_states = []
            record = {"variables": [str(self._orig(x)) for x in comp],
                      "status": "closed form", "seconds": 0.0}
            for compact, stars, star_full in states:
                self._star_full = star_full
                if record["status"] == "time limit":
                    status, solutions = "time limit", None  # same block, other DFE branch
                else:
                    status, solutions, seconds = self._solve_block(comp, eqs, compact)
                    record["seconds"] += seconds
                if solutions is None:
                    # no closed form (or not found in time): keep the block implicit,
                    # solved numerically on demand
                    record["status"] = status
                    compact, stars, star_full = dict(compact), dict(stars), dict(star_full)
                    for x in comp:
                        s = self._star(x)
                        compact[x], stars[s], star_full[s] = s, None, s
                        self._implicit_reason[s] = status
                    block = [(self._star(x), eqs[x].xreplace(compact)) for x in comp]
                    if block not in self._implicit_blocks:
                        self._implicit_blocks.append(block)
                    new_states.append((compact, stars, star_full))
                    continue
                for sol in solutions:
                    explicit = {x: v.xreplace(star_full) for x, v in sol.items()}
                    if not self._plausible(explicit):
                        continue
                    c2, s2, f2 = dict(compact), dict(stars), dict(star_full)
                    for x, v in sol.items():
                        if len(comp) == 1 and self._simplify_enabled and sp.count_ops(v) <= 200:
                            v = self._simp(v)
                        if self._inline(v):
                            c2[x] = v
                        else:
                            s = self._star(x)
                            c2[x], s2[s], f2[s] = s, v, v.xreplace(f2)
                    new_states.append((c2, s2, f2))
            self._block_log.append(record)
            if record["status"] == "time limit":
                names = record["variables"]
                warnings.warn(
                    f"The disease-free values of {names} were not found symbolically within "
                    f"dfe_timeout={self.dfe_timeout:g} s. They are kept implicit "
                    f"({', '.join(n + '_star' for n in names)}) and computed numerically by "
                    f"dfe_numeric() and R0_numeric(). Increase dfe_timeout (None: no limit) "
                    f"to keep trying symbolically.",
                    stacklevel=4,
                )
            if not new_states:
                raise DiseaseFreeEquilibriumError(
                    "No non-negative disease-free equilibrium was found for "
                    f"{[str(self._orig(x)) for x in comp]}. Pass it (or part of it) with dfe={{...}}."
                )
            if len(new_states) > 64:
                raise DiseaseFreeEquilibriumError(
                    "Too many disease-free equilibria; pass the one to use with dfe={...}."
                )
            states = new_states

        undetermined = [x for x in self._xu if x not in states[0][0]]

        def explicit_point(state):
            compact, _, star_full = state
            point = {x: v.xreplace(star_full) for x, v in compact.items()}
            for x in undetermined:
                point.setdefault(x, x)
            return point

        def nonzero(state):
            return sum(1 for v in explicit_point(state).values() if not sp.sympify(v).is_zero)

        states.sort(key=lambda st: -nonzero(st))
        compact, stars, star_full = states[0]
        self._stars, self._star_full = stars, star_full
        self._dfe_compact = dict(compact)
        for x in undetermined:
            self._dfe_compact.setdefault(x, x)
        chosen = explicit_point(states[0])
        if len(states) > 1:
            pretty = {str(self._orig(k)): self._orig(v) for k, v in self._dfe_compact.items()
                      if k in self._xu}
            warnings.warn(
                f"{len(states)} disease-free equilibria were found; using the one with "
                f"the most non-zero compartments: {pretty}. Pass dfe={{...}} to choose "
                f"another one (all are in model.dfe_candidates).",
                stacklevel=4,
            )
        for x in self._xi:
            residual = sp.simplify(self._f[x].xreplace(self._dfe_compact))
            if residual != 0:
                raise DiseaseFreeEquilibriumError(
                    f"d{self._orig(x)}/dt = {self._orig(residual)} at the disease-free point, "
                    f"so it is not an equilibrium. Check the infected compartments."
                )
        candidates = [{self._to_orig[k]: self._orig(v) for k, v in explicit_point(st).items()}
                      for st in states]
        return chosen, candidates

    def _compute_dfe_global(self, known):
        """Previous method: one sympy.solve call over all unknown DFE values."""
        unknown = [x for x in self._xu if x not in known]
        equations = []
        for x in self._xu:
            eq = sp.simplify(self._f[x].subs(known))
            if eq == 0:
                continue
            if not eq.free_symbols & set(unknown):
                raise DiseaseFreeEquilibriumError(
                    f"The given dfe values do not satisfy d{self._orig(x)}/dt = 0 "
                    f"(residual {self._orig(eq)})."
                )
            equations.append(eq)

        candidates = [{}]
        if equations:
            # Solve with assumption-free unknowns: positive symbols would make
            # SymPy discard legitimate zero components such as R = 0.
            free = {x: sp.Symbol(f"_{x.name}_dfe") for x in unknown}
            back = {d: x for x, d in free.items()}
            status, raw, seconds = solve_with_time_limit(
                [eq.xreplace(free) for eq in equations], list(free.values()), self.dfe_timeout)
            self._block_log = [{"variables": [str(self._orig(x)) for x in unknown],
                                 "status": "closed form" if status == "ok" else
                                 ("time limit" if status == "timeout" else "no closed form"),
                                 "seconds": seconds}]
            if status == "timeout":
                raise DiseaseFreeEquilibriumError(
                    f"The disease-free equilibrium was not found within dfe_timeout="
                    f"{self.dfe_timeout:g} s. Give it (or part of it) with dfe={{...}}, or "
                    f"increase dfe_timeout (None: no limit)."
                )
            if status != "ok":
                raw = []
            solutions = [{back[d]: sp.sympify(v).xreplace(back) for d, v in s.items()} for s in raw]
            solutions = [s for s in solutions if not any(sp.sympify(v).is_negative for v in s.values())]
            if not solutions:
                raise DiseaseFreeEquilibriumError(
                    "No non-negative disease-free equilibrium was found. "
                    "Pass it (or part of it) with dfe={...}."
                )
            solutions.sort(key=lambda s: -sum(1 for v in s.values() if not sp.sympify(v).is_zero))
            candidates = solutions

        def full(solution):
            point = dict(known)
            point.update({k: sp.simplify(v) for k, v in solution.items()})
            for x in unknown:
                point.setdefault(x, x)  # undetermined: left free (checked later)
            return point

        chosen = full(candidates[0])
        if len(candidates) > 1:
            pretty = {str(self._orig(k)): self._orig(v) for k, v in candidates[0].items()}
            warnings.warn(
                f"{len(candidates)} disease-free equilibria were found; using the one with "
                f"the most non-zero compartments: {pretty}. Pass dfe={{...}} to choose "
                f"another one (all are in model.dfe_candidates).",
                stacklevel=4,
            )

        for x in self._xi:
            residual = sp.simplify(self._f[x].subs(chosen))
            if residual != 0:
                raise DiseaseFreeEquilibriumError(
                    f"d{self._orig(x)}/dt = {self._orig(residual)} at the disease-free point, "
                    f"so it is not an equilibrium. Check the infected compartments."
                )

        orig_candidates = [
            {self._to_orig[k]: self._orig(v) for k, v in full(c).items()} for c in candidates
        ]
        return chosen, orig_candidates

    # -------------------------------------------------------------- matrices
    def _explicit(self, expr):
        """Replace the DFE symbols X^* by their explicit values (no simplification)."""
        return expr.xreplace(self._star_full) if self._star_full else expr

    @property
    def dfe_is_implicit(self) -> bool:
        """True if part of the DFE has no closed form (it is then solved numerically)."""
        return any(v is None for v in self._stars.values())

    @property
    def dfe_blocks(self) -> List[Dict[str, object]]:
        """How each block of the disease-free equilibrium was solved, in solving order.

        One dict per block with ``"variables"``, ``"status"`` (``"closed form"``,
        ``"no closed form"`` or ``"time limit"``, the last two solved
        numerically on demand) and ``"seconds"`` spent solving it symbolically.
        """
        return [dict(b) for b in self._block_log]

    def _build_matrices(self):
        # Jacobians are taken first and evaluated at the *compact* DFE, where
        # values without a short closed form are the symbols X^*. The matrices
        # stay small and R0 is simplified in that form; explicit values are
        # substituted only at the end, without simplification.
        Fv = sp.Matrix([self._F_terms[x] for x in self._xi])
        Vv = sp.Matrix([self._V_terms[x] for x in self._xi])
        F = self._simp(Fv.jacobian(self._xi).xreplace(self._dfe_compact))
        V = self._simp(Vv.jacobian(self._xi).xreplace(self._dfe_compact))

        free = (F.free_symbols | V.free_symbols) & set(self._xu)
        if free:
            names = sorted(str(self._orig(s)) for s in free)
            raise DiseaseFreeEquilibriumError(
                f"The disease-free value of {names} is not determined by the equations "
                f"(e.g. a closed population without births). Provide it with "
                f"dfe={{{names[0]!r}: ...}}."
            )

        det = sp.simplify(V.det())
        if det == 0:
            raise NextGenerationError(
                "The transition matrix V is singular at the disease-free equilibrium, so "
                "F V^-1 does not exist. Check the infected compartments and the equations."
            )
        for (i, j), entry in np.ndenumerate(np.array(F.tolist(), dtype=object)):
            if self._sign(entry) == -1:
                warnings.warn(
                    f"Entry F[{i},{j}] = {self._orig(entry)} is negative; new-infection terms "
                    f"should be non-negative.",
                    stacklevel=3,
                )
        self._Fc, self._Vc = F, V
        self._Kc = self._simp(F * self._inverse(V))
        if self._star_full:
            self._Fm, self._Vm, self._Km = (self._explicit(M) for M in (F, V, self._Kc))
        else:
            self._Fm, self._Vm, self._Km = F, V, self._Kc

    @staticmethod
    def _inverse(V):
        """Inverse of V, by substitution when V is triangular (common for staged infections)."""
        if V.is_lower or V.is_upper:
            return V.inv(method="LU")
        return V.inv()

    # -------------------------------------------------------------------- R0
    def _rank_one(self, K) -> bool:
        n = K.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(n):
                    for l in range(k + 1, n):
                        if sp.simplify(K[i, k] * K[j, l] - K[i, l] * K[j, k]) != 0:
                            return False
        return True

    def _parameter_samples(self, n_wanted=20, max_tries=600):
        """Random positive parameter values giving a meaningful DFE and K."""
        params = sorted({p for p in self._to_orig if p not in self._x}, key=lambda s: s.name)
        dfe_vals = sp.lambdify(params, [self._dfe[x] for x in self._xu] or [0], "numpy")
        K_vals = sp.lambdify(params, self._Km, "numpy")
        good, fallback = [], []
        for _ in range(max_tries):
            point = 10 ** self._rng.uniform(-2, 1, len(params))
            with np.errstate(all="ignore"):
                d = np.array(dfe_vals(*point), dtype=complex).ravel()
                k = np.array(K_vals(*point), dtype=complex)
            if not (np.all(np.isfinite(d)) and np.all(np.isfinite(k))):
                continue
            fallback.append(point)
            if np.all(d.real >= 0) and np.all(np.abs(d.imag) < 1e-9) and np.all(k.real >= -1e-12):
                good.append(point)
                if len(good) >= n_wanted:
                    break
        return params, (good or fallback[:n_wanted])

    def _dominant(self, eigs):
        if self.dfe_is_implicit:
            raise NextGenerationError(
                "Several eigenvalues of F V^-1 may dominate and part of the disease-free "
                "equilibrium has no closed form. Use model.R0_numeric(values)."
            )
        eigs_c, eigs = eigs, [self._explicit(e) for e in eigs]
        params, samples = self._parameter_samples()
        if not samples:
            raise NextGenerationError("Could not evaluate the eigenvalues numerically.")
        funcs = [sp.lambdify(params, e, "numpy") for e in eigs]
        wins: Dict[int, int] = {}
        for point in samples:
            with np.errstate(all="ignore"):
                vals = [complex(f(*point)) for f in funcs]
            radius = max(abs(v) for v in vals)
            tol = 1e-9 * max(1.0, radius)
            # Perron root: the real, non-negative eigenvalue of maximal modulus
            best = [i for i, v in enumerate(vals)
                    if abs(abs(v) - radius) <= tol and abs(v.imag) <= tol and v.real >= -tol]
            if best:
                best.sort(key=lambda i: -vals[i].real)
                wins[best[0]] = wins.get(best[0], 0) + 1
        if not wins:
            raise NextGenerationError(
                "The dominant eigenvalue of F V^-1 could not be identified. "
                "Use model.eigenvalues or model.R0_numeric(values)."
            )
        if len(wins) > 1:
            winners = [eigs_c[i] for i in sorted(wins)]
            warnings.warn(
                "The dominant eigenvalue of F V^-1 depends on the parameter values "
                "(e.g. competing strains), so R0 is returned as Max(...) of the "
                "eigenvalues that can dominate.",
                stacklevel=4,
            )
            return sp.Max(*winners)
        return eigs_c[next(iter(wins))]

    def _small_domain(self):
        """Indices of the compartments that receive new infections.

        If only these rows of F are non-zero, F = E E^T F and the non-zero
        eigenvalues of K = F V^-1 coincide with those of the smaller matrix
        K[rows, rows] (next-generation matrix with small domain; Diekmann,
        Heesterbeek & Roberts 2010, J. R. Soc. Interface 7:873-885).
        """
        n = self._Fc.shape[0]
        return [i for i in range(n) if any(sp.simplify(self._Fc[i, j]) != 0 for j in range(n))]

    def _compute_R0(self):
        rows = self._small_domain()
        K = self._Kc.extract(rows, rows)
        n_zero = self._Kc.shape[0] - len(rows)
        if K.shape == (1, 1):
            eigs = [self._simp(K[0, 0])]
            R0 = eigs[0]
        elif self._rank_one(K):
            R0 = self._simp(K.trace())
            eigs = [R0] + [sp.Integer(0)] * (K.shape[0] - 1)
        elif K.shape == (2, 2) and sp.simplify(K[0, 1] * K[1, 0]) == 0:
            # triangular: eigenvalues are the diagonal entries (e.g. competing strains)
            eigs = [self._simp(K[0, 0]), self._simp(K[1, 1])]
            nonzero = [e for e in eigs if e != 0]
            R0 = nonzero[0] if len(nonzero) == 1 else self._dominant(nonzero)
        elif K.shape == (2, 2):
            # Perron root of a non-negative 2x2 matrix, kept in the structured form
            # used in papers: ((a + d) + sqrt((a - d)^2 + 4bc)) / 2, real because
            # the discriminant is non-negative.
            a, b, c, d = (sp.factor(self._simp(e)) for e in K)
            disc = (a - d) ** 2 + 4 * b * c
            R0 = (a + d) / 2 + sp.sqrt(disc) / 2
            eigs = [R0, (a + d) / 2 - sp.sqrt(disc) / 2]
        else:
            try:
                eig_dict = K.eigenvals()
            except Exception as exc:  # pragma: no cover - depends on SymPy internals
                raise NextGenerationError(
                    f"SymPy could not compute the eigenvalues of F V^-1 ({exc}). "
                    f"Use model.R0_numeric(values) for numerical values."
                ) from exc
            eigs = [self._simp(e) for e, mult in eig_dict.items() for _ in range(mult)]
            if any(e.has(sp.CRootOf) for e in eigs):
                raise NextGenerationError(
                    "The eigenvalues of F V^-1 have no closed form. "
                    "Use model.R0_numeric(values) for numerical values."
                )
            nonzero = [e for e in eigs if e != 0]
            R0 = nonzero[0] if len(nonzero) == 1 else self._dominant(nonzero)
        eigs = list(eigs) + [sp.Integer(0)] * n_zero
        self._eigs_c_cache = [self._tidy(self._orig(e)) for e in eigs]
        self._R0c_cache = self._tidy(self._orig(R0))
        self._eigs_cache = [self._orig(self._explicit(self._pos(e))) for e in self._eigs_c_cache]
        self._R0_cache = self._orig(self._explicit(self._pos(self._R0c_cache)))

    @staticmethod
    def _tidy(expr):
        """Merge square roots split by the positivity assumptions: sqrt(a)*sqrt(b) -> sqrt(a*b)."""
        roots = any(not p.exp.is_Integer for p in expr.atoms(sp.Pow))
        return sp.powsimp(expr, force=True) if roots else expr

    @property
    def next_generation_matrix_small(self) -> sp.Matrix:
        """Next-generation matrix with small domain (rows/columns receiving new infections)."""
        rows = self._small_domain()
        return self._orig(self._explicit(self._Kc.extract(rows, rows)))

    # ---------------------------------------------------------- public API
    @property
    def R0(self) -> sp.Expr:
        """Basic reproduction number: spectral radius of ``F V^{-1}``."""
        if self._R0_cache is None:
            self._compute_R0()
        return self._R0_cache

    @property
    def eigenvalues(self) -> List[sp.Expr]:
        """Eigenvalues of the next-generation matrix (with multiplicity)."""
        if self._eigs_cache is None:
            self._compute_R0()
        return list(self._eigs_cache)

    @property
    def new_infections(self) -> Dict[sp.Symbol, sp.Expr]:
        """New-infection terms ``F_i`` of each infected compartment."""
        return {self._to_orig[x]: self._orig(t) for x, t in self._F_terms.items()}

    @property
    def transitions(self) -> Dict[sp.Symbol, sp.Expr]:
        """Transition terms ``V_i`` (so that ``dx_i/dt = F_i - V_i``)."""
        return {self._to_orig[x]: self._orig(sp.expand(t)) for x, t in self._V_terms.items()}

    @property
    def dfe(self) -> Dict[sp.Symbol, sp.Expr]:
        """Disease-free equilibrium used in the computation."""
        return {self._to_orig[x]: self._orig(self._dfe[x]) for x in self._x}

    @property
    def F(self) -> sp.Matrix:
        """Jacobian of the new-infection terms at the DFE (infected block)."""
        return self._orig(self._Fm)

    @property
    def V(self) -> sp.Matrix:
        """Jacobian of the transition terms at the DFE (infected block)."""
        return self._orig(self._Vm)

    @property
    def next_generation_matrix(self) -> sp.Matrix:
        """Next-generation matrix ``K = F V^{-1}``."""
        return self._orig(self._Km)

    K = next_generation_matrix

    def _values(self, values: Mapping[SymbolLike, float]) -> Dict[sp.Symbol, float]:
        out = {}
        for key, val in values.items():
            sym = to_symbol(key, self.symbols)
            if sym not in self._to_pos:
                raise ModelSpecificationError(f"'{sym}' is not a symbol of this model.")
            out[self._to_pos[sym]] = float(val)
        return out

    # ------------------------------------------------------- compact results
    @property
    def R0_compact(self) -> sp.Expr:
        """R0 written with the DFE values ``X^*`` that have no short closed form.

        For models whose disease-free equilibrium is simple (e.g. ``S = Lambda/mu``)
        this is the same as :attr:`R0`. Otherwise the symbols ``X^*`` are defined
        in :attr:`dfe_definitions`, in solving order.
        """
        if self._R0c_cache is None:
            self._compute_R0()
        return self._R0c_cache

    @property
    def dfe_compact(self) -> Dict[sp.Symbol, sp.Expr]:
        """Disease-free equilibrium with the symbols ``X^*`` of :attr:`dfe_definitions`."""
        return {self._to_orig[x]: self._orig(self._dfe_compact[x]) for x in self._x}

    @property
    def dfe_definitions(self) -> Dict[sp.Symbol, Optional[sp.Expr]]:
        """Definitions of the symbols ``X^*``, in solving order.

        Each value is written in terms of the parameters and earlier ``X^*``.
        ``None`` marks a value with no closed form, defined implicitly by
        ``dX/dt = 0`` and computed numerically by :meth:`dfe_numeric`.
        """
        return {s: (None if v is None else self._orig(v)) for s, v in self._stars.items()}

    @property
    def next_generation_matrix_compact(self) -> sp.Matrix:
        """``K = F V^{-1}`` written with the symbols ``X^*`` of :attr:`dfe_definitions`."""
        return self._orig(self._Kc)

    # ------------------------------------------------------ numerical results
    def _numeric_stars(self, vals: Dict[sp.Symbol, float]) -> Dict[sp.Symbol, sp.Float]:
        """Numerical values of the symbols X^* for the given parameter values."""
        vals = {k: sp.Float(v) for k, v in vals.items()}
        out: Dict[sp.Symbol, sp.Float] = {}
        done_blocks = set()
        for s, expr in self._stars.items():
            if s in out:
                continue
            if expr is None:
                for k, block in enumerate(self._implicit_blocks):
                    if k not in done_blocks and any(star == s for star, _ in block):
                        out.update({star: sp.Float(v) for star, v in
                                    self._solve_implicit_block(k, block, vals, out).items()})
                        done_blocks.add(k)
                        break
                continue
            value = expr.xreplace(out).xreplace(vals)
            self._check_missing(value)
            num = complex(sp.N(value))
            if abs(num.imag) > 1e-9 * max(1.0, abs(num)):
                raise DiseaseFreeEquilibriumError(
                    f"{s} = {num} is not real for these parameter values, so the "
                    f"disease-free equilibrium does not exist there."
                )
            out[s] = sp.Float(num.real)
        return out

    def _check_missing(self, expr):
        missing = sorted(str(self._orig(s)) for s in sp.sympify(expr).free_symbols
                         if s not in self._star_full)
        if missing:
            raise ModelSpecificationError(f"Missing values for parameters: {missing}")

    def _implicit_function(self, k, block):
        """Vector field of an implicit block, compiled once: f(stars, other symbols)."""
        if k not in self._implicit_cache:
            stars = [s for s, _ in block]
            others = sorted({sym for _, eq in block for sym in eq.free_symbols} - set(stars),
                            key=lambda sym: sym.name)
            func = sp.lambdify(stars + others, [eq for _, eq in block], "numpy")
            self._implicit_cache[k] = (stars, others, func)
        return self._implicit_cache[k]

    def _solve_implicit_block(self, k, block, vals, known):
        """Solve dX/dt = 0 for one block without closed form (integration + Newton)."""
        try:
            from scipy.integrate import solve_ivp
            from scipy.optimize import fsolve
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "Part of the disease-free equilibrium has no closed form; solving it "
                "numerically needs SciPy (pip install scipy)."
            ) from exc
        stars, others, func = self._implicit_function(k, block)
        missing = [sym for sym in others if sym not in known and sym not in vals]
        if missing:
            raise ModelSpecificationError(
                f"Missing values for parameters: {sorted(str(self._orig(m)) for m in missing)}")
        other_values = [float(known[sym]) if sym in known else float(vals[sym]) for sym in others]

        def field(_t, y):
            return np.asarray(func(*y, *other_values), dtype=float)

        y = np.ones(len(stars))
        t_end = 10.0
        for _ in range(14):  # integrate towards the stable state, then refine
            sol = solve_ivp(field, (0.0, t_end), y, method="LSODA", rtol=1e-8, atol=1e-12)
            y = sol.y[:, -1]
            if np.max(np.abs(field(0, y))) <= 1e-8 * max(1.0, float(np.max(np.abs(y)))):
                break
            t_end *= 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            root = fsolve(lambda z: field(0, z), y, xtol=1e-12)
        # accept by the residual, not by fsolve's flag: started at the root it
        # reports "no progress" (ier = 5) although the point is correct
        scale = max(1.0, float(np.max(np.abs(root))))
        residual = float(np.max(np.abs(field(0, root))))
        if not np.all(np.isfinite(root)) or residual > 1e-8 * scale or np.any(root < -1e-9):
            raise DiseaseFreeEquilibriumError(
                f"The disease-free values {[str(s) for s in stars]} could not be found "
                f"numerically for these parameter values."
            )
        return {s: float(v) for s, v in zip(stars, root)}

    def dfe_numeric(self, values: Mapping[SymbolLike, float]) -> Dict[sp.Symbol, float]:
        """Disease-free equilibrium for the given parameter values.

        Closed-form values are evaluated in solving order; blocks without a
        closed form are solved numerically (needs SciPy).
        """
        vals = {k: sp.Float(v) for k, v in self._values(values).items()}
        stars = self._numeric_stars(vals)
        out = {}
        for x in self._x:
            value = self._dfe_compact[x].xreplace(stars).xreplace(vals)
            self._check_missing(value)
            out[self._to_orig[x]] = float(sp.N(value))
        return out

    def R0_numeric(self, values: Mapping[SymbolLike, float]) -> float:
        """Spectral radius of ``F V^{-1}`` computed numerically.

        Useful for large models whose eigenvalues have no closed form, or whose
        disease-free equilibrium has none. ``values`` maps parameter names (or
        symbols) to numbers.
        """
        vals = {k: sp.Float(v) for k, v in self._values(values).items()}
        stars = self._numeric_stars(vals)
        K = self._Kc.xreplace(stars).xreplace(vals)
        missing = sorted(str(self._orig(s)) for s in K.free_symbols)
        if missing:
            raise ModelSpecificationError(f"Missing values for parameters: {missing}")
        eig = np.linalg.eigvals(np.array(K.evalf(), dtype=complex))
        return float(np.max(np.abs(eig)))

    # ------------------------------------------------------ DFE stability (A5)
    def _uninfected_jacobian(self):
        xu = [x for x in self._xu if self._dfe_compact.get(x) is not x]
        f = sp.Matrix([self._f[x] for x in xu])
        return xu, f.jacobian(xu).xreplace(self._dfe_compact)

    def dfe_stability(self, values: Optional[Mapping[SymbolLike, float]] = None,
                      n_samples: int = 200) -> Dict[str, object]:
        """Check condition (A5) of van den Driessche & Watmough (2002).

        R0 is a threshold only if the DFE is stable when the infection is
        absent: the Jacobian of the uninfected equations with respect to the
        uninfected compartments, at the DFE, must have eigenvalues with
        negative real part.

        With ``values``: returns ``{"eigenvalues", "stable", "dfe"}`` at that
        point. Without: samples ``n_samples`` random positive parameter sets
        (log-uniform on [0.01, 10]) and returns how many give a non-negative
        DFE (``"feasible"``) and how many of those are stable (``"stable"``).
        When the Jacobian is triangular after ordering the compartments, its
        symbolic eigenvalues are also returned (``"eigenvalues"``).
        """
        xu, J = self._uninfected_jacobian()
        if values is not None:
            vals = self._values(values)
            stars = self._numeric_stars(vals)
            Jn = J.xreplace(stars).xreplace(vals)
            self._check_missing(Jn)
            eig = np.linalg.eigvals(np.array(Jn.evalf(), dtype=complex)) if xu else np.array([])
            return {
                "eigenvalues": eig,
                "stable": bool(np.all(eig.real < 0)),
                "dfe": self.dfe_numeric(values),
            }

        result: Dict[str, object] = {"eigenvalues": self._triangular_eigenvalues(xu, J)}
        params = sorted((p for p in self._to_orig if p not in self._x), key=lambda s: s.name)
        feasible = stable = 0
        for _ in range(n_samples):
            point = {p: sp.Float(v) for p, v in zip(params, 10 ** self._rng.uniform(-2, 1, len(params)))}
            try:
                stars = self._numeric_stars(point)
                dfe = [float(sp.N(self._dfe_compact[x].xreplace(stars).xreplace(point))) for x in xu]
                Jn = np.array(J.xreplace(stars).xreplace(point).evalf(), dtype=complex)
            except (DiseaseFreeEquilibriumError, TypeError, ValueError, ZeroDivisionError):
                continue
            if not np.all(np.isfinite(dfe)) or min(dfe, default=0.0) < 0 or not np.all(np.isfinite(Jn)):
                continue
            feasible += 1
            if not xu or np.all(np.linalg.eigvals(Jn).real < 0):
                stable += 1
        result.update({
            "n_samples": n_samples,
            "feasible": feasible,
            "stable": stable,
            "stable_fraction": stable / feasible if feasible else float("nan"),
        })
        return result

    def _triangular_eigenvalues(self, xu, J):
        n = len(xu)
        edges = [(i, j) for i in range(n) for j in range(n) if i != j and J[i, j] != 0]
        comps = sp.utilities.iterables.strongly_connected_components((list(range(n)), edges))
        if any(len(c) > 1 for c in comps):
            return None
        return [self._orig(self._simp(J[i, i])) for i in range(n)]

    def sensitivity_indices(self, values: Optional[Mapping[SymbolLike, float]] = None):
        """Normalized forward sensitivity indices of R0.

        ``Upsilon_p = (dR0/dp) * p / R0`` for every parameter ``p`` in R0. A value
        of ``0.5`` means that a 1% increase of ``p`` increases R0 by 0.5%.
        Returns symbolic expressions, or floats if ``values`` is given.
        """
        R0 = self._pos(self.R0)
        params = sorted((s for s in R0.free_symbols), key=lambda s: s.name)
        out = {}
        for p in params:
            index = self._simp(sp.diff(R0, p) * p / R0)
            if values is not None:
                index = float(sp.N(index.subs(self._values(values))))
            else:
                index = self._orig(index)
            out[self._to_orig[p]] = index
        return out

    def prcc(self, distributions=None, **kwargs):
        """Global sensitivity of R0 by LHS-PRCC (Marino et al. 2008).

        ``distributions`` maps parameters to specs such as ``(low, high)`` or
        ``("loguniform", low, high)``. See :func:`pyr0compute.global_sensitivity.prcc`
        for all options (``n``, ``baseline``, ``spread``, ``fixed``, ``seed``).
        Needs SciPy.
        """
        from .global_sensitivity import prcc
        return prcc(self, distributions, **kwargs)

    def sobol_indices(self, distributions=None, **kwargs):
        """First-order and total Sobol indices of R0 (Saltelli et al. 2010).

        See :func:`pyr0compute.global_sensitivity.sobol` for all options
        (``n``, ``baseline``, ``spread``, ``fixed``, ``seed``, ``log_output``,
        ``confidence_level``). Needs SciPy >= 1.11.
        """
        from .global_sensitivity import sobol
        return sobol(self, distributions, **kwargs)

    def simulate(self, t_span=(0.0, 100.0), values=None, initial=None, **kwargs):
        """Integrate the ODE model; parameters not in ``values`` are drawn at random.

        By default the uninfected compartments start at the disease-free
        equilibrium and the infected ones at a small perturbation. Returns a
        :class:`~pyr0compute.simulation.SimulationResult` with the R0 of every
        run and the methods ``plot`` and ``plot_phase``. See
        :func:`pyr0compute.simulation.simulate` for all options (``n_runs``,
        ``ranges``, ``default_range``, ``R0_range``, ``seed``, ...). Needs SciPy.
        """
        from .simulation import simulate
        return simulate(self, t_span, values=values, initial=initial, **kwargs)

    def check_assumptions(self, values: Optional[Mapping[SymbolLike, float]] = None,
                          n_samples: int = 100, seed: int = 0, language: str = "en"):
        """Check assumptions (A1)-(A5) of van den Driessche & Watmough (2002).

        Under (A1)-(A5), Theorem 2 of the paper makes R0 a threshold: the DFE is
        locally asymptotically stable if R0 < 1 and unstable if R0 > 1. Each
        assumption is reported as ``"holds"``, ``"fails"`` or ``"undecided"``
        with the basis of the decision (by construction, symbolic proof for all
        positive parameters, at the given ``values``, or random samples).

        Returns an :class:`~pyr0compute.assumptions.AssumptionsReport`; its
        :meth:`~pyr0compute.assumptions.AssumptionsReport.to_latex` writes the
        report in LaTeX (``language="es"`` for Spanish: cumple / no cumple /
        no se pudo decidir). See :func:`pyr0compute.assumptions.check_assumptions`.
        """
        from .assumptions import check_assumptions
        return check_assumptions(self, values, n_samples=n_samples, seed=seed, language=language)

    def global_sensitivity(self, distributions=None, method: str = "prcc", **kwargs):
        """Global sensitivity analysis of R0: ``method="prcc"`` or ``"sobol"``."""
        method = method.lower()
        if method in ("prcc", "lhs-prcc", "lhs"):
            return self.prcc(distributions, **kwargs)
        if method == "sobol":
            return self.sobol_indices(distributions, **kwargs)
        raise ValueError("method must be 'prcc' or 'sobol'.")

    def latex(self, compact: Optional[bool] = None) -> str:
        """LaTeX code of R0.

        ``compact`` (default: True when the DFE has values ``X^*`` without a
        short closed form) writes :attr:`R0_compact` instead of the explicit R0.
        """
        if compact is None:
            compact = bool(self._stars)
        return sp.latex(self.R0_compact if compact else self.R0)

    def report(self) -> str:
        """Step-by-step description of the computation."""
        def pp(obj):
            return sp.pretty(obj, use_unicode=True)

        lines = ["=" * 64, "pyR0compute: next-generation matrix method", "=" * 64]
        lines.append(f"Variables:  {', '.join(map(str, self.variables))}")
        lines.append(f"Infected:   {', '.join(map(str, self.infected))}")
        lines.append(f"Parameters: {', '.join(map(str, self.parameters))} (detected automatically)")
        lines.append(f"\nNew-infection terms F_i ({self._new_infections_mode}):")
        for v, t in self.new_infections.items():
            lines.append(f"  {v}: {t}")
        if self._classification:
            lines.append("\nTerm classification:")
            for x, term, kind, why in self._classification:
                lines.append(f"  d{self._orig(x)}/dt  {kind}  {self._orig(term)}   ({why})")
        lines.append("\nTransition terms V_i:")
        for v, t in self.transitions.items():
            lines.append(f"  {v}: {t}")
        compact = bool(self._stars)
        lines.append("\nDisease-free equilibrium:")
        for v, val in (self.dfe_compact if compact else self.dfe).items():
            lines.append(f"  {v} = {val}")
        if compact:
            lines.append("where (solved block by block, in this order):")
            for s, val in self.dfe_definitions.items():
                if val is None:
                    why = (f"not found within dfe_timeout = {self.dfe_timeout:g} s"
                           if self._implicit_reason.get(self._pos(s)) == "time limit"
                           else "no closed form")
                    val = f"({why}; computed numerically)"
                lines.append(f"  {s} = {val}")
            F, V, K = (self._orig(M) for M in (self._Fc, self._Vc, self._Kc))
        else:
            F, V, K = self.F, self.V, self.K
        lines += ["\nF =", pp(F), "\nV =", pp(V), "\nK = F V^-1 =", pp(K)]
        try:
            lines += ["\nR0 =", pp(self.R0_compact if compact else self.R0)]
            if compact:
                lines.append("(model.R0 gives R0 with the explicit DFE values substituted)")
        except NextGenerationError as exc:
            lines += [f"\nR0: {exc}"]
        return "\n".join(lines)

    def report_latex(
        self,
        style: str = "document",
        standalone: bool = False,
        mat_str: str = "bmatrix",
    ) -> str:
        """Step-by-step description of the computation in LaTeX.

        Parameters
        ----------
        style
            ``"document"`` (default): LaTeX to paste into a paper or compile
            (needs ``amsmath``). ``"markdown"``: Markdown with ``$$...$$`` math,
            to render in Jupyter/VS Code with
            ``display(Markdown(model.report_latex("markdown")))``.
        standalone
            With ``style="document"``, wrap the result in a complete
            ``\\documentclass{article}`` file that compiles as is.
        mat_str
            Matrix environment: ``"bmatrix"`` (brackets), ``"pmatrix"``
            (parentheses) or ``"matrix"``.
        """
        if style not in ("document", "markdown"):
            raise ValueError("style must be 'document' or 'markdown'.")

        def tex(obj):
            return sp.latex(obj, mat_str=mat_str, mat_delim="")

        def names(symbols):
            return ", ".join(tex(s) for s in symbols)

        def block(rows):
            body = " \\\\\n".join(f"{lhs} &= {rhs}" for lhs, rhs in rows)
            if style == "markdown":
                return f"$$\n\\begin{{aligned}}\n{body}\n\\end{{aligned}}\n$$"
            return f"\\begin{{align*}}\n{body}\n\\end{{align*}}"

        def heading(text):
            return f"### {text}" if style == "markdown" else f"\\subsection*{{{text}}}"

        def inline(math):
            return f"${math}$"

        md = style == "markdown"
        out = []
        out.append("## pyR0compute: next-generation matrix method" if md
                   else "\\section*{pyR0compute: next-generation matrix method}")
        sep = "  \n" if md else " \\\\\n"
        out.append(sep.join([
            ("**Variables:** " if md else "\\textbf{Variables:} ") + inline(names(self.variables)),
            ("**Infected compartments:** " if md else "\\textbf{Infected compartments:} ")
            + inline(names(self.infected)),
            ("**Parameters (detected automatically):** " if md
             else "\\textbf{Parameters (detected automatically):} ") + inline(names(self.parameters)),
        ]))

        out.append(heading("Model"))
        out.append(block([(f"\\frac{{d{tex(v)}}}{{dt}}", tex(self.equations[v])) for v in self.variables]))

        out.append(heading(f"New-infection terms ({self._new_infections_mode})"))
        out.append(block([(f"\\mathcal{{F}}_{{{tex(v)}}}", tex(t)) for v, t in self.new_infections.items()]))

        if self._classification:
            out.append(heading("Term classification"))
            if md:
                rows = ["| Equation | Term | Class | Reason |", "|---|---|:---:|---|"]
                for x, term, kind, why in self._classification:
                    v = self._to_orig[x]
                    rows.append(f"| $d{tex(v)}/dt$ | ${tex(self._orig(term))}$ "
                                f"| $\\mathcal{{{kind}}}$ | {why} |")
                out.append("\n".join(rows))
            else:
                rows = ["\\begin{tabular}{llcl}", "\\hline",
                        "Equation & Term & Class & Reason \\\\", "\\hline"]
                for x, term, kind, why in self._classification:
                    v = self._to_orig[x]
                    why_tex = why.replace("_", "\\_")
                    rows.append(f"$d{tex(v)}/dt$ & ${tex(self._orig(term))}$ "
                                f"& $\\mathcal{{{kind}}}$ & {why_tex} \\\\")
                rows += ["\\hline", "\\end{tabular}"]
                out.append("\n".join(rows))

        out.append(heading("Transition terms"))
        out.append(block([(f"\\mathcal{{V}}_{{{tex(v)}}}", tex(t)) for v, t in self.transitions.items()]))

        compact = bool(self._stars)
        out.append(heading("Disease-free equilibrium"))
        out.append(block([(tex(v), tex(val))
                          for v, val in (self.dfe_compact if compact else self.dfe).items()]))
        if compact:
            out.append("The values marked with $^*$ are found block by block, in this order:")
            out.append(block([
                (tex(s), tex(val) if val is not None
                 else "\\text{root of } \\dot{" + tex(s.compartment) + "} = 0")
                for s, val in self.dfe_definitions.items()
            ]))
            F, V, K = (self._orig(M) for M in (self._Fc, self._Vc, self._Kc))
        else:
            F, V, K = self.F, self.V, self.K

        out.append(heading("Next-generation matrix"))
        out.append(block([
            ("F", tex(F)),
            ("V", tex(V)),
            ("K = F V^{-1}", tex(K)),
        ]))
        small = self._orig(self._Kc.extract(self._small_domain(), self._small_domain())) \
            if compact else self.next_generation_matrix_small
        if small.shape != self.K.shape:
            rows_ = ", ".join(tex(self.infected[i]) for i in self._small_domain())
            out.append(
                "Only " + inline(rows_)
                + " receive new infections, so $\\mathcal{R}_0$ is the spectral radius of the "
                "next-generation matrix with small domain:"
            )
            out.append(block([("K_{\\text{small}}", tex(small))]))

        out.append(heading("Basic reproduction number"))
        try:
            out.append(block([("\\mathcal{R}_0 = \\rho\\left(F V^{-1}\\right)",
                               tex(self.R0_compact if compact else self.R0))]))
        except NextGenerationError as exc:
            out.append(str(exc))

        text = "\n\n".join(out)
        if standalone and not md:
            text = ("\\documentclass{article}\n\\usepackage{amsmath}\n"
                    "\\usepackage[margin=2cm]{geometry}\n\\begin{document}\n\n"
                    f"{text}\n\n\\end{{document}}\n")
        return text

    def __repr__(self) -> str:
        return (
            f"R0Model(variables={list(self.variables)}, infected={list(self.infected)}, "
            f"parameters={list(self.parameters)})"
        )

    @classmethod
    def from_text(cls, text: str, infected, **kwargs) -> "R0Model":
        """Build a model from a text block (see :class:`R0Model`)."""
        return cls(text, infected, **kwargs)
