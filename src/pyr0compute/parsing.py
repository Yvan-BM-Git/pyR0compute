"""Safe parsing of ODE models written as plain text.

Names such as ``I``, ``S``, ``E``, ``N``, ``beta``, ``gamma`` or ``Lambda``
have special meanings in SymPy (imaginary unit, singleton registry, Euler's
number, ...). In epidemiology they are compartments and parameters, so every
identifier found in a model is turned into a plain :class:`sympy.Symbol`
unless it is one of a short list of mathematical functions.
"""

from __future__ import annotations

import keyword
import re
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

from .exceptions import ModelSpecificationError

ExprLike = Union[str, sp.Expr, int, float]
SymbolLike = Union[str, sp.Symbol]

#: Function names recognised inside text expressions.
FUNCTIONS: Dict[str, object] = {
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "tanh": sp.tanh,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "Min": sp.Min,
    "Max": sp.Max,
}

_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_TRANSFORMS = standard_transformations + (convert_xor,)
_GLOBALS = {
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
    "Function": sp.Function,
}

_DERIV_PATTERNS = (
    re.compile(r"^\s*d\s*([A-Za-z_]\w*)\s*/\s*d\s*t\s*=\s*(.+)$"),  # dS/dt = ...
    re.compile(r"^\s*([A-Za-z_]\w*)\s*'\s*=\s*(.+)$"),  # S' = ...
)
_DEF_PATTERN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$")  # N = S + I + R


def to_symbol(name: SymbolLike, table: Optional[Mapping[str, sp.Symbol]] = None) -> sp.Symbol:
    """Return the Symbol for ``name`` (a string or an existing Symbol)."""
    if isinstance(name, sp.Symbol):
        return name
    if isinstance(name, str):
        name = name.strip()
        if not _IDENT.fullmatch(name):
            raise ModelSpecificationError(f"'{name}' is not a valid variable name.")
        if table is not None and name in table:
            return table[name]
        return sp.Symbol(name)
    raise ModelSpecificationError(
        f"Expected a variable name or a SymPy Symbol, got {type(name).__name__}: {name!r}"
    )


def parse_expression(text: ExprLike, table: Optional[Mapping[str, sp.Basic]] = None) -> sp.Expr:
    """Parse ``text`` into a SymPy expression.

    ``table`` maps names to the objects that must be used for them (the model
    variables, already-created parameters or auxiliary definitions). Any other
    identifier becomes a new plain Symbol, except the functions in
    :data:`FUNCTIONS`. ``^`` is accepted as exponentiation.
    """
    if isinstance(text, sp.Basic):
        return text
    if isinstance(text, (int, float)):
        return sp.sympify(text)
    if not isinstance(text, str):
        raise ModelSpecificationError(f"Cannot interpret {text!r} as an expression.")

    table = dict(table or {})
    local: Dict[str, object] = {}
    code = text

    for name in set(_IDENT.findall(text)):
        if keyword.iskeyword(name):
            # e.g. 'lambda': rename in the code, keep the original symbol name
            alias = f"__kw_{name}__"
            code = re.sub(rf"\b{name}\b", alias, code)
            local[alias] = table.get(name, sp.Symbol(name))
        elif name in table:
            local[name] = table[name]
        elif name in FUNCTIONS:
            local[name] = FUNCTIONS[name]
        else:
            local[name] = sp.Symbol(name)

    try:
        expr = parse_expr(code, local_dict=local, global_dict=dict(_GLOBALS),
                          transformations=_TRANSFORMS, evaluate=True)
    except Exception as exc:  # pragma: no cover - message depends on SymPy
        raise ModelSpecificationError(f"Could not parse expression '{text}': {exc}") from exc
    return sp.sympify(expr)


def split_text_model(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split a text model into derivative lines and auxiliary definitions.

    Accepted line formats (``#`` starts a comment)::

        dS/dt = Lambda - beta*S*I/N - mu*S
        I' = beta*S*I/N - (gamma + mu)*I
        N = S + I + R            # auxiliary definition, substituted later
    """
    equations: List[Tuple[str, str]] = []
    definitions: List[Tuple[str, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for pattern in _DERIV_PATTERNS:
            match = pattern.match(line)
            if match:
                equations.append((match.group(1), match.group(2)))
                break
        else:
            match = _DEF_PATTERN.match(line)
            if match:
                definitions.append((match.group(1), match.group(2)))
            else:
                raise ModelSpecificationError(
                    f"Line {lineno} is not of the form 'dX/dt = ...', \"X' = ...\" "
                    f"or 'name = ...': {raw.strip()!r}"
                )
    if not equations:
        raise ModelSpecificationError("No equation of the form 'dX/dt = ...' was found.")
    return equations, definitions


def parse_text_model(
    text: str,
) -> Tuple[Dict[sp.Symbol, sp.Expr], Dict[str, sp.Symbol], Dict[str, sp.Expr]]:
    """Parse a whole text model.

    Returns ``(equations, symbols, definitions)``: ``equations`` maps each state
    variable to its right-hand side (with auxiliary definitions substituted),
    ``symbols`` maps every name used to its Symbol and ``definitions`` holds
    the auxiliary expressions such as ``N = S + I + R``.
    """
    raw_eqs, raw_defs = split_text_model(text)
    names = [name for name, _ in raw_eqs]
    duplicated = {n for n in names if names.count(n) > 1}
    if duplicated:
        raise ModelSpecificationError(f"Variables defined more than once: {sorted(duplicated)}")

    table: Dict[str, sp.Basic] = {name: sp.Symbol(name) for name in names}
    definitions: Dict[str, sp.Expr] = {}
    for name, rhs in raw_defs:
        if name in table:
            raise ModelSpecificationError(
                f"'{name}' is a state variable and cannot also be an auxiliary definition."
            )
        # earlier definitions may be used in later ones
        definitions[name] = parse_expression(rhs, {**table, **definitions})

    equations: Dict[sp.Symbol, sp.Expr] = {}
    for name, rhs in raw_eqs:
        equations[table[name]] = parse_expression(rhs, {**table, **definitions})

    symbols: Dict[str, sp.Symbol] = {}
    for expr in equations.values():
        for s in expr.free_symbols:
            symbols[s.name] = s
    symbols.update({name: table[name] for name in names})
    return equations, symbols, definitions


def unique_symbols(items: Iterable[sp.Basic]) -> Dict[str, sp.Symbol]:
    """Map names to symbols, refusing two different symbols with the same name."""
    out: Dict[str, sp.Symbol] = {}
    for item in items:
        for s in item.free_symbols:
            other = out.get(s.name)
            if other is not None and other != s:
                raise ModelSpecificationError(
                    f"Two different symbols are both named '{s.name}' (probably created "
                    f"with different assumptions, e.g. positive=True). Use a single symbol "
                    f"for each name."
                )
            out[s.name] = s
    return out
