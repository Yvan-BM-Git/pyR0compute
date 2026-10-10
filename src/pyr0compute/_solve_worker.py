"""Worker process for :func:`pyr0compute._solve.solve_with_time_limit`.

Run as ``python -m pyr0compute._solve_worker``: reads a pickled
``{"system": [...], "unknowns": [...]}`` from stdin, calls ``sympy.solve``
and writes a pickled ``(status, result)`` to stdout. Running it in its own
interpreter lets the parent stop a solve that does not finish, which is not
possible from inside the same Python process.
"""

import pickle
import sys


def main() -> None:
    data = pickle.load(sys.stdin.buffer)
    import sympy as sp

    try:
        result = ("ok", sp.solve(data["system"], data["unknowns"], dict=True))
    except NotImplementedError:
        result = ("no_closed_form", None)
    except Exception as exc:  # reported to the parent, which falls back
        result = ("error", repr(exc))
    pickle.dump(result, sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
