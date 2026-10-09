"""``sympy.solve`` with a time limit.

SymPy's polynomial solvers (Groebner bases, resultants) can run for hours on
coupled nonlinear blocks, and a pure-Python computation cannot be interrupted
safely from the same process. The solve therefore runs in a separate
interpreter (``python -m pyr0compute._solve_worker``) that is stopped when the
time limit is reached. A fresh interpreter is used instead of
``multiprocessing`` because, with the "spawn" start method (macOS, Windows),
``multiprocessing`` re-imports the user's script.

If the worker cannot be started (e.g. an embedded interpreter), the solve
falls back to the current process, interrupted with ``SIGALRM`` where that
is available (Unix, main thread) and without a limit otherwise.
"""

from __future__ import annotations

import os
import pickle
import signal
import subprocess
import sys
import threading
import time
import warnings
from typing import List, Optional, Sequence, Tuple

import sympy as sp

__all__ = ["solve_with_time_limit"]

_WORKER_UNAVAILABLE = False  # set once the worker has failed to start


def _package_root() -> str:
    """Directory that contains the ``pyr0compute`` package (for the worker's sys.path)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _solve_in_worker(system, unknowns, timeout: float) -> Tuple[str, object]:
    payload = pickle.dumps({"system": list(system), "unknowns": list(unknowns)},
                           protocol=pickle.HIGHEST_PROTOCOL)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_package_root()] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    proc = subprocess.Popen(
        [sys.executable, "-m", "pyr0compute._solve_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    try:
        out, err = proc.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return "timeout", None
    if proc.returncode != 0 or not out:
        return "unavailable", err.decode(errors="replace")[-500:]
    return pickle.loads(out)


class _Alarm(Exception):
    pass


def _solve_in_process(system, unknowns, timeout: Optional[float]) -> Tuple[str, object]:
    use_alarm = (
        timeout is not None
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if use_alarm:
        def handler(signum, frame):
            raise _Alarm()

        previous = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return "ok", sp.solve(list(system), list(unknowns), dict=True)
    except _Alarm:
        return "timeout", None
    except NotImplementedError:
        return "no_closed_form", None
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def solve_with_time_limit(
    system: Sequence[sp.Expr],
    unknowns: Sequence[sp.Symbol],
    timeout: Optional[float],
    in_process: bool = False,
) -> Tuple[str, Optional[List[dict]], float]:
    """Solve ``system = 0`` for ``unknowns``, giving up after ``timeout`` seconds.

    Returns ``(status, solutions, seconds)`` with ``status`` one of ``"ok"``,
    ``"no_closed_form"`` (SymPy raised NotImplementedError) or ``"timeout"``.
    ``timeout=None`` solves in the current process without a limit;
    ``in_process=True`` skips the worker (for blocks known to be easy).
    """
    global _WORKER_UNAVAILABLE
    start = time.perf_counter()
    if timeout is None or in_process:
        status, result = _solve_in_process(system, unknowns, None)
        return status, result, time.perf_counter() - start

    status, result = "unavailable", None
    if not _WORKER_UNAVAILABLE:
        try:
            status, result = _solve_in_worker(system, unknowns, timeout)
        except (OSError, pickle.PickleError, EOFError) as exc:
            status, result = "unavailable", repr(exc)
    if status == "error":
        status = "unavailable"  # e.g. an exception inside SymPy: retry here for the traceback
    if status == "unavailable":
        if not _WORKER_UNAVAILABLE:
            _WORKER_UNAVAILABLE = True
            warnings.warn(
                "pyR0compute could not start a separate Python process to solve the "
                "disease-free equilibrium with a time limit; solving in this process "
                "instead" + (" (interrupted with SIGALRM)." if hasattr(signal, "SIGALRM")
                             else " without a time limit."),
                RuntimeWarning, stacklevel=4,
            )
        remaining = max(0.1, timeout - (time.perf_counter() - start))
        status, result = _solve_in_process(system, unknowns, remaining)
    return status, result, time.perf_counter() - start
