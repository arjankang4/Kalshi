"""The pass/fail harness the validators and pinned-behaviour tests share.

WHY THIS IS NOT PYTEST. `strategy/test_quoting.py` states the requirement outright: "No pytest
dependency -- this has to be runnable in the minutes before a session." pytest is not installed
in `kalshi_bot` and there is no conftest, so every gate here is a plain module run as
`python -m <mod>` that prints its own findings and exits non-zero on failure. That property is
deliberate and this module preserves it.

WHY IT EXISTS. By 2026-08-24 the same four helpers had been copy-pasted into fifteen modules --
`check` byte-identical in six, `assert_true` in five, `approx` in four, `truthy` in three -- each
appending to its own module-level failure list. Fifteen copies of a predicate is fifteen places
for one to drift, and a test harness that drifts reports green for the wrong reason.

WHAT IT DELIBERATELY DOES NOT DO. It does not collect, discover, or reorder anything. The gates
read top-to-bottom as a printed argument and that is most of their value; a runner that hid the
ordering would cost more than it saved.

    from harness import approx, assert_true, check, finish, ok, truthy

    check("the ladder is three rungs", len(rungs), 3)
    finish()            # prints PASS / FAIL: <labels> and exits 0 / 1

`FAILURES` is module-level and therefore shared: a process that runs two gates in sequence
reports the union, which is what a caller checking one exit code wants. `reset()` clears it for
the rare in-process harness that needs to score gates separately (`notebooks/03` shells out
instead, so it does not).
"""
import sys

FAILURES = []


def check(label, got, want):
    """Exact equality. `got` and `want` are printed on every line, pass or fail.

    Printing the values even on a pass is not noise: these gates are read as evidence, and a
    number nobody can see is an assertion rather than a measurement.

    `!r` rather than `!s` deliberately. Kalshi carries counts as fixed-point STRINGS, so a line
    reading `got=10.00 want=10.00` can be a passing comparison of two floats or a failing
    comparison of a float against "10.00" -- and the failing one is the bug you are hunting.
    """
    good = got == want
    print(f"  {'ok  ' if good else 'FAIL'} {label}: got={got!r} want={want!r}")
    if not good:
        FAILURES.append(label)
    return good


def ok(label, got, want, tol=None):
    """`check` with an optional tolerance -- exact when `tol` is None, else |got-want| <= tol.

    The three model validators (`pricing/validate_greeks`, `rvol/validate`, `edgecal/validate`)
    grew this instead of two separate helpers because most of what they assert is numeric and
    only some of it is exact. A `None` on either side never passes a tolerance comparison.
    """
    good = (got == want) if tol is None else (
        got is not None and want is not None and abs(got - want) <= tol)
    print(f"  {'ok  ' if good else 'FAIL'} {label}: got={got!r} want={want!r}"
          + (f" tol={tol}" if tol else ""))
    if not good:
        FAILURES.append(label)
    return good


def approx(label, got, want, tol=1e-9):
    """Equality within `tol`. A `None` never passes -- absence is not approximately a number."""
    good = got is not None and abs(got - want) <= tol
    print(f"  {'ok  ' if good else 'FAIL'} {label}: got={got} want={want} tol={tol}")
    if not good:
        FAILURES.append(label)
    return good


def assert_true(label, cond, detail=""):
    """A property that holds, with optional evidence printed beside it."""
    print(f"  {'ok  ' if cond else 'FAIL'} {label}{(' ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(label)
    return bool(cond)


def truthy(label, cond, note=""):
    """`assert_true` with the note after a colon. Kept distinct because the three validators
    that use it print a measured value as the note and the colon reads better there."""
    print(f"  {'ok  ' if cond else 'FAIL'} {label}" + (f": {note}" if note else ""))
    if not cond:
        FAILURES.append(label)
    return bool(cond)


def failed():
    return list(FAILURES)


def reset():
    FAILURES.clear()


def report():
    """Print the verdict and return the exit code, without exiting.

    Separate from `finish` because several gates are also called as library functions from
    `notebooks/03` and from other validators, where a `sys.exit` would kill the caller.
    """
    print(f"\n{'PASS' if not FAILURES else 'FAIL: ' + ', '.join(FAILURES)}")
    return 0 if not FAILURES else 1


def finish():
    sys.exit(report())
