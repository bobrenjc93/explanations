"""Independent re-check of one arithmetic step of a worked solution (same rules as check_gen2.py)."""
import re
from decimal import Decimal as D
import mathgen2 as m2

num = r"\(?-?[0-9.]+\)?"
val = lambda t: D(t.strip("()"))
STEP = re.compile(rf"({num})([-+*/%])({num})(?::([0-9rp,]*))?(?:=(-?[0-9.]+))?")


def close6(a, b):
    """equal up to a couple of units in the 6th significant digit"""
    return a == b or abs(a - b) <= max(abs(a), abs(b)) * D("3e-5") or abs(a - b) < D("1e-8")


def check_step(st):
    """True / False for an arithmetic step, None for anything else (labels, restated equations, looked-up values)."""
    m = STEP.fullmatch(st)
    if not m:
        return None
    a, op, b, work, v = val(m.group(1)), m.group(2), val(m.group(3)), m.group(4), m.group(5)
    if op in "+-":
        return close6(a + b if op == "+" else a - b, D(v))
    if op == "*":
        ok = close6(a * b, D(v))
        if work:
            items = [int(x) for x in work.split(",")]
            (na, _), (nb, _) = m2.sig_int(a), m2.sig_int(b)
            ok &= items[-1] == na * nb
        return ok
    steps = []
    for t in work.split(",") if work else []:
        head, rem = t.split("r")
        d = int(head.split("p")[0])
        if "p" in head and int(head.split("p")[1]) != d * (int(b) if op == "%" else m2.sig_int(b)[0]):
            return False
        steps.append((d, int(rem)))
    if op == "%":
        return steps[-1][1] == int(a) % int(b) and int("".join(str(d) for d, _ in steps)) == int(a) // int(b)
    if not steps:
        return a == 0
    digs = "".join(str(d) for d, _ in steps)
    true = abs(a / b)
    want = true.scaleb(-true.adjusted() + len(digs) - 1)
    return abs(D(int(digs)) - want) < 1 and close6(a / b, D(v))
