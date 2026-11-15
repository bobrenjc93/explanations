"""Seven harder tiers on top of the seven calculator levels: big numbers, long expressions, linear and quadratic
equations, linear systems, sums and Newton's method.

Every problem is a dict with
    text    what the model reads: the old expression grammar plus x y z k, solve( ) and sum( )
    truth   the displayed answer from exact (or double precision) arithmetic, independent of any scratchpad
    kind    a sub-type label for the analysis
and solve(problem) writes a worked solution in the mathgen2 step language. All arithmetic in it is an ordinary
Tracer step (partial products, long division, Newton square roots ...); the algebra in between is written as
restated equations and labelled values:

    solve(3*(x-2)+4*x=5*(x+1)-7)   3*(-2)=-6;5*1=5;3*x-6+4*x=5*x+5-7;3+4=7;5+(-7)=-2;7*x-6=5*x-2;7-5=2;-2-(-6)=4;2*x=4;4/2:2r0=2
    solve(2*x^2-5*x-3=0)           5*5=25;(-5)^2=25;4*2=8;8*(-3)=-24;25-(-24)=49;d=49;2*2=4;sqrt(49)~7;...;5+7=12;12/4:3r0=3;2*3=6;-3/6:5r0=-0.5
                                   (the root where -b and the square root add up, then the other from x1*x2 = c/a; listed in ascending order)
    solve(2*x+3*y=7,x-y=1)         Cramer's rule: d=..;dx=..;dy=..; then dx/d, dy/d   (3x3: cofactors of the first row)
    sum(k^2+3*k,k,1,9)             k=1;<term>;k=2;<term>;<running sum>;...
    solve(x^3+2*x-5=0,1)           d=3*x^2+2;x=1;<f(x) by Horner>;<f'(x) by Horner>;<f/f'>;<x-f/f'>;x=1.4;...   until x repeats to 4 digits
                                   (iterates carry 2, then 3, then 4 significant digits)
"""
import math
import random
import re
from contextlib import contextmanager
from decimal import Decimal as D, ROUND_HALF_UP
from fractions import Fraction as F

import mathgen as mg
import mathgen2 as m2
from mathgen import Reject, num, B, fn, rint, nint, dec, pick
from mathgen2 import V, s, fmt4

NEW = "xyzkvu"
CHARS = m2.CHARS + NEW  # old ids are unchanged; the six new characters are appended
STOI = {c: i + 3 for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS) + 3
OLD = mg.NLEVELS
LEVELS = mg.LEVELS + ["L8 big numbers", "L9 long expressions", "L10 linear equations", "L11 quadratics", "L12 linear systems",
                      "L13 sums", "L14 Newton's method"]
NLEVELS = len(LEVELS)
P_MAX = 96    # BOS + text + '='
A_MAX = 40    # answer + EOS
T_MAX = 2304  # '[' worked solution ']' answer EOS
VARS = "xyz"
# Stretch problems: every tier's size knob set beyond its training range (more digits, operations, terms, iterations).
# They are never trained on; build_stretch.py makes a test set of them.
STRETCH = False
S_P_MAX, S_T_MAX = 260, 6000


def K(train, stretch):
    return stretch if STRETCH else train



@contextmanager
def big():
    """The new tiers show integers in full up to 10^18 (the seven old levels stop at 10^10)."""
    old = (mg.INT_SHOW, mg.FRAC_LIMIT, mg.POW_LIMIT, m2.INT_LIMIT, m2.INT_SHOW)
    e = K(18, 26)
    mg.INT_SHOW, mg.FRAC_LIMIT, mg.POW_LIMIT, m2.INT_LIMIT, m2.INT_SHOW = 10**e, 10 ** (e + 4), 10**e, D(10) ** (e + 4), D(10) ** e
    try:
        yield
    finally:
        mg.INT_SHOW, mg.FRAC_LIMIT, mg.POW_LIMIT, m2.INT_LIMIT, m2.INT_SHOW = old


class Tracer(m2.Tracer):
    """Direct calls collapse to one 'op=value' step when procedures are switched off (the values-only format),
    the way mathgen2.Tracer.val already does for expression nodes."""

    def _one(self, f, *a):
        first = len(self.steps)
        v = f(*a)
        if not m2.OPT["work"] and len(self.steps) > first:
            last = re.sub(r":[^=]*", "", self.steps[-1])
            del self.steps[first:]
            self.steps.append(last)
        return v

    def mul(self, a, b):
        return self._one(super().mul, a, b)

    def div(self, a, b):
        return self._one(super().div, a, b)

    def sqrt(self, x):
        return self._one(super().sqrt, x)

    def power(self, a, b, base_is_e):
        return self._one(super().power, a, b, base_is_e)


# ------------------------------------------------------------------ printing
def lin_text(pairs, const=None):
    """'7*x-6' from [(coefficient, name), ...] and a constant; zero terms are dropped."""
    out = ""
    for c, name in pairs:
        if c == 0:
            continue
        body = name if abs(c) == 1 else s(abs(c)) + "*" + name
        out += ("-" if c < 0 else ("+" if out else "")) + body
    if const is not None and (const != 0 or not out):
        out += ("-" if const < 0 else ("+" if out else "")) + s(abs(const))
    return out or "0"


def terms_text(terms):
    """One side of a linear equation as written: ('x', c) | ('c', c) | ('p', m, a, b) for m*(a*x+b)."""
    out = ""
    for t in terms:
        if t[0] == "c":
            body = s(abs(t[1]))
        elif t[0] == "x":
            body = "x" if abs(t[1]) == 1 else s(abs(t[1])) + "*x"
        else:
            body = s(abs(t[1])) + "*(" + lin_text([(t[2], "x")], t[3]) + ")"
        out += ("-" if t[1] < 0 else ("+" if out else "")) + body
    return out or "0"


def poly_text(coefs, var="x"):
    """'2*x^2-5*x-3' from coefficients, highest degree first."""
    out, n = "", len(coefs) - 1
    for i, c in enumerate(coefs):
        if c == 0:
            continue
        deg = n - i
        if deg == 0:
            body = s(abs(c))
        else:
            xp = var if deg == 1 else f"{var}^{deg}"
            body = xp if abs(c) == 1 else s(abs(c)) + "*" + xp
        out += ("-" if c < 0 else ("+" if out else "")) + body
    return out or "0"


def sub(text, var, x):
    """The expression with a number in place of the variable (negative numbers in parentheses)."""
    return text.replace(var, s(x) if x >= 0 else "(" + s(x) + ")")


# ------------------------------------------------------- L8: big numbers
def gen_big(rng):
    def add(r):
        lo, hi = K((8, 14), (15, 20))
        n = nint(r, lo, hi)
        for _ in range(r.randint(1, 3)):
            n = B(r.choice("+-"), n, nint(r, lo, hi))
        return "add/sub", n

    def mul(r):
        return "multiply", B("*", nint(r, *K((4, 8), (9, 11))), nint(r, *K((4, 8), (9, 11))))

    def div(r):
        return "divide", B("/", nint(r, *K((7, 12), (13, 16))), num(str(max(2, rint(r, *K((3, 6), (7, 8)))))))

    def divx(r):
        b, q = max(2, rint(r, *K((3, 6), (7, 8)))), max(2, rint(r, *K((3, 7), (8, 9))))
        return "divide", B("/", num(str(b * q)), num(str(b)))

    def mod(r):
        return "mod", fn("mod", nint(r, *K((7, 12), (13, 16))), num(str(max(2, rint(r, *K((3, 6), (7, 8)))))))

    def gcd(r):
        g, f = max(1, rint(r, *K((1, 4), (2, 4)))), K((3, 4), (5, 6))
        return "gcd/lcm", fn(r.choice(["gcd", "lcm"]), num(str(g * rint(r, *f))), num(str(g * rint(r, *f))))

    def power(r):
        return "power", B("^", nint(r, *K((2, 3), (4, 4))), num(str(r.randint(*K((3, 6), (4, 6))))))

    def mixed(r):
        def tree(q, nops):
            if nops == 0:
                return nint(q, *K((3, 6), (4, 7)))
            k = q.randint(0, nops - 1)
            return B(q.choices("+-*", weights=[2, 2, 3])[0], tree(q, k), tree(q, nops - 1 - k))

        return "mixed", tree(r, r.randint(*K((2, 3), (4, 5))))

    kind, tree = pick(rng, (3, add), (4, mul), (3, div), (1, divx), (1.5, mod), (2, gcd), (1.5, power), (3, mixed))(rng)
    return {"kind": kind, "text": mg.show(tree)}


# -------------------------------------------------- L9: long expressions
def gen_long(rng):
    def small(q):
        return dec(q, 1, 2, 1)

    def leaf(q):
        return pick(q, (3, small), (2, lambda z: nint(z, 1, 3)), (2, mg._sci_fn), (1, lambda z: B("^", small(z), num("2"))),
                    (1, lambda z: fn("sqrt", nint(z, 1, 3))), (0.5, lambda z: ("const", "pi")), (0.5, lambda z: dec(z, 2, 4, 2)))(q)  # fmt: skip

    def tree(q, nops):
        if nops == 0:
            return leaf(q)
        if nops >= 2 and q.random() < 0.18:  # a function of a whole sub-expression counts as one operation
            return fn(q.choice(["sqrt", "sqrt", "abs", "ln"]), tree(q, nops - 1))
        k = q.randint(0, nops - 1)
        return B(q.choices("+-*/", weights=[3, 3, 3, 1.5])[0], tree(q, k), tree(q, nops - 1 - k))

    n = rng.randint(*K((4, 9), (10, 14)))
    return {"kind": f"{n} operations", "text": mg.show(tree(rng, n))}


def truth_expr(p):
    return mg.fmt(mg.ev(mg.parse(p["text"])))


def trace_expr(p, tr):
    return fmt4(tr.val(mg.parse(p["text"])))


# ------------------------------------------------- L10: linear equations
def _cf(r, hard):
    if hard and r.random() < 0.35:
        v = D(r.randint(1, 199)) / 10
    elif hard:
        v = D(max(1, rint(r, 1, 3)))
    else:
        v = D(r.randint(1, 12))
    return v if r.random() < 0.6 else -v


def gen_linear(rng):
    hard = rng.random() < 0.4

    def term():
        k = rng.choices("xcp", weights=[4, 3.5, 2.5])[0]
        if k != "p":
            return (k, _cf(rng, hard))
        m = D(rng.randint(2, 12)) * rng.choice([1, 1, -1])
        a = D(1) if rng.random() < 0.6 else D(rng.randint(2, 9)) * rng.choice([1, 1, -1])
        return ("p", m, a, _cf(rng, hard))

    L, R = [term() for _ in range(rng.randint(*K((1, 4), (5, 7))))], [term() for _ in range(rng.randint(*K((1, 3), (4, 6))))]
    if all(t[0] == "c" for t in L + R):
        raise Reject
    n = len(L) + len(R) + sum(t[0] == "p" for t in L + R)
    return {"kind": f"{n} terms", "L": L, "R": R, "text": "solve(" + terms_text(L) + "=" + terms_text(R) + ")"}


def _reduce(side):
    p = q = F(0)
    for t in side:
        if t[0] == "x":
            p += F(t[1])
        elif t[0] == "c":
            q += F(t[1])
        else:
            p += F(t[1]) * F(t[2])
            q += F(t[1]) * F(t[3])
    return p, q


def truth_linear(P):
    (pl, ql), (pr, qr) = _reduce(P["L"]), _reduce(P["R"])
    if pl == pr:
        raise Reject
    return mg.fmt((qr - ql) / (pl - pr))


def trace_linear(P, tr):
    sides, expanded = [], False
    for side in (P["L"], P["R"]):  # 1. multiply out the brackets
        out = []
        for t in side:
            if t[0] == "p":
                expanded = True
                _, m, a, b = t
                out.append(("x", m if a == 1 else tr.mul(V(m), V(a)).d))
                out.append(("c", tr.mul(V(m), V(b)).d))
            else:
                out.append(t)
        sides.append(out)
    if expanded:
        tr.steps.append(terms_text(sides[0]) + "=" + terms_text(sides[1]))
    red, collected = [], False
    for side in sides:  # 2. collect the x terms and the constants on each side
        acc = {}
        for kind in "xc":
            vals = [t[1] for t in side if t[0] == kind]
            a = V(vals[0]) if vals else V(D(0))
            for c in vals[1:]:
                a, collected = tr.add(a, V(c), "+"), True
            acc[kind] = a.d
        red.append(acc)
    L, R = red
    if collected:
        tr.steps.append(lin_text([(L["x"], "x")], L["c"]) + "=" + lin_text([(R["x"], "x")], R["c"]))
    p, q, moved = L["x"], R["c"], False
    if R["x"] != 0:  # 3. x to the left, constants to the right
        p, moved = tr.add(V(L["x"]), V(R["x"]), "-").d, True
    if L["c"] != 0:
        q, moved = tr.add(V(R["c"]), V(L["c"]), "-").d, True
    if p == 0:
        raise Reject
    if moved:
        tr.steps.append(lin_text([(p, "x")]) + "=" + s(q))
    v = V(q) if p == 1 else tr.div(V(q), V(p))  # 4. divide
    if not tr.steps:
        raise Reject
    return fmt4(v)


# -------------------------------------------------------- L11: quadratics
def _dec(f):
    return D(f.numerator) / D(f.denominator)


def gen_quad(rng):
    if STRETCH:  # coefficients several times larger than any in training
        co = [F(rng.randint(10, 30)), F(rng.randint(31, 300) * rng.choice([1, -1])), F(rng.randint(41, 400) * rng.choice([1, -1]))]
    elif rng.random() < 0.45:  # built from two rational roots
        d1, d2 = rng.choice([1, 1, 1, 2, 4, 5]), rng.choice([1, 1, 1, 2])
        r1, r2 = F(rng.randint(-12 * d1, 12 * d1), d1), F(rng.randint(-12 * d2, 12 * d2), d2)
        a = F(r1.denominator * r2.denominator * rng.choice([1, 1, 2, 3]))
        co = [a, -a * (r1 + r2), a * r1 * r2]
    else:
        co = [F(rng.randint(1, 9)), F(rng.randint(-30, 30)), F(rng.randint(-40, 40))]
        if rng.random() < 0.2:
            co[rng.randint(1, 2)] += F(rng.randint(1, 9), 10)
    disc = co[1] ** 2 - 4 * co[0] * co[2]
    if disc < 0 and rng.random() < 0.7:
        raise Reject
    kind = "no real root" if disc < 0 else "double root" if disc == 0 else "rational roots" if _fsqrt(disc) is not None else "irrational roots"
    L, R = [_dec(c) for c in co], [D(0)] * 3
    if rng.random() < 0.45:  # not in standard form: terms on both sides
        for i in range(3):
            u = rng.random()
            if i and u < 0.35:
                R[i], L[i] = -L[i], D(0)
            elif u < 0.55:
                e = D(rng.randint(1, 9)) * rng.choice([1, -1])
                R[i], L[i] = e, L[i] + e
    # held out by root set: every arrangement and multiple of a test equation is a test problem
    return {"kind": kind, "L": L, "R": R, "text": "solve(" + poly_text(L) + "=" + poly_text(R) + ")", "key": f"quad {co[1] / co[0]} {co[2] / co[0]}"}


def _fsqrt(f):
    rn, rd = math.isqrt(f.numerator), math.isqrt(f.denominator)
    return F(rn, rd) if f >= 0 and rn * rn == f.numerator and rd * rd == f.denominator else None


def truth_quad(P):
    a, b, c = (F(l) - F(r) for l, r in zip(P["L"], P["R"]))
    disc = b * b - 4 * a * c
    if disc < 0:
        return "none"
    if disc == 0:
        return mg.fmt(-b / (2 * a))
    r = _fsqrt(disc)
    if r is not None:
        x1, x2 = (-b - r) / (2 * a), (-b + r) / (2 * a)
    else:
        q = -(float(b) + math.copysign(math.sqrt(disc), float(b))) / 2
        x1, x2 = sorted([q / float(a), float(c) / q])
    return mg.fmt(x1) + "," + mg.fmt(x2)


def trace_quad(P, tr):
    co, moved = [], False
    for l, r in zip(P["L"], P["R"]):  # 1. everything to the left
        if r != 0:
            co.append(tr.add(V(l), V(r), "-").d)
            moved = True
        else:
            co.append(l)
    a, b, c = co
    if moved:
        tr.steps.append(poly_text(co) + "=0")
    b2 = tr.power(V(b), V(D(2)), False) if b != 0 else V(D(0))  # 2. discriminant
    disc = tr.add(b2, tr.mul(tr.mul(V(D(4)), V(a)), V(c)), "-") if c != 0 else b2
    tr.steps.append("d=" + s(disc.d))
    if disc.d < 0:
        return "none"
    den = tr.mul(V(D(2)), V(a))
    if disc.d == 0:
        return fmt4(tr.div(V(-b), den))
    sq = tr.sqrt(disc)  # 3. the root where -b and the square root add up, then the other one from x1*x2 = c/a (no cancellation)
    far = tr.div(tr.add(V(-b), sq, "-" if b >= 0 else "+"), den)
    near = tr.div(V(c), tr.mul(V(a), far))
    lo, hi = (far, near) if b >= 0 else (near, far)
    return fmt4(lo) + "," + fmt4(hi)


# ---------------------------------------------------- L12: linear systems
def gen_system(rng):
    n = 3 if STRETCH or rng.random() >= 0.5 else 2
    hi = 30 if STRETCH or (n == 2 and rng.random() < 0.35) else 9  # stretch: 3x3 with two-digit coefficients
    A = [[rng.randint(-hi, hi) for _ in range(n)] for _ in range(n)]
    if any(all(v == 0 for v in row) for row in A):
        raise Reject
    if rng.random() < 0.6:
        sol = [rng.randint(-9, 9) for _ in range(n)]
        b = [sum(a * x for a, x in zip(row, sol)) for row in A]
    else:
        b = [rng.randint(-30, 30) for _ in range(n)]
    eqs = [lin_text([(D(a), v) for a, v in zip(row, VARS)]) + "=" + str(bi) for row, bi in zip(A, b)]
    return {"kind": f"{n}x{n}" + (" two-digit" if hi == 30 else ""), "A": A, "b": b, "text": "solve(" + ",".join(eqs) + ")"}


def _det(M):
    if len(M) == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    return sum((-1) ** j * M[0][j] * _det([row[:j] + row[j + 1 :] for row in M[1:]]) for j in range(len(M)))


def truth_system(P):
    A, b = P["A"], P["b"]
    d = _det(A)
    if d == 0:
        raise Reject
    return ",".join(mg.fmt(F(_det([row[:j] + [bi] + row[j + 1 :] for row, bi in zip(A, b)]), d)) for j in range(len(A)))


def _tdet(tr, M):
    """A determinant written out: a*d-b*c, or cofactors of the first row (each minor, then its product)."""
    if len(M) == 2:
        return tr.add(tr.mul(M[0][0], M[1][1]), tr.mul(M[0][1], M[1][0]), "-")
    t = [tr.mul(M[0][j], _tdet(tr, [row[:j] + row[j + 1 :] for row in M[1:]])) for j in range(3)]
    return tr.add(tr.add(t[0], t[1], "-"), t[2], "+")


def trace_system(P, tr):
    A = [[V(D(a)) for a in row] for row in P["A"]]
    b = [V(D(x)) for x in P["b"]]
    d = _tdet(tr, A)
    tr.steps.append("d=" + s(d.d))
    if d.d == 0:
        raise Reject
    dets = []
    for j in range(len(A)):
        dj = _tdet(tr, [row[:j] + [bi] + row[j + 1 :] for row, bi in zip(A, b)])
        tr.steps.append(f"d{VARS[j]}={s(dj.d)}")
        dets.append(dj)
    return ",".join(fmt4(tr.div(dj, d)) for dj in dets)


# ---------------------------------------------------------------- L13: sums
def _sum_term(r):
    """-> (term in k, smallest lower limit, largest upper limit)"""
    a, b, c, sg = r.randint(2, 9), r.randint(1, 9), r.randint(1, 20), r.choice("+-")
    return pick(
        r, (2, ("k^2", 0, 20)), (1, ("k^3", 0, 12)), (3, (f"{a}*k{sg}{b}", 0, 20)), (3, (f"k^2{sg}{a}*k", 0, 20)),
        (3, (f"{a}*k^2{sg}{b}*k+{c}", 0, 15)), (1.5, (f"k^3{sg}{a}*k", 0, 12)), (2, (f"k*(k+{b})", 0, 20)), (2, (f"(k+{b})^2", 0, 15)),
        (2, (f"(k+{b})*(k+{c})", 0, 15)), (2, (f"{b}/k", 1, 15)), (1.5, (f"{b}/k^2", 1, 12)), (1.5, (f"{b}/(k+{c})", 0, 15)),
        (1.5, (f"1/(k*(k+{b}))", 1, 12)), (1.5, (f"k/(k+{b})", 0, 12)), (1, (f"(k+{b})/(k+{c})", 0, 12)), (2, (f"{r.randint(2, 5)}^k", 0, 7)),
        (1.5, (f"k*{r.randint(2, 4)}^k", 0, 7)), (1, (f"{r.randint(2, 4)}^k{sg}{a}*k", 0, 7)), (1.5, ("k!", 0, 10)),
        (1.5, ("sqrt(k)", 0, 12)), (1, (f"sqrt(k+{b})", 0, 12)), (1, (f"{a}*sqrt(k)", 0, 12)),
    )  # fmt: skip


def gen_sum(rng):
    term, lo_min, hi_max = _sum_term(rng)
    lo = lo_min + rng.choice([0, 1, 1, 1, 2, 3, 4, 5, 6])
    if STRETCH:  # 13 to 24 terms (training: 3 to 12); the power and factorial terms cannot go that far and are left out
        if hi_max < 12:
            raise Reject
        hi = lo + rng.randint(13, 24) - 1
    else:
        hi = min(lo + rng.randint(3, 12) - 1, hi_max)
    if hi - lo < 2:
        raise Reject
    return {"kind": f"{hi - lo + 1} terms", "term": term, "lo": lo, "hi": hi, "text": f"sum({term},k,{lo},{hi})"}


def truth_sum(P):
    tot = F(0)
    for k in range(P["lo"], P["hi"] + 1):
        v = mg.ev(mg.parse(P["term"].replace("k", str(k))))
        tot = tot + v if isinstance(v, F) and isinstance(tot, F) else float(tot) + float(v)
    return mg.fmt(tot)


def trace_sum(P, tr):
    acc = None
    for k in range(P["lo"], P["hi"] + 1):
        tr.steps.append(f"k={k}")
        v = tr.val(mg.parse(P["term"].replace("k", str(k))))
        acc = v if acc is None else tr.add(acc, v, "+")
    return fmt4(acc)


# --------------------------------------------------- L14: Newton's method
def gen_newton(rng):
    deg = 3  # cubics only: the root of a small quadratic is a square-root table plus a little arithmetic, no iteration needed
    co = [D(1) if rng.random() < 0.5 else D(rng.randint(1, 3))] + [D(rng.randint(-15, 15)) for _ in range(deg)]
    x0 = D(rng.randint(-5, 5)) if rng.random() < 0.6 else D(rng.randint(-50, 50)) / 10
    # held out by polynomial: a test polynomial is never seen in training, from any starting point
    return {"kind": "cubic", "co": co, "x0": x0, "text": f"solve({poly_text(co)}=0,{s(x0)})",
            "key": "newton " + " ".join(str(F(c) / F(co[0])) for c in co[1:])}


def _deriv(co):
    n = len(co) - 1
    return [c * (n - i) for i, c in enumerate(co[:-1])]


def _horner(co, x):
    v = 0.0
    for c in co:
        v = v * x + c
    return v


def truth_newton(P):
    """The root that exact Newton iteration from the given start converges to."""
    co = [float(c) for c in P["co"]]
    dco = [float(c) for c in _deriv(P["co"])]
    x = float(P["x0"])
    for _ in range(60):
        d = _horner(dco, x)
        if d == 0:
            raise Reject
        xn = x - _horner(co, x) / d
        if abs(xn) > 1e6:
            raise Reject
        if abs(xn - x) <= 1e-13 * max(1.0, abs(xn)):
            return mg.fmt(xn)
        x = xn
    raise Reject


def _round_sig(d, n):
    return d if d == 0 else d.quantize(D(1).scaleb(d.adjusted() - (n - 1)), rounding=ROUND_HALF_UP)


def _horner_steps(tr, co, x):
    """A polynomial at x by Horner's rule: multiply the running value by x, add the next coefficient."""
    acc = V(co[0])
    for c in co[1:]:
        acc = x if acc.x and acc.d == 1 else tr.mul(acc, x)
        if c != 0:
            acc = tr.add(acc, V(c), "+")
    return acc


def trace_newton(P, tr):
    co, dco = P["co"], _deriv(P["co"])
    tr.steps.append("d=" + poly_text(dco))
    x, exact = P["x0"], True
    for i in range(1, 9):
        tr.steps.append("x=" + s(x))
        xv = V(x, exact)
        fx = _horner_steps(tr, co, xv)
        if fx.d == 0:
            return fmt4(V(x, False))
        dfx = _horner_steps(tr, dco, xv)
        if dfx.d == 0:
            raise Reject
        xn = tr.add(xv, tr.div(fx, dfx), "-")
        if _round_sig(xn.d, 4) == x:  # x no longer changes at 4 digits
            return fmt4(xn)
        # the next iterate carries one more digit each time (2, 3, then 4): Newton corrects the rest anyway
        x, exact = _round_sig(xn.d, min(4, i + 1)), False
    raise Reject


# --------------------------------------------------------------------- API
GENS = [gen_big, gen_long, gen_linear, gen_quad, gen_system, gen_sum, gen_newton]
TRUTH = [truth_expr, truth_expr, truth_linear, truth_quad, truth_system, truth_sum, truth_newton]
TRACE = [trace_expr, trace_expr, trace_linear, trace_quad, trace_system, trace_sum, trace_newton]


def sample(level, rng):
    """A well-formed problem of one of the new tiers (level 7..13)."""
    while True:
        try:
            with big():
                p = GENS[level - OLD](rng)
                p["truth"] = TRUTH[level - OLD](p)
        except (Reject, ZeroDivisionError, OverflowError, ValueError):
            continue
        if len(p["text"]) + 2 > K(P_MAX, S_P_MAX) or len(p["truth"]) + 1 > A_MAX:
            continue
        p["level"] = level
        return p


def solve(p, rng=None):
    """-> (worked solution without brackets, displayed answer derived from it). rng jitters Newton square-root estimates."""
    tr = Tracer(rng)
    with big():
        ans = TRACE[p["level"] - OLD](p, tr)
    return ";".join(tr.steps), ans


def held_out(p):
    """Test-set membership: by text, or by the problem's key where the same answer could be reached from many texts."""
    return mg.is_test(p.get("key", p["text"]))


TABLE = bytes(STOI.get(chr(i), 0) for i in range(256))


def encode(text):
    return list(text.encode("ascii").translate(TABLE))


def decode(ids):
    return "".join(ITOS.get(int(i), "") for i in ids)


if __name__ == "__main__":
    import sys

    rng = random.Random(1)
    for lv in range(OLD, NLEVELS):
        print(f"--- {LEVELS[lv]}")
        for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 4):
            p = sample(lv, rng)
            try:
                sc, ans = solve(p)
            except Reject:
                print(f"  {p['text']} = {p['truth']}   REJECT")
                continue
            print(f"  {p['text']} = {p['truth']}   {'' if ans == p['truth'] else '(chain says ' + ans + ')'}  [{p['kind']}, {len(sc)} chars]\n      [{sc[:700]}{'...' if len(sc) > 700 else ''}]")
