"""Scratchpad v2: a worked solution for every problem, with guard digits and real procedures.

The v1 scratchpad listed intermediate values to 4 significant digits and had no procedure for division,
roots, powers or gcd. A perfect model following it tops out at 94.9% on the benchmark. Here every step is

    <operation on printed values>[:<work>]=<value>        steps are joined by ';'

and values carry 6 significant digits (integers are never rounded). Work, by operation:

  a*b   partial products, most significant multiplier digit first, with a running Horner sum:
        p1,p2,s2,p3,s3,...   where s_i = 10*s_(i-1) + p_i    (omitted for one-digit multipliers)
  a/b   long division on the significands: one 'DrREM' (or 'DpPRODUCTrREM') per quotient digit, to 7 significant digits
  a%b   the same long division on integers, kept only for its remainder (mod, gcd, lcm)
  sqrt  one Newton step from a 4-digit estimate: sqrt(x)~s ; x/s:...=q ; sqrt(x)=(s+q)/2   (cube roots likewise)
  a^n   repeated multiplication for integer |n| <= 7 (then 1/p for negative n)
  ln    ln(mantissa) and exponent*ln(10), mantissa in [0.316, 3.16);   log: log10(mantissa)
  gcd   Euclid, every remainder by long division;  lcm = a*(b/gcd)
  10^x  10^(fractional part) from a 100-entry table, shifted;  2^x likewise, then times 2^(integer part)
  sin, cos off the 0.01 grid: grid value plus (x - x0) * slope;  ln off the 4-digit mantissa grid: ln(m0) + (m - m0)/m0
  e^x   e^(integer part) * e^(first decimal), two small tables;  n! = (n-1)! * n;  nPr, nCr by running products
Everything else (trig on the grid, inverse trig ...) is a single looked-up value.

The chain is self-consistent: every value is derived from the printed operands, so the final answer is the
last value rounded to 4 significant digits (agreement with the true answer is measured by check()).
"""
import math
import re
from decimal import Decimal as D, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP, getcontext

import mathgen as mg
from mathgen import Reject

getcontext().prec = 80
SIG = 6
PI, E = D("3.14159"), D("2.71828")
LN10 = 2.302585092994046
EXTRA = ":;~%{<"  # characters the v2 scratchpad adds to the vocabulary ("{" and "<" open a thinking block at lower effort)
CHARS = mg.CHARS + EXTRA
STOI = {c: i + 3 for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS) + 3


class V:
    """A value in the chain: the printed decimal and whether it is still exact (no rounding, no looked-up function)."""

    __slots__ = ("d", "x")

    def __init__(self, d, x=True):
        self.d, self.x = d, x


TIE_SAFE = True
# Each change made on the way to the final format is a switch, so the path can be replayed one step at a time.
OPT = {
    "work": True,          # write procedures (partial products, long division, Euclid, Newton, chains); False: one 'op=value' per node
    "split_pow": True,     # 10^x and 2^x from a 100-entry table for the fractional part
    "offgrid": True,       # sin/cos off the 0.01 grid and ln off the 4-digit mantissa grid by first-order correction
    "jitter": True,        # training only: perturb Newton starting estimates
    "split_exp": True,     # e^x = e^(whole part) * e^(first decimal)
    "chains": True,        # n! = (n-1)! * n, nPr / nCr as running products, cube roots by Newton
    "div_products": True,  # write the product in long-division steps for divisors of 3+ digits
}
RUNGS = {  # cumulative versions of the scratchpad, in the order they were introduced
    "values6": dict(work=False, split_pow=False, offgrid=False, jitter=False, split_exp=False, chains=False, div_products=False),
    "v2.0": dict(work=True, split_pow=False, offgrid=False, jitter=False, split_exp=False, chains=False, div_products=False),
    "v2.1": dict(work=True, split_pow=True, offgrid=True, jitter=False, split_exp=False, chains=False, div_products=False),
    "v2.2": dict(work=True, split_pow=True, offgrid=True, jitter=True, split_exp=True, chains=True, div_products=False),
    "v2.3": dict(work=True, split_pow=True, offgrid=True, jitter=True, split_exp=True, chains=True, div_products=True),
}


def configure(rung):
    OPT.update(RUNGS[rung])


def r6(x):
    """Integers are never rounded; anything else is rounded half-up to 6 significant digits.

    Tie-safe: an inexact result never ends in 0 or 5 (it takes the neighbour on the true value's side instead),
    so rounding the 6-digit value again to 4 digits can never be fooled by a false ...50 or ...00."""
    if x == x.to_integral_value():
        if abs(x) >= D(10) ** 12:
            raise Reject
        return x.to_integral_value()
    ulp = D(1).scaleb(x.adjusted() - (SIG - 1))
    y = x.quantize(ulp, rounding=ROUND_HALF_UP)
    if TIE_SAFE and y != x:
        down = x.quantize(ulp, rounding=ROUND_DOWN)  # toward zero
        up = down + ulp.copy_sign(x)
        if int(y.scaleb(-ulp.adjusted()).to_integral_value()) % 5 == 0:
            y = down if y == up else up
    if abs(y) >= D(10) ** 12:
        raise Reject
    return y if abs(y) >= D("1e-9") else D(0)


def rv(x, exact):
    """Round a freshly computed result; it stays exact only if its operands were and nothing was rounded away."""
    y = r6(x)
    return V(y, exact and y == x)


def s(x):
    t = format(x, "f")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return "0" if t in ("-0", "") else t


def rs(x):
    """Right-hand operand: negatives are parenthesised."""
    return f"({s(x)})" if x < 0 else s(x)


def fl(v):
    """Float result of a looked-up function -> 6 significant digits, never exact."""
    if v != v or v in (math.inf, -math.inf):
        raise Reject
    return V(r6(D(repr(v))) if abs(v) >= 1e-12 else D(0), False)


def sig_int(x):
    """|x| = n * 10**e with n free of trailing zeros."""
    _, digits, exp = abs(x).normalize().as_tuple()
    return int("".join(map(str, digits))), exp


def fmt4(v):
    """The calculator display: exact integers in full, everything else to 4 significant digits."""
    d = v.d
    if d == 0:
        return "0"
    if v.x and d == d.to_integral_value():
        if abs(d) >= D(10) ** 10:
            raise Reject
        return s(d)
    e = d.adjusted()
    if e >= 7:
        raise Reject
    if e < -4:  # too small to display: the true value is 0 up to the chain's rounding (e.g. sin of a multiple of pi)
        return "0"
    return s(d.quantize(D(1).scaleb(e - 3), rounding=ROUND_HALF_UP))


def mul_work(x, y):
    nx, _ = sig_int(x)
    ny, _ = sig_int(y)
    if nx == 0 or ny == 0:
        return ""
    a, b = (ny, nx) if len(str(nx)) < len(str(ny)) else (nx, ny)  # b: the multiplier, fewer digits (ties: 2nd operand)
    digits = [int(c) for c in str(b)]
    if len(digits) == 1:
        return ""
    items, run = [], 0
    for i, d in enumerate(digits):
        p = a * d
        items.append(str(p))
        run = run * 10 + p
        if i:
            items.append(str(run))
    return ",".join(items)


def long_div(X, Y, frac):
    """Digit-by-digit division of positive integers. -> (steps, quotient digits as int, fraction digits, remainder)

    A step is 'DrREM', or 'DpPRODrREM' when the divisor has 3+ digits and D >= 2: writing the product D*Y down turns the
    remainder into a plain subtraction (multiply-and-subtract in one go was the commonest way a division went wrong)."""
    steps, q, r, started, nsig, nfrac = [], 0, 0, False, 0, 0
    show = OPT["div_products"] and len(str(Y)) >= 3

    def step(d, r):
        steps.append(f"{d}p{d * Y}r{r}" if show and d >= 2 else f"{d}r{r}")

    for c in str(X):
        r = 10 * r + int(c)
        d = r // Y
        r -= d * Y
        q = 10 * q + d
        if d or started:
            started = True
            nsig += 1
            step(d, r)
    while frac and r and nsig < 7:
        r *= 10
        d = r // Y
        r -= d * Y
        q = 10 * q + d
        nfrac += 1
        if d or started:
            started = True
            nsig += 1
            step(d, r)
        if nfrac > 40:
            raise Reject
    return steps, q, nfrac, r


class Tracer:
    def __init__(self, rng=None):
        self.steps = []
        self.rng = rng

    def estimate(self, value):
        """4 significant digits of a root. In training the last digit is jittered by up to 5 units: a model that only ever
        sees perfect estimates learns to copy them through the Newton step instead of doing the division."""
        est = D(repr(value))
        ulp = D(1).scaleb(est.adjusted() - 3)
        est = est.quantize(ulp, rounding=ROUND_HALF_UP)
        if self.rng is not None and OPT["jitter"]:
            est = max(est + ulp * self.rng.randint(-5, 5), ulp)
        return V(est, False)

    # ------------------------------------------------------------ primitives
    def add(self, a, b, op):
        v = rv(a.d + b.d if op == "+" else a.d - b.d, a.x and b.x)
        self.steps.append(f"{s(a.d)}{op}{rs(b.d)}={s(v.d)}")
        return v

    def mul(self, a, b):
        v = rv(a.d * b.d, a.x and b.x)
        w = mul_work(a.d, b.d)
        self.steps.append(f"{s(a.d)}*{rs(b.d)}" + (f":{w}" if w else "") + f"={s(v.d)}")
        return v

    def div(self, a, b):
        if b.d == 0:
            raise Reject
        if a.d == 0:
            self.steps.append(f"0/{rs(b.d)}=0")
            return V(D(0), a.x and b.x)
        (nx, ex), (ny, ey) = sig_int(a.d), sig_int(b.d)
        steps, q, nfrac, r = long_div(nx, ny, True)
        v = rv(D(q).scaleb(ex - ey - nfrac) * (-1 if (a.d < 0) != (b.d < 0) else 1), a.x and b.x and r == 0)
        self.steps.append(f"{s(a.d)}/{rs(b.d)}:{','.join(steps)}={s(v.d)}")
        return v

    def rem(self, a, b):
        """Integer long division kept for its remainder."""
        steps, _, _, r = long_div(a, b, False)
        self.steps.append(f"{a}%{b}:{','.join(steps) or f'0r{a}'}")
        return r

    def sqrt(self, x):
        if x.d < 0:
            raise Reject
        if x.d == 0:
            self.steps.append("sqrt(0)=0")
            return V(D(0), False)
        est = self.estimate(math.sqrt(float(x.d)))
        self.steps.append(f"sqrt({s(x.d)})~{s(est.d)}")
        q = self.div(x, est)
        v = V(r6((est.d + q.d) / 2), False)
        self.steps.append(f"sqrt({s(x.d)})={s(v.d)}")
        return v

    def table(self, text, value):
        v = fl(value)
        self.steps.append(f"{text}={s(v.d)}")
        return v

    # ----------------------------------------------------------------- nodes
    def power(self, a, b, base_is_e):
        head = f"{rs(a.d)}^{rs(b.d)}"
        if b.x and b.d == b.d.to_integral_value() and abs(b.d) <= 7 and not base_is_e:
            n = int(b.d)
            if n == 0:
                v = V(D(1), a.x)
            elif n == 1:
                v = a
            else:
                if a.d == 0 and n < 0:
                    raise Reject
                m = V(abs(a.d), a.x)
                p = m
                for _ in range(abs(n) - 1):
                    p = self.mul(p, m)
                if n < 0:
                    p = self.div(V(D(1)), p)
                v = V(-p.d, p.x) if a.d < 0 and n % 2 else p
            self.steps.append(f"{head}={s(v.d)}")
            return v
        if b.d == D("0.5"):
            v = self.sqrt(a)
        elif b.d == D("1.5"):
            v = self.mul(a, self.sqrt(a))
        elif OPT["chains"] and b.d == D("0.333333") and a.d > 0:  # cube root: one Newton step, (2s + x/s^2) / 3
            est = self.estimate(float(a.d) ** (1 / 3))
            self.steps.append(f"{head}~{s(est.d)}")
            q = self.div(self.div(a, est), est)
            v = self.div(self.add(self.mul(est, V(D(2))), q, "+"), V(D(3)))
        elif OPT["split_exp"] and base_is_e and b.d == b.d.quantize(D("0.1")) and abs(b.d) < 30:
            n = int(b.d.to_integral_value(rounding=ROUND_FLOOR))
            frac = b.d - n
            if frac == 0:  # whole powers of e are a 15-entry table
                return self.table(head, math.exp(n))
            v = self.mul(self.table(f"2.71828^{rs(D(n))}", math.exp(n)), self.table(f"2.71828^{s(frac)}", math.exp(float(frac))))
        elif OPT["split_pow"] and a.x and a.d in (D(10), D(2)) and b.d == b.d.quantize(D("0.01")) and abs(b.d) < 40:
            n = int(b.d.to_integral_value(rounding=ROUND_FLOOR))
            frac = b.d - n
            t = fl(math.pow(float(a.d), float(frac)))
            if a.d == 10:  # 10^x = 10^frac shifted by the integer part
                v = V(r6(t.d.scaleb(n)), False)
                self.steps.append(f"{head}:{s(t.d)}={s(v.d)}")
                return v
            if n == 0:
                return self.table(head, math.pow(2.0, float(frac)))
            self.steps.append(f"2^{s(frac)}={s(t.d)}")
            v = self.mul(t, V(D(2) ** n))
        else:
            if a.d < 0 or (a.d == 0 and b.d <= 0):
                raise Reject
            try:
                return self.table(head, math.pow(float(a.d), float(b.d)))
            except (OverflowError, ValueError):
                raise Reject
        self.steps.append(f"{head}={s(v.d)}")
        return v

    def trig(self, name, x):
        """sin/cos at an argument off the 0.01 grid: nearest grid point plus the first-order correction."""
        x0 = x.d.quantize(D("0.01"), rounding=ROUND_HALF_UP)
        delta = x.d - x0
        f = {"sin": math.sin, "cos": math.cos, "tan": math.tan}[name]
        if not OPT["offgrid"] or delta == 0 or name == "tan" or abs(x0) > D("9.99"):
            return self.table(f"{name}({s(x.d)})", f(float(x.d)))
        s0 = lambda: self.table(f"sin({s(x0)})", math.sin(float(x0)))
        c0 = lambda: self.table(f"cos({s(x0)})", math.cos(float(x0)))
        if name == "sin":
            main, slope = s0(), c0()
            v = self.add(main, self.mul(V(delta, False), slope), "+")
        else:
            main, slope = c0(), s0()
            v = self.add(main, self.mul(V(delta, False), slope), "-")
        self.steps.append(f"{name}({s(x.d)})={s(v.d)}")
        return v

    def fn(self, name, args):
        x = args[0]
        if name == "sqrt":
            return self.sqrt(x)
        if name == "abs":
            v = V(abs(x.d), x.x)
            self.steps.append(f"abs({s(x.d)})={s(v.d)}")
            return v
        if name in ("gcd", "lcm", "mod"):
            a, b = args[0].d, args[1].d
            if a != a.to_integral_value() or b != b.to_integral_value() or a < 0 or b <= 0:
                raise Reject
            a, b = int(a), int(b)
            if name == "mod":
                r = self.rem(a, b)
                self.steps.append(f"mod({a},{b})={r}")
                return V(D(r))
            hi, lo = a, b
            while True:
                r = self.rem(hi, lo)
                if r == 0:
                    break
                hi, lo = lo, r
            self.steps.append(f"gcd({a},{b})={lo}")
            if name == "gcd":
                return V(D(lo))
            v = self.mul(V(D(a)), self.div(V(D(b)), V(D(lo))))
            self.steps.append(f"lcm({a},{b})={s(v.d)}")
            return v
        xf = float(x.d)
        try:
            if name in ("ln", "log"):
                if x.d <= 0:
                    raise Reject
                # x = m * 10**e with the mantissa centred on 1 (0.316 <= m < 3.16), so that adding e*ln(10) never cancels
                e = x.d.adjusted()
                m = x.d.scaleb(-e)
                if m >= D("3.16228"):
                    m, e = m / 10, e + 1
                f = math.log if name == "ln" else math.log10
                m0 = m.quantize(D(1).scaleb(m.adjusted() - 3), rounding=ROUND_HALF_UP)
                if OPT["offgrid"] and m0 != m:  # more than 4 significant digits: ln(m) ~ ln(m0) + (m - m0) / m0
                    base = self.table(f"ln({s(m0)})", math.log(float(m0)))
                    v = self.add(base, self.div(V(m - m0, False), V(m0, False)), "+")
                    if e:
                        v = self.add(v, fl(e * LN10), "+")
                    if name == "log":
                        v = self.div(v, V(D("2.30259"), False))
                    self.steps.append(f"{name}({s(x.d)})={s(v.d)}")
                    return v
                if e == 0:
                    return self.table(f"{name}({s(x.d)})", f(xf))
                lm = fl(f(float(m)))
                le = fl(e * LN10) if name == "ln" else V(D(e))
                v = V(r6(lm.d + le.d), False)
                self.steps.append(f"{name}({s(x.d)}):{s(lm.d)}" + (f",{s(le.d)}" if name == "ln" else "") + f"={s(v.d)}")
                return v
            if name in ("sin", "cos", "tan"):
                return self.trig(name, x)
            f = {"asin": math.asin, "acos": math.acos, "atan": math.atan}[name]
            return self.table(f"{name}({s(x.d)})", f(xf))
        except (ValueError, OverflowError):
            raise Reject

    def val(self, node):
        t = node[0]
        if t == "num":
            return V(D(node[2]))
        if t == "const":
            return V(PI if node[1] == "pi" else E, False)
        kids = [self.val(c) for c in (node[2] if t == "fn" else [c for c in node[1:] if isinstance(c, tuple)])]
        first = len(self.steps)
        v = self.apply(node, kids)
        if not OPT["work"]:  # values only: keep the node's last step and drop its work
            last = re.sub(r":[^=]*", "", self.steps[-1])
            del self.steps[first:]
            self.steps.append(last)
        return v

    def apply(self, node, kids):
        t = node[0]
        if t == "bin":
            a, b = kids
            op = node[1]
            if op in "+-":
                return self.add(a, b, op)
            if op == "*":
                return self.mul(a, b)
            if op == "/":
                return self.div(a, b)
            return self.power(a, b, node[2] == ("const", "e"))
        if t == "fn":
            return self.fn(node[1], kids)
        if t == "neg":
            return self.add(V(D(0)), kids[0], "-")
        if t == "fact":
            n = int(kids[0].d)
            if not 0 <= n <= 13:
                raise Reject
            return self.fact(n)
        if t in ("nCr", "nPr"):
            n, r = int(kids[0].d), int(kids[1].d)
            if not 0 <= r <= n:
                raise Reject
            if not OPT["chains"]:
                v = V(D(math.comb(n, r) if t == "nCr" else math.perm(n, r)))
                self.steps.append(f"{n}n{t[1:]}{r}={s(v.d)}")
                return v
            p = V(D(n if r else 1))  # n (n-1) ... (n-r+1), one factor at a time
            for k in range(1, r):
                p = self.mul(p, V(D(n - k)))
            if t == "nCr" and r >= 2:
                p = self.div(p, self.fact(r))
            self.steps.append(f"{n}n{t[1:]}{r}={s(p.d)}")
            return p
        raise KeyError(t)

    def fact(self, n):
        """n! = (n-1)! * n, with (n-1)! looked up."""
        if n < 2 or not OPT["chains"]:
            v = V(D(math.factorial(n)))
        else:
            prev = V(D(math.factorial(n - 1)))
            self.steps.append(f"{n - 1}!={s(prev.d)}")
            v = self.mul(prev, V(D(n)))
        self.steps.append(f"{n}!={s(v.d)}")
        return v


def solve(expr, rng=None):
    """-> (scratchpad text without brackets, displayed answer derived from it). rng jitters Newton starting points."""
    tr = Tracer(rng)
    v = tr.val(mg.parse(expr))
    return ";".join(tr.steps), fmt4(v)


def encode(text):
    return [STOI[c] for c in text]


def decode(ids):
    return "".join(ITOS.get(int(i), "") for i in ids)


if __name__ == "__main__":
    import random
    import sys

    rng = random.Random(1)
    for lv in range(mg.NLEVELS):
        print(f"--- {mg.LEVELS[lv]}")
        for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 5):
            e, a, _ = mg.sample(lv, rng)
            try:
                sc, ans = solve(e)
            except Reject:
                print(f"  {e} = {a}   REJECT")
                continue
            print(f"  {e} = {a}   {'' if ans == a else '(chain says ' + ans + ')'}\n      [{sc}]")
