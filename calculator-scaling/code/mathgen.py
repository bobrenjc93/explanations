"""Synthetic calculator curriculum: arithmetic -> TI-84-style scientific expressions.

Every problem is an expression tree. One evaluator defines the semantics:
rational arithmetic is exact (fractions.Fraction), anything transcendental is a
float, and the display rule is "integers in full, everything else rounded to 4
significant digits". A separate recursive-descent parser (parse) re-reads the
printed string so the printer's precedence/parenthesisation can be cross-checked.
"""
import math
import random
import zlib
from fractions import Fraction as F

LEVELS = [
    "L1 add/sub",
    "L2 long add/sub",
    "L3 mul/div",
    "L4 PEMDAS",
    "L5 dec/pow/root",
    "L6 scientific",
    "L7 composite",
]
NLEVELS = len(LEVELS)

CHARS = "0123456789+-*/^().,!=[]" + "abcdegilmnopqrstCP"
PAD, BOS, EOS = 0, 1, 2
STOI = {c: i + 3 for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS) + 3

# Size limits of the calculator (mathgen3 raises them for its big-number tier; the seven levels here never reach them).
INT_SHOW = 10**10   # integers are displayed in full below this
FRAC_LIMIT = 10**13  # exact rationals are abandoned beyond this
POW_LIMIT = 10**12

SIN_RANGE = 0  # > 0: half of the direct sin/cos arguments are drawn from 10 < |x| <= SIN_RANGE (the wide pool; off by default)

P_MAX = 30  # BOS + expression + '=' (left padded)
A_MAX = 12  # answer + EOS (right padded)
C_MAX = 48  # '[' scratch ']' answer EOS (right padded)


class Reject(Exception):
    pass


# ---------------------------------------------------------------- display rule
def _place(digs, e):
    """digs = 4 significant digits d.ddd, e = decimal exponent."""
    if e >= 3:
        return digs + "0" * (e - 3)
    if e >= 0:
        return (digs[: e + 1] + "." + digs[e + 1 :]).rstrip("0").rstrip(".")
    return ("0." + "0" * (-e - 1) + digs).rstrip("0")


def fmt(v):
    if isinstance(v, F):
        if v.denominator == 1:
            if abs(v.numerator) >= INT_SHOW:
                raise Reject
            return str(v.numerator)
        a = abs(v)
        e = len(str(a.numerator)) - len(str(a.denominator))
        while a >= F(10) ** (e + 1):
            e += 1
        while a < F(10) ** e:
            e -= 1
        n = int(a / F(10) ** (e - 3) + F(1, 2))  # round half away from zero
        if n >= 10000:
            n //= 10
            e += 1
        digs = str(n)
    else:
        if v != v or v in (math.inf, -math.inf):
            raise Reject
        a = abs(v)
        if a < 1e-9:
            return "0"
        m, ex = f"{a:.3e}".split("e")
        e = int(ex)
        digs = m.replace(".", "")
    if e >= 7 or e < -4:
        raise Reject
    return ("-" if v < 0 else "") + _place(digs, e)


# ------------------------------------------------------------------- evaluator
def _int(v):
    if not isinstance(v, F) or v.denominator != 1:
        raise Reject
    return v.numerator


def _pow(a, b):
    if isinstance(a, F) and isinstance(b, F) and b.denominator == 1 and abs(b) <= 64:
        if a == 0 and b < 0:
            raise Reject
        r = a ** int(b)
        if abs(r.numerator) > POW_LIMIT or r.denominator > POW_LIMIT:
            raise Reject
        return r
    af, bf = float(a), float(b)
    if af < 0 or (af == 0 and bf <= 0):
        raise Reject
    try:
        return math.pow(af, bf)
    except (OverflowError, ValueError):
        raise Reject


def _fn(name, args):
    if name == "abs":
        return abs(args[0])
    if name in ("gcd", "lcm", "mod"):
        a, b = _int(args[0]), _int(args[1])
        if a < 0 or b <= 0:
            raise Reject
        if name == "gcd":
            return F(math.gcd(a, b))
        if name == "lcm":
            return F(a * b // math.gcd(a, b))
        return F(a % b)
    x = float(args[0])
    try:
        if name == "sqrt":
            return math.sqrt(x)
        if name == "sin":
            return math.sin(x)
        if name == "cos":
            return math.cos(x)
        if name == "tan":
            return math.tan(x)
        if name == "asin":
            return math.asin(x)
        if name == "acos":
            return math.acos(x)
        if name == "atan":
            return math.atan(x)
        if name == "ln":
            return math.log(x)
        if name == "log":
            return math.log10(x)
    except (ValueError, OverflowError):
        raise Reject
    raise KeyError(name)


def ev(node):
    t = node[0]
    if t == "num":
        return node[1]
    if t == "const":
        return math.pi if node[1] == "pi" else math.e
    if t == "bin":
        op, a, b = node[1], ev(node[2]), ev(node[3])
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        elif op == "*":
            r = a * b
        elif op == "/":
            if b == 0:
                raise Reject
            r = a / b
        else:
            r = _pow(a, b)
        if isinstance(r, F) and (abs(r.numerator) > FRAC_LIMIT or r.denominator > FRAC_LIMIT):
            raise Reject
        return r
    if t == "fn":
        return _fn(node[1], [ev(a) for a in node[2]])
    if t == "neg":
        return -ev(node[1])
    if t == "fact":
        n = _int(ev(node[1]))
        if not 0 <= n <= 13:
            raise Reject
        return F(math.factorial(n))
    if t in ("nCr", "nPr"):
        n, r = _int(ev(node[1])), _int(ev(node[2]))
        if not 0 <= r <= n:
            raise Reject
        return F(math.comb(n, r) if t == "nCr" else math.perm(n, r))
    raise KeyError(t)


# --------------------------------------------------------------------- printer
PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 4}


def show(node, ctx=0, right=False, lead=True):
    """ctx: parent precedence, right: right operand, lead: first thing in its group."""
    t = node[0]
    if t == "num":
        s = node[2]
        return s if s[0] != "-" or (lead and ctx < 4) else "(" + s + ")"
    if t == "const":
        return node[1]
    if t == "fn":
        return node[1] + "(" + ",".join(show(a) for a in node[2]) + ")"
    if t == "fact":
        return show(node[1], 5) + "!"
    if t in ("nCr", "nPr"):
        return show(node[1], 5) + "n" + t[1:] + show(node[2], 5, True, False)
    if t == "neg":
        if lead and ctx < 4:
            return "-" + show(node[1], 3, False, False)
        return "(-" + show(node[1], 3, False, False) + ")"
    op = node[1]
    p = PREC[op]
    wrap = p < ctx or (p == ctx and right) or (op == "^" and ctx >= 4)
    inner_lead = lead or wrap
    if op == "^":
        s = show(node[2], 5, False, inner_lead) + "^" + show(node[3], 5, True, False)
    else:
        s = show(node[2], p, False, inner_lead) + op + show(node[3], p, True, False)
    return "(" + s + ")" if wrap else s


# ---------------------------------------------------------------------- parser
NAMES = ["asin", "acos", "atan", "sqrt", "sin", "cos", "tan", "log", "abs", "gcd", "lcm", "mod", "ln", "pi", "e"]


def parse(s):
    pos = 0

    def peek(k=1):
        return s[pos : pos + k]

    def expr():
        nonlocal pos
        n = term()
        while peek() in ("+", "-") and peek() != "":
            op = s[pos]
            pos += 1
            n = ("bin", op, n, term())
        return n

    def term():
        nonlocal pos
        n = unary()
        while peek() in ("*", "/") and peek() != "":
            op = s[pos]
            pos += 1
            n = ("bin", op, n, unary())
        return n

    def unary():
        nonlocal pos
        if peek() == "-":
            pos += 1
            c = unary()
            if c[0] == "num" and c[2][0] != "-":
                return num("-" + c[2])
            return ("neg", c)
        return power()

    def power():
        nonlocal pos
        b = postfix()
        if peek() == "^":
            pos += 1
            return ("bin", "^", b, postfix())
        return b

    def postfix():
        nonlocal pos
        a = atom()
        while True:
            if peek() == "!":
                pos += 1
                a = ("fact", a)
            elif peek(3) in ("nCr", "nPr"):
                t = peek(3)
                pos += 3
                a = (t, a, atom())
            else:
                return a

    def atom():
        nonlocal pos
        if peek() == "(":
            pos += 1
            n = expr()
            assert s[pos] == ")", (s, pos)
            pos += 1
            return n
        if peek().isdigit():
            j = pos
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            n = num(s[pos:j])
            pos = j
            return n
        for name in NAMES:
            if s.startswith(name, pos) and not s.startswith("nCr", pos) and not s.startswith("nPr", pos):
                pos += len(name)
                if name in ("pi", "e"):
                    return ("const", name)
                assert s[pos] == "(", (s, pos)
                pos += 1
                args = [expr()]
                while s[pos] == ",":
                    pos += 1
                    args.append(expr())
                assert s[pos] == ")", (s, pos)
                pos += 1
                return ("fn", name, args)
        raise ValueError(f"cannot parse {s!r} at {pos}")

    n = expr()
    if pos != len(s):
        raise ValueError(f"trailing input in {s!r} at {pos}")
    return n


def calc(s):
    """The reference calculator: string in, display string out."""
    return fmt(ev(parse(s)))


# ------------------------------------------------------------ tree construction
def num(s):
    return ("num", F(s), s)


def B(op, a, b):
    return ("bin", op, a, b)


def fn(name, *args):
    return ("fn", name, list(args))


def rint(rng, lo, hi):
    """Integer with a uniformly chosen number of digits in [lo, hi]."""
    d = rng.randint(lo, hi)
    return rng.randint(0 if d == 1 else 10 ** (d - 1), 10**d - 1)


def nint(rng, lo, hi):
    return num(str(rint(rng, lo, hi)))


def dec(rng, sig_lo=1, sig_hi=3, max_dec=2, neg=0.0):
    """Decimal literal with sig_lo..sig_hi significant digits and up to max_dec decimals."""
    n = rint(rng, sig_lo, sig_hi)
    d = rng.randint(0, max_dec)
    s = str(n) if d == 0 else f"{n / 10**d:.{d}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if rng.random() < neg and F(s) != 0:
        s = "-" + s
    return num(s)


def pick(rng, *weighted):
    r = rng.random() * sum(w for w, _ in weighted)
    for w, f in weighted:
        r -= w
        if r < 0:
            return f
    return weighted[-1][1]


# ---------------------------------------------------------------------- levels
def gen_L1(rng):
    return B(rng.choice("+-"), nint(rng, 1, 3), nint(rng, 1, 3))


def gen_L2(rng):
    if rng.random() < 0.6:
        a, b = rint(rng, 3, 8), rint(rng, 3, 8)
        if max(a, b) < 1000:
            a = rint(rng, 4, 8)
        return B(rng.choice("+-"), num(str(a)), num(str(b)))
    n = nint(rng, 1, 4)
    for _ in range(rng.randint(2, 3)):
        n = B(rng.choice("+-"), n, nint(rng, 1, 4))
    return n


def gen_L3(rng):
    r = rng.random()
    if r < 0.5:
        a, b = nint(rng, 1, 4), nint(rng, 1, 3)
        return B("*", a, b) if rng.random() < 0.5 else B("*", b, a)
    if r < 0.67:
        b, q = max(1, rint(rng, 1, 3)), rint(rng, 1, 3)
        return B("/", num(str(b * q)), num(str(b)))
    if r < 0.85:
        return B("/", nint(rng, 1, 5), num(str(max(1, rint(rng, 1, 3)))))
    return fn("mod", nint(rng, 1, 5), num(str(max(1, rint(rng, 1, 3)))))


def _arith_tree(rng, nops, leaf):
    if nops == 0:
        return leaf(rng)
    k = rng.randint(0, nops - 1)
    op = rng.choices("+-*/", weights=[3, 3, 3, 1.2])[0]
    return B(op, _arith_tree(rng, k, leaf), _arith_tree(rng, nops - 1 - k, leaf))


def gen_L4(rng):
    def leaf(r):
        return nint(r, 1, 2) if r.random() < 0.9 else nint(r, 3, 3)

    return _arith_tree(rng, rng.randint(2, 4), leaf)


def gen_L5(rng):
    def decimal_op(r):
        return B(r.choice("+-*/"), dec(r, 1, 3, 2, 0.15), dec(r, 1, 3, 2, 0.15))

    def int_pow(r):
        if r.random() < 0.4:
            return B("^", nint(r, 1, 3), num("2"))
        return B("^", num(str(r.randint(2, 30))), num(str(r.randint(2, 7))))

    def neg_pow(r):
        return B("^", num(str(r.randint(2, 20))), num(str(-r.randint(1, 4))))

    def dec_pow(r):
        return B("^", dec(r, 1, 2, 1), num(str(r.randint(2, 4))))

    def root(r):
        if r.random() < 0.3:
            return fn("sqrt", num(str(r.randint(0, 316) ** 2)))
        return fn("sqrt", nint(r, 1, 4) if r.random() < 0.7 else dec(r, 2, 4, 2))

    def recip(r):
        return B("/", num("1"), dec(r, 1, 3, 2, 0.1))

    def signed(r):
        a, b = dec(r, 1, 2, 0, 1.0), dec(r, 1, 2, 0, 0.5)
        return B(r.choice("+-*"), a, b)

    def absval(r):
        return fn("abs", B(r.choice("+-"), dec(r, 1, 3, 1, 0.3), dec(r, 1, 3, 1, 0.3)))

    def frac_pow(r):
        return B("^", nint(r, 1, 3), pick(r, (2, num("0.5")), (1, B("/", num("1"), num("3"))), (1, num("1.5"))))

    def sq_neg(r):
        return B("^", num(str(-r.randint(1, 30))), num(str(r.randint(2, 3))))

    g = pick(
        rng, (4, decimal_op), (3, int_pow), (1, neg_pow), (1, dec_pow), (3, root), (1, recip), (2, signed),
        (1, absval), (1, frac_pow), (1, sq_neg),
    )  # fmt: skip
    return g(rng)


def _sci_arg(rng, name):
    """A plain numeric argument in the function's interesting range."""
    if SIN_RANGE and name in ("sin", "cos") and rng.random() < 0.5:
        return num(f"{rng.choice([1, -1]) * rng.randint(1000, SIN_RANGE * 100) / 100:.2f}".rstrip("0").rstrip("."))
    if name in ("sin", "cos", "tan"):
        return num(f"{rng.randint(-999, 999) / 100:.2f}".rstrip("0").rstrip("."))
    if name in ("asin", "acos"):
        return num(f"{rng.randint(-100, 100) / 100:.2f}".rstrip("0").rstrip("."))
    if name == "atan":
        return dec(rng, 1, 3, 2, 0.3)
    if name in ("ln", "log"):
        while True:
            n = dec(rng, 1, 4, 2)
            if n[1] > 0:
                return n
    if name == "sqrt":
        return dec(rng, 1, 4, 2)
    raise KeyError(name)


def _sci_fn(rng, names=("sin", "cos", "tan", "ln", "log", "asin", "acos", "atan")):
    name = rng.choice(names)
    return fn(name, _sci_arg(rng, name))


def gen_L6(rng):
    def trig(r):
        return _sci_fn(r, ("sin", "cos", "tan"))

    def logs(r):
        return _sci_fn(r, ("ln", "log"))

    def inv(r):
        return _sci_fn(r, ("asin", "acos", "atan"))

    def exp_e(r):
        return B("^", ("const", "e"), num(f"{r.randint(-50, 99) / 10:.1f}".rstrip("0").rstrip(".")))

    def exp_10(r):
        return B("^", num(r.choice(["10", "2"])), num(f"{r.randint(-300, 600) / 100:.2f}".rstrip("0").rstrip(".")))

    def fact(r):
        return ("fact", num(str(r.randint(0, 13))))

    def comb(r):
        n = r.randint(1, 30)
        return (r.choice(["nCr", "nPr"]), num(str(n)), num(str(r.randint(0, min(n, 8)))))

    def gcdlcm(r):
        if r.random() < 0.6:
            g = r.randint(2, 40)
            a, b = g * r.randint(1, 25), g * r.randint(1, 25)
        else:
            a, b = r.randint(1, 999), r.randint(1, 999)
        return fn(r.choice(["gcd", "lcm"]), num(str(a)), num(str(b)))

    def consts(r):
        c = ("const", r.choice(["pi", "e"]))
        x = dec(r, 1, 3, 1)
        return pick(r, (2, B("*", x, c)), (1, B("/", x, c)), (1, B("/", c, x)), (1, B("^", c, num(str(r.randint(2, 4))))),
                    (1, B("+", c, x)), (0.3, c))  # fmt: skip

    g = pick(rng, (4, trig), (3, logs), (2, inv), (2, exp_e), (2, exp_10), (1, fact), (2, comb), (2, gcdlcm), (2, consts))
    return g(rng)


def gen_L7(rng):
    def small(r):
        return dec(r, 1, 2, 1)

    def hyp(r):
        return fn("sqrt", B("+", B("^", nint(r, 1, 2), num("2")), B("^", nint(r, 1, 2), num("2"))))

    def lin_f(r):
        return B(r.choice("+-"), B("*", small(r), _sci_fn(r)), small(r))

    def two_f(r):
        return B(r.choice("+-*/"), _sci_fn(r), _sci_fn(r))

    def f_sq(r):
        return B("^", _sci_fn(r), num("2"))

    def f_of_g(r):
        inner = pick(r, (1, lambda q: fn("sqrt", dec(q, 1, 3, 1))), (1, lambda q: B("^", small(q), num("2"))),
                     (1, lambda q: B("*", small(q), ("const", "pi"))), (1, lambda q: B("+", small(q), small(q))),
                     (1, lambda q: B("/", nint(q, 1, 2), nint(q, 1, 2))))(r)  # fmt: skip
        return fn(r.choice(["sin", "cos", "ln", "log", "sqrt", "atan"]), inner)

    def compound(r):
        g = B("^", B("+", num("1"), num(f"{r.randint(1, 25) / 100:.2f}".rstrip("0"))), num(str(r.randint(2, 30))))
        return g if r.random() < 0.5 else B("*", nint(r, 2, 4), g)

    def quad(r):
        a, b, c = r.randint(1, 9), r.randint(-20, 20), r.randint(-20, 9)
        disc = B("-", B("^", num(str(b)), num("2")), B("*", B("*", num("4"), num(str(a))), num(str(c))))
        top = B(r.choice("+-"), num(str(-b)), fn("sqrt", disc))
        return B("/", top, B("*", num("2"), num(str(a))))

    def ident(r):
        x = small(r)
        return pick(r, (1, B("+", B("^", fn("sin", x), num("2")), B("^", fn("cos", x), num("2")))),
                    (1, fn("ln", B("^", ("const", "e"), x))), (1, fn("log", B("^", num("10"), x))),
                    (1, B("^", ("const", "e"), fn("ln", x))), (1, B("^", fn("sqrt", x), num("2"))))  # fmt: skip

    def geom(r):
        x = small(r)
        pi = ("const", "pi")
        return pick(r, (1, B("*", pi, B("^", x, num("2")))), (1, B("*", B("*", num("2"), pi), x)),
                    (1, B("*", B("/", num("4"), num("3")), B("*", pi, B("^", x, num("3"))))),
                    (1, B("*", B("*", num("2"), pi), fn("sqrt", B("/", x, num("9.8"))))))  # fmt: skip

    def degrees(r):
        d = r.choice([0, 15, 30, 45, 60, 75, 90, 120, 135, 150, 180, 210, 270, 360]) if r.random() < 0.5 else r.randint(0, 360)
        return fn(r.choice(["sin", "cos", "tan"]), B("/", B("*", num(str(d)), ("const", "pi")), num("180")))

    def mixed(r):
        def leaf(q):
            return pick(q, (3, small), (2, _sci_fn), (1, lambda z: B("^", small(z), num("2"))),
                        (1, lambda z: fn("sqrt", nint(z, 1, 3))), (0.5, lambda z: ("const", "pi")))(q)  # fmt: skip

        return _arith_tree(r, r.randint(2, 3), leaf)

    g = pick(rng, (2, hyp), (3, lin_f), (3, two_f), (1.5, f_sq), (3, f_of_g), (2, compound), (1.5, quad), (1.5, ident),
             (2, geom), (1.5, degrees), (3, mixed))  # fmt: skip
    return g(rng)


GENS = [gen_L1, gen_L2, gen_L3, gen_L4, gen_L5, gen_L6, gen_L7]


# ------------------------------------------------------------------ scratchpad
def scratch(tree):
    """Supervised chain-of-thought: intermediate values in evaluation order."""
    if tree[0] == "bin" and tree[1] == "*" and tree[2][0] == "num" and tree[3][0] == "num":
        a, b = tree[2][1], tree[3][1]
        if a.denominator == 1 and b.denominator == 1 and min(a, b) >= 10:
            a, b = int(a), int(b)
            return [str(a * int(d) * 10**i) for i, d in enumerate(reversed(str(b))) if d != "0"]
    out = []

    def walk(n, root):
        t = n[0]
        if t in ("num", "const"):
            return
        kids = n[2] if t == "fn" else [c for c in n[1:] if isinstance(c, tuple)]
        for c in kids:
            walk(c, False)
        if not root:
            out.append(fmt(ev(n)))

    walk(tree, True)
    return out


def is_test(expr):
    return zlib.crc32(expr.encode()) % 16 == 0


def sample(level, rng):
    """-> (expression, answer, scratch list); retries until the problem is well formed."""
    while True:
        tree = GENS[level](rng)
        try:
            ans = fmt(ev(tree))
            sc = scratch(tree)
        except (Reject, ZeroDivisionError, OverflowError):
            continue
        expr = show(tree)
        if len(expr) + 2 > P_MAX or len(ans) + 1 > A_MAX:
            continue
        if len(ans) + 3 + sum(len(x) + 1 for x in sc) > C_MAX:
            continue
        return expr, ans, sc


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS.get(int(i), "") for i in ids)


if __name__ == "__main__":
    import sys

    rng = random.Random(0)
    for lv in range(NLEVELS):
        print(f"--- {LEVELS[lv]}")
        for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 12):
            e, a, sc = sample(lv, rng)
            print(f"  {e} = {a}" + (f"      [{','.join(sc)}]" if sc else ""))
