"""Classify a problem string into a human-readable sub-skill (for error analysis)."""
import re


def subtype(level, e):
    if level == 0:
        return "a+b" if "+" in e else "a-b"
    if level == 1:
        n_ops = len(re.findall(r"(?<=\d)[+-]", e))
        if n_ops == 1:
            d = max(len(x) for x in re.split(r"[+-]", e) if x)
            return f"2 operands, {'<=5' if d <= 5 else '6-8'} digits"
        return "3-4 operands"
    if level == 2:
        if e.startswith("mod"):
            return "mod(a,b)"
        if "*" in e:
            a, b = e.split("*")
            return f"a*b, smaller has {min(len(a), len(b))} digit" + ("s" if min(len(a), len(b)) > 1 else "")
        a, b = e.split("/")
        return "a/b exact" if int(a) % int(b) == 0 else "a/b inexact"
    if level == 3:
        return f"{len(re.findall(r'(?<=[0-9)])[-+*/]', e))} operations"
    if level == 4:
        if e.startswith("sqrt"):
            return "sqrt(x)"
        if e.startswith("abs"):
            return "abs(a±b)"
        if "^" in e:
            return "powers"
        if e.startswith("1/"):
            return "1/x"
        return "decimal/signed " + ("+-" if not re.search(r"[*/]", e) else "*/")
    if level == 5:
        for name in ("asin", "acos", "atan", "sin", "cos", "tan", "ln", "log", "gcd", "lcm"):
            if e.startswith(name):
                return {"asin": "inverse trig", "acos": "inverse trig", "atan": "inverse trig", "sin": "sin/cos/tan",
                        "cos": "sin/cos/tan", "tan": "sin/cos/tan", "ln": "ln/log", "log": "ln/log", "gcd": "gcd/lcm",
                        "lcm": "gcd/lcm"}[name]
        if "!" in e:
            return "n!"
        if "nCr" in e or "nPr" in e:
            return "nCr/nPr"
        if e.startswith("e^") or e.startswith("10^") or e.startswith("2^"):
            return "e^x, 10^x, 2^x"
        return "pi/e constants"
    return "composite"
