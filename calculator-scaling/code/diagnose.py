"""The first fault in a model's worked solution: every arithmetic step is re-checked independently; if all of them are
right, the first place the solution departs from the reference names the failure."""
import re
from stepcheck import check_step, STEP

OPS = {"*": "a multiplication", "/": "a long division", "%": "a long division", "+": "an addition or subtraction", "-": "an addition or subtraction"}


def checked(st):
    """check_step, with a malformed step (work that does not parse) counted as wrong arithmetic."""
    try:
        return check_step(st)
    except Exception:
        return False


def depart(step):
    """What kind of step the reference has where the model first departs from it."""
    if step is None:
        return "extra or missing steps"
    if re.fullmatch(r"(d[xyz]?|k|x)=.*", step):
        return "a labelled value (d=, x=, k=)"
    if any(c in step for c in "xyz") and "=" in step:
        return "a restated equation"
    if "~" in step:
        return "a root estimate"
    if STEP.fullmatch(step):
        return "the operands of a step"
    return "a looked-up or summary value"


def diagnose(pred, reference):
    rsteps = reference[1 : reference.rindex("]")].split(";")
    cut = "]" not in pred  # never closed its block: decoding hit the length budget
    steps = pred[1:].split(";")[:-1] if cut else pred[1 : pred.rindex("]")].split(";")
    for i, st in enumerate(steps):
        if checked(st) is False:
            op = STEP.fullmatch(st).group(2)
            return "wrong arithmetic: " + OPS[op], i / max(len(rsteps), 1)
    if cut:
        return "ran past the length budget with every step right", None
    for i, st in enumerate(steps):
        r = rsteps[i] if i < len(rsteps) else None
        if st != r:
            if r is not None and "~" in r and "~" in st:
                continue  # a different starting estimate is not an error
            return "departs at " + depart(r), i / max(len(rsteps), 1)
    if len(steps) < len(rsteps):
        return "departs at extra or missing steps", 1.0
    if pred.split("]")[-1] == reference.split("]")[-1]:
        return "follows the reference exactly; its 6-digit arithmetic rounds to a different answer", 1.0
    return "final answer miscopied", 1.0


def fault_offset(pred, reference):
    """Token offset, in the model's own output, of the step diagnose() blames (None when there is no such step)."""
    rsteps = reference[1 : reference.rindex("]")].split(";")
    cut = "]" not in pred
    steps = pred[1:].split(";")[:-1] if cut else pred[1 : pred.rindex("]")].split(";")
    off = [1]
    for st in steps:
        off.append(off[-1] + len(st) + 1)
    for i, st in enumerate(steps):
        if checked(st) is False:
            return off[i]
    if cut:
        return None
    for i, st in enumerate(steps):
        r = rsteps[i] if i < len(rsteps) else None
        if st != r and not (r is not None and "~" in r and "~" in st):
            return off[i]
    return off[-1]
