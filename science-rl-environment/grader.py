"""Programmatic grading of a model completion against a problem's reference answer.

Three failure modes are kept apart, because for RL they are different signals
and collapsing them into one scalar throws away the diagnosis:

    format_ok   an answer line was found and parsed at all
    unit_ok     the unit is dimensionally what was asked for
    value_ok    the magnitude is within tolerance

A model that computes correctly and writes `4.7` with no unit is failing at
formatting; one that writes `4.7 M` for a 4.7 mM answer is failing at
chemistry. Both score zero, and the component flags say which happened.

The reward is binary by default. Shaping is available and documented, but the
default is strict for a reason given in the README: every shaped term is a
reward-hacking surface, and partial credit for "right order of magnitude"
rewards a guess that lands in the right decade.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import units
from families import Problem, Target

# A number: optional sign, digits, optional decimal, optional exponent in any of
# the forms 1e-3 / 1E-3 / 1 x 10^-3 / 1*10^-3 / 1 10^-3. Thousands separators
# are tolerated because models emit them; they are stripped before float().
_NUM = r"""
    [-+\u2212]?                       # sign, incl. unicode minus
    (?:\d{1,3}(?:,\d{3})+|\d+)        # 1,200 or 1200
    (?:\.\d+)?                        # decimal part
    (?:
        \s*(?:[eE]\s*[-+\u2212]?\d+)  # e-3
      | \s*(?:[x\u00d7*]\s*10\s*\^?\s*[-+\u2212]?\d+)   # x 10^-3
    )?
"""
_NUM_RE = re.compile(_NUM, re.X)

_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s+)?answer\s*[:=]\s*(.+?)(?=\n|$)", re.I | re.S
)
_INSUFFICIENT_RE = re.compile(
    r"\b(insufficient|not\s+enough\s+information|underdetermined|cannot\s+be\s+determined)\b",
    re.I,
)


@dataclass
class Score:
    reward: float
    format_ok: bool
    unit_ok: bool
    value_ok: bool
    order_ok: bool
    trap_hit: bool
    abstained: bool
    detail: list[str] = field(default_factory=list)
    extracted: str = ""

    def as_dict(self) -> dict:
        return {
            "reward": self.reward,
            "format_ok": self.format_ok,
            "unit_ok": self.unit_ok,
            "value_ok": self.value_ok,
            "order_ok": self.order_ok,
            "trap_hit": self.trap_hit,
            "abstained": self.abstained,
            "extracted": self.extracted,
            "detail": self.detail,
        }


def _clean_number(text: str) -> float:
    t = text.replace(",", "").replace("\u2212", "-").strip()
    t = re.sub(r"\s*[x\u00d7*]\s*10\s*\^?\s*", "e", t)
    t = re.sub(r"\s*[eE]\s*", "e", t)
    return float(t)


def extract_answer_line(completion: str) -> tuple[str, str]:
    """Return (answer_text, how). `how` is 'boxed', 'answer_line', 'last_number' or 'none'.

    Extraction order matters: an explicit marker beats a trailing number, and the
    LAST marker beats earlier ones, because models restate intermediate results
    in the same form while working.
    """
    boxed = _BOXED_RE.findall(completion)
    if boxed:
        return boxed[-1].strip(), "boxed"
    lines = _ANSWER_RE.findall(completion)
    if lines:
        return lines[-1].strip(), "answer_line"
    nums = _NUM_RE.findall(completion)
    if nums:
        # Fall back to the tail of the completion, so a stray number mid-reasoning
        # is not mistaken for the answer.
        tail = completion.strip().splitlines()[-1] if completion.strip() else ""
        return tail.strip(), "last_number"
    return "", "none"


def parse_fields(answer_text: str, n_expected: int) -> list[tuple[float, str]]:
    """Split an answer into (value, unit) pairs.

    Multi-value answers are separated by ';' (the contract stated in the prompt)
    with ',' and ' and ' accepted as fallbacks. A single '+/-' pair is also split,
    since that is how an uncertainty is naturally written.
    """
    text = answer_text.strip().strip(".")
    text = re.sub(r"^(?:is|are|=|approximately|about|~|\u2248)\s*", "", text, flags=re.I)
    if n_expected > 1:
        parts = re.split(r"\s*;\s*", text)
        if len(parts) < n_expected:
            parts = re.split(r"\s*(?:\u00b1|\+/-|\+-)\s*|\s*,\s*|\s+and\s+", text)
    else:
        parts = [text]
    out: list[tuple[float, str]] = []
    for part in parts:
        if not part.strip():
            continue
        mo = _NUM_RE.search(part)
        if not mo:
            continue
        try:
            value = _clean_number(mo.group(0))
        except ValueError:
            continue
        unit = part[mo.end():].strip()
        unit = re.sub(r"^(?:of|in)\s+", "", unit, flags=re.I)
        unit = unit.strip("()[]").strip()
        unit = re.sub(r"\s*\(.*?\)\s*$", "", unit).strip()
        out.append((value, unit))
    return out


def _match_target(value: float, unit: str, target: Target) -> tuple[bool, bool, bool, str]:
    """Return (unit_ok, value_ok, order_ok, reason) for one field."""
    unit_text = unit if unit else ""
    # A dimensionless target accepts a bare number; a dimensioned one does not.
    if not unit_text:
        try:
            target_dims = units.parse_unit(target.unit)[1]
        except units.UnitError:
            target_dims = units.DIMENSIONLESS
        if target_dims != units.DIMENSIONLESS:
            return False, False, False, f"missing unit (expected [{units.fmt_dims(target_dims)}])"
        unit_text = "1"
    unit_ok = units.same_dimension(unit_text, target.unit)
    ok, reason = units.close(value, unit_text, target.value, target.unit, target.rel_tol)
    order_ok = False
    if unit_ok:
        try:
            got = units.to_si(value, unit_text)[0]
            want = units.to_si(target.value, target.unit)[0]
            order_ok = want != 0 and 0.1 <= abs(got / want) <= 10
        except units.UnitError:
            order_ok = False
    return unit_ok, ok, order_ok, reason


def score(problem: Problem, completion: str, *, format_bonus: float = 0.0,
          shaped: bool = False) -> Score:
    """Grade a completion. Returns a Score; `reward` is what the trainer consumes.

    format_bonus  small credit for emitting a parseable answer at all, even when
                  wrong. Keeps early training from being a flat zero signal.
    shaped        if True, adds partial credit for a correct unit and correct
                  order of magnitude. Off by default -- see README section 4.
    """
    answer_text, how = extract_answer_line(completion)
    detail: list[str] = []
    abstained = bool(_INSUFFICIENT_RE.search(answer_text)) or (
        how in ("none", "last_number") and bool(_INSUFFICIENT_RE.search(completion))
    )

    # --- unanswerable instances: abstention is the correct answer -------------
    if problem.insufficient is not None:
        if abstained:
            missing_words = {w for w in re.findall(r"[a-z]+", problem.insufficient.lower())
                             if len(w) > 3}
            named = bool(missing_words) and any(
                w in completion.lower() for w in missing_words
            )
            detail.append(f"correct abstention; missing quantity named: {named}")
            return Score(reward=1.0, format_ok=True, unit_ok=True, value_ok=True,
                         order_ok=True, trap_hit=False, abstained=True,
                         detail=detail, extracted=answer_text)
        detail.append(f"answered a problem with no computable answer "
                      f"(missing: {problem.insufficient})")
        return Score(reward=0.0, format_ok=(how != "none"), unit_ok=False, value_ok=False,
                     order_ok=False, trap_hit=False, abstained=False,
                     detail=detail, extracted=answer_text)

    # --- abstention on an answerable instance --------------------------------
    if abstained:
        detail.append("abstained on an answerable problem")
        return Score(reward=0.0, format_ok=True, unit_ok=False, value_ok=False,
                     order_ok=False, trap_hit=False, abstained=True,
                     detail=detail, extracted=answer_text)

    if how == "none":
        detail.append("no answer found: expected a line 'ANSWER: <value> <unit>'")
        return Score(reward=0.0, format_ok=False, unit_ok=False, value_ok=False,
                     order_ok=False, trap_hit=False, abstained=False,
                     detail=detail, extracted="")

    fields = parse_fields(answer_text, len(problem.targets))
    if len(fields) < len(problem.targets):
        detail.append(f"expected {len(problem.targets)} value(s), parsed {len(fields)} "
                      f"from {answer_text!r}")
        return Score(reward=format_bonus if fields else 0.0, format_ok=bool(fields),
                     unit_ok=False, value_ok=False, order_ok=False, trap_hit=False,
                     abstained=False, detail=detail, extracted=answer_text)

    unit_flags, value_flags, order_flags = [], [], []
    for (value, unit), target in zip(fields, problem.targets):
        u_ok, v_ok, o_ok, reason = _match_target(value, unit, target)
        unit_flags.append(u_ok)
        value_flags.append(v_ok)
        order_flags.append(o_ok)
        detail.append(f"{target.key}: got {value:.6g} {unit or '(no unit)'}, "
                      f"expected {target.value:.6g} {target.unit} -> {reason}")

    unit_ok = all(unit_flags)
    value_ok = all(value_flags)
    order_ok = all(order_flags)

    trap_hit = False
    if problem.trap is not None and fields:
        for value, unit in fields:
            _, t_ok, _, _ = _match_target(value, unit or problem.trap.unit, problem.trap)
            if t_ok and not value_ok:
                trap_hit = True
                detail.append(f"trap hit: {problem.trap_note}")
                break

    if value_ok and unit_ok:
        reward = 1.0
    elif shaped:
        reward = 0.3 * unit_ok + 0.2 * order_ok + format_bonus
    else:
        reward = format_bonus
    return Score(reward=min(reward, 1.0), format_ok=True, unit_ok=unit_ok, value_ok=value_ok,
                 order_ok=order_ok, trap_hit=trap_hit, abstained=False,
                 detail=detail, extracted=answer_text)


def reference_completion(problem: Problem) -> str:
    """The completion a perfect solver would emit. Used by the self-test to prove
    the grader accepts its own reference answer -- the check that catches a
    generator and grader that disagree."""
    if problem.insufficient is not None:
        return (f"The {problem.insufficient} is not given and cannot be derived.\n"
                f"ANSWER: INSUFFICIENT {problem.insufficient}")
    body = "; ".join(f"{t.value:.6g} {t.unit}" for t in problem.targets)
    return f"{problem.solution}\nANSWER: {body}"


def trap_completion(problem: Problem) -> str | None:
    if problem.trap is None:
        return None
    return (f"Applying the shortcut.\nANSWER: "
            + "; ".join(
                f"{problem.trap.value:.6g} {problem.trap.unit}" if t.key == problem.trap.key
                else f"{t.value:.6g} {t.unit}"
                for t in problem.targets
            ))
