"""Minimal dimensional-analysis engine for unit-aware answer grading.

Why this exists: a science answer is a number *and* a unit, and the two failure
modes are different. "4.7" when the answer is 4.7 mM is a formatting failure;
"4.7 M" is a thousand-fold error. A grader that string-matches units cannot tell
`mg/mL` from `g/L` (identical) or `mM` from `mmol/L` (identical), and will
happily accept `mol` where `mol/L` was required. So units are reduced to SI base
dimensions and compared there.

Stdlib only, so the grader runs wherever the rollouts do.

    >>> to_si(4.7, "mg/mL")
    (4.7, (-3, 1, 0, 0, 0, 0, 0))
    >>> same_dimension("mg/mL", "g/L")
    True
    >>> same_dimension("mM", "mmol/L")
    True
    >>> same_dimension("mM", "mol")
    False

Conventions that differ from strict SI, chosen to match how the quantities are
written in a lab:

  M      molar, mol/L (so the SI prefix on `mM` works as expected)
  xg     relative centrifugal field, as a multiple of standard gravity.
         Plain `g` is grams, so an RCF answer must be written `9503 xg`.
         `x g`, `× g` and `g-force` are normalised to `xg` on input.
  rpm    revolutions per minute, reduced to s^-1 (a revolution is dimensionless)
  cells, counts, reads, AU
         dimensionless counting units, kept distinct from `1` only for display
"""

from __future__ import annotations

import re

# SI base dimensions, in fixed order.
DIM_NAMES = ("m", "kg", "s", "A", "K", "mol", "cd")
Dims = tuple  # tuple[int, ...] of length 7


def D(m: int = 0, kg: int = 0, s: int = 0, A: int = 0, K: int = 0, mol: int = 0, cd: int = 0) -> Dims:
    return (m, kg, s, A, K, mol, cd)


DIMENSIONLESS = D()

# symbol -> (factor to SI, dimensions)
_UNITS: dict[str, tuple[float, Dims]] = {
    "": (1.0, DIMENSIONLESS),
    "1": (1.0, DIMENSIONLESS),
    # base
    "m": (1.0, D(m=1)),
    "g": (1e-3, D(kg=1)),
    "s": (1.0, D(s=1)),
    "A": (1.0, D(A=1)),
    "K": (1.0, D(K=1)),
    "mol": (1.0, D(mol=1)),
    "cd": (1.0, D(cd=1)),
    # volume / concentration
    "L": (1e-3, D(m=3)),
    "M": (1e3, D(mol=1, m=-3)),  # 1 M = 1 mol/L = 1000 mol/m^3
    # time
    "min": (60.0, D(s=1)),
    "h": (3600.0, D(s=1)),
    "hr": (3600.0, D(s=1)),
    "day": (86400.0, D(s=1)),
    "wk": (604800.0, D(s=1)),
    "yr": (3.15576e7, D(s=1)),
    # derived
    "Hz": (1.0, D(s=-1)),
    "N": (1.0, D(kg=1, m=1, s=-2)),
    "Pa": (1.0, D(kg=1, m=-1, s=-2)),
    "J": (1.0, D(kg=1, m=2, s=-2)),
    "W": (1.0, D(kg=1, m=2, s=-3)),
    "C": (1.0, D(A=1, s=1)),
    "V": (1.0, D(kg=1, m=2, s=-3, A=-1)),
    "cal": (4.184, D(kg=1, m=2, s=-2)),
    "atm": (101325.0, D(kg=1, m=-1, s=-2)),
    "bar": (1e5, D(kg=1, m=-1, s=-2)),
    "mmHg": (133.322387415, D(kg=1, m=-1, s=-2)),
    "Torr": (133.322368421, D(kg=1, m=-1, s=-2)),
    # mass of a molecule
    "Da": (1.66053906660e-27, D(kg=1)),
    # rotation and acceleration fields
    "rpm": (1.0 / 60.0, D(s=-1)),
    "xg": (9.80665, D(m=1, s=-2)),
    # dimensionless, kept as named units so a bare number is not required
    "cells": (1.0, DIMENSIONLESS),
    "counts": (1.0, DIMENSIONLESS),
    "reads": (1.0, DIMENSIONLESS),
    "copies": (1.0, DIMENSIONLESS),
    "molecules": (1.0, DIMENSIONLESS),
    "particles": (1.0, DIMENSIONLESS),
    "AU": (1.0, DIMENSIONLESS),
    "fold": (1.0, DIMENSIONLESS),
    "doublings": (1.0, DIMENSIONLESS),
    "cycles": (1.0, DIMENSIONLESS),
    "samples": (1.0, DIMENSIONLESS),
    "animals": (1.0, DIMENSIONLESS),
    "participants": (1.0, DIMENSIONLESS),
    "pH": (1.0, DIMENSIONLESS),
    "percent": (1e-2, DIMENSIONLESS),
    "%": (1e-2, DIMENSIONLESS),
}

# SI prefixes. Looked up only after an exact symbol match fails, so `M` stays
# molar, `min` stays minutes, and `h` stays hours.
_PREFIXES: dict[str, float] = {
    "y": 1e-24, "z": 1e-21, "a": 1e-18, "f": 1e-15, "p": 1e-12,
    "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3,
    "c": 1e-2, "d": 1e-1, "da": 1e1, "hecto": 1e2,
    "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15,
}

_TOKEN_RE = re.compile(r"^([A-Za-zµμ%]+)(?:\^?\(?(-?\d+)\)?)?$")


class UnitError(ValueError):
    """Raised when a unit string cannot be parsed. Grading treats this as a
    unit failure rather than a crash, so a model that invents a unit scores
    zero instead of taking down the rollout."""


def _lookup(symbol: str) -> tuple[float, Dims]:
    if symbol in _UNITS:
        return _UNITS[symbol]
    for plen in (2, 1):  # 'da' before 'd'
        if len(symbol) > plen:
            pre, rest = symbol[:plen], symbol[plen:]
            if pre in _PREFIXES and rest in _UNITS:
                factor, dims = _UNITS[rest]
                return factor * _PREFIXES[pre], dims
    raise UnitError(f"unknown unit symbol: {symbol!r}")


def _normalise(text: str) -> str:
    t = text.strip()
    t = t.replace("×", "*").replace("·", "*").replace("⋅", "*")
    t = re.sub(r"\bg[- ]?force\b", "xg", t, flags=re.I)
    t = re.sub(r"(?<![A-Za-z])[x×]\s*g(?![A-Za-z])", "xg", t)
    t = re.sub(r"\bper\b", "/", t)
    t = re.sub(r"\s*/\s*", "/", t)
    t = re.sub(r"\s*\*\s*", "*", t)
    # superscript digits occasionally survive copy-paste
    for sup, plain in zip("¹²³⁰⁴⁵⁶⁷⁸⁹⁻",
                          "123074567890-"):
        t = t.replace(sup, plain)
    return t


def parse_unit(text: str) -> tuple[float, Dims]:
    """Parse a unit string into (factor to SI, dimension tuple)."""
    t = _normalise(text)
    if t == "":
        return 1.0, DIMENSIONLESS
    factor = 1.0
    dims = [0] * len(DIM_NAMES)
    # Everything after the first '/' is a denominator, so 'mol/L/s' == 'mol/(L*s)'.
    for position, chunk in enumerate(t.split("/")):
        sign = 1 if position == 0 else -1
        for token in re.split(r"[*\s]+", chunk.strip()):
            if not token or token == "1":
                continue
            mo = _TOKEN_RE.match(token)
            if not mo:
                raise UnitError(f"cannot parse unit token: {token!r} (in {text!r})")
            sym, exp = mo.group(1), int(mo.group(2) or 1) * sign
            f, d = _lookup(sym)
            factor *= f ** exp
            for i in range(len(dims)):
                dims[i] += d[i] * exp
    return factor, tuple(dims)


def to_si(value: float, unit: str) -> tuple[float, Dims]:
    """Convert a (value, unit) pair to its SI magnitude and dimensions."""
    factor, dims = parse_unit(unit)
    return value * factor, dims


def same_dimension(unit_a: str, unit_b: str) -> bool:
    try:
        return parse_unit(unit_a)[1] == parse_unit(unit_b)[1]
    except UnitError:
        return False


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between units of the same dimension. Raises UnitError otherwise."""
    fa, da = parse_unit(from_unit)
    fb, db = parse_unit(to_unit)
    if da != db:
        raise UnitError(f"incommensurable units: {from_unit!r} [{fmt_dims(da)}] "
                        f"vs {to_unit!r} [{fmt_dims(db)}]")
    return value * fa / fb


def fmt_dims(dims: Dims) -> str:
    """Human-readable dimension signature, for grader diagnostics."""
    parts = [f"{n}^{e}" if e != 1 else n for n, e in zip(DIM_NAMES, dims) if e]
    return "·".join(parts) if parts else "dimensionless"


def close(value_a: float, unit_a: str, value_b: float, unit_b: str, rel_tol: float) -> tuple[bool, str]:
    """Compare two quantities. Returns (ok, reason).

    Dimension mismatch is reported distinctly from magnitude mismatch: the two
    are different model errors and collapsing them loses the diagnostic.
    """
    try:
        si_a, da = to_si(value_a, unit_a)
        si_b, db = to_si(value_b, unit_b)
    except UnitError as exc:
        return False, f"unit_error: {exc}"
    if da != db:
        return False, (f"dimension_mismatch: got [{fmt_dims(da)}] from {unit_a!r}, "
                       f"expected [{fmt_dims(db)}] from {unit_b!r}")
    if si_b == 0:
        ok = abs(si_a) <= rel_tol
    else:
        ok = abs(si_a - si_b) / abs(si_b) <= rel_tol
    if ok:
        return True, "ok"
    ratio = (si_a / si_b) if si_b else float("inf")
    return False, f"value_mismatch: ratio {ratio:.4g} (tolerance {rel_tol:.3g})"
