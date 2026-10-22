#!/usr/bin/env python3
"""Validate the environment itself. Run before every RL job.

An RL environment with a wrong grader is worse than no environment: it trains
the policy on noise, confidently, and the loss curve looks fine while it does
it. These checks are the closest thing to a proof that the reward means what it
claims.

Per family, per difficulty, over many seeds:

  reference-accept   the generator's own reference answer scores 1.0.
                     Catches a generator and grader that disagree -- the defect
                     that silently inverts the reward.
  perturb-reject     the reference answer scaled by 1.5 scores 0.0.
                     Catches a tolerance so loose it accepts anything.
  tolerance-edge     the reference nudged just inside tolerance scores 1.0 and
                     just outside scores 0.0. Catches an off-by-one in the
                     comparison and a tolerance of 0 or infinity.
  unit-required      the right number with a dimensionally wrong unit scores
                     0.0 with unit_ok False. Catches unit checking being
                     skipped for that family's answer.
  no-unit-rejected   a bare number for a dimensioned answer scores 0.0.
  format-detected    a completion with no answer line scores 0.0 with
                     format_ok False, distinct from a wrong answer.
  trap-scored-zero   the designed wrong method scores 0.0 and sets trap_hit.
                     Catches a trap that coincides with the right answer.
  abstain-penalised  abstaining on an answerable problem scores 0.0.
  determinism        the same (family, seed, difficulty) renders identically.
  seed-split         train and eval seed ranges do not overlap.

Unanswerable instances get their own pass: abstention scores 1.0, a confident
number scores 0.0, and the generator leaves no computable answer behind.

    python3 selftest.py              # default: 60 seeds per family per difficulty
    python3 selftest.py --seeds 300  # slower, before a long run
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import families as F
import grader as G
import units as U

from families import EVAL_SEED_BASE


class Failures:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.checks = 0

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.items.append(message)

    def report(self) -> int:
        print(f"\n{self.checks} assertion(s) run")
        if not self.items:
            print("PASS")
            return 0
        print(f"{len(self.items)} FAILURE(S):\n")
        for item in self.items[:40]:
            print(f"  - {item}")
        if len(self.items) > 40:
            print(f"  ... and {len(self.items) - 40} more")
        print("\nFAIL")
        return 1


def dimensioned(unit: str) -> bool:
    try:
        return U.parse_unit(unit)[1] != U.DIMENSIONLESS
    except U.UnitError:
        return False


def wrong_unit_for(unit: str) -> str:
    """A unit guaranteed to be dimensionally incompatible with `unit`."""
    for candidate in ("s", "m", "mol", "K", "kg", "A"):
        if not U.same_dimension(candidate, unit):
            return candidate
    return "s"


def check_answerable(fam: str, seed: int, diff: str, fail: Failures) -> None:
    p = F.sample(fam, seed, diff)
    tag = f"{fam}/{diff}/{seed}"

    # The reference answer must score full marks.
    ref = G.reference_completion(p)
    s = G.score(p, ref)
    fail.check(s.reward == 1.0,
               f"{tag}: reference answer scored {s.reward} (expected 1.0): {s.detail}")
    fail.check(s.value_ok and s.unit_ok and s.format_ok,
               f"{tag}: reference flags value={s.value_ok} unit={s.unit_ok} fmt={s.format_ok}")

    # Targets must be finite and non-degenerate, or the comparison is meaningless.
    for t in p.targets:
        fail.check(t.value == t.value and abs(t.value) != float("inf"),
                   f"{tag}: target {t.key} is not finite ({t.value})")
        fail.check(0 < t.rel_tol < 0.5, f"{tag}: target {t.key} rel_tol {t.rel_tol} out of range")
        fail.check(abs(t.value) > 1e-12, f"{tag}: target {t.key} is ~0, tolerance is meaningless")

    # A 1.5x error must be rejected.
    bad = "ANSWER: " + "; ".join(f"{t.value*1.5:.6g} {t.unit}" for t in p.targets)
    s_bad = G.score(p, bad)
    fail.check(s_bad.reward == 0.0, f"{tag}: 1.5x perturbation scored {s_bad.reward}")

    # Tolerance edges, probed on the first target only (the others move with it).
    t0 = p.targets[0]
    inside = [t0.value * (1 + t0.rel_tol * 0.5)] + [t.value for t in p.targets[1:]]
    outside = [t0.value * (1 + t0.rel_tol * 3.0)] + [t.value for t in p.targets[1:]]
    s_in = G.score(p, "ANSWER: " + "; ".join(
        f"{v:.10g} {t.unit}" for v, t in zip(inside, p.targets)))
    s_out = G.score(p, "ANSWER: " + "; ".join(
        f"{v:.10g} {t.unit}" for v, t in zip(outside, p.targets)))
    fail.check(s_in.reward == 1.0, f"{tag}: just-inside-tolerance scored {s_in.reward}")
    fail.check(s_out.reward == 0.0, f"{tag}: 3x-tolerance error scored {s_out.reward}")

    # Dimensionally wrong unit on the right number.
    if dimensioned(t0.unit):
        wu = wrong_unit_for(t0.unit)
        s_wu = G.score(p, "ANSWER: " + "; ".join(
            f"{t.value:.6g} {wu if i == 0 else t.unit}" for i, t in enumerate(p.targets)))
        fail.check(s_wu.reward == 0.0 and not s_wu.unit_ok,
                   f"{tag}: wrong unit {wu!r} scored {s_wu.reward} unit_ok={s_wu.unit_ok}")

        # Bare number for a dimensioned answer.
        s_nu = G.score(p, "ANSWER: " + "; ".join(
            f"{t.value:.6g}" + ("" if i == 0 else f" {t.unit}")
            for i, t in enumerate(p.targets)))
        fail.check(s_nu.reward == 0.0,
                   f"{tag}: bare number for [{U.fmt_dims(U.parse_unit(t0.unit)[1])}] "
                   f"scored {s_nu.reward}")

    # Missing answer line is a format failure, not a wrong answer.
    s_nf = G.score(p, "I worked through it and the result follows from the given quantities.")
    fail.check(s_nf.reward == 0.0 and not s_nf.format_ok,
               f"{tag}: unformatted completion fmt_ok={s_nf.format_ok} reward={s_nf.reward}")

    # The trap must be wrong, and recognised as the trap.
    trap = G.trap_completion(p)
    if trap is not None:
        s_t = G.score(p, trap)
        fail.check(s_t.reward == 0.0,
                   f"{tag}: trap ({p.trap_note}) scored {s_t.reward} -- "
                   f"trap coincides with the correct answer")
        fail.check(s_t.trap_hit, f"{tag}: trap not flagged as trap_hit")

    # Abstaining on an answerable problem earns nothing.
    s_ab = G.score(p, "ANSWER: INSUFFICIENT molecular weight")
    fail.check(s_ab.reward == 0.0 and s_ab.abstained,
               f"{tag}: abstention on answerable problem scored {s_ab.reward}")

    # The prompt must state the answer contract and expose every given.
    rendered = p.render()
    fail.check("ANSWER:" in rendered, f"{tag}: rendered prompt lacks the answer contract")
    fail.check("INSUFFICIENT" in rendered, f"{tag}: rendered prompt lacks the abstention option")
    for g in p.givens:
        fail.check(g.label in rendered, f"{tag}: given {g.label!r} missing from prompt")

    # Determinism.
    again = F.sample(fam, seed, diff)
    fail.check(again.render() == rendered, f"{tag}: not deterministic for a fixed seed")
    fail.check([t.value for t in again.targets] == [t.value for t in p.targets],
               f"{tag}: targets not deterministic for a fixed seed")


def check_insufficient(fam: str, seed: int, diff: str, fail: Failures) -> None:
    p = F.sample(fam, seed, diff, insufficient=True)
    tag = f"{fam}/{diff}/{seed}/insufficient"
    fail.check(p.insufficient is not None, f"{tag}: insufficient flag not set")
    fail.check(p.targets == [], f"{tag}: unanswerable problem still carries targets")
    fail.check(p.trap is None, f"{tag}: unanswerable problem still carries a trap")
    fail.check(p.insufficient not in [g.label for g in p.givens],
               f"{tag}: the dropped quantity {p.insufficient!r} is still in the givens")

    s_ok = G.score(p, G.reference_completion(p))
    fail.check(s_ok.reward == 1.0 and s_ok.abstained,
               f"{tag}: correct abstention scored {s_ok.reward}")

    s_num = G.score(p, "Working through it gives a single value.\nANSWER: 42 mL")
    fail.check(s_num.reward == 0.0,
               f"{tag}: confident number on an unanswerable problem scored {s_num.reward}")

    for phrasing in ("ANSWER: INSUFFICIENT", "There is not enough information to answer.",
                     "ANSWER: cannot be determined from the information given"):
        s_alt = G.score(p, phrasing)
        fail.check(s_alt.reward == 1.0,
                   f"{tag}: abstention phrasing {phrasing[:34]!r} scored {s_alt.reward}")


def check_units_module(fail: Failures) -> None:
    equivalences = [("mg/mL", "g/L"), ("mM", "mmol/L"), ("umol", "mol"), ("M", "mol/L"),
                    ("mL", "cm^3"), ("min", "s"), ("rpm", "Hz"), ("xg", "m/s^2"),
                    ("M^-1 cm^-1", "L/mol/cm"), ("umol/min/mg", "mol/s/kg")]
    for a, b in equivalences:
        fail.check(U.same_dimension(a, b), f"units: {a} should match {b} dimensionally")
    distinctions = [("mM", "mol"), ("kDa", "g/mol"), ("mg", "mL"), ("1/h", "h")]
    for a, b in distinctions:
        fail.check(not U.same_dimension(a, b), f"units: {a} should NOT match {b}")
    conversions = [(1, "M", "mM", 1000.0), (10000, "rpm", "Hz", 166.6667),
                   (1, "umol", "mol", 1e-6), (1, "h", "min", 60.0)]
    for v, src, dst, want in conversions:
        got = U.convert(v, src, dst)
        fail.check(abs(got - want) / want < 1e-4, f"units: {v} {src} -> {dst} gave {got}, want {want}")
    fail.check(not U.close(1, "flurbles", 1, "mol", 0.01)[0],
               "units: unknown unit should fail closed, not raise")


def check_number_parsing(fail: Failures) -> None:
    cases = [
        ("ANSWER: 1.5e-3 mol", 1.5e-3), ("ANSWER: 1.5 x 10^-3 mol", 1.5e-3),
        ("ANSWER: 1,200 mol", 1200.0), ("ANSWER: -4.2 mol", -4.2),
        ("ANSWER: 0.075 mol", 0.075), ("ANSWER: approximately 3.4 mol", 3.4),
    ]
    for text, want in cases:
        fields = G.parse_fields(G.extract_answer_line(text)[0], 1)
        fail.check(bool(fields) and abs(fields[0][0] - want) < 1e-12,
                   f"parse: {text!r} gave {fields}, want {want}")
    # The LAST marker wins: models restate intermediate values in the same form.
    multi = "ANSWER: 1 mol\nOn reflection that was the intermediate.\nANSWER: 2 mol"
    fail.check(G.parse_fields(G.extract_answer_line(multi)[0], 1)[0][0] == 2.0,
               "parse: last answer line should win")
    # Boxed takes precedence.
    fail.check(G.extract_answer_line(r"ANSWER: 1 mol \boxed{7 mol}")[1] == "boxed",
               "parse: boxed form should take precedence")
    # Two fields from a +/- form.
    two = G.parse_fields("25.0 +/- 0.7 umol", 2)
    fail.check(len(two) == 2 and abs(two[1][0] - 0.7) < 1e-9,
               f"parse: '25.0 +/- 0.7 umol' gave {two}")


def check_seed_split(fail: Failures) -> None:
    train = {F.sample("dilution", s, "medium").render() for s in range(200)}
    held = {F.sample("dilution", EVAL_SEED_BASE + s, "medium").render() for s in range(200)}
    overlap = train & held
    fail.check(not overlap,
               f"seed split: {len(overlap)} prompt(s) appear in both train and eval seed ranges")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=60, help="seeds per family per difficulty")
    ap.add_argument("--family", action="append", help="restrict to these families")
    args = ap.parse_args(argv)

    fams = args.family or sorted(F.FAMILIES)
    unknown = [f for f in fams if f not in F.FAMILIES]
    if unknown:
        print(f"unknown family/families: {unknown}", file=sys.stderr)
        return 2

    fail = Failures()
    check_units_module(fail)
    check_number_parsing(fail)
    check_seed_split(fail)

    traps = Counter()
    for fam in fams:
        for diff in F.DIFFICULTIES:
            for seed in range(args.seeds):
                check_answerable(fam, seed, diff, fail)
            for seed in range(max(4, args.seeds // 6)):
                check_insufficient(fam, seed, diff, fail)
        p = F.sample(fam, 0, "medium")
        traps[fam] = 1 if p.trap is not None else 0
        print(f"  checked {fam:<30} {len(F.DIFFICULTIES)*args.seeds:>5} instance(s)"
              f"{'' if p.trap else '   (no trap defined)'}")

    missing_traps = [f for f, has in traps.items() if not has]
    fail.check(not missing_traps,
               f"families without a trap answer: {missing_traps} -- "
               "no diagnostic for which wrong method the policy learned")

    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
