#!/usr/bin/env python3
"""The trainer-facing surface: sample batches, grade rollouts, emit datasets.

Two integration styles, both supported:

    # in-process, for a trainer that can import Python
    from env import make_batch, grade
    batch = make_batch(1024, split="train")
    rewards = [grade(p, completion).reward for p, completion in zip(batch, rollouts)]

    # out-of-process, for a trainer that passes JSON around
    python3 env.py emit --n 50000 --split train --out train.jsonl
    python3 env.py grade --problems train.jsonl --completions rollouts.jsonl

A problem is a pure function of (family, seed, difficulty, insufficient), so an
emitted record needs only those four fields to be regenerated exactly. That is
what makes the dataset cheap to ship and makes `verify` possible: regenerate
every record and confirm the stored answer still matches, which catches a
generator change silently invalidating a dataset you are still training on.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import families as F
import grader as G
from families import EVAL_SEED_BASE, GENERATOR_VERSION, Problem

DEFAULT_INSUFFICIENT_RATE = 0.10


def make_batch(
    n: int,
    *,
    split: str = "train",
    seed: int = 0,
    families: list[str] | None = None,
    difficulties: tuple[str, ...] = F.DIFFICULTIES,
    insufficient_rate: float = DEFAULT_INSUFFICIENT_RATE,
) -> list[Problem]:
    """Sample `n` problems, balanced across families.

    `insufficient_rate` is the fraction with no computable answer, where correct
    abstention is the reward-maximising action. Do not set it to zero: with
    every problem answerable, always emitting a number is optimal and the
    policy's ability to recognise an underdetermined problem is trained away.
    """
    fams = families or sorted(F.FAMILIES)
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    base = EVAL_SEED_BASE if split == "eval" else 0
    rng = random.Random(f"batch|{split}|{seed}|{n}")
    out: list[Problem] = []
    for i in range(n):
        fam = fams[i % len(fams)]
        diff = difficulties[(i // len(fams)) % len(difficulties)]
        insufficient = rng.random() < insufficient_rate
        out.append(F.sample(fam, base + seed * n + i, diff, insufficient=insufficient))
    rng.shuffle(out)
    return out


def grade(problem: Problem, completion: str, **kwargs) -> G.Score:
    """Grade one rollout. Thin pass-through, kept so trainers import one module."""
    return G.score(problem, completion, **kwargs)


# ------------------------------------------------------------ serialisation


def to_record(problem: Problem, split: str) -> dict:
    return {
        "id": f"{problem.family}|{problem.difficulty}|{problem.seed}"
              f"{'|insufficient' if problem.insufficient else ''}",
        "generator_version": GENERATOR_VERSION,
        "split": split,
        "family": problem.family.replace("__insufficient", ""),
        "difficulty": problem.difficulty,
        "seed": problem.seed,
        "insufficient": problem.insufficient,
        "prompt": problem.render(),
        "targets": [asdict(t) for t in problem.targets],
        "trap": asdict(problem.trap) if problem.trap else None,
        "trap_note": problem.trap_note,
        "solution": problem.solution,
        "meta": problem.meta,
    }


def from_record(rec: dict) -> Problem:
    """Regenerate the problem a record describes, and check it still matches.

    Raises if the stored answer disagrees with the regenerated one, which means
    the generators changed under a dataset emitted by an earlier version.
    """
    problem = F.sample(rec["family"], rec["seed"], rec["difficulty"],
                       insufficient=bool(rec.get("insufficient")))
    stored = [(t["key"], round(t["value"], 9)) for t in rec["targets"]]
    fresh = [(t.key, round(t.value, 9)) for t in problem.targets]
    if stored != fresh:
        raise ValueError(
            f"record {rec['id']} no longer matches its generator "
            f"(record version {rec.get('generator_version')}, current {GENERATOR_VERSION}). "
            f"stored={stored} regenerated={fresh}. Re-emit the dataset."
        )
    return problem


# -------------------------------------------------------------------- CLI


def cmd_emit(args: argparse.Namespace) -> int:
    batch = make_batch(args.n, split=args.split, seed=args.seed,
                       insufficient_rate=args.insufficient_rate)
    out = Path(args.out) if args.out else None
    stream = out.open("w", encoding="utf-8") if out else sys.stdout
    try:
        for p in batch:
            stream.write(json.dumps(to_record(p, args.split), ensure_ascii=False) + "\n")
    finally:
        if out:
            stream.close()
    if out:
        counts = Counter(p.family.replace("__insufficient", "") for p in batch)
        n_insuff = sum(1 for p in batch if p.insufficient)
        print(f"wrote {len(batch)} problem(s) to {out}")
        print(f"  families: {len(counts)}, unanswerable: {n_insuff} "
              f"({n_insuff/len(batch):.1%}), split: {args.split}")
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    problems = {}
    for line in Path(args.problems).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            problems[rec["id"]] = rec
    rows = []
    for line in Path(args.completions).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rec = problems.get(row["id"])
        if rec is None:
            print(f"skipping unknown problem id: {row['id']}", file=sys.stderr)
            continue
        problem = from_record(rec)
        s = grade(problem, row["completion"], format_bonus=args.format_bonus,
                  shaped=args.shaped)
        rows.append({"id": row["id"], "family": rec["family"],
                     "difficulty": rec["difficulty"], **s.as_dict()})
    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    _summarise(rows)
    return 0


def _summarise(rows: list[dict]) -> None:
    if not rows:
        print("no rollouts graded")
        return
    n = len(rows)
    print(f"\n{n} rollout(s) graded\n")
    print(f"  mean reward        {sum(r['reward'] for r in rows)/n:.3f}")
    for flag in ("format_ok", "unit_ok", "value_ok", "order_ok", "trap_hit", "abstained"):
        print(f"  {flag:<18} {sum(1 for r in rows if r[flag])/n:.3f}")
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    print("\n  per family:")
    for fam in sorted(by_fam):
        rs = by_fam[fam]
        acc = sum(r["reward"] for r in rs) / len(rs)
        trap = sum(1 for r in rs if r["trap_hit"]) / len(rs)
        print(f"    {fam:<32} reward {acc:.3f}  trap {trap:.3f}  (n={len(rs)})")
    # The diagnostic worth acting on: a high trap rate names the wrong procedure
    # the policy has settled into, which a scalar reward curve never shows.
    worst = max(by_fam.items(), key=lambda kv: sum(1 for r in kv[1] if r["trap_hit"]) / len(kv[1]))
    rate = sum(1 for r in worst[1] if r["trap_hit"]) / len(worst[1])
    if rate > 0.1:
        print(f"\n  highest trap rate: {worst[0]} at {rate:.1%} -- inspect `trap_note` "
              f"for that family; the policy has learned a specific wrong method")


def cmd_demo(args: argparse.Namespace) -> int:
    fams = [args.family] if args.family else sorted(F.FAMILIES)
    for fam in fams:
        p = F.sample(fam, args.seed, args.difficulty)
        print("=" * 78)
        print(p.render())
        print("-" * 78)
        print("reference:", p.solution)
        ref = G.score(p, G.reference_completion(p))
        print(f"  reference scores {ref.reward}")
        trap = G.trap_completion(p)
        if trap:
            ts = G.score(p, trap)
            print(f"  trap ({p.trap_note}) scores {ts.reward}, trap_hit={ts.trap_hit}")
        bare = G.score(p, f"ANSWER: {p.targets[0].value:.6g}")
        print(f"  correct number, no unit: {bare.reward} (unit_ok={bare.unit_ok})")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    print(f"generator version {GENERATOR_VERSION}, {len(F.FAMILIES)} families, "
          f"{len(F.DIFFICULTIES)} difficulties\n")
    print(f"  {'family':<32} {'distinct/'+str(args.n):>12}  {'targets':>7}  trap")
    total_small = 0
    for fam in sorted(F.FAMILIES):
        seen = {F.sample(fam, s, "medium").render() for s in range(args.n)}
        p = F.sample(fam, 0, "medium")
        small = len(seen) < 0.9 * args.n
        total_small += small
        print(f"  {fam:<32} {len(seen):>12}  {len(p.targets):>7}  "
              f"{'yes' if p.trap else 'no':<4}{'  <-- narrow' if small else ''}")
    print(f"\n  A family whose distinct count saturates below the seed count has a "
          f"small problem\n  space and is memorisable; widen its sampling grid.")
    if total_small:
        print(f"  {total_small} family/families flagged narrow.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    ok = bad = 0
    for line in Path(args.problems).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        try:
            from_record(rec)
            ok += 1
        except ValueError as exc:
            bad += 1
            if bad <= 10:
                print(f"  {exc}")
    print(f"\n{ok} record(s) verified against the current generators, {bad} stale")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit", help="write a JSONL dataset of problems")
    e.add_argument("--n", type=int, default=1000)
    e.add_argument("--split", choices=("train", "eval"), default="train")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--insufficient-rate", type=float, default=DEFAULT_INSUFFICIENT_RATE)
    e.add_argument("--out", help="output path; omit for stdout")
    e.set_defaults(func=cmd_emit)

    g = sub.add_parser("grade", help="score rollouts against an emitted dataset")
    g.add_argument("--problems", required=True, help="JSONL from `emit`")
    g.add_argument("--completions", required=True, help='JSONL of {"id":..., "completion":...}')
    g.add_argument("--out", help="write per-rollout scores here")
    g.add_argument("--format-bonus", type=float, default=0.0)
    g.add_argument("--shaped", action="store_true", help="partial credit (see README section 4)")
    g.set_defaults(func=cmd_grade)

    d = sub.add_parser("demo", help="print one problem per family with grading behaviour")
    d.add_argument("--family")
    d.add_argument("--difficulty", choices=F.DIFFICULTIES, default="medium")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("stats", help="report problem-space size per family")
    s.add_argument("--n", type=int, default=3000)
    s.set_defaults(func=cmd_stats)

    v = sub.add_parser("verify", help="check an emitted dataset against current generators")
    v.add_argument("--problems", required=True)
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
