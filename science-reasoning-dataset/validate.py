#!/usr/bin/env python3
"""Validate a science-reasoning corpus: schema, duplicates, train/eval contamination, mixture drift.

Run before every training job. Exits non-zero on any ERROR.

    python3 validate.py                 # every *.jsonl here; eval_*.jsonl is the eval split
    python3 validate.py --strict        # warnings fail too
    python3 validate.py --train sft_core.jsonl --eval eval_held_out.jsonl

Checks that earn their place (each has caught a real class of defect):

  schema         structural conformance, including provenance completeness
  ids            uniqueness, and that only dpo-* records carry chosen/rejected
  system-prompt  one system prompt across the corpus, or a deliberate set of them
  rubric         every record is gradable: non-empty must_contain and must_not_contain
  duplicates     exact and near-duplicate prompts within a split, which silently
                 reweight the mixture away from the declared table
  contamination  near-duplicate prompts across train and eval, which inflate every
                 reported metric; this is the check whose absence is unrecoverable
  mixture        category shares against the declared target
  numerics       numeric_answers coverage, which determines how much of the eval
                 can be graded mechanically rather than by model or human

Only the standard library is used, so this runs anywhere the data does. If
`jsonschema` is importable it is used for the schema pass; otherwise a built-in
structural check runs and the report says so.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

# Declared target mixture, from README.md. Keep the two in sync by hand --
# a script that reads the README is a script that silently accepts a typo there.
TARGET_MIX = {
    "quantitative-inference": 0.16,
    "statistical-reasoning": 0.16,
    "experimental-design": 0.13,
    "data-interpretation": 0.12,
    "calibration-and-abstention": 0.11,
    "false-premise-correction": 0.10,
    "derivation": 0.08,
    "evidence-appraisal": 0.08,
    "computation-workflow": 0.04,
    "constraint-retention": 0.02,
}

SHINGLE = 5
NEAR_DUP_WITHIN = 0.70
NEAR_DUP_ACROSS = 0.50  # stricter across splits: contamination is costlier than redundancy
MIXTURE_TOLERANCE = 0.05
MIXTURE_MIN_N = 500  # below this, mixture drift is expected and reported as INFO

REQUIRE_ORIGIN_DETAIL = {"model-generated", "model-generated-human-edited", "public-dataset"}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []

    def add(self, level: str, msg: str) -> None:
        self.rows.append((level, msg))

    error = lambda self, m: self.add("ERROR", m)  # noqa: E731
    warn = lambda self, m: self.add("WARN", m)  # noqa: E731
    info = lambda self, m: self.add("INFO", m)  # noqa: E731

    @property
    def n_errors(self) -> int:
        return sum(1 for lvl, _ in self.rows if lvl == "ERROR")

    def dump(self) -> None:
        width = max((len(lvl) for lvl, _ in self.rows), default=5)
        for lvl, msg in self.rows:
            print(f"{lvl:<{width}}  {msg}")
        counts = Counter(lvl for lvl, _ in self.rows)
        print(
            f"\n{counts['ERROR']} error(s), {counts['WARN']} warning(s), "
            f"{counts['INFO']} note(s)"
        )


def load(paths: list[Path], rep: Report) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        if not path.exists():
            rep.error(f"{path}: not found")
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                rep.error(f"{path}:{lineno}: invalid JSON -- {exc.msg} at col {exc.colno}")
                continue
            if not isinstance(obj, dict):
                rep.error(f"{path}:{lineno}: top-level value is {type(obj).__name__}, expected object")
                continue
            obj["_src"] = f"{path.name}:{lineno}"
            records.append(obj)
    return records


def check_schema(records: list[dict], schema_path: Path, rep: Report) -> None:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        rep.info("jsonschema not installed; running built-in structural check instead")
        _check_schema_builtin(records, rep)
        return

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(schema)
    for rec in records:
        payload = {k: v for k, v in rec.items() if k != "_src"}
        for err in sorted(validator.iter_errors(payload), key=lambda e: list(e.path)):
            loc = "/".join(str(p) for p in err.path) or "<root>"
            rep.error(f"{rec['_src']}: schema at {loc}: {err.message}")
    rep.info(f"schema: validated {len(records)} record(s) against {schema_path.name}")


def _check_schema_builtin(records: list[dict], rep: Report) -> None:
    required = {"id", "category", "skill", "difficulty", "messages", "rubric", "provenance"}
    roles = {"system", "user", "assistant"}
    for rec in records:
        src = rec["_src"]
        missing = required - rec.keys()
        if missing:
            rep.error(f"{src}: missing required field(s): {', '.join(sorted(missing))}")
            continue
        if rec["category"] not in TARGET_MIX:
            rep.error(f"{src}: unknown category {rec['category']!r}")
        if not re.fullmatch(r"(sft|dpo|eval)-[a-z0-9-]+-\d{4}", rec["id"]):
            rep.error(f"{src}: id {rec['id']!r} does not match the required pattern")
        msgs = rec["messages"]
        if not isinstance(msgs, list) or len(msgs) < 2:
            rep.error(f"{src}: messages must be a list of at least 2 turns")
        else:
            for i, m in enumerate(msgs):
                if not isinstance(m, dict) or m.keys() != {"role", "content"}:
                    rep.error(f"{src}: messages[{i}] must have exactly role and content")
                elif m["role"] not in roles:
                    rep.error(f"{src}: messages[{i}] has unknown role {m['role']!r}")
                elif not str(m["content"]).strip():
                    rep.error(f"{src}: messages[{i}] content is empty")
        prov = rec.get("provenance")
        if isinstance(prov, dict):
            if prov.get("source") in REQUIRE_ORIGIN_DETAIL and not prov.get("origin_detail"):
                rep.error(
                    f"{src}: provenance.source is {prov['source']!r} so origin_detail is required "
                    "(generating model + prompt id, or dataset + split + licence)"
                )
        else:
            rep.error(f"{src}: provenance must be an object")


def check_ids_and_roles(records: list[dict], rep: Report) -> None:
    seen: dict[str, str] = {}
    for rec in records:
        rid, src = rec.get("id"), rec["_src"]
        if rid in seen:
            rep.error(f"{src}: duplicate id {rid!r} (first seen at {seen[rid]})")
        else:
            seen[rid] = src

        is_pref = str(rid).startswith("dpo-")
        has_pair = "chosen" in rec or "rejected" in rec
        if is_pref and not ("chosen" in rec and "rejected" in rec):
            rep.error(f"{src}: dpo record must carry both chosen and rejected")
        if has_pair and not is_pref:
            rep.error(f"{src}: non-dpo record carries chosen/rejected; give it a dpo-* id or remove them")

        # An SFT record needs a gold completion; a preference or eval record must not
        # have one, or the trainer/grader will silently use the wrong field.
        last_role = rec["messages"][-1]["role"] if isinstance(rec.get("messages"), list) and rec["messages"] else None
        if str(rid).startswith("sft-") and last_role != "assistant":
            rep.error(f"{src}: sft record must end with an assistant turn")
        if (is_pref or str(rid).startswith("eval-")) and last_role == "assistant":
            rep.error(f"{src}: {str(rid).split('-')[0]} record must not end with an assistant turn")


def check_system_prompts(records: list[dict], rep: Report) -> None:
    prompts: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        msgs = rec.get("messages") or []
        if not msgs or msgs[0].get("role") != "system":
            rep.warn(
                f"{rec['_src']}: no system turn. Train with the system prompt you deploy with, "
                "or vary it deliberately -- an accidental mix teaches the model the prompt is optional."
            )
            continue
        prompts[msgs[0]["content"]].append(rec["_src"])
    if len(prompts) > 1:
        rep.warn(f"system prompt: {len(prompts)} distinct variants across the corpus")
        for text, srcs in sorted(prompts.items(), key=lambda kv: -len(kv[1])):
            rep.info(f"  {len(srcs):>4} record(s): {text[:72]}...")
    elif prompts:
        rep.info("system prompt: single variant across the corpus")


def check_rubrics(records: list[dict], rep: Report) -> None:
    thin = []
    for rec in records:
        rub = rec.get("rubric") or {}
        if not rub.get("must_contain"):
            rep.error(f"{rec['_src']}: rubric.must_contain is empty -- the record is not gradable")
        if not rub.get("must_not_contain"):
            rep.error(f"{rec['_src']}: rubric.must_not_contain is empty -- no failure mode is specified")
        if len(rub.get("must_contain") or []) < 3:
            thin.append(rec["_src"])
    if thin:
        rep.warn(f"rubric: {len(thin)} record(s) with fewer than 3 must_contain criteria: {', '.join(thin[:5])}")


def shingles(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    if len(toks) < SHINGLE:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def prompt_text(rec: dict) -> str:
    return "\n".join(m["content"] for m in rec.get("messages", []) if m.get("role") == "user")


def check_duplicates(records: list[dict], label: str, rep: Report) -> None:
    exact: dict[str, str] = {}
    sigs: list[tuple[str, set[str]]] = []
    for rec in records:
        text = prompt_text(rec)
        key = re.sub(r"\s+", " ", text.strip().lower())
        if key in exact:
            rep.error(f"{rec['_src']}: exact duplicate prompt of {exact[key]} (within {label})")
        else:
            exact[key] = rec["_src"]
        sigs.append((rec["_src"], shingles(text)))

    hits = 0
    for (src_a, sh_a), (src_b, sh_b) in combinations(sigs, 2):
        j = jaccard(sh_a, sh_b)
        if j >= NEAR_DUP_WITHIN:
            rep.warn(f"near-duplicate within {label}: {src_a} ~ {src_b} (Jaccard {j:.2f})")
            hits += 1
    rep.info(
        f"duplicates ({label}): {len(records)} record(s), {hits} near-duplicate pair(s) "
        f"at Jaccard >= {NEAR_DUP_WITHIN}"
    )


def check_contamination(train: list[dict], evalset: list[dict], rep: Report) -> None:
    if not train or not evalset:
        rep.info("contamination: skipped (need both a train and an eval split)")
        return
    tr = [(r["_src"], shingles(prompt_text(r))) for r in train]
    ev = [(r["_src"], shingles(prompt_text(r))) for r in evalset]
    worst = 0.0
    hits = 0
    for src_e, sh_e in ev:
        for src_t, sh_t in tr:
            j = jaccard(sh_e, sh_t)
            worst = max(worst, j)
            if j >= NEAR_DUP_ACROSS:
                rep.error(
                    f"contamination: eval {src_e} ~ train {src_t} (Jaccard {j:.2f}). "
                    "Decontaminate the TRAINING set against the frozen eval set, not the reverse."
                )
                hits += 1
    rep.info(
        f"contamination: {len(ev)} eval x {len(tr)} train prompt(s) compared, "
        f"max Jaccard {worst:.2f}, {hits} violation(s) at threshold {NEAR_DUP_ACROSS}"
    )


def check_mixture(records: list[dict], rep: Report) -> None:
    if not records:
        return
    n = len(records)
    counts = Counter(r.get("category") for r in records)
    level = rep.info if n < MIXTURE_MIN_N else rep.warn
    if n < MIXTURE_MIN_N:
        rep.info(
            f"mixture: n={n} is below {MIXTURE_MIN_N}, so drift from the target table is expected; "
            "reported as notes, not warnings"
        )
    print_rows = []
    for cat, target in sorted(TARGET_MIX.items(), key=lambda kv: -kv[1]):
        share = counts.get(cat, 0) / n
        delta = share - target
        flag = "" if abs(delta) <= MIXTURE_TOLERANCE else "  <-- off target"
        print_rows.append(f"  {cat:<28} target {target:5.1%}  actual {share:5.1%}  ({delta:+.1%}){flag}")
        if abs(delta) > MIXTURE_TOLERANCE:
            level(f"mixture: {cat} is {delta:+.1%} off target ({share:.1%} vs {target:.1%})")
    unknown = set(counts) - set(TARGET_MIX)
    for cat in sorted(unknown):
        rep.error(f"mixture: category {cat!r} is not in the declared target mixture")
    rep.info("mixture table:\n" + "\n".join(print_rows))


def check_numerics(records: list[dict], label: str, rep: Report) -> None:
    with_nums = [r for r in records if (r.get("rubric") or {}).get("numeric_answers")]
    if not records:
        return
    rep.info(
        f"numerics ({label}): {len(with_nums)}/{len(records)} record(s) carry ground-truth numbers "
        f"({len(with_nums)/len(records):.0%} mechanically gradable)"
    )
    for rec in with_nums:
        for i, na in enumerate(rec["rubric"]["numeric_answers"]):
            if not isinstance(na.get("value"), (int, float)):
                rep.error(f"{rec['_src']}: numeric_answers[{i}].value is not a number")
            if not isinstance(na.get("rel_tol"), (int, float)) or na.get("rel_tol", 0) <= 0:
                rep.error(f"{rec['_src']}: numeric_answers[{i}].rel_tol must be a positive number")
            if not str(na.get("unit", "")).strip():
                rep.error(
                    f"{rec['_src']}: numeric_answers[{i}] has no unit. A ground-truth number "
                    "without a unit cannot be checked against an answer that carries one."
                )


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent
    # Flat layout: every *.jsonl beside this script is training data, except the
    # eval_* files. Splitting on the filename keeps a frozen eval set from being
    # swept into training by a careless glob.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--train",
        nargs="*",
        type=Path,
        default=sorted(p for p in root.glob("*.jsonl") if not p.name.startswith("eval_")),
    )
    ap.add_argument("--eval", nargs="*", type=Path, default=sorted(root.glob("eval_*.jsonl")))
    ap.add_argument("--schema", type=Path, default=root / "schema.json")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = ap.parse_args(argv)

    rep = Report()
    train = load(list(args.train), rep)
    evalset = load(list(args.eval), rep)
    allrecs = train + evalset
    if not allrecs:
        rep.error("no records loaded")
        rep.dump()
        return 1

    rep.info(f"loaded {len(train)} train record(s) from {len(args.train)} file(s), {len(evalset)} eval record(s)")

    check_schema(allrecs, args.schema, rep)
    check_ids_and_roles(allrecs, rep)
    check_system_prompts(allrecs, rep)
    check_rubrics(allrecs, rep)
    check_duplicates(train, "train", rep)
    check_duplicates(evalset, "eval", rep)
    check_contamination(train, evalset, rep)
    check_mixture(train, rep)
    check_numerics(train, "train", rep)
    check_numerics(evalset, "eval", rep)

    rep.dump()
    failed = rep.n_errors > 0 or (args.strict and any(lvl == "WARN" for lvl, _ in rep.rows))
    print("FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
