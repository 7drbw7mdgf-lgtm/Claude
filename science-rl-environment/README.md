# Science RL environment — verifiable-answer problems with programmatic graders

A reinforcement-learning environment for scientific quantitative reasoning. Every
problem has an answer a program can check, so the reward comes from a grader
rather than a rubric or a judge model.

This is the companion to `../science-reasoning-dataset/`, and it exists because
that corpus cannot do this job. SFT and DPO on 20k–100k examples shape *how* a
model answers within the distribution it already has. They do not make it better
at multi-step quantitative reasoning, because nothing in the objective rewards
getting the number right. Here the objective is exactly that.

One flat folder, no subdirectories:

```
README.md       this file: design, reward, integration, limitations
units.py        dimensional-analysis engine for unit-aware grading
families.py     13 parameterised problem generators with closed-form solutions
grader.py       answer extraction, unit-aware comparison, reward
env.py          batch sampling, dataset emit/verify, rollout grading (CLI)
selftest.py     validates the environment itself — run before every job
```

Current state: `python3 selftest.py` → **PASS**, 53,815 assertions in 2.8 s;
`--seeds 150` → **PASS**, 134,453 assertions. Stdlib only, no dependencies.

```bash
python3 selftest.py                                   # validate the environment
python3 env.py demo --family buffer_ph                # see a problem and its grading
python3 env.py stats                                  # problem-space size per family
python3 env.py emit --n 50000 --split train --out train.jsonl
python3 env.py grade --problems train.jsonl --completions rollouts.jsonl
```

---

## 1. Why generators rather than a problem set

Each family is a function of a seed, not a list of items. A fixed set of a few
thousand science problems is memorisable, and under RL memorisation is the
cheapest policy that collects the reward — so a static set measures recall after
the first epoch, not reasoning.

`env.py stats` reports the distinct-prompt count per family and flags any family
whose count saturates below the seed count. That check earned its place
immediately: the first version of `power_sample_size` had **16** distinct
problems and `allometric_dose` had **32**, because both sampled from short
`rng.choice` lists. Both now sample from continuous grids and exceed 99% distinct
over 2,000 seeds.

Difficulty is a knob (`easy`/`medium`/`hard`) rather than a curation job. Harder
instances add irrelevant givens — a distractor is a cheap way to test whether the
policy selects the quantities it needs rather than consuming everything present.

## 2. What a problem carries

```python
Problem(
  family, difficulty, seed,
  context, givens, question,       # prompt is assembled from these
  targets,    # reference answer(s): value + unit + relative tolerance
  trap,       # the answer a specific named wrong method produces
  solution,   # reference working, not shown to the model
  insufficient,  # set when a required quantity was removed
)
```

The generator knows the answer because it chose the inputs, which is what makes
the reward verifiable without a solver.

**Prompts are assembled from a `givens` list, not written as prose.** That is
what makes the `insufficient` wrapper sound: it removes one *required* given
mechanically, leaving a genuinely unanswerable problem. Independence of the
required givens is ensured by construction per family, not proved — a
hand-written prose prompt could not support this at all.

## 3. Units are graded dimensionally

A science answer is a number *and* a unit, and `4.7 mM` versus `4.7 M` is a
thousand-fold error, not a formatting quibble. String comparison cannot do this:
`mg/mL` and `g/L` are identical, `mM` and `mmol/L` are identical, and `mol` where
`mol/L` was asked is wrong. So `units.py` reduces unit strings to SI base
dimensions and compares there.

Conventions that differ from strict SI, chosen to match how quantities get
written in a lab: `M` is molar (so the prefix in `mM` works), `rpm` reduces to
s⁻¹, and **relative centrifugal field is `xg`** because plain `g` is grams —
`x g`, `× g` and `g-force` are normalised on input.

Unknown units fail closed rather than raising, so a model that invents a unit
scores zero instead of ending the rollout.

## 4. Reward design

Three outcomes are tracked separately, because for RL they are different signals
and one scalar throws away the diagnosis:

| Flag | Meaning |
|---|---|
| `format_ok` | an answer was found and parsed at all |
| `unit_ok` | the unit is dimensionally what was asked for |
| `value_ok` | the magnitude is within the target's relative tolerance |
| `trap_hit` | the answer matches the designed wrong method |
| `abstained` | the model declined to give a number |

**The default reward is binary**: 1.0 when every target matches in both value and
unit, 0.0 otherwise. `--shaped` adds partial credit for a correct unit and
correct order of magnitude, and `--format-bonus` gives small credit for a
parseable answer. Both are off by default, deliberately: every shaped term is a
reward-hacking surface, and order-of-magnitude credit pays for a guess that lands
in the right decade. Use `--format-bonus` early if the policy cannot yet produce
the answer contract at all and the reward is a flat zero; drop it once format
compliance is above roughly 90%, or you are paying for formatting indefinitely.

### Trap answers are the diagnostic that matters

Every family defines the answer a specific wrong procedure produces — Km + [S]
replaced by Km, the acid/base ratio inverted, relative errors added linearly
instead of in quadrature, rpm transferred between rotors without the radius
correction, sensitivity reported as positive predictive value. Traps score zero
like any miss, but they are logged, and `env.py grade` reports the per-family
trap rate.

This is the most informative number the environment produces. A reward curve
tells you the policy is wrong; a 40% trap rate on `uncertainty_propagation` tells
you it is adding relative errors linearly. That names the defect, and `trap_note`
says what it is.

A trap can accidentally coincide with the correct answer — at high prevalence the
PPV really does approach the sensitivity. `families.sample()` redraws while that
holds, deterministically, and drops the trap rather than scoring a wrong method
as correct if 64 redraws fail. The self-test caught this case too.

### Unanswerable problems are 10% of every batch

`insufficient_rate` defaults to 0.10 and **should not be set to zero**. With every
problem answerable, always emitting a number is the reward-maximising policy, and
the model's ability to recognise an underdetermined problem is trained away — the
exact calibration regression that makes an RL-tuned model confidently invent a
value for a quantity it was never given.

On these instances abstention scores 1.0 and a confident number scores 0.0.
Several phrasings are accepted (`INSUFFICIENT`, "not enough information",
"cannot be determined"), and the grader records whether the model named the
missing quantity. Abstaining on an *answerable* problem scores 0.0, so the
pressure runs both ways — the same both-directions principle as the preference
pairs in the sibling corpus.

## 5. Integration

A problem is a pure function of `(family, seed, difficulty, insufficient)`, so an
emitted record needs only those four fields to regenerate exactly. Two
consequences: datasets are cheap to ship, and `env.py verify` can regenerate
every record and confirm the stored answer still matches. That catches a
generator change silently invalidating a dataset you are still training on —
records carry `generator_version`, and `from_record` raises rather than
mis-grading.

In-process:

```python
from env import make_batch, grade
batch = make_batch(1024, split="train")
prompts = [p.render() for p in batch]
rewards = [grade(p, c).reward for p, c in zip(batch, rollouts)]
```

Out-of-process: `emit` to JSONL, run rollouts, `grade` a
`{"id":..., "completion":...}` file. Grading only reads the final `ANSWER:` line,
so the policy may use tools, code execution or long chains of thought freely —
the environment is agnostic to how the answer was reached.

**Train and eval seeds are disjoint by construction**: eval seeds start at
`EVAL_SEED_BASE = 1_000_000`, one constant shared by the emitter, the grader and
the self-test. The self-test asserts no prompt appears in both ranges — the check
that failed on the first run and exposed the narrow-problem-space bug.

## 6. The self-test is the load-bearing part

An RL environment with a wrong grader is worse than no environment: it trains the
policy on noise, confidently, while the loss curve looks healthy. `selftest.py`
is the closest thing available to a proof that the reward means what it claims.
Per family, per difficulty, over many seeds:

- **reference-accept** — the generator's own answer scores 1.0. Catches a
  generator and grader that disagree, the defect that silently inverts the reward.
- **perturb-reject** — the answer × 1.5 scores 0.0.
- **tolerance-edge** — just inside tolerance scores 1.0, 3× tolerance scores 0.0.
- **unit-required** — right number, dimensionally wrong unit → 0.0, `unit_ok` false.
- **no-unit-rejected** — a bare number for a dimensioned answer → 0.0.
- **format-detected** — no answer line → 0.0 with `format_ok` false, distinct
  from a wrong answer.
- **trap-scored-zero** — the designed wrong method → 0.0 with `trap_hit` set.
- **abstain-penalised** — abstention on an answerable problem → 0.0.
- **determinism** — a fixed seed renders identically and yields identical targets.
- **seed-split** — train and eval prompt sets do not intersect.

Plus a pass over unanswerable instances, the unit engine's equivalences and
distinctions, and number parsing (`1.5e-3`, `1.5 × 10⁻³`, `1,200`, `25.0 +/- 0.7`,
last-marker-wins, `\boxed{}` precedence).

It found three real defects on its first run: two narrow problem spaces and one
trap that coincided with the truth. That is the expected yield, and the reason to
run it before every job rather than once.

## 7. Known limitations

- **Thirteen families, all single-step or two-step.** This trains arithmetic
  discipline and unit hygiene, not the long multi-step reasoning that drives
  frontier science performance. A real curriculum needs families with 5–20
  dependent steps, where a mid-chain error propagates — and those are much harder
  to generate with a closed-form reference answer.
- **Verifiable ≠ representative.** Problems with a single checkable number are a
  biased sample of science. Design critique, mechanism, evidence appraisal and
  calibration under genuine uncertainty are most of real scientific reasoning and
  none of it is gradable this way. That is what the sibling rubric corpus covers,
  and why neither artefact replaces the other.
- **Reward hacking is bounded but not eliminated.** The answer contract can be
  satisfied by guessing within tolerance for families whose answers cluster, and
  a policy could learn the distribution of answers per family rather than the
  method. Mitigations in place: wide continuous sampling, tolerances at 1–2%, and
  trap-rate monitoring. Not measured yet: whether a policy trained here improves
  on held-out problems from families it never saw.
- **No partial-credit signal for correct method with arithmetic slip.** Binary
  reward treats a sign error and a complete misunderstanding identically. This is
  the right default for correctness but it makes the gradient sparse early; the
  `--format-bonus` and `--shaped` options are the available mitigations and both
  have the costs described in section 4.
- **The `insufficient` wrapper assumes given-independence.** Required givens are
  marked by hand per family. If a dropped quantity were derivable from the
  remainder, the instance would be answerable and abstention would be wrongly
  rewarded. Checked by construction, not proved.
- **Tolerances are judgement calls.** 1% for exact arithmetic, 2% for
  propagated uncertainty, 5% for sample-size formulas where rounding conventions
  differ. These are defensible, not derived.
- **No solution-trace supervision.** The `solution` field holds reference working
  and is currently unused by the reward. Using it for process supervision —
  rewarding the derivation, not just the answer — is the obvious next step and is
  not built.
