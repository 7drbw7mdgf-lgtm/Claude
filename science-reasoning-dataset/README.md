# Science-reasoning training set

A corpus specification and a hand-written seed set for post-training a model on
**scientific reasoning** — the quantitative and inferential competences, not
domain recall.

This is the dataset layer for `../pulse-training-and-architecture-guide.md`.
That guide specifies the pipeline (§A2 SFT mixture, §A3 preference pairs, §A7
evaluation gate) and deliberately leaves the data unspecified. This directory is
the data: the taxonomy, the record schema, the provenance rules, the grading
protocol, and 34 worked records to annotate the rest against.

```
schema.json              record schema (JSON Schema draft-07)
seed/sft_core.jsonl      14 SFT records, core reasoning skills
seed/sft_calibration.jsonl  6 SFT records, calibration / appraisal / computation
seed/dpo_pairs.jsonl      8 preference pairs
eval/held_out.jsonl       6 frozen evaluation prompts (no gold completions)
tools/validate.py         schema, duplicate, contamination and mixture checks
```

Current state: `python3 tools/validate.py` → **PASS**, 0 errors, 0 warnings.

---

## 1. What this trains, and what it does not

**Target competences.** Getting the arithmetic right with units; propagating
uncertainty; separating statistical significance from magnitude; noticing when a
design cannot answer the question asked of it; correcting a false premise that
arrives with social pressure attached; and saying "I don't know, here is how to
find out" instead of producing a plausible number.

**Not target competences.** Factual recall, and citation. Those belong to
retrieval and to the deterministic verification layer (guide §B1, §B3, §B5). The
architectural premise is that the model is the thing that *reasons, retrieves and
computes*, not the thing that *knows* — so several records train the model to
decline a number it cannot source and route to retrieval instead. A corpus that
rewarded confident recall would train directly against that architecture.

The sharpest instance is `sft-calib-0016`: the correct answer to "what is the Ki
of staurosporine against CDK2" is a refusal to state one, with a specific reason
and a route to ChEMBL. Note that two records reference connectors being
unreachable in the session where they were written; genericise those lines before
reusing the corpus elsewhere (§7).

---

## 2. Taxonomy and target mixture

Ten categories. Shares are the **target for the full corpus**, not for the seed
set — `tools/validate.py` reports drift and, below n = 500, treats it as
expected rather than as a warning.

| Share | Category | What it trains |
|---|---|---|
| 16% | `quantitative-inference` | units, uncertainty propagation, scaling, physical plausibility bounds |
| 16% | `statistical-reasoning` | power, multiplicity, regression to the mean, interval interpretation |
| 13% | `experimental-design` | controls, confounding, blocking, unit of replication |
| 12% | `data-interpretation` | aggregation reversal, selection effects, base rates |
| 11% | `calibration-and-abstention` | known vs contested vs unknown; abstain and route |
| 10% | `false-premise-correction` | correct the premise, including under social pressure |
| 8% | `derivation` | derive the result *and* state its domain of validity |
| 8% | `evidence-appraisal` | study design, surrogate endpoints, preprint status |
| 4% | `computation-workflow` | compute in code, with assertions that can fail |
| 2% | `constraint-retention` | hold a format across turns, and when truth outranks it |

Two design choices worth stating explicitly, because they differ from the
obvious allocation:

**`derivation` asks for the domain of validity, not the derivation.** A model
that reproduces `SE = σ/√n` has memorised a result. A model that can say *which
assumption it rests on and what happens when that fails* has the competence. In
`sft-deriv-0011` the derivation is four lines and the failure analysis — correlated
observations, pseudoreplication, infinite variance, systematic error — is the rest
of the record. Grade the second part.

**`computation-workflow` is small but load-bearing.** Its records require
runnable code whose assertions encode the specific error the calculation invites
(an inverted Henderson–Hasselbalch ratio; a radius-dependent RCF). The point is
not that the model can write Python — it is that the check is *in* the artefact,
so a wrong answer fails loudly instead of being asserted fluently.

---

## 3. Record schema

See `schema.json`. Every record carries:

- `id` — `(sft|dpo|eval)-<slug>-<4 digits>`. The prefix is load-bearing:
  `validate.py` enforces that only `dpo-*` records carry `chosen`/`rejected`,
  that `sft-*` records end with an assistant turn, and that `dpo-*` and `eval-*`
  records do not.
- `category`, `skill`, `difficulty` — `skill` is finer-grained than `category`
  and exists so evaluation can be sliced by competence rather than by topic.
- `messages` — including the system turn. **Use the system prompt you deploy
  with** (guide §A6): one variant across the corpus, or a deliberate set. The
  validator reports how many distinct ones it finds.
- `rubric.must_contain` / `must_not_contain` — the gradable content, written as
  independently markable properties. `must_not_contain` names the *specific*
  failure this record is built to penalise, not generic badness.
- `rubric.numeric_answers` — ground truth with units and relative tolerance,
  where grading can be mechanical rather than judgemental. 57% of seed train
  records carry these.
- `provenance` — source, spec version, review status. `origin_detail` is
  **required** for anything model-generated or dataset-derived, and the
  validator fails without it.

### Why rubrics rather than only gold completions

A gold completion scores a model by similarity to one way of being right. These
prompts have many. The rubrics let an eval ask *did the answer do the six things
that make it correct*, and — more usefully — *did it do the one thing that makes
it wrong*, which similarity scoring cannot see. `must_not_contain` is where most
of the signal is.

---

## 4. Preference pairs: what makes a pair teach anything

Eight pairs in `seed/dpo_pairs.jsonl`. Three rules, from guide §A3:

**1. The rejected response must be one the model would actually produce.** DPO
moves probability mass away from the rejected response; if the model assigns it
no mass, the gradient teaches nothing. So the rejected responses here are *good*
— fluent, structurally sensible, often numerically correct. `dpo-quant-0107`'s
rejected response has flawless arithmetic and is rejected only for not checking
whether the input was physically possible. `dpo-verbose-0104`'s contains no false
statement at all and is rejected purely for hedging, unranked causes and a
non-answer close.

> In production, **sample the rejected responses from your own SFT checkpoint**
> (k = 4–8 at T ≈ 0.8–1.0, label against the rubrics, pair best against worst,
> discard prompts where all samples are equivalent). The hand-written rejections
> here are scaffolding for the annotation standard, not a substitute for that
> step.

**2. Pairs must run in both directions.** `dpo-refusal-0105` and
`dpo-refusal-0106` are matched deliberately: the first rejects an *over-refusal*
of a routine AMR-surveillance question, the second rejects an *over-completion*
that projects a human response rate from a mouse xenograft. A single-direction
signal turns "reduce unnecessary refusal" into "reduce refusal", and "be
appropriately cautious" into "hedge everywhere". Both failures are visible in
deployed models and both come from one-sided preference data.

**3. The difference between chosen and rejected must be the thing you are
training.** Not length, not formatting, not politeness. Where the rejected
response is longer or shorter for reasons unrelated to the lesson, the pair
teaches length.

**Caution on `dpo-calib-0102` and `dpo-refusal-0106`:** their rejected responses
contain fabricated figures and a fabricated-looking citation, by construction.
Those strings are training targets for *rejection* and must never be lifted as
source material. The `reviewer_note` on each says so; keep that note attached.

---

## 5. Evaluation

`eval/held_out.jsonl` is **frozen**. Six prompts, no gold completions — rubrics
and ground-truth numbers only. Decontaminate the *training* set against it, never
the reverse; `validate.py` enforces this at Jaccard ≥ 0.50 on 5-gram shingles of
the user turns, stricter than the 0.70 it uses for within-split redundancy,
because contamination is the more expensive error.

Each eval record has a **discriminating item** named in its `reviewer_note` — the
criterion that separates a model which reasoned from one that pattern-matched:

| Record | Pattern-match answer | Discriminating item |
|---|---|---|
| `eval-quant-9001` | propagates in quadrature correctly | notices a shared calibration offset *cancels* in a ratio |
| `eval-stats-9002` | "n is small" | identifies optional stopping, and the upward bias in the effect estimate |
| `eval-design-9003` | "use a mixed model" | says the correct analysis will likely be inconclusive at n = 3 |
| `eval-premise-9004` | "a CI is about the procedure" | concedes the numerical coincidence under a flat prior without conflating the two |
| `eval-calib-9005` | declines to give the number | spots that "most recently approved" is itself unverifiable |
| `eval-interp-9006` | recites the base-rate fallacy | computes PPV ≈ 1.9% and frames it per 100,000 screened |

Grading protocol, in order of preference:

1. **Mechanical**, where `numeric_answers` exists: extract the quantity, compare
   within `rel_tol`, require the unit. Cheapest and the least drift.
2. **Rubric-based model grading** for `must_contain` / `must_not_contain`, one
   criterion per call, reporting per-criterion rates rather than a single score.
   A collapsed score hides which competence moved.
3. **Human adjudication** on a stratified sample, with agreement reported. Guide
   §A1 sets the bar: below ~0.8 Cohen's κ the criteria are too vague to train
   against — fix the rubric, don't average over the disagreement.

Report **over-refusal and over-completion separately**, per guide §A7. One number
lets the two trade off invisibly, which is exactly the failure this corpus is
built to catch.

---

## 6. Scaling the seed to a full corpus

The seed set's job is to fix the annotation standard. To scale:

1. **Write the near-miss pairs first.** The matched pairs in §4 are the expensive
   part and the part that teaches the decision surface. Unmatched examples teach
   a prior.
2. **Generate candidates, then adjudicate.** Model-generated prompts are fine;
   model-generated *rubrics* are not — the rubric is the ground truth, and a
   rubric written by the model being graded inherits its blind spots. Set
   `provenance.source` honestly and fill `origin_detail`; the validator fails
   without it.
3. **Keep the preference prompt distribution aligned with the SFT distribution**
   (guide §A3). A large shift between stages produces a reward signal the policy
   cannot act on.
4. **Re-run `tools/validate.py --strict` before every job.** Deduplicate within
   the training set too: repeated examples silently reweight the mixture away
   from the table in §2, so the mixture you trained on is not the one you
   declared.
5. **Grow `experimental-design` and `derivation` first** — the validator flags
   both as under-target in the seed set, and they are the two categories where
   good records are hardest to write, so budget for them early.

---

## 7. Known limitations

- **n = 34.** This is an annotation standard, not a training corpus. §2's shares
  are aspirational at this size and the validator says so.
- **Single author, single pass.** Every record is `human-written` and marked
  `reviewed` or `adjudicated` by its author. No independent annotator has labelled
  these, so the κ validation that guide §A1 requires has not been done. Do that
  before scaling: it is the step that tells you whether the rubrics are writable
  by someone other than their author.
- **Two records are environment-specific.** `sft-calib-0016` and `dpo-calib-0102`
  name connectors as unreachable behind a proxy 403, which was true in the session
  where they were written. Genericise to "if retrieval is unavailable, say so and
  mark the answer unverified" before reuse.
- **Domain skew.** Life-sciences-heavy, reflecting the Pulse target domain, with
  some physical-chemistry and general statistics. The reasoning skills are
  domain-general; the surface content is not, and a model trained only on this
  will have seen those skills in one dress.
- **No multi-turn records yet.** `constraint-retention` is at 2% with a single
  single-turn record. Constraint *retention* properly needs several turns with an
  intervening distraction, which none of these have.
- **No grading harness.** §5 specifies the protocol; the code that runs it
  against model outputs is not here. The mechanical tier is the cheapest to build
  and should come first.
