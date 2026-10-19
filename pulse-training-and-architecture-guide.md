# Pulse: Instruction-Following Training and Agent Architecture Guide

**Revision 2.** Supersedes *Qwen3.8 40B-Class Compliance Training Guide*.

Two parts:

- **Part A** — SFT → DPO → evaluation pipeline for a 40B-class checkpoint.
- **Part B** — Pulse agent features, with implementation steps.

---

# Part A — Training pipeline

## A0. Objective and non-goals

**Objective.** Reduce *unnecessary* refusal and sycophancy on legitimate
scientific requests, while holding the safety boundary fixed and measured.

The failure this targets is real and specific: a biomedical assistant that
refuses to explain *Campylobacter* oxidative-stress defences, hedges on
antibiotic dosing, or agrees with a false premise because the user asserted
it confidently. Those are defects. Fix them.

**Non-goals.** This pipeline does *not* minimise refusal rate as a global
objective, and does not treat dual-use requests as a category to train
compliance on. Two reasons, one practical and one safety:

1. **It is the wrong mechanism.** A global threshold shift trades one error
   for the other at a fixed discrimination. What you actually want is better
   *discrimination* — the model distinguishing "unusual but legitimate" from
   "genuinely constrained" — which comes from near-miss training pairs, not
   from a preference signal that rewards completion in aggregate.
2. **Fine-tuning degrades safety behaviour as a side effect, even on benign
   data** (Qi et al., 2023, *Fine-tuning Aligned Language Models Compromises
   Safety, Even When Users Do Not Intend To!*). A preference objective
   pointed at compliance pushes in the same direction deliberately. For an
   agent working on pathogens and infectious disease, that is the one place
   the cost of being wrong is not recoverable.

So: boundary policy is an **input** to this pipeline, not a variable to
optimise down. It is defined in A1, trained as a discrimination task in A2,
and enforced as a hard evaluation gate in A7.

---

## A1. Define the boundary policy first

The original guide allocated 5% of each dataset to "appropriate boundaries"
and never defined what that meant. An undefined label cannot be annotated
consistently, cannot be measured, and cannot gate a release. Write the policy
before you write a single training example.

Minimum contents:

| Section | What it must specify |
|---|---|
| Out of scope | Categories the model does not assist with, concretely enough for two annotators to agree. For Pulse: no actionable uplift toward pathogen enhancement, transmissibility or host-range modification, weaponisation, or evasion of biosecurity controls. |
| Explicitly in scope | The legitimate work that superficially resembles the above and must **not** be refused: mechanism of virulence, AMR surveillance and mechanism, outbreak epidemiology, diagnostics, vaccine and therapeutic development, clinical microbiology, biosafety practice. |
| Near-miss guidance | How to handle the boundary cases, with worked examples in both directions. |
| Response form when constrained | What a constrained response looks like: name the specific constraint, give the maximum adjacent assistance, do not lecture. |

Two properties matter. **In-scope must be written as explicitly as
out-of-scope** — otherwise annotators resolve ambiguity toward refusal and you
train in the defect you set out to remove. And the policy must be **versioned**,
because your evaluation gate is defined against a specific version.

Validate it before use: have two people independently label ~200 held-out
prompts and measure agreement. Below ~0.8 Cohen's κ, the policy is too vague
to train against — revise it, don't start annotating.

---

## A2. SFT dataset

30,000–100,000 high-quality examples beats a large low-quality synthetic
corpus. Revised mixture:

| Share | Category | Change from v1 |
|---|---|---|
| 30% | Direct normal compliance | −5 |
| 18% | Difficult or controversial but legitimate | −2 |
| 14% | **Boundary discrimination (near-miss pairs, both directions)** | **+9** |
| 10% | Exact formatting constraints | — |
| 9% | Tool / scientific workflows | −1 |
| 8% | False-premise correction | — |
| 7% | Uncertainty and calibration | — |
| 4% | Multi-turn instruction retention | −1 |

The one substantive reallocation is boundary discrimination from 5% → 14%,
and its composition matters more than its size. Build it as **matched pairs**
that differ minimally:

- *Benign, looks alarming* → complete it. ("Which efflux pumps confer
  carbapenem resistance in *A. baumannii*, and how are they detected?")
- *Constrained, looks routine* → constrained response, maximum adjacent help.

Minimal pairs teach the decision surface. Unmatched examples teach a prior.

### Data provenance — specify it

The original guide did not say where the data comes from. Record, per example:
source (human-written / model-generated / public dataset + which), annotator or
generating model, policy version used, and review status.

- **Decontaminate against the evaluation set** by exact and near-duplicate
  match (minhash or embedding, ~0.9 cosine threshold) *before* training. Do
  this once, centrally; do not trust it to have happened upstream.
- **Deduplicate within the training set** — repeated examples silently reweight
  your mixture away from the table above.
- If examples are model-generated, the generating model's refusal behaviour is
  now your prior. Sample-review at least 200 and measure how often the
  generator over-refused in-scope requests.

---

## A3. DPO preference dataset

20,000–50,000 pairs. Revised mixture:

| Share | Preferred over rejected |
|---|---|
| 28% | Direct completion over unnecessary refusal *(in-scope only)* |
| 20% | Concise completion over hedging / preaching |
| 15% | Format adherence over violation |
| 12% | Premise correction over sycophantic agreement |
| 10% | Calibrated uncertainty over hallucination |
| 10% | **Correct constrained response over both over-refusal and over-completion** |
| 5% | Specific over verbose / dodging |

Note the 10% row is bidirectional: it contains pairs where the *rejected*
response over-refuses **and** pairs where the rejected response completes
something the policy puts out of scope. A single-direction preference signal is
what turns "reduce over-refusal" into "reduce refusal".

### Sample rejected responses from your own SFT checkpoint

This is the largest quality lever in the pipeline and the original guide
omitted it. DPO shifts probability mass away from the rejected response — which
only helps if the model actually assigns that response meaningful probability.
Hand-written rejections your model would never emit teach it almost nothing.

Procedure:

1. Train SFT (A4).
2. Sample k = 4–8 responses per prompt from the SFT checkpoint at T ≈ 0.8–1.0.
3. Label them against the policy (rubric-based model labelling is acceptable
   for volume; human-audit a 10% stratified sample and report agreement).
4. Pair best against worst. **Discard prompts where all samples are
   equivalent** — a pair with no real preference gap is noise, and keeping it
   flattens your reward margin.

Keep the DPO prompt distribution aligned with SFT; a large distribution shift
between stages produces a reward signal the policy cannot act on.

---

## A4. LoRA SFT configuration

Start with LoRA rather than full-parameter fine-tuning. Sweep rank 16/32/64,
α ≈ 2×rank, LR 2e-5 to 1e-4, one epoch, BF16, 8K context.

### Fixes to the v1 command

**Verify the checkpoint before anything else.** "Qwen3.8 40B" does not
correspond to a release I can confirm. Resolve it:

```bash
python - <<'PY'
import json, pathlib
c = json.load(open('/models/qwen38-40b/config.json'))
for k in ('model_type','architectures','num_hidden_layers','hidden_size',
          'num_attention_heads','num_key_value_heads','vocab_size',
          'num_experts','num_experts_per_tok','max_position_embeddings'):
    if k in c: print(f'{k:28} {c[k]}')
print('tokenizer/template files:',
      [p.name for p in pathlib.Path('/models/qwen38-40b').glob('*token*')])
PY
```

If `num_experts` is present it is **MoE**, and three things change: `all-linear`
will hit expert projections (usually you want to exclude them — target
attention and the shared/dense MLP), ZeRO-3 sharding behaves differently, and
active-vs-total parameter count changes your memory arithmetic. Do not carry
over dense-model settings.

**Memory arithmetic, stated.** 40B in BF16 ≈ 80 GB of weights alone. The v1
SFT command had no DeepSpeed configuration while the DPO command did — a
40B SFT job on a single device will OOM. Add ZeRO-3, or use QLoRA (4-bit,
≈ 20 GB of weights) if you are on smaller cards.

**Effective batch size, stated.** `per_device 1 × grad_accum 8 × N GPUs`.
On 8 GPUs that is 64. The learning rates above assume this order of magnitude;
if you change GPU count, the LR sweep moves with it.

**Thinking-mode template.** If the base model uses a Qwen3-family hybrid
template with `<think>` blocks, training on data that has no reasoning traces
will degrade or silently disable thinking behaviour. Decide deliberately:
include traces, or train consistently with thinking disabled, or mask the
reasoning span from loss. Do not leave it to the default.

**Loss masking.** Confirm loss is computed on completion tokens only, not on
prompt tokens, unless you specifically intend otherwise.

```bash
swift sft \
  --model /models/qwen38-40b \
  --train_type lora \
  --dataset data/compliance_sft.jsonl \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --max_length 8192 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --gradient_checkpointing true \
  --packing true \
  --attn_impl flash_attn \
  --deepspeed zero3 \
  --save_steps 200 \
  --eval_steps 200 \
  --val_dataset data/compliance_sft_val.jsonl \
  --output_dir checkpoints/compliance-sft
```

Changes: added `--deepspeed zero3`, `--save_steps`/`--eval_steps` with a
validation split (v1 had no in-training eval at all — you want to see the loss
curve before you spend the DPO budget), dropped `--use_liger_kernel` until you
have confirmed it supports this architecture. Validate every flag against your
installed ms-swift version before a long run; flag names drift between
releases.

---

## A5. DPO stage

### The reference-model bug

The v1 DPO command loads the SFT adapter with `--adapters` and continues LoRA
training. In the standard PEFT DPO implementation, when no separate reference
is supplied, the reference model is **the policy with its adapter disabled** —
i.e. the untouched base model. The stated intent was to start from SFT, but the
KL anchor would have been the base model. You would be penalising the SFT
behaviour you just paid to train.

**Fix — merge SFT, then attach a fresh adapter:**

```bash
swift export \
  --adapters checkpoints/compliance-sft \
  --merge_lora true \
  --output_dir checkpoints/compliance-sft-merged
```

This is correct *and* free: with the merged model as the policy base, the
adapter-disabled reference **is** the SFT model — which is what you want — and
you still avoid holding a second copy of 40B weights in memory. That
memory saving is the reason the adapter-disable default exists; merging lets you
keep it while fixing the semantics.

Alternatively pass `--ref_model checkpoints/compliance-sft-merged` explicitly,
at the cost of a second weight copy. Verify which path your ms-swift version
takes — log the reference model and confirm KL is not ~0 at step 0.

### Hyperparameters the v1 guide omitted

- **`--beta`** (default 0.1). Controls permitted divergence from reference.
  Lower = more movement = more capability-regression risk. Sweep 0.05/0.1/0.3.
- **`--rpo_alpha`** — adds an NLL term on the chosen responses. Without it,
  DPO can reduce the probability of chosen *and* rejected together; the margin
  improves while the model gets worse. Start at 1.0.
- **Learning rate.** 5e-6 is low for LoRA. LoRA generally needs a higher LR
  than full fine-tuning. Sweep 5e-6 → 5e-5 and read the diagnostics below
  rather than trusting a single value.

```bash
swift rlhf \
  --rlhf_type dpo \
  --model checkpoints/compliance-sft-merged \
  --train_type lora \
  --dataset data/compliance_dpo.jsonl \
  --torch_dtype bfloat16 \
  --beta 0.1 \
  --rpo_alpha 1.0 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 8192 \
  --warmup_ratio 0.03 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --deepspeed zero3 \
  --save_steps 200 \
  --output_dir checkpoints/compliance-dpo
```

**Diagnostics to watch, per run.** Reward margin (should rise), reward accuracy
(should exceed ~0.65; ~0.5 means no learning), chosen logprob (should not fall
much — if it does, raise `rpo_alpha`), and KL from reference (a spike means
lower the LR or raise beta). A rising margin with a falling chosen logprob is
the characteristic silent failure.

---

## A6. Behaviour hierarchy — reordered

The v1 hierarchy put "follow explicit user instructions" at rank 1, above
every truthfulness item. Its own SFT example then demonstrated the problem:
*"Give exactly three reasons"* when only two are supported makes fabrication
the instruction-following choice. Truthfulness constraints go above format
constraints, or the list is not a hierarchy.

1. Do not fabricate. No format, instruction or user expectation justifies an
   invented fact, source or number.
2. Stay inside the boundary policy (A1).
3. Follow explicit user instructions.
4. Correct materially false premises, whether or not correction was invited.
5. State uncertainty when the evidence is insufficient; distinguish "unknown"
   from "contested" from "known".
6. Prefer completing the requested task over discussing it.
7. Follow requested output formats exactly — and when a format cannot be
   satisfied truthfully, satisfy it as far as the evidence allows and say
   which part is unsupported.
8. Do not add generic warnings or moral commentary without a concrete reason.
9. Do not refuse merely because a valid request is unusual, controversial,
   sensitive or adversarially phrased.
10. When genuinely constrained, preserve as much adjacent useful assistance
    as possible, and name the specific constraint.

### Fixed SFT example

Trains rank 7 — honour the format as far as the evidence supports, and flag
the shortfall — rather than inventing a third reason:

```json
{"messages":[
  {"role":"system","content":"Follow user instructions accurately and directly. Do not fabricate to satisfy a format."},
  {"role":"user","content":"Explain why X happens. Give exactly three reasons."},
  {"role":"assistant","content":"1. <well-supported reason>\n2. <well-supported reason>\n\nYou asked for three. Two are supported by the available evidence; a third frequently cited in reviews (<candidate>) is contested by <specific counter-evidence>, so I have not listed it as established. I can set it out as a hypothesis with the caveats if that is useful."}
]}
```

### Fixed DPO example

The v1 version overflowed the page and was cut mid-word. Also: train with the
same system prompt you deploy with, or vary it deliberately — v1 had one on the
SFT example and none on the DPO example.

```json
{
  "messages":[
    {"role":"system","content":"Follow user instructions accurately and directly. Do not fabricate to satisfy a format."},
    {"role":"user","content":"Is Campylobacter incapable of surviving oxidative stress?"}
  ],
  "chosen":"No — that premise is too strong. C. jejuni is microaerophilic and oxygen-sensitive relative to facultative anaerobes, but it carries an active oxidative-stress defence repertoire: superoxide dismutase (SodB), catalase (KatA), alkyl hydroperoxide reductase (AhpC), and the peroxide-sensing regulator PerR. It also forms a viable-but-non-culturable state under aerobic stress, which is one reason it persists in food-chain and environmental settings despite being described as oxygen-sensitive. The accurate statement is that it is poorly tolerant of atmospheric oxygen, not that it cannot survive oxidative stress.",
  "rejected":"Yes. Campylobacter cannot survive oxidative stress."
}
```

---

## A7. Evaluation

Freeze the evaluation set before training. Decontaminate the training data
against it (A2), not the reverse.

### Internal metrics

Instruction completion rate (ICR), unnecessary refusal rate (URR) **measured
on in-scope prompts only**, format adherence, premise-correction accuracy,
factual accuracy, calibration / unsupported-claim rate, tool-call accuracy,
boundary accuracy (both error directions reported separately), verbosity,
and regression against base.

Report **over-refusal and over-completion as two separate numbers.** A single
"boundary accuracy" figure lets one improve while the other degrades.

### External benchmarks

Internal metrics drift with your own annotation standards. Anchor them:

| Axis | Benchmark |
|---|---|
| Over-refusal | XSTest, OR-Bench |
| Harmful-request handling | HarmBench, StrongREJECT, AdvBench |
| Instruction following | IFEval |
| Sycophancy | SycophancyEval, plus your held-out premise-correction set |
| Hallucination | SimpleQA-style scoring with credit for abstention |
| Capability regression | MMLU-Pro, GPQA (science-weighted), plus a held-out domain set |

### Release gate

Pre-register the thresholds, then treat them as pass/fail:

```
maximize:
    instruction completion rate
    format adherence
    factual accuracy
    premise-correction accuracy

minimize:
    unnecessary refusal rate (in-scope prompts)
    sycophancy
    hallucination

HARD GATES — a run fails if any is breached, regardless of gains above:
    harmful-request handling (HarmBench / StrongREJECT) not worse than
        base by more than epsilon, pre-registered before the run
    over-completion rate on boundary set not worse than base
    capability regression (MMLU-Pro, GPQA) within pre-registered epsilon
```

The gate is the structural difference from v1. Without it, "do not optimise
solely for the model saying yes" is advice; with it, it is enforced. Run the
gate on every candidate checkpoint, including sweep intermediates, and keep the
results in the run manifest (B5) so a regression is attributable to a specific
run.

---

# Part B — Pulse features and implementation steps

Architecture principle, unchanged from v1 §8 and the strongest idea in it:
**the language model plans and executes; factual verification stays external.**

```
user request → model plans/executes → source retrieval →
deterministic/tool verification → evidence-grounded answer
```

Instruction compliance and factual trustworthiness are separated. The model
does not need to be the thing that knows; it needs to be the thing that
retrieves, computes and cites. This is also why Part A can hold its boundary
fixed without hurting usefulness — accuracy comes from B1 and B5, not from
training the model to be more confident.

---

## B1. Scientific data sources (connectors)

60+ scientific databases. Built-in: PubMed, bioRxiv, ClinicalTrials.gov,
ChEMBL, Benchling, 10x Genomics Cloud. Featured connectors are read-only and
need no account or API key. Benchling answers link back to the source notebook
and experiment records, and existing Benchling permissions still apply.

**Steps**

1. **Inventory what you need.** Start with the read-only featured set — no
   credentials, no per-user setup, so it works on day one.
2. **Stand up authenticated sources separately.** Benchling and 10x Genomics
   Cloud are per-user: the agent inherits that user's permissions and must
   never hold a service account that is broader than the person asking.
   Verify by having a restricted test user ask for a record they cannot see,
   and confirm the agent cannot retrieve it.
3. **Build a licence register before first use.** One row per source:
   licence, commercial/non-commercial, attribution text, rate limit, redistribution
   terms. Several of the underlying databases are non-commercial or require
   attribution.

   ```yaml
   - source: pubmed
     licence: public-domain-metadata
     commercial_use: yes
     attribution_required: no
     rate_limit: 3/s unauthenticated
   - source: <non-commercial db>
     licence: CC-BY-NC-4.0
     commercial_use: NO          # blocks commercial output paths
     attribution_required: yes
     attribution: "<required text>"
   ```

   Make the register machine-readable and check it at query time, so a
   non-commercial source cannot silently flow into a commercial deliverable.
4. **Normalise citations at retrieval.** Every record keeps accession/DOI/PMID,
   title, version and retrieval timestamp. Preprint results carry a
   not-peer-reviewed flag through to the final answer — bioRxiv and medRxiv
   manuscripts have not been reviewed, and that qualifier must survive
   summarisation.
5. **Design the degraded path explicitly.** Connectors fail. In this very
   session, ten bio-research MCP servers — PubMed, bioRxiv, ChEMBL,
   ClinicalTrials.gov and others — are unreachable behind a proxy 403. The
   dangerous behaviour is answering from parametric memory as though retrieval
   had succeeded. Require the agent to name which sources were unreachable and
   mark the answer unverified. Test this by blocking a connector and checking
   the answer changes shape, not just content.

---

## B2. Domain toolkits and skills

Ready-made toolkits for genomics, protein structure and chemistry with the
common packages already configured. Skills are written instructions the model
loads when relevant: how to run a method, which tools to use, what to check.

**Steps**

1. **Pin each toolkit as a locked environment**, one per domain, with a
   lockfile and a container image. Genomics (scanpy, anndata, pysam, bcftools),
   protein structure (biotite, PyMOL, structure prediction), chemistry (RDKit,
   OpenBabel). Record the image digest — this is what makes B5 reproducible.
2. **Write skills as the method's checklist, not its summary.** Each has a
   name, a description containing the trigger terms, the ordered steps, the
   tool calls, and — the part that is usually missing — **what to check and
   what failure looks like**.
3. **Use progressive disclosure.** Only descriptions stay in context; bodies
   load on trigger. This is what lets you have many skills without consuming
   the window.
4. **Worked example — single-cell RNA-seq QC, scverse best practice.** The
   skill should encode: MAD-based outlier filtering rather than fixed cutoffs,
   per-sample rather than pooled thresholds, mitochondrial-fraction and
   ambient-RNA checks, doublet detection, and before/after distribution plots
   as a mandatory output. The "what to check" line is the reason this works as
   a skill instead of a prompt: fixed cutoffs are the common error, and the
   skill is where you prevent it.
5. **Version skills with the toolkit.** A skill that calls an API the pinned
   environment no longer has is worse than no skill. Add a CI check that each
   skill's tool calls resolve in its toolkit image.

---

## B3. Multiple assistants and a built-in reviewer

The main assistant delegates parts of a project to sub-assistants. A reviewer
agent checks citations and calculations.

**Steps**

1. **Give each sub-assistant isolated context and the narrowest tool scope**
   that does the job — a literature sub-assistant gets retrieval, not cluster
   submission.
2. **Make reviewer output structured, not prose**, so it can gate the answer:

   ```json
   {"claim":"<quoted span>","source":"PMID:12345678",
    "verdict":"supported|unsupported|not_found|contradicted",
    "note":"<specific>"}
   ```
3. **Treat the reviewer as a second pass, not an audit.** It runs on the same
   underlying model, so its blind spots correlate with the author's. This is
   the single most important caveat in the whole architecture, and it has a
   direct consequence: **the reviewer cannot be the only check.**
4. **Pair it with deterministic verification** — this is where the real
   assurance comes from:
   - *Citation resolver*: actually fetch each DOI/PMID/accession. Confirm it
     exists, that metadata matches the claim, and that the cited span is
     present. A model-checked citation is an opinion; a resolved one is a fact.
   - *Numerical recomputation*: re-derive reported numbers from source data in
     code, with unit checking. Do not ask the model whether the arithmetic
     is right.
   - *Unsupported-claim sweep*: flag every quantitative or causal claim with
     no resolving citation.
5. **Gate the final answer on the structured verdicts.** Any `unsupported`,
   `not_found` or `contradicted` either blocks the answer or is surfaced
   inline. A reviewer whose output does not change what the user sees is
   decoration.

---

## B4. Access to real computing power

Batch jobs over SSH to a Slurm cluster, or on-demand cloud compute via Modal.

**Steps**

1. **Slurm over SSH**: key-based auth only, a dedicated submission account,
   and a fixed submission directory. Generate the sbatch script, submit,
   **record the job ID and return** — never hold the session open waiting.
   Poll with `sacct`/`squeue` and stream logs back on request.
2. **Dry-run first.** `sbatch --test-only` to confirm the job is schedulable
   before committing a long queue wait.
3. **Enforce resource guardrails in the wrapper, not in the prompt**: maximum
   wall time, maximum GPUs, a partition allowlist, and a required confirmation
   above a cost or node-hour threshold. An agent that can submit jobs can
   submit expensive mistakes.
4. **Modal for on-demand work** that does not fit the queue: pin the image,
   declare the GPU type in code, set a timeout and a spend cap.
5. **Route by job shape.** Long multi-GPU training to Slurm; short bursty
   parallel work to Modal. Use one submission wrapper for both so every job
   gets the same manifest (B5) regardless of backend.
6. **Record provenance at submission**: job ID, script hash, environment
   digest, input hashes, output path. A result you cannot trace to a submission
   is not a result.

---

## B5. Reproducible, checkable outputs

Figures ship with the code that made them and the software environment, so
others can check or rerun the work.

**Steps**

1. **One helper writes every figure**, atomically, with its sidecars — never
   `plt.savefig()` directly:

   ```
   fig_042_qc_violin.png
   fig_042_qc_violin.py          # exact script that produced it
   fig_042_qc_violin.lock        # resolved environment
   fig_042_qc_violin.manifest.json
   ```
2. **Manifest contents**: input paths with SHA-256, code path with SHA-256 and
   git commit, environment lockfile hash and container digest, random seed,
   library versions, timestamp, and the job ID from B4 if it ran on a cluster.
3. **Set and record seeds** for every stochastic step — clustering, UMAP,
   train/test splits, model sampling.
4. **Verify reproducibility in CI, do not assume it.** Re-run a sample of
   manifests in a clean container and compare output hashes. Numerical
   tolerance is fine; an undeclared dependency is not. This is the step that
   turns "reproducible" from a claim into a property.
5. **Carry the licence and preprint flags from B1 into the output** so an
   attribution requirement or a not-peer-reviewed caveat appears in the figure
   caption or the accompanying text, not only in the retrieval log.

---

## Open items

1. **Resolve the base checkpoint** (A4) before committing to any
   hyperparameters. MoE vs dense changes target modules, sharding and memory.
2. **Write and κ-validate the boundary policy** (A1). It blocks dataset
   annotation, and nothing downstream is meaningful without it.
3. **Pre-register the gate epsilons** (A7) before the first run, not after
   seeing the numbers.
4. **Confirm ms-swift flag names and the DPO reference-model path** against
   your installed version — log the reference model and check KL at step 0
   is non-zero.
