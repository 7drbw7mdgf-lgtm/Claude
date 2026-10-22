"""Parameterised problem families with closed-form reference solutions.

Each family is a generator, not a fixed item list. One family yields an
unbounded stream of instances from a seed, which matters for RL in two ways:
memorising the training set is not a shortcut to reward, and difficulty is a
knob rather than a curation job.

Every instance carries:

  targets     the reference answer(s), computed in closed form from the sampled
              inputs. This is what makes the reward verifiable: the generator
              knows the answer because it chose the inputs.
  trap        the answer produced by a specific, named wrong method. Scored as
              a miss like any other, but logged separately -- trap-hit rate is
              the most informative diagnostic in the whole environment, because
              it says *which* wrong procedure the policy has learned.
  solution    the reference working. Not shown to the model; used for debugging
              graders and for distilling traces if you want SFT data later.

Prompts are assembled from a `givens` list rather than written as prose, so the
`insufficient` wrapper can remove a required quantity mechanically and leave a
genuinely unanswerable problem. Independence of the required givens is ensured
by construction per family, not proved.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import NormalDist

# ---------------------------------------------------------------- data model


@dataclass(frozen=True)
class Target:
    """One reference answer. `rel_tol` is relative, and chosen per family: tight
    where the arithmetic is exact, looser where rounding conventions differ."""

    key: str
    value: float
    unit: str
    rel_tol: float = 0.01


@dataclass(frozen=True)
class Given:
    label: str
    value: str
    required: bool = True


@dataclass
class Problem:
    family: str
    difficulty: str
    context: str
    givens: list[Given]
    question: str
    targets: list[Target]
    solution: str
    trap: Target | None = None
    trap_note: str = ""
    insufficient: str | None = None
    seed: int = -1
    meta: dict = field(default_factory=dict)

    def render(self) -> str:
        lines = [self.context, "", "Given:"]
        for g in self.givens:
            lines.append(f"- {g.label}: {g.value}")
        lines += ["", self.question, ""]
        if len(self.targets) == 1:
            lines.append("Give your final answer on the last line, exactly in this form:")
            lines.append("ANSWER: <value> <unit>")
        else:
            order = "; ".join(f"<{t.key}>" for t in self.targets)
            lines.append(
                f"Give your final answer on the last line as {len(self.targets)} "
                f"values in this order ({order}), exactly in this form:"
            )
            lines.append("ANSWER: " + "; ".join("<value> <unit>" for _ in self.targets))
        lines.append(
            "If the information given is insufficient to compute the answer, write instead:"
        )
        lines.append("ANSWER: INSUFFICIENT <the missing quantity>")
        return "\n".join(lines)


# ------------------------------------------------------------------- helpers

# Bumped whenever a generator changes in a way that alters the answer for a
# given seed. Stamped into emitted records so a stale dataset is detectable
# rather than silently mis-graded.
GENERATOR_VERSION = "1.0"

# Train seeds live below this, eval seeds at or above it. A single constant, so
# the split cannot drift between the emitter, the grader and the self-test.
EVAL_SEED_BASE = 1_000_000

COMPOUNDS = [
    ("glucose", 180.16), ("sucrose", 342.30), ("Tris base", 121.14),
    ("sodium chloride", 58.44), ("EDTA (free acid)", 292.24), ("glycine", 75.07),
    ("HEPES", 238.30), ("DTT", 154.25), ("imidazole", 68.08), ("urea", 60.06),
]


def _g(x: float, sig: int = 4) -> str:
    """Format a number for a prompt without leaking precision cues."""
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e5:
        s = f"{x:.{sig}g}"
    else:
        s = f"{x:.{sig}g}"
    return s


def _pick_round(rng: random.Random, lo: float, hi: float, step: float) -> float:
    n = int(round((hi - lo) / step))
    return round(lo + step * rng.randint(0, n), 10)


# ------------------------------------------------------------------ families


def molarity_from_mass(rng: random.Random, difficulty: str) -> Problem:
    name, mw = rng.choice(COMPOUNDS)
    conc_mM = _pick_round(rng, 2, 250, 0.5)
    vol_mL = _pick_round(rng, 25, 1000, 5)
    mass_mg = conc_mM * 1e-3 * (vol_mL * 1e-3) * mw * 1e3  # mM -> M, mL -> L, g -> mg
    givens = [
        Given("Compound", name),
        Given("Molecular weight", f"{mw} g/mol"),
        Given("Target concentration", f"{_g(conc_mM)} mM"),
        Given("Final volume", f"{vol_mL} mL"),
    ]
    if difficulty == "hard":
        givens.append(Given("Balance readability", "0.1 mg", required=False))
        givens.append(Given("Storage temperature", f"{rng.choice([-20, 4, 25])} C", required=False))
    return Problem(
        family="molarity_from_mass",
        difficulty=difficulty,
        context=f"You are preparing an aqueous solution of {name}.",
        givens=givens,
        question=f"What mass of {name} do you weigh out?",
        targets=[Target("mass", mass_mg, "mg", 0.01)],
        solution=(f"m = c x V x MW = {conc_mM} mM x {vol_mL} mL x {mw} g/mol "
                  f"= {conc_mM*1e-3:.6g} mol/L x {vol_mL*1e-3:.4g} L x {mw} g/mol "
                  f"= {mass_mg/1e3:.6g} g = {mass_mg:.6g} mg"),
        trap=Target("mass", mass_mg * 1000, "mg", 0.01),
        trap_note="volume left in mL (factor 1000)",
        meta={"mw": mw, "conc_mM": conc_mM, "vol_mL": vol_mL},
    )


def dilution(rng: random.Random, difficulty: str) -> Problem:
    stock = _pick_round(rng, 10, 1000, 2.5)  # mM
    fold = rng.choice([4, 5, 8, 10, 20, 25, 40, 50])
    target = stock / fold
    final_mL = _pick_round(rng, 5, 100, 0.5)
    v_stock_mL = target * final_mL / stock
    givens = [
        Given("Stock concentration", f"{_g(stock)} mM"),
        Given("Required concentration", f"{_g(target)} mM"),
        Given("Required final volume", f"{final_mL} mL"),
    ]
    if difficulty != "easy":
        givens.append(Given("Diluent", "buffer, pH 7.4", required=False))
    return Problem(
        family="dilution",
        difficulty=difficulty,
        context="You are diluting a concentrated stock.",
        givens=givens,
        question="What volume of stock do you need?",
        targets=[Target("stock_volume", v_stock_mL, "mL", 0.01)],
        solution=(f"C1V1 = C2V2 -> V1 = C2 V2 / C1 = {_g(target)} x {final_mL} / {_g(stock)} "
                  f"= {v_stock_mL:.6g} mL"),
        trap=Target("stock_volume", stock * final_mL / target, "mL", 0.01),
        trap_note="C1/C2 inverted",
        meta={"fold": fold},
    )


def serial_dilution(rng: random.Random, difficulty: str) -> Problem:
    step = rng.choice([2, 3, 4, 5, 10])
    n = rng.randint(3, 10)
    start = _pick_round(rng, 0.5, 500, 0.5)  # uM
    final = start / step ** n
    givens = [
        Given("Starting concentration", f"{_g(start)} uM"),
        Given("Dilution at each step", f"1:{step}"),
        Given("Number of serial steps", f"{n}"),
    ]
    if difficulty == "hard":
        givens.append(Given("Volume transferred each step", f"{rng.choice([10, 20, 50])} uL",
                            required=False))
    return Problem(
        family="serial_dilution",
        difficulty=difficulty,
        context="You perform a serial dilution, carrying forward at each step.",
        givens=givens,
        question="What is the concentration in the final tube?",
        targets=[Target("final_concentration", final, "uM", 0.01)],
        solution=f"c = c0 / step^n = {_g(start)} / {step}^{n} = {final:.6g} uM",
        trap=Target("final_concentration", start / (step * n), "uM", 0.01),
        trap_note="step x n instead of step^n",
        meta={"step": step, "n": n},
    )


def concentration_units(rng: random.Random, difficulty: str) -> Problem:
    name, mw = rng.choice(COMPOUNDS)
    mg_per_mL = _pick_round(rng, 0.2, 40, 0.01)
    mM = mg_per_mL / mw * 1000
    givens = [Given("Compound", name), Given("Molecular weight", f"{mw} g/mol"),
              Given("Measured concentration", f"{_g(mg_per_mL)} mg/mL")]
    if difficulty == "hard":
        givens.append(Given("Purity", "> 98%", required=False))
    # Two target units, so the same physical quantity is asked for in a unit the
    # model has to convert into rather than read off.
    want_uM = rng.random() < 0.5
    return Problem(
        family="concentration_units",
        difficulty=difficulty,
        context=f"A stock of {name} is quantified by mass.",
        givens=givens,
        question=f"Express the concentration in {'uM' if want_uM else 'mM'}.",
        targets=[Target("concentration", mM * (1000 if want_uM else 1),
                        "uM" if want_uM else "mM", 0.01)],
        solution=(f"c = (mass/volume) / MW = {_g(mg_per_mL)} g/L / {mw} g/mol "
                  f"= {mM/1000:.6g} mol/L = {mM:.6g} mM"),
        trap=Target("concentration", mg_per_mL / mw, "mM", 0.01),
        trap_note="mg/mL treated as g/L without the 1000x to mM",
        meta={"mw": mw},
    )


def buffer_ph(rng: random.Random, difficulty: str) -> Problem:
    buf, pka = rng.choice([
        ("sodium phosphate (pKa2)", 7.20), ("Tris-HCl", 8.06),
        ("acetate", 4.76), ("HEPES", 7.55), ("MES", 6.15), ("bicarbonate (pKa1)", 6.35),
    ])
    offset = _pick_round(rng, 0.10, 0.70, 0.05) * rng.choice([-1, 1])
    ph = round(pka + offset, 2)
    total_mM = _pick_round(rng, 10, 200, 1)
    vol_mL = _pick_round(rng, 50, 1000, 10)
    stock_M = rng.choice([0.5, 1.0])
    ratio = 10 ** (ph - pka)
    n_total_mmol = total_mM * vol_mL / 1000
    n_base = n_total_mmol * ratio / (1 + ratio)
    v_base_mL = n_base / stock_M
    givens = [
        Given("Buffer system", buf),
        Given("pKa", f"{pka}"),
        Given("Target pH", f"{ph}"),
        Given("Target total buffer concentration", f"{total_mM} mM"),
        Given("Final volume", f"{vol_mL} mL"),
        Given("Stock concentration of both conjugate species", f"{stock_M} M"),
    ]
    if difficulty == "hard":
        givens.append(Given("Temperature of use", f"{rng.choice([4, 25, 37])} C", required=False))
    # Inverted ratio: the classic error, and it gives a specific wrong volume.
    n_base_inv = n_total_mmol * (1 / ratio) / (1 + 1 / ratio)
    return Problem(
        family="buffer_ph",
        difficulty=difficulty,
        context=("You are preparing a buffer from separate stocks of the conjugate acid and "
                 "conjugate base. Use the Henderson-Hasselbalch equation and the pKa given."),
        givens=givens,
        question="What volume of the conjugate BASE stock do you need?",
        targets=[Target("base_volume", v_base_mL, "mL", 0.01)],
        solution=(f"ratio = 10^(pH-pKa) = 10^({ph}-{pka}) = {ratio:.6g}; "
                  f"n_total = {total_mM} mM x {vol_mL} mL = {n_total_mmol:.6g} mmol; "
                  f"n_base = n_total x r/(1+r) = {n_base:.6g} mmol; "
                  f"V = n/stock = {v_base_mL:.6g} mL"),
        trap=Target("base_volume", n_base_inv / stock_M, "mL", 0.01),
        trap_note="acid/base ratio inverted",
        meta={"pka": pka, "ph": ph},
    )


def centrifugation(rng: random.Random, difficulty: str) -> Problem:
    r_src = _pick_round(rng, 5.0, 11.0, 0.1)
    r_dst = _pick_round(rng, 11.5, 20.0, 0.1)
    rpm_src = _pick_round(rng, 2000, 15000, 100)
    rcf = 1.118e-5 * r_src * rpm_src ** 2
    rpm_dst = math.sqrt(rcf / (1.118e-5 * r_dst))
    givens = [
        Given("Published setting", f"{rpm_src} rpm"),
        Given("Rotor radius in the published protocol", f"{_g(r_src)} cm"),
        Given("Radius of your rotor", f"{_g(r_dst)} cm"),
    ]
    if difficulty != "easy":
        givens.append(Given("Spin duration", f"{rng.choice([5, 10, 15, 20])} min", required=False))
    return Problem(
        family="centrifugation",
        difficulty=difficulty,
        context=("You are reproducing a centrifugation step on a rotor with a different radius. "
                 "Use RCF = 1.118e-5 x r[cm] x rpm^2."),
        givens=givens,
        question="What speed in rpm reproduces the published relative centrifugal field?",
        targets=[Target("speed", rpm_dst, "rpm", 0.01)],
        solution=(f"RCF = 1.118e-5 x {_g(r_src)} x {rpm_src}^2 = {rcf:.6g} xg; "
                  f"rpm = sqrt(RCF / (1.118e-5 x {_g(r_dst)})) = {rpm_dst:.6g} rpm"),
        trap=Target("speed", float(rpm_src), "rpm", 0.01),
        trap_note="rpm transferred unchanged, ignoring radius",
        meta={"rcf": rcf},
    )


def beer_lambert(rng: random.Random, difficulty: str) -> Problem:
    eps = _pick_round(rng, 5000, 60000, 100)  # M^-1 cm^-1
    path = rng.choice([0.1, 0.2, 0.5, 1.0])
    a = _pick_round(rng, 0.10, 0.95, 0.005)
    df = rng.choice([1, 2, 5, 10, 20, 50]) if difficulty != "easy" else 1
    conc_cuvette_M = a / (eps * path)
    conc_stock_uM = conc_cuvette_M * df * 1e6
    givens = [
        Given("Absorbance", f"{_g(a)}"),
        Given("Molar extinction coefficient", f"{eps} M^-1 cm^-1"),
        Given("Path length", f"{path} cm"),
    ]
    if df != 1:
        givens.append(Given("Dilution factor applied before reading", f"{df}-fold"))
    if difficulty == "hard":
        givens.append(Given("Reading wavelength", f"{rng.choice([260, 280, 340, 405])} nm",
                            required=False))
    return Problem(
        family="beer_lambert",
        difficulty=difficulty,
        context="You quantify a protein spectrophotometrically. Use A = eps x c x l.",
        givens=givens,
        question="What is the concentration of the undiluted stock, in uM?",
        targets=[Target("concentration", conc_stock_uM, "uM", 0.015)],
        solution=(f"c_cuvette = A/(eps l) = {_g(a)}/({eps} x {path}) = {conc_cuvette_M:.6g} M; "
                  f"stock = c x DF = {conc_cuvette_M:.6g} x {df} = {conc_stock_uM:.6g} uM"),
        trap=(Target("concentration", conc_cuvette_M * 1e6, "uM", 0.015) if df != 1 else
              Target("concentration", a * eps * path * 1e6, "uM", 0.015)),
        trap_note="dilution factor omitted" if df != 1 else "multiplied instead of divided by eps·l",
        meta={"df": df},
    )


def michaelis_menten(rng: random.Random, difficulty: str) -> Problem:
    vmax = _pick_round(rng, 5, 200, 1)  # umol/min/mg
    km = _pick_round(rng, 2, 200, 1)  # uM
    s = rng.choice([0.2, 0.33, 0.5, 1, 2, 3, 4, 10]) * km
    v = vmax * s / (km + s)
    givens = [
        Given("Vmax", f"{vmax} umol/min/mg"),
        Given("Km", f"{km} uM"),
        Given("Substrate concentration", f"{_g(s)} uM"),
    ]
    if difficulty == "hard":
        givens.append(Given("Enzyme concentration in the assay", f"{rng.choice([5, 10, 25])} nM",
                            required=False))
        givens.append(Given("Assay temperature", "30 C", required=False))
    return Problem(
        family="michaelis_menten",
        difficulty=difficulty,
        context=("An enzyme follows Michaelis-Menten kinetics under steady-state conditions. "
                 "Assume no inhibition and no substrate depletion."),
        givens=givens,
        question="What is the initial reaction rate?",
        targets=[Target("rate", v, "umol/min/mg", 0.01)],
        solution=(f"v = Vmax [S]/(Km + [S]) = {vmax} x {_g(s)}/({km} + {_g(s)}) "
                  f"= {v:.6g} umol/min/mg"),
        trap=Target("rate", vmax * s / km, "umol/min/mg", 0.01),
        trap_note="Km + [S] replaced by Km (first-order form used outside its range)",
        meta={"s_over_km": s / km},
    )


def diagnostic_predictive_value(rng: random.Random, difficulty: str) -> Problem:
    sens = round(rng.uniform(0.85, 0.999), 4)
    spec = round(rng.uniform(0.85, 0.9995), 4)
    # Log-uniform prevalence: the interesting regime is rare disease, and a
    # linear grid puts almost no mass there.
    prev = round(10 ** rng.uniform(-4.0, -1.5), 6)
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    givens = [
        Given("Sensitivity", f"{sens*100:.4g}%"),
        Given("Specificity", f"{spec*100:.4g}%"),
        Given("Prevalence in the screened population", f"{prev*100:.4g}%"),
    ]
    if difficulty != "easy":
        givens.append(Given("Number screened", f"{rng.choice([50, 100, 250])},000", required=False))
    return Problem(
        family="diagnostic_predictive_value",
        difficulty=difficulty,
        context="A screening test is applied to an unselected population.",
        givens=givens,
        question=("A sample returns positive. What is the probability it is a true positive "
                  "(the positive predictive value), as a percentage?"),
        targets=[Target("ppv", ppv * 100, "percent", 0.02)],
        solution=(f"PPV = sens x p / (sens x p + (1-spec)(1-p)) = "
                  f"{sens} x {prev} / ({sens} x {prev} + {1-spec:.6g} x {1-prev:.6g}) "
                  f"= {ppv:.6g} = {ppv*100:.6g}%"),
        trap=Target("ppv", sens * 100, "percent", 0.02),
        trap_note="sensitivity reported as PPV (base rate ignored)",
        meta={"prev": prev},
    )


def power_sample_size(rng: random.Random, difficulty: str) -> Problem:
    d = round(rng.uniform(0.15, 1.20), 3)
    power = _pick_round(rng, 0.70, 0.95, 0.01)
    alpha = rng.choice([0.05, 0.01, 0.10])
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    n = 2 * (z_a + z_b) ** 2 / d ** 2
    givens = [
        Given("Expected standardised effect size (Cohen's d)", f"{d}"),
        Given("Two-sided alpha", f"{alpha}"),
        Given("Target power", f"{power*100:.0f}%"),
        Given("Design", "two independent groups of equal size, normal approximation"),
    ]
    if difficulty == "hard":
        givens.append(Given("Expected attrition", "none", required=False))
    return Problem(
        family="power_sample_size",
        difficulty=difficulty,
        context=("You are sizing a two-group comparison. Use the normal-approximation formula "
                 "n_per_group = 2(z_(1-alpha/2) + z_power)^2 / d^2."),
        givens=givens,
        question="How many participants are needed PER GROUP? Round to the nearest whole participant.",
        targets=[Target("n_per_group", n, "participants", 0.05)],
        solution=(f"z_a={z_a:.4f}, z_b={z_b:.4f}; "
                  f"n = 2({z_a:.4f}+{z_b:.4f})^2/{d}^2 = {n:.6g} -> {math.ceil(n)} per group"),
        trap=Target("n_per_group", n / 2, "participants", 0.05),
        trap_note="factor of 2 omitted (one-sample formula used for a two-group design)",
        meta={"d": d, "power": power},
    )


def exponential_decay(rng: random.Random, difficulty: str) -> Problem:
    mode = rng.choice(["remaining", "time", "rate_constant"])
    t_half = _pick_round(rng, 0.25, 48, 0.05)  # h
    k = math.log(2) / t_half
    givens = [Given("Half-life", f"{_g(t_half)} h")]
    if mode == "remaining":
        t = round(t_half * _pick_round(rng, 0.5, 5.0, 0.1), 3)
        frac = math.exp(-k * t) * 100
        givens.append(Given("Elapsed time", f"{_g(t)} h"))
        question = "What percentage of the initial amount remains?"
        targets = [Target("remaining", frac, "percent", 0.01)]
        sol = (f"k = ln2/t_half = {k:.6g} /h; remaining = exp(-k t) = exp(-{k:.6g} x {t}) "
               f"= {frac/100:.6g} = {frac:.6g}%")
        trap = Target("remaining", max(0.0, 100 * (1 - t / t_half / 2)), "percent", 0.01)
        trap_note = "linear decay instead of exponential"
    elif mode == "time":
        frac = round(10 ** rng.uniform(-2.0, -0.1), 4)
        t = -math.log(frac) / k
        givens.append(Given("Target fraction remaining", f"{frac*100:.4g}%"))
        question = "How long must elapse?"
        targets = [Target("time", t, "h", 0.01)]
        sol = (f"k = ln2/t_half = {k:.6g} /h; t = -ln(f)/k = -ln({frac})/{k:.6g} = {t:.6g} h")
        trap = Target("time", t_half * frac * 10, "h", 0.01)
        trap_note = "proportional scaling of the half-life"
    else:
        # The requested output unit varies, so the step is a conversion as well
        # as a formula. The grader compares dimensionally, so any correct unit
        # would pass -- the variation is there to widen the problem space.
        out_unit, per_hour = rng.choice([("1/h", 1.0), ("1/min", 1 / 60), ("1/day", 24.0)])
        question = f"What is the first-order rate constant, expressed in {out_unit}?"
        targets = [Target("rate_constant", k * per_hour, out_unit, 0.01)]
        sol = (f"k = ln2/t_half = 0.6931/{_g(t_half)} h = {k:.6g} /h "
               f"= {k*per_hour:.6g} {out_unit}")
        trap = Target("rate_constant", (1 / t_half) * per_hour, out_unit, 0.01)
        trap_note = "1/t_half used instead of ln2/t_half"
    if difficulty == "hard":
        givens.append(Given("Initial amount", f"{rng.choice([10, 25, 100])} mg", required=False))
    return Problem(
        family="exponential_decay",
        difficulty=difficulty,
        context="A quantity decays by a first-order process.",
        givens=givens,
        question=question,
        targets=targets,
        solution=sol,
        trap=trap,
        trap_note=trap_note,
        meta={"mode": mode},
    )


def uncertainty_propagation(rng: random.Random, difficulty: str) -> Problem:
    op = rng.choice(["product", "quotient"])
    a = _pick_round(rng, 2.0, 50.0, 0.1)
    da = round(a * rng.choice([0.01, 0.02, 0.03, 0.05]), 4)
    b = _pick_round(rng, 2.0, 50.0, 0.1)
    db = round(b * rng.choice([0.01, 0.02, 0.03, 0.05]), 4)
    rel = math.hypot(da / a, db / b)
    if op == "product":
        val = a * b
        unit, q = "AU", f"the product of the two readings"
    else:
        val = a / b
        unit, q = "", "the ratio of the first reading to the second"
    unc = val * rel
    givens = [
        Given("First quantity", f"{_g(a)} +/- {_g(da)} AU"),
        Given("Second quantity", f"{_g(b)} +/- {_g(db)} AU" if op == "product"
              else f"{_g(b)} +/- {_g(db)} AU"),
        Given("Error type", "independent random errors, 1 sigma"),
    ]
    if difficulty == "hard":
        givens.append(Given("Replicates per quantity", f"{rng.choice([3, 4, 6])}", required=False))
    return Problem(
        family="uncertainty_propagation",
        difficulty=difficulty,
        context=("Two independently measured quantities are combined. Propagate the "
                 "uncertainty to 1 sigma."),
        givens=givens,
        question=f"Report {q} and its uncertainty.",
        targets=[Target("value", val, unit or "1", 0.01),
                 Target("uncertainty", unc, unit or "1", 0.02)],
        solution=(f"rel = sqrt(({da}/{a})^2 + ({db}/{b})^2) = {rel:.6g}; "
                  f"value = {val:.6g}; uncertainty = {val:.6g} x {rel:.6g} = {unc:.6g}"),
        trap=Target("uncertainty", val * (da / a + db / b), unit or "1", 0.02),
        trap_note="relative errors added linearly instead of in quadrature",
        meta={"op": op},
    )


def allometric_dose(rng: random.Random, difficulty: str) -> Problem:
    species, factor = rng.choice([("mouse", 12.3), ("rat", 6.2), ("monkey", 3.1), ("dog", 1.8)])
    dose = _pick_round(rng, 0.5, 200, 0.5)  # mg/kg
    weight = _pick_round(rng, 45, 110, 0.5)  # kg
    hed = dose / factor
    total_mg = hed * weight
    # Two modes: the scaling alone, or the scaling followed by a mass calculation.
    total_mode = rng.random() < 0.5
    givens = [
        Given("Species studied", species),
        Given("Dose administered", f"{_g(dose)} mg/kg"),
        Given("Body-surface-area conversion factor for this species", f"{factor}"),
        Given("Human body weight", f"{_g(weight)} kg", required=total_mode),
    ]
    if total_mode:
        question = "What is the total human equivalent dose for this person, in mg?"
        targets = [Target("total_dose", total_mg, "mg", 0.01)]
        solution = (f"HED = {_g(dose)}/{factor} = {hed:.6g} mg/kg; "
                    f"total = {hed:.6g} x {_g(weight)} = {total_mg:.6g} mg")
        trap = Target("total_dose", dose * weight, "mg", 0.01)
        trap_note = "mg/kg dose transferred with no allometric scaling"
    else:
        question = "What is the human equivalent dose, in mg/kg?"
        targets = [Target("hed", hed, "mg/kg", 0.01)]
        solution = f"HED = {_g(dose)}/{factor} = {hed:.6g} mg/kg"
        trap = Target("hed", float(dose), "mg/kg", 0.01)
        trap_note = "dose transferred on a mg/kg basis with no scaling"
    return Problem(
        family="allometric_dose",
        difficulty=difficulty,
        context=("Convert an animal dose to the human equivalent dose by body-surface-area "
                 "scaling (HED = animal dose / conversion factor)."),
        givens=givens,
        question=question,
        targets=targets,
        solution=solution,
        trap=trap,
        trap_note=trap_note,
        meta={"species": species, "total_mode": total_mode},
    )


FAMILIES = {
    fn.__name__: fn
    for fn in (
        molarity_from_mass, dilution, serial_dilution, concentration_units,
        buffer_ph, centrifugation, beer_lambert, michaelis_menten,
        diagnostic_predictive_value, power_sample_size, exponential_decay,
        uncertainty_propagation, allometric_dose,
    )
}

DIFFICULTIES = ("easy", "medium", "hard")


def make_insufficient(problem: Problem, rng: random.Random) -> Problem:
    """Remove one required quantity, making the problem genuinely unanswerable.

    This family exists to counter a specific, documented RL failure: when every
    problem has a computable answer, always producing a number is the
    reward-maximising policy, and the model's ability to say "not enough
    information" is trained away. Here the reward for a confident number is
    zero and the reward for correct abstention is one.
    """
    required = [i for i, g in enumerate(problem.givens) if g.required]
    if not required:
        return problem
    drop = rng.choice(required)
    missing = problem.givens[drop]
    problem.givens = [g for i, g in enumerate(problem.givens) if i != drop]
    problem.insufficient = missing.label
    problem.targets = []
    problem.trap = None
    problem.trap_note = ""
    problem.family = f"{problem.family}__insufficient"
    problem.solution = (f"Unanswerable: {missing.label} is not given and cannot be derived "
                        f"from the remaining quantities.")
    problem.meta = {**problem.meta, "missing": missing.label}
    return problem


def _trap_is_distinguishable(problem: Problem) -> bool:
    """True when the trap answer is far enough from the truth to be a real miss.

    Some parameter draws make a wrong method accidentally right -- at high
    prevalence the PPV approaches the sensitivity, so "report the sensitivity"
    lands inside tolerance. Keeping such an instance would score a wrong
    procedure as correct and, worse, reward it. The guard lives here rather than
    in each family because it is the same condition everywhere.
    """
    if problem.trap is None:
        return True
    for t in problem.targets:
        if t.key != problem.trap.key:
            continue
        if t.value == 0:
            return abs(problem.trap.value) > t.rel_tol
        gap = abs(problem.trap.value - t.value) / abs(t.value)
        return gap > 3 * t.rel_tol
    return True


def sample(family: str, seed: int, difficulty: str = "medium",
           insufficient: bool = False) -> Problem:
    """Build one problem. Deterministic in (family, seed, difficulty, insufficient).

    Redraws while the sampled parameters make the trap answer indistinguishable
    from the correct one. The redraw is seeded deterministically, so the result
    is still a pure function of the arguments.
    """
    if family not in FAMILIES:
        raise KeyError(f"unknown family: {family!r}")
    problem = None
    for attempt in range(64):
        rng = random.Random(f"{family}|{difficulty}|{seed}|{insufficient}|{attempt}")
        problem = FAMILIES[family](rng, difficulty)
        if _trap_is_distinguishable(problem):
            break
    else:
        # Exhausted the redraws: keep the instance but drop the unusable trap
        # rather than silently scoring a wrong method as correct.
        problem.trap = None
        problem.trap_note = ""
    problem.seed = seed
    if insufficient:
        rng = random.Random(f"{family}|{difficulty}|{seed}|drop")
        problem = make_insufficient(problem, rng)
    return problem
