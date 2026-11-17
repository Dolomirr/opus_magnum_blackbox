# Research history: from a memorizing overfit run to a generalizing agent

Scope: the training-research line of work — tier-1
generalization, the autodiscovery experiments, and the tier-2 "anywhere" family. This document
records what was tried, the limitations discovered, what each limitation says about the
architecture in the context of the problem that was live at the time, and the decisions that
followed.

Numbers are kept out of the prose on purpose: every phase carries **Data** blocks (tables and
lists) with the exact measurements, and the text refers to them. 

---

## 1. Timeline at a glance

| phase | headline | decision that followed |
|---|---|---|
| overfit era (tier-0) | machinery works; no way to measure learning | build a real curriculum |
| tier-1 v1 | train solved, held-out zero — memorization, plus a broken ruler | enumerate the whole space; honest eval |
| tier-1 v2 | held-out performance jumps to a strong level; gap collapses to a hard slice | move to autodiscovery |
| autodiscovery dial | performance scales with supervision; one slice owns the loss | stratified self-imitation |
| discovery ingredients | all three signals fail; a buried observation in the buffers | amplify self-imitation |
| stratified SIL | the lost slice recovers from zero to roughly half | accept spec-derived labels; harden the stack |
| hardening matrix | all label-free signals fail in distinct ways | keep the label; widen the family |
| tier-2 mixed bank | held-out collapses to zero despite healthy train | augment fix, error taxonomy, static arms |
| curriculum split | width was the ceiling; per-concept banks and rehearsal fix the rest | 256-wide model becomes the working default |
| follow-up batch | width closed at 256; bootstrap cost measured; handoff held-out plateaus ~0.6 | tier-2 near-done; multi-reference supervision the one open lever; tier-3 next |
| feeding + procedural cycle | cycled/solver-fed cloning validated; features dead; coverage law and no-extrapolation confirmed | recipe locked; tier-3 goes solver-first |

---

## 2. Phase 1 — the overfit era (before tier-1)

The full pipeline existed (token codec, grammar mask, small transformer, hand-written
REINFORCE) and exactly one training level (tier-0: two-atom typed molecule, solution-placed
entrypoints, cycling on) with one hand-written solution. Training was behavior cloning on that
solution plus RL consolidation. It proved the machinery end to end and nothing else.

**Limitation discovered:** a curriculum of one level and one known solution cannot measure
learning — there was no held-out signal at all.

**What it says about the setup:** the bottleneck was the dataset, not the algorithm. Every
later phase is a variation on "what does the training distribution have to contain for the
model to generalize".

---

## 3. Phase 2 — tier-1 v1: memorization and a broken ruler

Family: 6x6 board, fixed 1-cell typed entrance/exit, one length-1 arm, two deliveries
(`target 2`). A generator produced a small sampled bank (see Data 3.1). Behavior cloning
early-stopped with the training set fully replayed, and both pipeline runs ended with the
training bank perfectly solved and the held-out bank at zero.

**Limitations discovered:**

1. **A small sampled bank is a lookup table.** The transformer memorized it outright during
   cloning; moderate token accuracy was enough for perfect greedy replay of every level.
2. **Once the training distribution is solved, REINFORCE has no signal.** Advantages vanish,
   the entropy bonus becomes the only active objective, and the policy random-walks between
   memorized modes: greedy train behavior flipped between perfect and zero between evals, and
   the longer run collapsed completely mid-stage before recovering.
3. **The eval metric was measuring one level, not the bank.** All eval episodes drew a single
   shared variant (pickled replicas carry one RNG stream; the stream advances one draw per
   check). Reported held-out zeros were many copies of one unsolved level, and the occasional
   perfect rows were the same re-solved level.
4. **The same RNG coupling corrupted training batches**, making each iteration cover a handful
   of levels instead of a full independent sample.

**What it says about the architecture:** nothing about the network was wrong — the failure was
distributional. A small transformer will happily key absolute input configurations to outputs
when the input space is small and supervision is dense. The "saturated RL" dynamic (no gradient
once solved) is a property of the objective, not of the task.

**Decisions:** independent replica seeding; exhaustive per-variant greedy evaluation;
best-eval checkpointing; enumerate the whole task space.

**Data 3.1**

- Bank: 48 train + 16 eval levels; family as above; one `.oms` reference solution per level.
- BC early stop: after four evaluation checks; loss ≈ 0.46 (gate 0.5); token accuracy ≈ 0.835;
  train greedy 1.00.
- Runs: 400+600 and 600+1000 iterations; both ended train 1.00 / eval-bank 0.00.
- Saturation evidence: value loss → ~0.02; greedy train flipped 1.00 ↔ 0.00 across evals; the
  longer run hit sampled solve 0.00 and reward ≈ +0.15 at iteration 1040 before recovering.
- Eval artifact: 8 slots drew 1 distinct level per check (verified mechanically); the stream
  produced a fixed level sequence over checks.

---

## 4. Phase 3 — tier-1 v2: coverage fixes generalization

The 6x6 single-arm space is exactly 428 ordered (entrance, exit) pairs at hex distance 1–2.
The bank became all of them, split train/eval (Data 4.1). Consolidation gained an eval-patience
early stop and a much lower entropy coefficient.

Result: behavior cloning alone generalized strongly; RL added a solid refinement; the final
gap between train and held-out was small; and the leftover failures were structured — one hard
distance class — rather than random. A cleaner later split showed the class axis was distance
only: an early "edge/corner clustering" reading was a confound (Data 4.3).

**Limitation discovered:** none new — this run established the positive baseline. The hard
slice (distance-1 pairs: entrance and exit adjacent, two candidate arm bases) remained at
roughly three-quarters solved while everything else was near-complete, identically on train and
held-out.

**What it says about the architecture:** with full input coverage the small model computes
(E, X) → solution as a function of geometry rather than a lookup, and BC alone does most of the
work; RL's role is refinement. Generalization is an input-distribution property.

**Data 4.1**

- Bank: 342 train / 86 eval = all 428 pairs; distance split 170 d1 / 258 d2.
- BC (no early stop, 100 epochs): train greedy 0.728, eval 0.581, peak eval 0.640.
- rl-shape (400 it): train 0.830, eval 0.721.
- rl-consolidate: best eval 0.814 @ iter 610; early stop @ 810.
- Final probe of `best_eval.pt`: train 293/342 (0.856); eval 70/86 (0.814).
- Eval by class: d1 25/34 (0.74); d2 45/52 (0.87). Train by class: d1 105/136 (0.77);
  d2 188/206 (0.91).
- Wall time ≈ 13 min; consolidation entropy coefficient 0.001 (was 0.007).

---

## 5. Phase 4 — the autodiscovery dial (Nov 12)

Question: how much of the result survives when the cloning bootstrap shrinks? A pipeline key
`autodiscovery = k` (1..10) scaled BC epochs linearly down to zero at k=10; everything else was
fixed. Runs at k = 1, 5, 7, 8, 9, plus k=9 with doubled patience and a longer ceiling.

**Findings.** Held-out performance scales with the amount of supervision, and the loss is not
spread evenly: one slice owns all of it. Splitting best checkpoints by distance class showed
distance-2 performance constant across every supervision level while distance-1 collapsed from
over half to zero (Data 5.2). The extended k=9 run bought almost nothing further — a plateau,
not a slow climb.

Probing the mechanism: for the large majority of d1 eval pairs, *both* candidate arm bases
solve with the canonical program, so base choice was not the blocker. The consistent
explanation is mode lock-in: the policy learned the longer-distance swing rule and applied it
everywhere; on adjacent pairs that swing releases the atom past the exit, the episode earns
nothing distinguishable from garbage, and no gradient says "one turn fewer would work". The
rare behavior existed in the sampling tail but was never reinforced.

**Limitations discovered:**

1. **Policy gradient consolidates modes; it does not generate them.** What matters is not the
   probability of generating a new behavior but of it being reinforced — and these diverge
   completely for coordinated multi-token novelties under terminal rewards.
2. **Terminal rewards make near-misses invisible**, so the reward landscape around a dominant
   mode is flat zero.
3. **Cloning ignition is load-bearing.** With cloning fully skipped, zero episodes solved across
   the whole run; entropy annealed steadily with no signal anywhere.

**What it says about the architecture:** the k=9 failure was not capability — rare-slice
solutions existed in the sampling tail. It was an exploration/credit-assignment property of the
training objective, and it will worsen as action spaces grow unless rare successes get
amplified and the mode can be escaped.

**Decision:** amplify what sampling already finds — self-imitation — before adding more
exploration pressure.

**Data 5.1 — dial summary** (BC-end train/eval; best eval; RL gain = best minus BC-end; stop)

| k | BC-end train | BC-end eval | best eval | RL gain | stop |
|---|---|---|---|---|---|
| 1 (canonical lineage) | 0.728 | 0.581 | 0.814 | +0.233 | iter 810 |
| 5 | 0.629 | 0.605 | 0.733 | +0.128 | 600 |
| 7 | 0.556 | 0.512 | 0.593 | +0.081 | 600 |
| 8 | 0.518 | 0.523 | 0.523 | +0.000 | 600 |
| 9 | 0.330 | 0.291 | 0.419 | +0.128 | 1120 |
| 9 extended (patience 60, ceiling 3000) | 0.330 | 0.291 | 0.453 | +0.163 | 2110 |

**Data 5.2 — best-checkpoint eval by distance class** (d1 out of 34, d2 out of 52)

| k | d1 | d2 |
|---|---|---|
| 1 | 25/34 | 45/52 |
| 5 | 16/34 | 47/52 |
| 7 | 3/34 | 48/52 |
| 8 | 0/34 | 45/52 |
| 9 ext | 0/34 | 39/52 |

**Data 5.3 — mechanism probe**

- d1 pairs where both candidate bases solve: 26/34; exactly one: 8/34; none: 0/34.
- Extended k=9 run: patience 60 (≈600 stale evals allowed), best +0.034 over the short run.
- k=10 (no BC): zero solved episodes in ≈19,200; entropy annealed 1.09 → 0.86.
- Overall entropy across dial runs: ≈ 0.44–0.50; the canonical regime used 0.005/0.007 and the
  later canonical default 0.001.

---

## 6. Phase 5 — discovery ingredients: three failures and a buried success (Nov 12)

Matrix at k=9: progress shaping, mask-uniform ε-sampling, self-imitation, the full combo, and
the combo with cloning skipped. Every arm scored zero on the rare slice; the majority slice
held; cloning-skip stayed at zero throughout.

The buried result: the self-imitation buffers captured a handful of *distinct* rare-slice
levels solved by plain sampling, both runs, early — discovery was happening; conversion was
not. The amplification weight was far too small against the dominant gradient.

Component verdicts: ε-sampling perturbs nearly every episode and a single uniform token breaks
a working program — it added majority-slice noise only. Shaping was structurally blind on the
rare slice: the entrance respawns an atom every step, pinning the final distance at the initial
distance unless a delivery lands.

**Limitations discovered:**

1. **A learning signal that is quiet relative to the dominant gradient is equivalent to no
   signal.**
2. **Token-level exploration cannot assemble coordinated new programs**; it only damages
   working ones.
3. **Partial-credit shaping is blind exactly where the rare mode lives**, because the respawn
   keeps the distance signal pinned.

**What it says about the architecture:** the path from "sampled once by luck" to "in the
policy" requires amplification whose weight competes with the main gradient; the base recipe
has none, so lucky discoveries evaporate.

**Data 6.1 — matrix results** (rare slice = d1, 34 eval variants; majority = d2, 52)

| arm | shaping | ε | sil coef | d1 eval final | d2 eval final | eval-bank best | first rare-slice level buffered |
|---|---|---|---|---|---|---|---|
| baseline k=9 | — | — | — | 0/34 | 36/52 | 0.453 | — |
| full | 0.3 | 0.05 | 0.1 | 0/34 | 30/52 | 0.419 | iter 105 (3 levels) |
| shaping only | 0.3 | 0 | 0 | 0/34 | 33/52 | 0.419 | — |
| ε only | 0 | 0.05 | 0 | 0/34 | 33/52 | 0.395 | — |
| sil only | 0 | 0 | 0.1 | 0/34 | 33/52 | 0.407 | iter 35 (6 levels) |
| full, no BC | 0.3 | 0.05 | 0.1 | 0/34 | 0/52 | 0.000 | — |

**Data 6.2 — supporting measurements**

- Buffer composition: full run — majority + 3 rare levels; sil-only — majority + 6 rare.
- ε-sampling perturbation: ≈95% of episodes receive at least one uniform token.
- Shaping blindness: respawn pins the distance signal unless a delivery lands (structural, not
  a tuning issue).
- No-BC arm: ≈19,200 sampled episodes, zero solves.
- Stability: last-5 eval spread ≤ 0.023 everywhere; no transient rare-slice eval win in any arm.

---

## 7. Phase 6 — stratified self-imitation breaks the lock-in (Nov 12/13)

Change: amplify — raise the imitation coefficient, raise the replay batch, and draw half of
every replay minibatch from buffered rare-slice rows. Two arms: a moderate coefficient and a
heavier one.

Result: the lost slice recovered from zero to roughly half on held-out data with the majority
slice intact; the heavier coefficient was strictly worse on every metric. The flywheel is the
important part — converting the first few rare solutions shifted the policy, which shifted what
sampling finds, which fed the buffer: rare levels ever-buffered went from a handful to nearly
the whole rare slice. The agent taught itself a mode that thin cloning never demonstrated.

**Limitations discovered:**

1. **Amplification has a balance band.** The heavier coefficient was worse everywhere; too much
   imitation stops complementing the RL gradient and starts distorting the policy.
2. **A within-family generalization gap appeared** in the recovered slice (train well above
   held-out).
3. **Stability cost**: eval spread roughly tripled versus baseline.

**What it says about the architecture:** replay decouples the learning rate of a behavior from
its sampling probability. Combined with phase 4, the working division of labor is now: cloning
= ignition, policy gradient = refinement, stratified replay = mode construction. Each has a
distinct failure when removed.

**Data 7.1**

| metric | baseline k=9 | moderate (0.4) | heavier (0.5) |
|---|---|---|---|
| rare-slice eval final / best | 0/34 / — | 17/34 / 20/34 | 13/34 / 14/34 |
| majority-slice eval final / best | 0.692 / 0.692 | 0.673 / 0.712 | 0.635 / 0.673 |
| eval-bank best | 0.453 | 0.663 | 0.570 |
| train rare-slice (best) | 0/136 | 95/136 | — |
| rare-slice levels ever buffered | 6 | 101/136 | 97/136 |
| first rare level buffered | — | iter 35 | iter 35 |
| iterations (shape + consolidate) | 400+1710 | 400+1080 | 400+920 |
| eval last-5 spread | 0.012 | 0.081 | 0.035 |

Buffer union (moderate arm): 306 distinct train levels — 101 rare + 205 majority; eval-bank
levels never entered the buffer.

---

## 8. Phase 7 — hardening: three label-free signals fail (Nov 12/13)

Goal: remove the human-chosen distance label from replay stratification, and test temperature
as an exploration knob. All against the standing baseline (best 0.663).

| arm | intent | result |
|---|---|---|
| scarcity weighting (replay ∝ inverse solve count) | derive rarity from the buffer itself | spread replay over hundreds of rarely-solved levels; rare slice barely converted; peaked in the wrong stage |
| sampling temperature T = 2 | fatten the policy tail | sampling quality collapsed; rare slice untouched; buffer grew slowest |
| failure-driven replay ("replay what you fail greedily") | concentrate on the agent's own weaknesses | majority slice best-ever; rare slice collapsed |

**Limitations discovered:**

1. **Proportional weighting dilutes.** The working scheme concentrated — half of every minibatch
   from one group, guaranteed. Scarcity spread replay across hundreds of levels, giving each
   rare level a thin, self-decaying slice.
2. **Blur breaks working programs.** High temperature (like the earlier ε result) damages the
   policy's own sampling quality; the policy's tail at normal temperature is the exploration
   channel that works.
3. **"Replay what you fail" repairs the dominant mode; it cannot build a rare one.** Early
   failures are mostly majority-mode failures, so the concentration budget went there (best
   majority result ever recorded) while the rare group plateaued without shrinking. Repair and
   discovery need different signals.

**What it says about the architecture:** the loop needs exactly one piece of outside
information — which slice of the input space is the rare one — so replay can concentrate there.
It does not need solutions or schemas; that label is computable from level data, which is why
it was accepted as fair curriculum input. A genuinely label-free version would need a novelty
signal over solution space (e.g., clustering buffered programs), not a performance signal —
noted for the future, not built.

**Data 8.1**

- Determinism check: a rerun of the baseline config reproduced identical metrics (1480
  iterations), validating comparability.
- Scarcity: best eval 0.419 @ iter 330 (inside the shaped stage); early stop @ 600; final rare
  9/34, majority 21/52; consolidation sampled solve ≈ 0.33 vs baseline ≈ 0.50.
- Temperature 2.0: sampled solve 0.055 (shaped) / 0.022 (consolidate); rare 0/34 at best and
  final; entropy 0.61–0.67; buffer slowest to fill; zero dead-ends (mechanism safe, signal poor).
- Failure-driven replay: best rare 6/34, majority 38/52 (best majority recorded); failure group
  trajectory 0 → 40 → plateau 45–55 → max 71 → 53 at stop (never shrank); train rare 19/136 at
  best, 29/136 at final.

---

## 9. Phase 8 — tier-2 "anywhere": the generalization collapse (Nov 14/15)

Family widened: any entrance/exit positions on 6x6, arms of length 1–3, one or two arms with a
handoff when a single arm cannot reach. Measured structure of the ordered-pair space: about
half the pairs are single-arm and about half require a handoff (full class table in Data 9.6).
Bank: sampled-stratified (composition in Data 9.1), every instance simulator-verified.

First finding was a **data bug with research weight**: the augmentation shuffled tick order
inside each handle, corrupting cloning targets for tick-sensitive programs. Before the fix,
cloning reached high token accuracy with near-zero greedy solve; after fixing it (handles
relabeled, triplets emitted in ascending tick order), the same budget produced a large
improvement — the fix was found by investigating a symptom, not by tuning.

Runs after the fix (warm start from a tier-1 checkpoint that was itself weaker than the best
available one, and was further diluted by tier-2 cloning):

- Training-bank performance climbed under RL + replay while held-out stayed at zero — per-pair
  memorization without transfer.
- Cloning alone memorized: high token accuracy, small fraction of train levels solved
  greedily, effectively zero held-out.
- Zero-shot transfer of the tier-1 checkpoint onto tier-2: almost nil.

**Limitations discovered:**

1. **The generalization collapse.** The tier-1 precedent — cloning generalizing by itself
   under full coverage — did not transfer to this family.
2. **Teacher-forced metrics stopped being proxies.** Near-perfect next-token accuracy coexisted
   with near-zero greedy solve: over longer, timing-sensitive programs, single-token errors
   compound, and next-token accuracy says nothing about generation.
3. **Learnability tracks how *forced* a solution is.** Long-distance pairs (near-unique
   solutions — geometry forces the plan) were memorized and generalized at high rates in the
   fresh run, while the middle-distance handoff class failed almost completely — and the worst
   sub-slice (distance 5) split its reference solutions between two mirrored arm-ordering
   families chosen by the solver's ranking. Where many valid solutions exist and a rank rule
   picks one, the target mapping is locally inconsistent and cannot be interpolated.
4. **Warm start was doubly handicapped** (a weaker source checkpoint plus dilution by tier-2
   cloning), though still mildly positive, not harmful.
5. **The tier-1 sizing rule ("a few dozen instances per mode") was wrong.** It assumed
   tier-1-like internal uniformity. Tier-2 modes carry high internal variance (base, ring,
   swing, handoff cell, which arm extends); what matters is coverage *and* consistency of the
   target mapping, not instance count.

**What it says about the architecture:** a small model imitates a deterministic solver's
*choices* only where those choices are simple or forced. A rank-based reference solver over a
multi-modal solution space produces a target function with discontinuities, and cloning on it
produces a memorizer. This is a property of the supervision construction, and it predicts which
slices fail before any training.

**Data 9.1 — bank composition**

- Train 450 = 150 single-arm (`target 2`) + 300 handoff (`target 1`); eval 120 = 40 + 80;
  0 rejected.
- Strata (distance × class, train/eval): 1s 39/10, 2s 59/16, 3h 93/25, 3s 20/6, 4h 62/16,
  4s 28/7, 5h 84/23, 6h 26/7, 6s 4/1, 7h 20/5, 8h 10/3, 9h 4/1, 10h 1/0.

**Data 9.2 — augment fix, measured on identical cloning budgets**

| metric | before fix | after fix |
|---|---|---|
| loss | 0.338 | 0.099 |
| token accuracy | 0.867 | 0.965 |
| greedy train solve | 0.022 | 0.111 |

Whole train bank re-verified after the fix (450 entries → 7200 variants). A stale-handle bug in
the same code path (shuffled two-arm variants emitting handle 0 after handle 1) was fixed in
the same change.

**Data 9.3 — mixed-bank runs after the fix**

| | warm start | fresh |
|---|---|---|
| BC end: train / eval greedy | 0.111 / ≈0 | 0.111 / ≈0 |
| best eval-bank | 0.050 (6/120) | 0.025 (3/120) |
| train greedy best | 0.322 | 0.291 |
| end | early stop @ 600 | early stop @ 820 |

Zero-shot tier-1 transfer onto tier-2: 2/450 train, 0/120 eval. Warm-start source note: the
tier-1 canonical run reached 0.535 that session while a stronger 0.663 checkpoint existed in
an earlier run directory — the weaker one was used, an avoidable handicap.

**Data 9.4 — per-distance × class, warm run at its best checkpoint** (train | eval)

| slice | train | eval |
|---|---|---|
| single d1 | 6/39 | 1/10 |
| single d2 | 10/59 | 2/16 |
| single d3 | 0/20 | 0/6 |
| single d4 | 0/28 | 0/7 |
| single d6 | 0/4 | 0/1 |
| single total | 16/150 (0.107) | 3/40 (0.075) |
| handoff d3 | 14/93 | 3/25 |
| handoff d4 | 3/62 | 0/16 |
| handoff d5 | 0/84 | 0/23 |
| handoff d6 | 0/26 | 0/7 |
| handoff d7 | 5/20 | 0/5 |
| handoff d8 | 7/10 | 0/3 |
| handoff d9 | 3/4 | 0/1 |
| handoff d10 | 1/1 | 0/0 |
| handoff total | 33/300 (0.110) | 3/80 (0.037) |

Fresh run: single total 43/150 (0.287) train / 3/40 eval; handoff total 54/300 (0.180) train /
0/80 eval. Notable: fresh-run long handoffs (d7–d10) solved far better than short ones on train
(d7 15/20, d8 10/10, d9 3/4, d10 1/1) — see the analysis below for why that inversion matters.

**Data 9.5 — SIL stratification telemetry** (mechanism verified working)

| | warm | fresh |
|---|---|---|
| replay rows in consolidation | 200 | 420 |
| rare-group size min/max/mean | 1 / 115 / 81.5 | 0 / 106 / 65.3 |
| new buffered levels | 204 (56% handoff) | 196 (54% handoff) |
| replay picks (top-5 rows) | 564 handoff / 1072 single | 1342 handoff / 2292 single |

**Data 9.6 — pair-space structure and the d5 anomaly**

- 1260 ordered pairs; single-arm-able 656 (ring 1: 428, ring 2: 120, ring 3: 108); handoff 604.
- d5 reference solutions split between mirrored arm-ordering families: `(arm, arm_piston)`
  31 pairs vs `(arm_piston, arm)` 51 pairs (plus 12/1 length variants) — the solver's rank
  choice flips between neighboring pairs.
- Fresh-run d5 handoff: 8/84 train, 0/23 eval — the worst sub-slice, exactly where the mapping
  is most inconsistent.

---

## 10. Phase 9 — static arms, error taxonomy, curriculum split, width (Nov 15/16)

**Static arms.** The simulator gained real length-parameterized arms as separate blueprints
(a wire-format insertion that shifts later categories but preserves every existing token, with
checkpoint remapping on warm start). Per-placement length overrides also exist in the
simulator, but a static length field is not representable in the token grammar — adding one
would be a wire-format break, so it was not used. `arm_piston` fell back to a ranked-last
fallback for a small minority of pairs. Single-arm cycles became pure-rotation programs,
structurally identical to tier-1. Retrain: the same wall (held-out stayed near zero).

**Error taxonomy** (greedy decode of every variant against references, best checkpoint): the
overwhelming majority of failures were *wrong arm length*; wrong base position was a distant
second; wrong program/ticks were negligible. Among length mismatches, about half still placed
at least one arm on a reference base — the model often knew where to stand but not how long
the arm must be; the other half had cascading base errors, since a different ring means a
different cell. Even the tier-1-identical near slices failed with a longer arm where a short
one belongs.

**Limitations discovered:**

1. **Two orthogonal concepts introduced at once (length selection + handoff topology) were
   neither learned.** The length concept exists only as token embeddings with no geometric
   feature, so the ring-to-geometry association must be inferred from data; mixed with
   topology, at this bank size, both stalled.
2. **Time-navigation was explicitly rejected as the cause.** The program template was imitated
   correctly — failures carried the right action multiset in the wrong arrangement, or the
   wrong hardware. The temporal dimension was not the blocker; the spatial one was.
3. **Teacher-forced accuracy is decision-grade only for short, forgiving programs** (already
   observed; here it became the standard argument for greedy probes).

**What it says about the architecture:** a small model imitates a deterministic solver's
choices only where those choices are simple or forced; and it learns a concept that exists
only as a token embedding only when the training distribution isolates it.

**Curriculum split round 1:** stage A (single-arm bank only, small model) produced a large
held-out jump — roughly seven times the mixed-bank result — though below the plan's
thresholds; its trajectory was still improving when patience stopped it. The wide-model arm was
invalid: its cloning loss *diverged* (rose steadily mid-training while accuracy stayed flat)
under the shared recipe — an optimization failure that says nothing about capacity, later fixed
by a gentler cloning recipe. Stage B (handoff-only after warm start) failed through
**catastrophic forgetting**: at its best checkpoint it solved a fair share of handoffs and
*none* of the single-arm bank — stage A's hard-won skill was erased by cloning on the new bank
alone.

**Limitations discovered:** one new concept per training phase is learnable; two at once are
not; and stacking phases without rehearsal erases prior phases. Both are ordinary properties of
small transformers under sequential fine-tuning; the fixes are curriculum design and rehearsal,
not new mechanisms.

**Curriculum split round 2 (config-only):** extending stage A pushed held-out well past the
thresholds; the wide model with a fair cloning recipe (lower cloning learning rate, more
epochs) reached near-perfect single-arm generalization *from cloning alone* — its RL stage
first collapsed the policy and then only recovered, never exceeding the cloned state; and the
rehearsed handoff stage retained the single-bank skill while lifting handoffs past the target
threshold.

**Limitations discovered in round 2:**

1. **Width was the binding constraint for the length concept — quantified.** On identical data,
   the small model saturated with residual length errors and a lower ceiling; the wide model
   eliminated length errors entirely and reached near-ceiling held-out performance from cloning
   alone. The concept lives in token embeddings with no geometric feature, and extracting that
   relation is what the extra capacity buys.
2. **Width does not fix sparse classes.** The long-distance handoff slices hold only a few
   dozen pairs each; even the wide model solves them poorly. That residual is a
   data-distribution limit (coverage), not a model limit.
3. **When cloning reaches the ceiling, RL is pure risk.** The wide model's RL stage cratered a
   near-perfect policy within a few iterations, then recovered to equal-but-never-better.
   Best-eval checkpointing and patience are the only guards; an explicit "skip RL when
   cloning-end eval is near ceiling" rule is a cheap future improvement.

**What it says about the architecture:** the jump between the two widths is a capacity cliff
for concept-from-embedding inference, not a gradual scaling effect — and at this problem's
scale (60–70-token programs, a few hundred levels), the wider model is effectively free
(per-iteration time unchanged). Below that width, curriculum and data compensate but slowly and
imperfectly.

**Data 10.1 — static-arms retrain**

- BC end: train greedy 0.158, eval 0.033; best eval-bank 0.058; same wall as the mixed bank.
- Taxonomy over 570 variants (train + eval), best checkpoint: solved 57; of 513 failures —
  wrong blueprint 415, wrong base 48, wrong program/ticks 28, invalid/other remainder.
- Among blueprint mismatches: bases fully shared with the reference in 60 cases, one of two in
  151, none in 204.
- Worst examples: tier-1-identical near slices failing with a longer arm where a short one
  belongs (dozens of cases across d1/d2).

**Data 10.2 — round 1**

- Stage A (single bank, small model): BC end 0.147 train / 0.160 eval; RL first eval 0.031
  (the crater), best 0.427 @ iter 1300, final 0.420 @ 1500 (patience stop). Slices at best:
  ring 1 41/85, ring 2+3 15/46; by distance — d1 10/34, d2 31/51, d3 7/18, d4 4/24, d6 4/4.
- Wide-model arm: BC loss trajectory 0.36 → 0.52 → 0.57 → 0.84 → 0.63/0.70 (accuracy flat
  ≈ 0.94); train greedy 0.002; best eval 0.015; early stop @ 600. Recipe at the time: cloning
  lr 1e-3, 56 effective epochs.
- Stage B (handoff-only after warm start): BC end 0.265 train / 0.044 eval; best combined eval
  0.079 @ 55; at its best checkpoint — handoff 20/121 (d3 7/38, d4 10/25, d5 2/34, d6 0/10,
  d7 0/8, d8 1/4, d9 0/2), single 0/131.

**Data 10.3 — round 2**

- R1 (extend stage A): first eval 0.389 → best 0.763 @ 2000 → final 0.725 @ 2400 (0.44 s/iter).
  Slices at best: ring 1 63/85 (d1 19/34, d2 44/51), ring 2+3 37/46 (d3 13/18, d4 20/24,
  d6 4/4); failures 11 wrong-blueprint / 7 wrong-base / 13 wrong-program.
- R2 (wide model, fair recipe): BC end loss 0.000, accuracy 1.000, train 1.000, eval 0.962;
  RL first eval 0.038 @ 5 (the crater), best 0.962 @ 105, final 0.947 @ 800 (0.38 s/iter).
  Slices at best: ring 1 81/85 (d1 30/34, d2 51/51), ring 2+3 45/46 (d3 18/18, d4 23/24,
  d6 4/4); failures 3 wrong-base / 2 wrong-program; zero wrong-blueprint.
- R3 (handoff + rehearsal, wide model, warm from R2): BC end on the mixed diet: train 0.975,
  eval 0.762; RL first eval 0.377, best 0.790 @ 95, final 0.770 @ 800. Slices at best:
  single 127/131 (d1 31/34, d2 51/51, d3 17/18, d4 24/24, d6 4/4); handoff 72/121
  (d3 26/38, d4 18/25, d5 20/34, d6 5/10, d7 1/8, d8 2/4, d9 0/2).
- R3 handoff failure classes by distance: d3 wrong-base 9 / blueprint 2 / program 1;
  d4 blueprint 3 / base 4; d5 blueprint 8 / base 4 / program 2; d6 blueprint 3 / base 1 /
  program 1; d7 blueprint 6 / base 1; d8 blueprint 2; d9 blueprint 2.

---

## 11. Phase 10 — the follow-up batch: width closed, bootstrap cost measured, handoff plateau mapped (Nov 16)

One preparation phase (per-level manifest weights, RL-skip rule, canonical handoff ranking,
coverage probe script) plus a single unattended chain of six runs: a 384-wide probe on the
single bank, a cloning-budget dial at the working width, and two handoff interventions
(rare-slice oversampling, canonical references) against the standing mixed-rehearsal baseline.

**Findings.**

1. **Width is closed: 256 is the sweet spot.** The 384-wide model *underperformed* 256 under
   the same recipe — its cloning stage reached a lower held-out rate and its best checkpoint
   stayed below the 256 model's cloning-alone result, with the hardest slice (d1) worse, not
   better. Combined with the earlier cliff, the picture is: 128 cannot extract the
   length-selection concept even with full coverage and heavy RL; 256 extracts it from cloning
   alone; 384 does not improve on 256. The capacity story is a step function at 256, not a
   slope. (Caveat kept honestly: recipe sensitivity has burned one width comparison before —
   the first 256 attempt diverged until its cloning recipe was fixed — so "384 worse" means
   "384 worse under the recipe that works for 256", not "384 useless".)
2. **The bootstrap cost of a new mechanic is now measured.** At the working width, on the
   single-arm bank: roughly a third of the cloning budget already yields full-quality results;
   at about an eighth of the budget the discovery stack (RL + ring-stratified replay) rescues
   a half-learned policy to near-parity. A new datapoint on RL's role came for free: RL adds
   *nothing* at ceiling (again) but adds enormously below it — its largest recorded gain on
   this family.
3. **Handoff held-out generalization has a plateau at roughly six-tenths, and it is robust.**
   Neither oversampling the weak classes nor canonicalizing the mirrored reference topologies
   moved held-out handoff beyond a few points over the baseline — and the canonical run shows
   the earlier "d5 anomaly" framing was too narrow: the held-out gap is spread across all
   handoff distances beyond d3, while the training slices are solved essentially perfectly
   under every intervention. Three different levers now produce the same signature: the model
   memorizes every trained handoff pair and generalizes the two-arm placement geometry only
   partially to unseen positions. This looks like a structural property of the
   two-arm-mapping function at this capacity, not a data or consistency artifact.
4. **The best tier-2 agent to date is the oversampling run** — slightly better than the
   previous best on combined held-out and clearly better on full-space coverage, with the
   single-bank skill essentially retained.
5. **The bootstrap cost of a mechanic is a data-volume cliff, not a smooth curve.** A
   follow-up bank-size sweep (subsampling the single bank at 25/50/75/100 % of its 525
   solutions, everything else fixed, eval always the complete 131-variant bank) showed held-out
   performance that is essentially zero at a quarter coverage, a third at half coverage, most
   of the way at three-quarters, and full only at full coverage (Data 11.5). Passes were held
   fixed, so the binding variable is the number of *distinct* solutions, not total exposure.
   The earlier "a few dozen instances per mode" rule was wrong in both directions: too little
   for this concept, and coverage near the top still buys real held-out performance.

**What it says about the architecture:** the stack's remaining tier-2 weakness is not
capacity (384 did not help), not supervision volume (oversampling did not help held-out), and
not reference consistency (canonicalization did not help) — it is that one arbitrary verified
solution per level teaches a single-point target for a genuinely multi-modal function, and the
model learns the trained points perfectly while interpolating the rest at ~0.6. The one untried
lever that targets exactly this signature is multi-reference supervision: emit several
distinct verified topologies per handoff pair, so cloning sees the solution *set* rather than
one ranked member.

**Data 11.1 — batch summary**

| run | BC end: train / eval | best eval | final eval | end |
|---|---|---|---|---|
| 384-wide, single bank | 0.899 / 0.809 | 0.878 | 0.870 | early stop @ 800 |
| dial k=9 (22 BC epochs) | 0.571 / 0.496 | 0.893 | 0.863 | early stop @ 1160 |
| dial k=7 (67 BC epochs) | 0.971 / 0.863 | 0.947 | 0.947 | early stop @ 1250 |
| dial k=5 (105 BC epochs) | 1.000 / 0.962 | 0.962 | 0.947 | early stop @ 800 |
| oversample (E-c) | 0.975 / 0.762 | 0.810 combined | 0.802 | early stop @ 1500 |
| canonical (E-d) | 0.952 / 0.726 | 0.798 combined | 0.790 | early stop @ 1770 |

**Data 11.2 — width comparison on the single bank** (identical bank and recipe family)

| model | cloning-alone held-out | best held-out | d1 slice at best | path |
|---|---|---|---|---|
| 128d | ≈ 0.16 | 0.763 | 19/34 (0.56) | ~3,700 RL iterations |
| 256d | 0.962 | 0.962 | 30/34 (0.88) | cloning alone |
| 384d | 0.809 | 0.878 | 27/34 (0.79) | cloning + 800 RL iters |

**Data 11.3 — dial at 256d, single bank** (full budget = k=1 ≈ 190 epochs)

| k | effective BC epochs | BC-end held-out | best after RL + replay | RL gain |
|---|---|---|---|---|
| 9 | 22 | 0.496 | 0.893 | +0.397 |
| 7 | 67 | 0.863 | 0.947 | +0.084 |
| 5 | 105 | 0.962 | 0.962 | 0.000 |

**Data 11.4 — handoff arms** (held-out handoff bank, 121 variants; single-bank held-out for
retention; full space = all 1260 pairs)

| arm | handoff held-out | d5 held-out | d7–d10 held-out | single held-out | full-space total |
|---|---|---|---|---|---|
| baseline (R3, mixed rehearsal) | 72/121 (0.595) | 20/34 (0.588) | 3/14 | 127/131 (0.969) | 1172/1260 (0.930) |
| oversample (E-c) | 80/121 (0.661) | 22/34 (0.647) | 5/14 | 124/131 (0.947) | 1211/1260 (0.961) |
| canonical (E-d) | 73/121 (0.603) | 17/34 (0.500) | 4/14 | 128/131 (0.977) | 1205/1260 (0.956) |

Handoff training slices under both interventions: solved at 0.998–1.000. The plateau is
held-out only.

**Data 11.5 — bank-size sweep** (single bank; subsample fractions of the 525 train solutions,
fixed recipe m256 / 111 BC epochs, eval always the complete 131-variant bank)

| train solutions | share of space | BC-end held-out | best held-out (RL + replay) |
|---|---|---|---|
| 131 | 25% | 0.061 | 0.076 |
| 262 | 50% | 0.237 | 0.351 |
| 394 | 75% | 0.786 | 0.840 |
| 525 | 100% | 0.962 | 0.962 |

All subsampled runs reached near-perfect *train* solve (≥ 0.958), i.e. they memorized their
subsets; the differences are purely held-out generalization. RL + replay added little at the
thin end (0.061 → 0.076), confirming the count of distinct solutions, not passes, is the
binding variable at fixed budget.

## 12. Phase 11 — feeding mechanism, procedural BC, and the tier-3 unlock (Nov 16–17)

Three linked efforts: the bank-size sweep follow-ups (why sparse banks fail), the
feeding/features confirmation cycle, and the procedural solver-fed cloning confirmation.

**Findings.**

1. **Distinct-instance feeding is the mechanism (confirmed, twice).** Cycled feeding — fresh
   seeded subsets of the full pool per epoch, budget equal to the failed static run — recovered
   nearly all of the full-bank result on tier-2, and on tier-1 produced a fresh 256-wide
   baseline clearly above the old 128-wide lineage from BC alone. Feeding controls *budget*;
   it does not control *coverage* — cycling inside a partial pool is by definition the static
   partial run.
2. **Encoder side-channel features do not substitute for coverage (rejected).** Neither a bare
   entrance→exit distance scalar nor a class-explicit per-ring reachability vector moved the
   sparse-bank result; both landed near the failed baseline. Even handing the model the
   relational answer does not generalize — representation is not the bottleneck, distinct
   instances are.
3. **A standing recipe rule, observed twice:** 256-wide behavior cloning diverges at lr 1e-3
   (loss rising mid-training while token accuracy stays flat, and per-token independence makes
   greedy ≈ accuracy^length) and works at 1e-4. Every m256 BC config uses 1e-4.
4. **Procedural solver-fed cloning is validated end-to-end.** Training from zero with no
   materialized train bank — every epoch's supervision freshly solved from sampled pairs —
   matched bank-fed quality exactly, with zero solver failures across all epochs. The tier-3
   data path exists.
5. **The coverage law holds under procedural feeding.** Fresh specs drawn from a half-coverage
   pool beat the static half-pool only modestly and stayed far from full-space performance;
   and the well-covered classes degraded too — the class-competition signature, now confirmed
   a third time.
6. **No cross-class extrapolation exists.** Trained on ring-1 pairs only, the model solved the
   seen classes at high rate and *never placed a long arm at all* on unseen classes — every
   unseen-class failure was a wrong-blueprint, despite RL rolling out over the full space.
   Classes absent from generation are never discovered later.
7. **Incidental infrastructure proof:** an accidentally mis-configured rerun reproduced a
   historical run value-identically across all its metrics rows with the new feeding/features
   plumbing compiled in but inert — the strongest determinism and no-regression evidence to
   date.

**What it says about the architecture:** the data question is now precisely characterized.
Feeding (fresh instances per epoch) solves budget; coverage (distinct pairs from the space)
solves generalization; representation shortcuts solve neither; and class completeness is
mandatory. Together these convert tier-3's prerequisite from "an enumerable bank" into "a fast
verified solver plus an enumerated config space" — positions and solutions come from the
solver, class quotas come from enumeration, and the bank survives only as an eval/rollout
artifact.

**Data 12.1 — feeding vs features confirmations** (static sparse baseline 0.076; full bank
0.962; recovery bar 0.8; eval = complete 131-variant bank)

| arm | setup | best held-out |
|---|---|---|
| cycled (C1) | 131-of-525 fresh subset/epoch | **0.95** |
| features weak (C2) | static 131 + distance scalar | 0.130 |
| features strong (C2) | static 131 + distance + ring reachability | 0.137 |
| tier-1 modern (R1) | full tier-1 bank, m256, cycled BC at quarter budget | **0.9419** (BC-end alone 0.907; old 128d lineage 0.814) |

**Data 12.2 — procedural confirmation** (from zero, no train bank, solver-fed, 131 pairs/epoch,
same recipe; pools per plan)

| run | pool | best held-out | slices at best |
|---|---|---|---|
| A pfull | full 525-pair space | **0.954** | d1 30/34, d2 51/51, d3 18/18, d4 22/24, d6 4/4 |
| B phalf | stratified 50% (262 pairs) | 0.435 | d1 8/34, d2 27/51, d3 7/18, d4 13/24, d6 2/4 |
| C pring1 | ring-1 pairs only | 0.595 | d1 29/34, d2 49/51; d3/d4/d6 **0/46** |

Static coverage-curve references: 131 → 0.076, 262 → 0.351, 394 → 0.840, 525 → 0.962.
C's unseen-class failures: all 46 wrong-blueprint. B's failures: 42 wrong-base, 19
wrong-ticks, 13 wrong-blueprint. All runs logged `bc_solve_failures = 0` on every epoch.

**Data 12.3 — supporting facts**

- R1 arm 1 (diverged): BC loss 0.18 → 0.81 rising, token acc ≈ 0.93, train greedy 0.00–0.03;
  0.93^56 ≈ 0.02 = its greedy rate. Config fix: lr 1e-3 → 1e-4.
- Determinism proof: 1664 metrics rows reproduced value-identically (timestamps stripped) with
  the new plumbing inert.
- Augmentation generates no variants for single-placement solutions (nothing to shuffle) —
  fresh per-epoch pairs are the only BC diversity source on simple solutions.

## 13. Measurement-methodology limitations (bugs that changed conclusions)

Recorded separately because they are about the *ruler*, not the learner. Each changed a
conclusion before it was found.

1. **Eval RNG coupling (v1).** All eval episodes drew one shared level per check; every
   reported eval rate before the fix was a single-variant measurement in disguise.
2. **Train-batch coupling (v1).** Each batch drew correlated waves from one shared stream,
   cutting effective batch diversity to a handful of levels per iteration.
3. **Augment tick-shuffle corruption (tier-2).** Cloning targets were internally inconsistent
   (see Data 9.2). Found by investigating a symptom — the rule that fell out: investigate
   anomalies before tuning knobs.
4. **Teacher-forced accuracy is not a solve metric.** From tier-2 onward, near-perfect token
   accuracy coexisted with near-zero greedy solve. Only per-variant greedy probes are
   decision-grade.
5. **Slice-based attribution needs care.** The v2 "edge/corner clustering" claim was an
   artifact of confounding; a proper distance × edge split showed distance was the only real
   axis.

---

## 14. Architecture conclusions (the distilled list)

1. **Policy gradient consolidates; it does not generate.** The gradient is an expectation over
   current behavior. What matters for learning a new behavior is its *reinforcement*
   probability, not its *generation* probability — and these diverge completely for coordinated
   multi-token novelties under terminal rewards (phase 4–5 evidence).
2. **Replay/imitation decouples learning from sampling probability.** Self-imitation converts a
   rare success into a persistent supervised target. Its weight must compete with the main
   gradient — a balance band exists, and the heavier setting was strictly worse (Data 7.1).
   Its group must be the *rare concept*: a failure signal cannot identify it (Data 8.1).
3. **Cloning ignition is load-bearing but thin is enough** — a heavily reduced cloning budget
   still ignited tier-1 once the discovery stack was in place; zero cloning ignites nothing
   (Data 5.3, 6.1).
4. **Coverage and consistency of the input→target mapping dominate instance counts.** Full
   enumeration made tier-1 learnable; the tier-2 collapse came from partial coverage plus a
   rank-based reference solver whose choices flip between neighboring inputs (Data 9.5–9.6).
5. **Learnability tracks solution forcedness.** Where geometry leaves one answer, even few
   examples generalize; where many valid solutions exist and a rank rule picks one,
   interpolation fails (Data 9.4, 9.6).
6. **Teacher-forced accuracy is decision-grade only for short, forgiving programs.** Per-variant
   greedy probes are the only trustworthy metric for timing-sensitive sequences.
7. **Saturated RL drifts.** Once the training distribution is solved, entropy is the only
   active objective and the policy random-walks. Patience-based early stop plus best-eval
   checkpointing turn this from a failure mode into a non-event.
8. **One concept at a time, with rehearsal.** Two orthogonal concepts in one mixed bank
   produced interference (neither learned); stacking phases without rehearsal erased the prior
   phase; mixed-diet rehearsal fixed the second and per-class banks fixed the first.
9. **128-dim was a capacity cliff for concept-from-embedding inference; 256d removed it.** The
   length-selection concept was unlearnable at the smaller width even with full coverage, and
   trivially learnable at the larger one from cloning alone (Data 10.3). Width was not the
   limit for sparse-class coverage — that is a data property.
10. **Spec-derived labels are fair curriculum inputs.** The one piece of outside information
    the loop needs is "which slice is the rare one", computable from level data. Three attempts
    to derive it from the agent's own signals (scarcity, failure-driven replay) or to avoid it
    via blur (temperature) failed in distinct, diagnostic ways (Data 8.1).
11. **When cloning reaches the ceiling, RL adds nothing but risk.** The guard pattern
    (best-eval checkpointing + patience) is mandatory; an explicit skip rule is a cheap future
    improvement (Data 10.3). Refined in phase 10: below ceiling, RL + replay is the strongest
    refinement tool on record (Data 11.3) — the rule is "skip at ceiling, not always".
12. **Width is a step function, not a slope.** The length-selection concept was unlearnable at
    128, free at 256, and no better at 384 (Data 11.2). Capacity questions on this family are
    settled by one bounded probe per width, not by scaling defaults.
13. **One reference solution per level teaches a single-point target for a multi-modal
    function.** Perfect training-slice memorization with ~0.6 held-out interpolation survived
    oversampling and reference canonicalization (Data 11.4) — the remaining tier-2 gap is a
    property of the supervision shape, and multi-reference emission is the matching lever.
14. **Feeding solves budget; coverage solves generalization.** Fresh subsets per epoch recover
    full-bank quality at a fraction of the budget (Data 12.1), but a partial pool fails under
    any feeding (Data 12.2, B) — the two axes are independent and both must be satisfied.
15. **Representation shortcuts do not substitute for instances.** Even class-explicit encoder
    features left the sparse-bank result at the baseline (Data 12.1) — giving the model the
    relation is not giving it the mapping.
16. **No cross-class extrapolation.** Classes absent from the training generation are never
    learned — not by RL over the full space, not from adjacent classes (Data 12.2, C). Every
    class of the config space must be generated from the start.
17. **Ignition is an operational threshold on sampled solve rate, not on greedy accuracy.**
    Cloning's role is to move the sampling distribution into the reward-bearing region; once
    sampled episodes solve at all, RL + stratified replay do the mode construction. At full
    coverage, about an eighth of the cloning budget sufficed — greedy held-out was only half
    at that point, yet the discovery stack closed nearly all of the rest (Data 11.3, k=9).
    Below the threshold the failure is binary, not slow (zero solves across the whole no-BC
    run, Data 5.3); at the ceiling, RL adds nothing (Data 11.3, k=5). Crucially, ignition is
    cheap in *passes*, not in *coverage*: the dial holds coverage fixed and varies passes,
    while the bank-size sweep shows thin coverage is rescued by nothing (Data 11.5). The two
    axes must not be conflated when budgeting a new mechanic.
18. **A new mechanic needs complete class enumeration plus near-complete within-class
    coverage.** Within a class, training "starts working" around half coverage, becomes
    reliable around three-quarters, and reaches full quality only at complete coverage
    (Data 11.5, 12.2 B); classes absent entirely contribute exactly zero (Data 12.2, C).
    Derived planning consequence for tier-3 (inference, not measurement): the space to
    enumerate is level *configs* — small, so complete coverage is free — while positions live
    in the solutions the solver emits. That makes the binding choice the per-config topology
    policy: one canonical solution per config reduces the task to a small memorized lookup
    (the tier-1 v1 lesson — works, but teaches no transferable skill); freshly varying
    positions per epoch under procedural feeding is contradictory supervision on identical
    inputs (the d5 mechanism, but universal); the middle ground is a small fixed set of
    canonical topologies per config, which is exactly the multi-reference lever of
    conclusion 13.

---

## 15. Decisions ledger

**Chosen:** sampled-stratified then enumerated banks; exhaustive per-variant eval; best-eval
checkpointing; eval patience early stop; low entropy in consolidation; autodiscovery dial;
stratified replay at the moderate coefficient with the spec-derived rare-class label; static
length-arm blueprints (wire insertion, not a grammar change; warm-start id remapping);
warm start via model-weights-only loading; per-concept enumerated banks; mixed-diet rehearsal;
the wider model as the working default; **cycled BC feeding** (`bc_cycle`) wherever per-epoch
budget is below bank size; **procedural solver-fed BC** (`bc_source = "procedural"`) as the
tier-3 data path; **BC lr 1e-4 on the 256-wide model** (standing rule); the modern tier-1
lineage (0.9419) as the tier-1 anchor.

**Rejected, with the evidence that rejected them:** small sampled banks (memorized, v1);
ε-uniform exploration (breaks working programs, no rare-slice gain — Data 6.1); progress
shaping as a rare-slice signal (structurally blind where the entrance respawns — Data 6.1);
scarcity weighting (dilution — Data 8.1); high temperature (blurs sampling quality — Data 8.1);
failure-driven replay (repairs the majority, misses the rare — Data 8.1); pure RL without a
seed (zero signal — Data 5.3, 6.1); skipping the eval bank (unmeasurable); tier-0 as a warmup
stage (distribution mismatch); procedural per-reset generation (deferred — bank sizes remain
tractable); static-length grammar fields (wire-format break; blueprint insertion chosen
instead); sharing the wider model across incompatible checkpoints (shape mismatch is a hard
error by design); **384-wide default** (below 256 under the same recipe — Data 11.2);
**further handoff tuning via oversampling or reference canonicalization** (held-out plateau
robust to both — Data 11.4); **encoder side-channel features** (class-explicit input did not
move the sparse-bank result — Data 12.1).

**Open at the time of writing:** multi-reference supervision for handoff pairs (emit several
distinct verified topologies per level; targets the single-point-supervision signature
directly) — the one untried tier-2 lever, and a candidate to build into the tier-3 solver from
the start. Tier-3 (unfixed entrypoints) is next under the solver-first design: enumerated
config space with per-class quotas, procedural BC, eval/rollout-only banks, per-concept staging
with rehearsal, warm start from the tier-2 mixed checkpoint (design detail in
`notes/tier3_readiness.md`).

---

## 16. Artifact index

Session reports (some later removed from the repo; their content is preserved above and in
this session's transcript): `MODE_TOKEN.md` (mode-token naming diagnosis), `REPORT.md`
(tier-1 hardening matrix), `TIER1_DISCOVERY_TEST.md` (discovery matrix + stratified-SIL retry),
`RAPORT.md` (failure-driven replay retry), `ANALYSIS.md` (narrative summary as of tier-1 v2).

Plans: `.opencode/plan/` — `tier1_generalization.md`, `tier1_generalization_v2.md`,
`tier1_autodiscovery_test.md`, `tier1_discovery_test.md`, `tier2_anywhere.md`,
`tier2_curriculum_split.md`, `tier2_followup_experiments.md`, `bc_feeding_and_features.md`,
`procedural_bc_confirmation.md`. Stage reports for the last three live in
`.opencode/report/`.

Key run directories (under `runs/`):

| run | phase | result |
|---|---|---|
| `20261112-032904` / `033417` | tier-1 v1 | memorization; eval artifact found here |
| `20261112-060521` | tier-1 v2 | 0.814 eval; canonical lineage |
| `20261112-1337…16:30` | dial k=5/7/9/8 + k9-ext | supervision curve; rare-slice decomposition |
| `20261112-1943…20:14` | discovery matrix | all rare-slice arms zero; buffers caught a few |
| `20261112-222033` / `223452` | stratified SIL | lock-in broken |
| `20261112-232912` | hardening baseline | historical reference (0.663 best) |
| `20261113-025144 ×2` | scarcity / temperature | both failed |
| `20261113-050709` | failure-driven replay | majority best-ever, rare collapsed |
| `20261114-232314` | tier-1 canonical retrain | 0.535; warm-start source for tier-2 v1 |
| `20261114-234… Nov 15 00:xx` | tier-2 mixed bank | collapse; augment bug; post-fix probe |
| `20261115-193447` | static-arms retrain | same wall; taxonomy taken here |
| `20261115-213832` / `221850` | split round 1: A / wide | A 0.427; wide BC diverged |
| `20261115-224309` | split round 1: B | handoff partial, singles wiped |
| `20261115-230641` | round 2: extend A | 0.763 best @ 2000 |
| `20261115-235504` | round 2: wide model | 0.962 from cloning alone |
| `20261116-005140` | round 2: rehearsed handoff | handoff 0.595, single 0.969 |
| `20261116-060738` | follow-up: 384-wide probe | 0.878 — below 256 |
| `20261116-063808` / `065509` / `071930` | follow-up: dial at 256 | bootstrap cost measured |
| `20261116-074844` | follow-up: oversample | best tier-2 agent; full-space 0.961 |
| `20261116-084635` | follow-up: canonical refs | held-out unchanged — plateau confirmed |
| `runs/batch_F3` | follow-up batch chain log | — |
| `20261116-0941…1042` ×3 | bank-size sweep s25/s50/s75 | coverage cliff (Data 11.5) |
| `20261116-213524` | cycled feeding (C1) | 0.076 → 0.95 at equal budget |
| `20261116-215830` / `220644` | features weak / strong (C2) | 0.130 / 0.137 — rejected |
| `20261116-234310` / `20261117-002502` | R1 tier-1 modern (diverged / fixed) | 0.02 / **0.9419** |
| `20261116-235714` | accidental R2 | 1664-row value-identical reproduction |
| `20261117-015433` / `021653` / `024935` | procedural A / B / C | 0.954 / 0.435 / 0.595 |

Config lineage: `configs/tier1_simple_arms.toml` (canonical tier-1), `tier1_disc_sil2.toml`
(best tier-1 stack), `tier1_ad*.toml` (dial), `tier1_disc_{full,shaping,eps,sil,full_k10,sil2,
sil3}.toml` (discovery matrix), `tier2_anywhere.toml` and `tier2_fresh.toml` (mixed bank),
`tier2_single*.toml`, `tier2_handoff*.toml`, `tier2_handoff_mix.toml` (curriculum split).
