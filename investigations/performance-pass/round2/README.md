# Round 2: an audit of the performance pass, and what v7 still loses to v6

Asked for by the user on 2026-09-28: check every claim of the first pass (`../../../BENCHMARK.md`,
`../README.md`) three times against its raw data, set its method against current Julia practice,
and find what else brings v7 up to v6 on the models still behind. Nothing here is applied to `src/`
or `lib/`; every change is a diff in `diffs/`, on top of round 1's `ENGINE-T-ReactiveMP.diff`
(`src/` and `lib/` are unchanged since the revision round 1 measured, `d50fac9c`).

**Status (2026-09-28): in progress.** Done: the audit (§1), the method (§2), the v6/FULLT baseline
(§3) and the prototypes (§4), each through the root suite. Running: the paired v6/Q benchmark of
every model. Left: its results and the posterior gate on them, RxInfer's suite and `make test-all`
on Q, hmm's remaining bytes, and one confirmation run on an idle machine.

The machine was in use by the user throughout, so every timing here is **indicative**; the method
below is built for that, and the figures to publish come from one confirmation run on an idle
machine. Allocation counts, bytes, GC counts, JET reports and the correctness gate do not depend
on load.

## 1. The audit of round 1

Every number in round 1's tables was recomputed from `../results/final*/models.tsv` and
`micro.tsv`. **The tables are transcribed correctly**; the conclusions drawn from them are not all
supported.

### 1.1 "Faster than v6 in steady state on six of seven iterative benchmarks" does not hold as stated

- **n = 10⁴ is decided by where a GC pause fell.** Round 1's per-iteration figure is
  (min T(2I) − min T(I)) / I, the two minima from different samples. v6's T(2I) had a GC pause in
  rounds 2 and 3 of run 2 (GC share 16.6% and 17.5%) and none in round 1: 16.8 and 17.2 ms against
  11.5 ms per iteration. FULLT's never did (14.2–16.3 ms). The claimed 0.86× compares a paused v6
  with an unpaused FULLT; GC-free against GC-free, FULLT is **1.29× slower**. The same artifact
  gives FULL, FULLT and FULLTPC, which allocate identical setup bytes (248.7 MB), setups of 158,
  221 and 257 ms, and v6 11.2 ms in run 1 against 16.8 ms in run 2.
- **End to end, v7 is slower at every iteration count benchmarked**, except the streaming filter
  and nl: FULLT/v6 total time at I and 2I is gmm 1.36/1.15, iid 1.34/1.29, hmm 1.10/1.03, iid@10⁴
  1.30/1.13. "Faster in steady state" is true of the marginal cost of one more iteration only, and
  break-even is at about 40 iterations (gmm), 36 (hmm), never (iid at n = 10³).
- **Two slower benchmarks are left out of the count**: iid@100 (1.03×) and iid+defaults, RxInfer's
  defaults (1.21×). It is 6 of 9, and 5 of 9 without the n = 10⁴ artifact.
- **The derived rows amplify noise.** Setup + I × per-iteration ≠ T(I), since `summarise.py` takes
  medians of each derived row separately; between rounds the derived rows vary by 3–45% where
  totals vary by 0.5–5%. With three rounds, ±5–10% (hmm 0.95×) is within noise.
- **The main table mixes runs**: v6 from run 2, HEAD/P12/E4 from run 1.

### 1.2 First inference

- "≈ 9 s → under 1 s on models shaped like the workload" holds for iid, iid@100 and betabern only.
  ssm1, the BP state-space model *in* the workload, goes 13.6 → 6.6 s; iid+defaults stays at 2.2 s
  (the workload does not exercise `free_energy = true` or the session); "29–60% on the others"
  leaves out the filter (73%).
- **The fair comparison is FULLT against v6, neither with a workload: v7 is 1.13–1.30× slower.**
  RxInfer 5.5.2 has no workload (checked), and v6 would gain from one too.
- The precompile cost has no data behind it, and the two figures given contradict each other
  (≈ 21 s against 10 s in `BENCHMARK.md`; 5 → 70 s in `../notes/engine.md`).

### 1.3 "Steady state 4–20% faster with the workload": the cause (new)

Julia PR #61474 (merged 2026-05-21, backported to 1.13; checked) loads sysimage and package-image
objects already marked old, so a full collection does not mark them. What the JIT compiles lives
on the ordinary heap and is marked every time. Measured here (`jit_heap.jl`): a full collection
takes ≈ 30 ms after loading and 55–57 ms after the first inference, which adds ≈ 27 MB of live
heap, on v6 and v7 alike. A workload moves that into the package image; the effect is largest on
the GC-bound runs, as round 1 saw (iid@10⁴ 0.81×, @10⁵ 0.86×; nil on linreg). **v6 would gain the
same from a workload**, so FULLTPC-against-v6 ratios mix the engine's gains with the workload's.
The "4–20%" range is itself selective: FULLTPC's T(I) at 10⁴–10⁵ is 2–3% slower and its setup
12–17% slower.

### 1.4 The correctness gate is weaker than "bit-identical"

- `summarise1` stored `(nameof(type), mean, cov)`: type parameters dropped, no covariance for
  Categorical and DirichletCollection, and a Categorical's `mean` is one scalar, so each HMM state
  posterior was checked by one number.
- Run 2's reference variant is not recorded and `post/` is not committed; "153/153 identical to
  HEAD" holds only by transitivity (FULLT = FULL in run 2, FULL = A in run 1).
- P4 was checked on models without a form constraint only; it changes "check last" for a
  variable of degree ≥ 3 with one.
- The HMM benchmark sits at a symmetric fixed point (every entry of `q(A)` equal, uniform state
  posteriors), so its messages are uninformative; it may not represent an HMM's cost.

### 1.5 Micro benchmarks

- The headline "27 ns rule call" is `map/direct`; the engine's call crosses the stream barrier,
  `map/barrier`, 40 ns and 3 allocations under T.
- **Every product carried a fixed ≈ 500 ns** that no diff touched: the rebuild of the product's
  `Message` from a value of unknown type after the fold (`src/message.jl:394–421`). v6 has the
  same code, so removing it gains *over* v6 (Q1 below).
- The micro suite never ran on v6, and v6's setup was never broken down against v7's.

### 1.6 Inside the recommended diffs

- **P4** changes semantics as above; with `GenericProd`, partial products can grow into nested
  `ProductOf` trees. It needs a test with a form-constrained variable of degree ≥ 3.
- **P9**'s stack guard is one process-wide, non-atomic counter: threads race on it, a yield in a
  guarded region corrupts it, and `throw(task.exception)` loses the backtrace. It must be per task
  (not prototyped: a task-local lookup costs ≈ 2 × 30 ns per lazy subscription, so the per-thread
  counter with a sticky child task is the candidate).
- **P1**'s counter span ids print `0000…` in the compact display, collide across processes and
  contradict the docstring (fixed in Q, below).
- **P5b**'s global alias cache takes a lock and a hash per lookup and goes stale under Revise; Q3
  below replaces it on the path that matters.
- Smaller: README:28 labels FULLT "FULL" and `pagedata.json` relabels the variants; README and
  BENCHMARK disagree on where the page data is; 124/126 ns and 95/97 ns; "v6 within 6% between
  runs" is false for hmm and gmm setup (−11%, −8%); `rocket_stack.tsv` shows 1 024 → 4 096 links,
  not 1 030 → 3 264; P12's setup is worse than HEAD's on gmm and hmm, unmentioned.

## 2. Method of round 2

- **Minima, of one quantity each** (the user's choice: the minimum is what is achievable, and load
  only adds to it). Per-iteration is (min over samples of T(2I) − gctime) − (the same at I),
  divided by I: both minima GC-free-equivalent, so a pause's placement does not decide it. Setup
  is min(T(I) − gctime) − I × per-iteration, from the same minima.
- **Paired runs**: for each model the variants run back to back in fresh processes, round after
  round, the order rotating (`driver2.sh`); the ratio in a round is between samples minutes apart,
  and a change is called faster or slower only when its range over rounds excludes 1.
- **Every sample recorded raw** (`bench_models2.jl`): wall time, the thread's CPU time, GC time,
  bytes, GC collections, a 20 ms calibration loop just before it and the load average; 11 samples
  per iteration count per process, I and 2I alternating; twice round 1's iteration counts.
- **Deterministic evidence first**: allocation profiles by type and line (`alloc_profile.jl`),
  inclusive CPU profiles by function (`prof_setup.jl`, `cmp_prof.py`), line profiles
  (`prof_lines.jl`), JET, and a node creation and activation micro suite (`bench_setup_micro.jl`).
- **The gate compares distributions**: every parameter of every posterior and the free energies
  (`compare_variants2.jl`), naming the reference on every line.
- No callbacks are set anywhere: v6 draws a `uuid4()` span id for every rule call whenever any
  callback is (`callbacks.jl:180` in 6.5.0), so timing iterations through callbacks would slow v6
  alone.

## 3. Where v7 still loses to v6 (FULLT, round 1's recommended set)

The baseline, `driver2.sh` with v6 and FULLT, 5 paired rounds, all 17 models
(`results/base_models.tsv`, summarised in `results/base_tables.md`):

- **On the minimum total, FULLT is slower than v6 everywhere except the streaming filter (0.64×),
  nl (0.96×) and iid at n = 10⁵ (0.94×)**: ssm1 1.33×, ssm2 1.25×, betabern 1.45× (1.61× at
  5·10⁴), iid 1.35× (1.49× at 10⁴), gmm 1.17×, hmm 1.06×, and 1.19–1.47× on the scaling cases.
  Every range over the five rounds excludes 1.
- **Per iteration (GC-free-equivalent) FULLT is level or ahead**: gmm 0.86×, hmm 0.97×, nl 0.61×;
  iid 1.09×.
- **Setup is the gap**, 1.4–2.0× v6's (iid 19.3 against 12.0 ms, gmm 33.1 against 16.2, nl 27.7
  against 15.7, hmm 14.3 against 9.0).

What the gap is made of, from inclusive CPU profiles of the same `infer` on both
(`prof_setup.jl`, `cmp_prof.py`; ssm1, gmm, iid, hmm, nl):

- **`factornode`**: 0.34 µs per node on v6, 1.6 µs (NMV) to 5 µs (NormalMixture) on FULLT. v6
  took the interfaces in GraphPPL's order and the factorisation as positions, with an
  `alias_interface` method generated per node type. v7 resolves every interface by name, builds a
  `Dict{Any, Any}`, sorts groups, checks group lengths and converts the factorisation's keys back
  to positions for every node, in a function that never specialises on the node type
  (`factornode(fform::F) where {F}` binds `F = DataType`) — and RxInfer converts GraphPPL's
  positions into keys first.
- **Dependency resolution at activation**: the default or declared scheme is worked out again for
  every interface of every node (`declared_dependencies` alone is 2.4 ms of gmm's `infer`).
- **The `MessageMapping` constructor**: `Val(logscales)` from a run-time `Bool` makes its type a
  run-time construction for every interface.
- Variable activation, the equality chain and the stream wiring are v6's structure and cost about
  what v6's do.

Bytes, exactly (`bytes_per_iter.jl`): FULLT makes as many allocations per iteration as v6 on iid
(21 000) but 31% more bytes (945 against 722 KB), because **every `DeferredMessage` holds a copy of
its `MessageMapping`**: the mapping is immutable and stored inline, the factor node inside it, 96 B
against v6's 72.

The GC, measured (`live_heap.jl`, `jit_heap.jl`): a full collection with the model alive costs the
same on both (iid@10⁴: 114 ms v6, 110 ms Q), so the live graph is not heavier. A full collection
after the first inference costs 55–57 ms against 30 ms right after loading, the JIT's output being
on the heap, the same on both (§1.3).

## 4. The prototypes (`diffs/`, on top of round 1's `ENGINE-T-ReactiveMP.diff`)

Each passes the root suite (15 339 pass, 6 known broken; `diffs/Q-ReactiveMP.diff` is all of
them). Setup figures are per node from `bench_setup_micro.jl` (1 000 nodes of one shape, minimum
of 15 samples).

| diff | what | measured |
|---|---|---|
| `Q1Q4Q5-message` (Q1) | a barrier after the product's fold: the form constraint and the rebuild run on the product's concrete type, and a constraint that returns its input keeps the product | the fixed ≈ 500 ns per product is gone; gmm's products take less than v6's (3.1 against 4.6 ms per `infer`) |
| `Q3-creation-plan` (Q3) | a creation plan per (node type, interface keys, factorisation): where each interface comes from, the clusters and their keys, resolved and checked once; `FactorNode` keeps it | creation 3.4 → 0.71 µs per NMV node, 10.3 → 1.4 µs per NormalMixture node |
| `Q3-creation-plan` (Q3b) | the dependencies of each target kept in the plan as positions, per declaration | activation 5.4 → 4.9 µs per NMV node, 18.1 → 12.0 µs per NormalMixture node |
| `Q1Q4Q5-message` (Q4) | `logscales ? Val(true) : Val(false)`, so the mapping's type is inferred | fewer setup allocations; no measurable time |
| `Q1Q4Q5-message` (Q5) | `MessageMapping` a `mutable struct` of `const` fields, allocated once per interface, so a `DeferredMessage` holds a pointer (16 B instead of 96–104) | bytes per iteration iid 945 → 785 KB (v6 722), gmm 1 927 → 1 511 KB (v6 1 905), hmm 3 345 → 3 225 KB (v6 2 708) |
| `Q-audit-cold-path` | `audit_rule`'s error and warning behind `@noinline` calls, its checks still inlined | was ≈ 2% of iid's samples on its own lines |
| `Q-span-ids` | span ids from the counter spread by an odd multiplier and a per-session salt drawn in `__init__` | unique in a session, distinct in their first digits (the compact display), apart across sessions; no draw per call |

**Rejected**: a function barrier per interface in `activate_messages!` (5.7 against 4.9 µs per
node). JET shows why: `NodeInterface.variable::AbstractVariable` makes the inbound stream types
`Any`, so everything inside the barrier stays dynamic and the barrier adds a dispatch.

## 5. Where it stands

- **Per iteration, v7 is at or ahead of v6** on every model but iid at n ≤ 10³ (1.09×, CPU work in
  the products and materialisation, not GC or allocation), and the prototypes do not change that
  much; with Q5 v7 allocates less than v6 on gmm and 9% more on iid.
- **Setup is where v7 loses, and the prototypes take most of it back**: in the profiled `infer`,
  before Q5, v7 against v6 went from 1.28× to 1.12× on ssm1, 1.33× to 1.10× on gmm, 1.35× to 1.22×
  on iid, and 1.04× on hmm and nl. The first model of the paired run gives Q 1.11× v6 on ssm1
  (FULLT 1.33× in the baseline).
- **What remains** is mostly the reactive wiring built for every node, dynamic because the
  variables are held abstractly (`NodeInterface.variable::AbstractVariable`), which v6 pays too,
  and RxInfer's own per-node work.
- **Not prototyped, recommended**: P9's counter per thread or task; a test of P4 with a
  form-constrained variable of degree ≥ 3; a degree-2 fast path in the equality chain (beats v6 on
  state-space models; changes the product events callbacks see); a lazily allocated
  `AnnotationDict`; precompile workloads per rule package and one with `free_energy = true` and the
  session in RxInfer's; log scale `nothing` for data and constants when untracked.

