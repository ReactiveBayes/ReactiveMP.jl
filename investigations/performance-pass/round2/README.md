# Round 2: an audit of the performance pass, and what v7 still loses to v6

Asked for by the user on 2026-09-28: check every claim of the first pass (`../../../BENCHMARK.md`,
`../README.md`) three times against its raw data, set its method against current Julia practice,
and find what else brings v7 up to v6 on the models still behind. Nothing here is applied to `src/`
or `lib/`; every change is a diff in `diffs/`, on top of round 1's `ENGINE-T-ReactiveMP.diff`
(`src/` and `lib/` are unchanged since the revision round 1 measured, `d50fac9c`).

**Status (2026-09-28): measured and confirmed on an idle machine.** The recommended set is **F**:
round 1's engine changes (typed `RuleSpec`, P1, P2, P5a, P5b; P4 reduced to its typing, §4), the
round-2 prototypes (Q1, Q3, Q3b, Q4, Q5, the audit cold path, the span ids, the context once per
node, the typed interface vector), RxInfer's P6 and round 2's plugin changes, and Rocket's P7
*without* P9. **F is faster than v6 on every model**: 0.53–0.92× end to end, every paired
round below 1 (§6). Its posteriors and free energies are bitwise those of round 1's set on every
model, and v6's to within 3·10⁻¹² (§6). The root suite (with Aqua, 10 new items) and RxInfer's
suite (14 713) pass on it.

Until §6 the machine was in use by the user, so those timings are **indicative**; the method
below is built for that. §6 is the confirmation run, on the idle machine. Allocation counts, bytes, GC counts, JET reports and the correctness gate do not depend
on load.

## F, the recommended set, from the diffs

| package | on top of | diffs, in order |
|---|---|---|
| ReactiveMP | `d50fac9c` (= `src/` and `lib/` at HEAD) | `diffs/ALL-round1-and-round2-ReactiveMP.diff`, all of it in one; or `../diffs/ENGINE-T-ReactiveMP.diff`, `diffs/Q-ReactiveMP.diff`, `diffs/R-context-once-union-interfaces-ReactiveMP.diff` |
| RxInfer | `4ca9eb88` | `../diffs/P6-RxInfer.diff`, `../diffs/P6bug-RxInfer.diff`, `diffs/R-plugin-RxInfer.diff` |
| Rocket | 1.10.0 (`eba18dfc`) | `../diffs/P7-Rocket.diff` only (not P9; §4) |
| GraphPPL | 4.8.0 (`4cc7c6c`) | `../diffs/P8rel-GraphPPL.diff` |

`diffs/P9args-Rocket.diff` is P9 in the argument form, on top of P7 and P9, if an automatic stack
limit is wanted after all. Variants are built with `mkvariant2.sh` (or `mkv6.jl` for v6) and
measured with `driver2.sh`, `summarise2.py` and `compare_variants2.jl`.

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

| `R-context-once-union-interfaces` | activation builds the node's rule context once and gives it to every mapping, as `node_context`'s docstring already said (`message_mapping`, `marginal_mapping`); a node with groups keeps its interfaces in a `Vector{Union{NodeInterface, IndexedNodeInterface}}`, a small union the compiler splits, instead of a `Vector{Any}`; the creation plan records the node's declaration and is not used after a redefinition (Revise); `FactorNode.plan` is typed | activation 4.8 → 4.45 µs per NMV node, 11.4 → 10.6 µs per NormalMixture node; 10 new test items |
| `R-plugin-RxInfer` | RxInfer's per-node plugin: whether the node is declared (`applicable`) and its groups hoisted out of the per-edge loop, the interfaces collected into a vector of a declared element type rather than widened by `map` at run time, and the activation options built by a function of the node's algorithm and postprocessor through the positional constructor, not by a keyword call on `Any`-typed values | `set_rmp_factornode!` 3.6 → 1.7 ms on ssm1 (v6 2.5 ms): the per-edge `map` cost v6 as much as it cost v7 |
| (the equality chain) | **round 1's P4 reverted to HEAD's semantics**, keeping only its typed `ChainOutboundMapping{C}`: every partial product goes through `compute_product_of_messages` again, the form constraint checked on each. P4's two-message step let an unsupported `ProductOf` reach the chain's `missing` boundary, where `prod(::GenericProd, ::ProductOf, ::Missing)` is a BayesBase ambiguity, so RxInfer's "undefined functional form" hint turned into a `MethodError`: 1 of RxInfer's 14 713 tests failed on round 1's set (FULLT), which round 1 never ran RxInfer's suite on. With P1 and Q1 the full product of a pair costs little more than the two-message one | RxInfer's suite passes again |

**Rejected**: a function barrier per interface in `activate_messages!` (5.7 against 4.9 µs per
node). JET shows why: `NodeInterface.variable::AbstractVariable` makes the inbound stream types
`Any`, so everything inside the barrier stays dynamic and the barrier adds a dispatch.

### The stack guard (round 1's P9), measured

Round 1 adopted P9 because an intermediate engine variant overflowed the stack on a 1 000-link
chain without it. Measured here with round 1's `stack_depth.jl` (8 MiB task stack, no
`limit_stack_depth`), and paired on the setup-bound models (`results/p9_*`):

| | longest ssm1 chain | ssm1 | betabern | gmm |
|---|---|---|---|---|
| v6 | 1 344 | | | |
| v7 without the guard (Rocket with P7 only) | 2 304 | 1 | 1 | 1 |
| v7 with P9 as round 1 wrote it (a closure per lazy subscription) | 3 776 | 1.077 | 1.023 | 1.017 |
| v7 with P9 taking the function and its arguments, no closure | 3 776 | 1.063 | 1.013 | 1.000 |

- v7 without the guard already takes 1.7× v6's chains: the typed `RuleSpec` and round 2's
  changes made each link's frames shallower than the variant round 1 measured.
- The guard's remaining cost is the `try`/`finally` around every lazy subscription; round 1's
  own `rocket_stack.tsv` shows it (+7.5% on ssm1 with `limit_stack_depth = 100`), though round 1
  timed it only without that option.
- **Recommendation: leave P9 out.** v7 handles longer chains than v6 without it, and
  `limit_stack_depth`, which v6 users already know, covers the rest. If an automatic limit is
  wanted after all, take the argument form and make its counter per task.

## 5. Where it stands, and what the design costs

- **v7 is faster than v6** with F: 0.53–0.92× on every model, in steady state and in setup (§6).
- **The design's checks cost nothing measurable any more.** The name-based, validated node API is
  resolved once per node shape (the creation plan) instead of once per node, and the rule call's
  checks (`check_services`, the diagnostics, the log-scale declaration) fold into a 27–40 ns call.
  What made v7 slower was not the design but a handful of mechanical costs, each fixed with an
  ordinary Julia practice:
  - work repeated per node that depends only on the node's shape: cache it per shape;
  - containers whose element type Julia must work out at run time (`[x...]`, `map` over
    `Any`-typed values): declare the element type, a small `Union` where two kinds mix;
  - values built from run-time types in hot or per-node code (`Val(b)`, keyword calls with
    `Any`-typed values, closures over run-time-typed captures): branch to static values, call
    positional constructors behind a function barrier, pass arguments instead of capturing them;
  - immutable structs holding large immutable structs, copied into every object that holds them:
    make the shared one mutable with `const` fields;
  - per-object work done per use (the rule context per mapping): hoist it to where it is shared.
- **What remains v7's price** is compilation: first inference 1.03–1.35× v6 without a workload.
- **Not prototyped, recommended**:
  - a precompile workload in RxInfer (round 1's P11, extended with `free_energy = true` and the
    session) and in the rule packages with many nodes (DiscreteTransition, Delta);
  - a degree-2 fast path in the equality chain (every state-space model's chain variables; it
    changes the product events callbacks see);
  - a lazily allocated `AnnotationDict`;
  - log scale `nothing` for data and constants when log scales are not tracked;
  - hmm's remaining allocation (80 against 72 MB), to profile;
  - the BayesBase ambiguity `prod(::GenericProd, ::ProductOf, ::Missing)`, upstream.

## 6. The confirmation run (idle machine)

`driver2.sh` with v6, FULLT (round 1's set) and F, all 17 models, 5 paired rounds, 11 samples per
iteration count (5 for iid@10⁵ and betabern@5·10⁴), and GC-off samples besides
(`results/final/`: `models.tsv`, `models_gcoff.tsv`, `tables.md`, `tables_gcoff.md`). The ratio is
the variant's minimum over v6's in each round; min, median and max over the five rounds:

| model | v6 T(I) | FULLT / v6 | F / v6 | F: per iteration / setup (no GC) |
|---|---|---|---|---|
| ssm1 (Kalman smoother, BP) | 40.8 ms | 1.31, 1.32, 1.37 | **0.84, 0.87, 0.90** | one pass |
| ssm2 (2-D Kalman) | 76.2 ms | 1.24, 1.25, 1.26 | **0.85, 0.85, 0.86** | one pass |
| betabern | 49.5 ms | 1.41, 1.42, 1.44 | **0.80, 0.81, 0.81** | one pass |
| iid, I = 20 | 22.1 ms | 1.25, 1.29, 1.37 | **0.86, 0.89, 0.91** | 0.49 / 10.0 ms (v6 0.52 / 11.6) |
| nl (Delta), I = 10 | 44.3 ms | 0.93, 0.96, 0.97 | **0.71, 0.72, 0.74** | 1.49 / 17.4 ms (v6 2.96 / 14.7) |
| hmm, I = 20 | 55.1 ms | 1.03, 1.06, 1.07 | **0.88, 0.90, 0.92** | 1.99 / 9.5 ms (v6 2.29 / 9.4) |
| gmm, I = 20 | 59.3 ms | 1.21, 1.21, 1.24 | **0.82, 0.83, 0.85** | 1.63 / 17.1 ms (v6 2.19 / 15.5) |
| streaming filter | 6.2 ms | 0.62, 0.65, 0.71 | **0.45, 0.53, 0.56** | |
| ssm1 + defaults | 41.6 ms | 1.31, 1.33, 1.35 | **0.86, 0.88, 0.88** | |
| iid + defaults | 22.4 ms | 1.30, 1.32, 1.35 | **0.89, 0.92, 0.93** | |
| ssm1@100 | 4.3 ms | 1.27, 1.28, 1.30 | **0.85, 0.86, 0.88** | |
| ssm1@10⁴ | 494 ms | 1.46, 1.47, 1.50 | **0.88, 0.88, 0.89** | |
| iid@100 | 2.8 ms | 1.19, 1.24, 1.26 | **0.86, 0.88, 0.89** | |
| iid@10⁴ | 266 ms | 1.21, 1.49, 1.50 | **0.85, 0.86, 0.87** | 6.19 / 104 ms (v6 7.03 / 126) |
| iid@10⁵ | 7.14 s | 0.92, 0.92, 0.93 | **0.73, 0.73, 0.74** | 111 ms / 1.04 s (v6 164 ms / 1.88 s) |
| betabern@5·10⁴ | 560 ms | 1.27, 1.28, 1.68 | **0.61, 0.66, 0.86** | one pass |
| linreg (v6 fails) | — | 189 ms | 145 ms | 5.20 / 41.3 ms (FULLT 6.04 / 67.9) |

- **Allocation**: F allocates less than v6 on ssm1, ssm2, betabern, gmm and at scale, and about as
  much on iid; more on hmm (80 against 72 MB at I = 20).
- **GC-off** samples (the work alone) give the same ratios within 1–2 points: the gains are the
  engine's, not the collector's.
- **First inference** is where v7 still loses, with no precompile workload on either side:
  1.03–1.35× v6 (hmm 12.4 against 9.9 s, filter 6.5 against 4.8 s), F within a few percent of
  FULLT. The typed design compiles more specialisations; a precompile workload (round 1's P11) is
  the remedy, and v6 has none.
- **Correctness** (`results/final/posteriors_check.txt`, `compare_variants2.jl`, full parameters):
  F against FULLT, 80 of 85 dumps bitwise identical; the 5 others are the streaming filter, whose
  dump was its streams' description, not its values (round 1's dump had the same flaw). The
  filter's history compared separately (`results/final/post/*_filter_check.jls`) is bitwise
  identical for F, FULLT and v6. F against v6: bitwise identical on every model except gmm
  (9·10⁻¹³) and hmm (3·10⁻¹²).

