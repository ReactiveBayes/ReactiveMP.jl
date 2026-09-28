# The end-of-refactor performance pass (2026-09-26)

Benchmarks, profiles and prototypes for ReactiveMP v7 with RxInfer, Rocket and GraphPPL, asked
for by the user before the release. **Nothing here is applied to `src/` or `lib/`**: every change
is a diff in `diffs/`, and the user decides what goes in. The published report (charts, the full
decision table) is the private artifact linked from the session; this file is its record.

Measured at ReactiveMP `d50fac9c`, RxInfer `4ca9eb88` (`refactor/reactivemp-v7`), Rocket master
(1.10.0 plus CI-only commits), GraphPPL 4.8.0 (`4cc7c6c`); v6 is ReactiveMP 6.5.0 with RxInfer
5.5.2, on the same Rocket and GraphPPL versions. Julia 1.13.0, M-series Mac, 10 cores.

## Verdict

- HEAD was **1.5–4.6× slower than v6** in steady state, 2.2–3.2× in setup, and 15–30% slower to
  first inference. The cause is not the rule system's design: the rule body is 49 ns of an 873 ns
  rule call. The rest came from the rule's result being typed `Any` (`RuleSpec.body::Function`):
  - a callback event built with nobody listening (≈ 300 ns);
  - `Message{D, L}` built from `Any` (≈ 380 ns, `_compute_sparams`);
  - `uuid4()` span ids whenever callbacks are set (≈ 760 ns);
  - the 18-field `RuleSpec` boxed at each dynamic call.

  Activation cost ≈ 25 µs per node, a third of it in `input_names`.
- **The rule's result was untyped because `RuleSpec.body::Function` erases it.** Typing the spec
  (`RuleSpec{B, P, S, L, A}`, variant T, the user's follow-up question) makes the body call static:
  a rule call takes 27 ns and 2 allocations, against 69 ns and 5 with the function barrier (P3) this
  pass first tried. It is 0.83–0.92× P3 per iteration, 0.62–0.66× at n ≥ 10⁴, and costs nothing in
  compile time. It reverses §3.14's toy-based decision; see `notes/engine.md` § T.
- With every recommended change and the typed spec (FULL), v7 is **faster than v6 in steady state on
  six of the seven iterative benchmarks**:
  - gmm 0.80×, hmm 0.95×, nonlinear SSM 0.62×, streaming filter 0.59×;
  - iid at n = 10⁴ 0.86× and at 10⁵ 0.94×;
  - iid at n = 10³ is the exception, at 1.09×.

  The setup-bound belief-propagation models run at 1.25–1.43× v6.
- With a PrecompileTools workload in RxInfer, the first inference drops from 8.7 to 0.69 s (iid),
  7.9 to 0.53 s (Beta–Bernoulli) and 13.6 to 6.6 s (Kalman smoother). Steady state is also 4–20%
  faster.
- Posteriors and free energies are **bit-identical** to HEAD in both final runs (204/204 and
  153/153 comparisons).

## Final runs (`results/final/` and `results/final2/`, `driver.sh`, three rounds each, sequential, idle machine)

The second run measured v6, FULL with the P3 barrier, FULL with the typed spec (FULLT) and FULLT
with the workload (FULLTPC):

| | v6 | FULL (P3) | FULLT | FULLTPC |
|---|---|---|---|---|
| iid VMP, per iteration (n = 1 000) | 0.65 ms | 0.83 ms | 0.71 ms | 0.57 ms |
| iid VMP, per iteration (n = 10 000) | 16.8 ms | 23.2 ms | 14.5 ms | 11.8 ms |
| iid VMP, per iteration (n = 100 000) | 218 ms | 310 ms | 204 ms | 176 ms |
| Gaussian mixture, per iteration | 3.02 ms | 2.66 ms | 2.41 ms | 2.32 ms |
| HMM, per iteration | 3.11 ms | 3.11 ms | 2.96 ms | 2.85 ms |
| nonlinear SSM (Delta), per iteration | 3.66 ms | 2.42 ms | 2.26 ms | 1.87 ms |
| streaming filter, 1 000 points | 9.0 ms | 5.9 ms | 5.3 ms | 4.4 ms |
| loopy linear regression, per iteration | fails | 8.95 ms | 7.87 ms | 7.86 ms |
| Kalman smoother (BP), one infer | 59 ms | 79 ms | 78 ms | 75 ms |
| Beta–Bernoulli (BP), one infer | 71 ms | 103 ms | 102 ms | 96 ms |
| iid setup | 18.1 ms | 25.9 ms | 25.5 ms | 23.8 ms |
| first inference, iid | 7.5 s | 8.7 s | 8.8 s | 0.69 s |
| one rule call (NMV, BP) | — | 67 ns / 5 allocs | 27 ns / 2 | 28 ns / 2 |
| allocations per iteration (iid, n = 10⁴) | 7.2 MB | 13.3 MB | 9.4 MB | 9.4 MB |

The first run, which covers HEAD and the intermediate variants:

Median over rounds of each round's minimum. Per iteration = (T(2I) − T(I)) / I.

| | v6 | HEAD (A) | P1+P2 | engine (E4) | FULL | FULL + workload |
|---|---|---|---|---|---|---|
| iid VMP, per iteration (n = 1 000) | 0.67 ms | 3.08 ms | 1.13 ms | 0.83 ms | 0.76 ms | 0.66 ms |
| iid VMP, per iteration (n = 10 000) | 11.2 ms | 48.3 ms | 25.0 ms | 19.2 ms | 23.8 ms | 22.7 ms |
| Gaussian mixture, per iteration | 2.9 ms | 7.4 ms | 3.3 ms | 3.0 ms | 2.8 ms | 2.7 ms |
| HMM, per iteration | 3.0 ms | 5.1 ms | 3.3 ms | 3.2 ms | 3.3 ms | 3.2 ms |
| nonlinear SSM (Delta), per iteration | 3.6 ms | 5.5 ms | 2.8 ms | 2.5 ms | 2.3 ms | 1.9 ms |
| streaming filter, 1 000 points | 9.1 ms | 13.6 ms | 7.2 ms | 6.1 ms | 5.6 ms | 4.6 ms |
| Kalman smoother (BP), one infer | 62 ms | 120 ms | 115 ms | 78 ms | 78 ms | 74 ms |
| Beta–Bernoulli (BP), one infer | 70 ms | 159 ms | 155 ms | 104 ms | 103 ms | 97 ms |
| loopy linear regression, per iteration | fails (`DomainError`) | 20.7 ms | 12.1 ms | 10.6 ms | 9.3 ms | 8.9 ms |
| iid setup | 18.5 ms | 41.1 ms | 40.5 ms | 26.7 ms | 26.6 ms | 24.2 ms |
| first inference, iid | 7.5 s | 8.7 s | 8.7 s | 8.9 s | 8.8 s | 0.68 s |
| one rule call (NMV, BP) | — | 873 ns / 15 allocs | 181 ns / 11 | 69 ns / 5 | 69 ns / 5 | 69 ns / 5 |

Every row is in `results/final/tables.md`; the raw data is in `models.tsv` and `micro.tsv`.

## The options (see the report for the full table)

**Release** (all non-breaking unless noted; checked by the suites named in *Correctness*):

- ReactiveMP and MessagePassingRulesBase:
  - **P1**: lazy callback events and counter span ids.
  - **P2**: a constructor barrier.
  - **T**: a typed `RuleSpec{B, P, S, L, A}` (`T-typed-RuleSpec.diff`), with `RuleResult` carrying
    the spec's type. It supersedes **P3, P3b, P3c**, the `execute_rule(then, spec, …)` barrier.
  - **TS**: an optional `scratch_type` declaration for rules with scratch (`TS-typed-scratch.diff`,
    on top of T). An untyped scratch costs ≈ 40 ns and 3 allocations per call; typed, a scratch
    rule costs what any other rule does. BIFM, the only user, gains 5–7%.
  - **P4**: the equality chain multiplies partial products with the two-message product and applies
    the form constraint once per outbound message. Semantically, "check last" now means the whole
    product; the results are identical under RxInfer's default.
  - **P5a, P5b**: activation caches.
  - The combined diff is `ENGINE-T-ReactiveMP.diff` (`ENGINE-ReactiveMP.diff` is the P3 version).
- RxInfer:
  - **P11**: a precompile workload. It adds a dependency, and RxInfer precompiles in about 21 s
    instead of 10 s.
  - **P6d**: `benchmark = true` stops inflating what it measures by 23%; needs P1.
  - **P6a, P6c**: hygiene.
  - **the `iterate(::InferenceResult)` bug fix**.
- Rocket (1.11):
  - **P7**: counters in `collectLatest`/`GenericUpdatesStatus`, and mutable wrappers.
  - **P9**: `stackguarded`, which roughly triples the longest chain handled without
    `limit_stack_depth`; it is new public API. Its counter must become per task before release,
    and it needs a unit test.
  - The engine changes need P9: without it the 1 000-link chain graph overflows the stack.
- GraphPPL (4.9):
  - **C1**: `apply_meta!` was quadratic: 4.5 → 0.47 s at n = 3 000.
  - **C6**: matrix-variable constraints were quadratic.
  - **C2, C3, C7**: small fixes, including `ConstraintStack` without a 1 024-element deque block.
  - The recommended set is `P8rel-GraphPPL.diff`.

**Later or optional:**

- GraphPPL **C4** (typed extras; soft-breaking; take with a workload).
- GraphPPL **C5** (`created_by` as an `Expr`; soft-breaking; fewer specialisations, no time).

**No:**

- Rocket typed listeners (P7b).
- An emission stack guard (P9b), which costs 11–25% per iteration.
- RxInfer's typed `GraphVariableRef` (P6b), which regresses setup 3×.
- A default `free_energy = Float64`.
- GraphPPL's storage redesign (P8b), which gains ≲ 2%.
- A custom rule dispatch (P10): resolution is already static, 26 ns, and about 1% of compile time.
- The P3 barrier, superseded by the typed spec.

**Next round (measured, not prototyped):**

- **Allocation volume:** with the typed spec, v7 allocates 1.3× v6's bytes per iteration. Per rule
  call, what's left is the `Message` and its `AnnotationDict`; per update, the `DeferredMessage`
  and boxed `CountingReal`s.
- **Setup,** still 1.4–2.2× v6:
  - `Val(logscales)` from a runtime `Bool`;
  - the free-energy setup for deterministic nodes, 58 µs per node;
  - a resolution plan per node type.
- **Workloads per rule package.**
- **Rewriting the engine's generated functions**, measured first.
- **Log scales when tracked:** `UndefinedLogScale` boxes the spec, and `product_logscale` calls
  `applicable` per product.

## Correctness

- **Posteriors and free energies** are bit-identical to HEAD:
  - final run: 204/204 comparisons (17 cases × 3 rounds × 4 variants);
  - during development: every prototype on ssm1, iid, hmm, nl, linreg, betabern, gmm and the
    streaming filter.
- **Suites:**
  - ReactiveMP root on E4 and FULL: 15 339 pass, 6 known broken (Aqua skipped). On the typed spec
    (T), with Aqua: the same.
  - On T: TestUtils 147, and every node package (Delta 336, DiscreteTransition 2 128,
    Autoregressive 12 855, GaussianCoupling 293, Probit 398, GCV 463, SoftDot 928,
    ContinuousTransition 519, Pólya 108, BIFM 419, Flow 1 020, Approximations).
  - MessagePassingRulesBase: 617.
  - StandardMessagePassingRules: 10 139, 1 known broken.
  - GraphPPL on the recommended set plus C7: 76 803, 1 known broken.
  - Rocket on P7+P9: 12 295.
  - RxInfer on P6: 14 713.

## Files

- **Benchmarks:**
  - `bench_micro.jl`: rule calls, products, Rocket primitives, engine graph loops.
  - `bench_models.jl`: end to end through RxInfer, v6 or v7.
  - `bench_creation.jl`: model creation by stage.
  - `bench_rxinfer_opts.jl`.
  - `driver.sh`: the final run; `summarise.py` turns it into `tables.md` and `summary.json`.
- **Analyses:**
  - `prof_model.jl`, `prof_graph.jl`, `prof_activate.jl`, `prof_creation.jl`, `prof_streaming.jl`;
  - `jet_engine.jl`, `jet_rxinfer.jl`, `jet_graphppl.jl`;
  - `snoop_ttfx.jl`;
  - `stack_depth.jl`, `stack_frames.jl`, `rocket_fastpaths.jl`.
- **Correctness:** `compare_variants.jl` (all variants against a reference),
  `compare_posteriors.jl` (a pair), `dump_graph_extras.jl`.
- **Variants:** `mkvariant.sh` and `mkenv.jl` rebuild a variant from sibling worktrees and the
  diffs.
  - The scratch worktrees are gone after the session.
  - Recreate a variant by applying its diffs to the recorded revisions:
    - E4 is `ENGINE-ReactiveMP.diff`.
    - FULL is E4 plus `P7-Rocket.diff`, `P9-Rocket.diff`, `P8rel-GraphPPL.diff`,
      `P6-RxInfer.diff` and `P6bug-RxInfer.diff`.
    - FULLPC is FULL plus `P11-RxInfer.diff`.
    - FULLT and FULLTPC are the same with `ENGINE-T-ReactiveMP.diff` in place of the engine diff.
- **Notes:** `notes/engine.md`, `notes/rocket.md`, `notes/graphppl.md`, `notes/rxinfer.md`.
- **Results:** `results/`, with the final runs in `results/final/` and `results/final2/`, and the
  typed spec against the barrier in `results/typed_spec_vs_barrier.tsv`.
- **Report template:** `report/performance-pass.template.html`. The data is injected from
  `results/final/pagedata.json`.
