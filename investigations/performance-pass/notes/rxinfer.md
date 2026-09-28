# RxInfer (P6): what RxInfer's own code costs, and what fixing it buys

Scope: RxInfer's glue code on its `refactor/reactivemp-v7` branch (`4ca9eb88`), on top of ReactiveMP
with P1 (lazy callback events, counter span ids) and P2 (constructor barriers) applied: the
baseline variant is **P12**. The fixes live in variant **X1** (RxInfer worktree with P6, same
ReactiveMP). Julia 1.13.0, M-series Mac, 10 cores. **Other forks were benchmarking on the same machine
throughout (load average 8–18), so every timing here is indicative only.** Allocation figures are
exact and unaffected by the load.

## Headline

**RxInfer's own code is a small share of `infer`.** Its self time is 1.0% of iid VMP (n = 1000,
50 iterations), 3.0% of the ssm1 BP smoother (n = 1000) and 0.4–0.6% of streaming filters. The
per-iteration data feed is already concretely typed (`Vector{DataVariable{…}}`), and no RxInfer
code runs per message. What RxInfer glue does cost is setup, a few hundred µs per 1000 variables,
and a constant ~140 µs floor per `infer` call that GraphPPL dominates (below). So P6 is mostly
hygiene, plus **one real win: `benchmark = true`**.

## Option costs on the baseline (P12; `bench_rxinfer_opts.jl`, first run on a quieter machine)

| case | iid, 10 it (ms) | ssm1 (ms) | note |
|---|---|---|---|
| base: `session = nothing`, `free_energy = Float64` | 53.0 | 120.9 | |
| `session` default | 54.4 (+2.7%) | 121.9 (+0.8%) | a constant ≈ 10–14 µs per call plus strings; median much noisier (semaphore, `Dates.now`, UUID) |
| `free_energy = true` (`Real`) | 54.1 (+2%) | 123.4 (+2%, with session) | within noise of `Float64`; on a tiny model no difference |
| `free_energy = false` | 46.2 (−13%) | 103.2 (−15%) | the free energy itself is engine work (`score`), not RxInfer's |
| `benchmark = true` | 64.7 (**+22%**), 81.7 MB vs 51.9 MB | | every engine event built for a handler that records 8 of them |
| `trace = true` | 66.9 (+26%) | | expected: it records every event |
| `returnvars` KeepEach / one KeepLast | = base | | |

A tiny model (Beta–Bernoulli, 3 points) takes ≈ 140 µs and ~2 100 allocations per `infer`
(`results/rxinfer_tiny.txt`); that floor matters for code calling `infer` in a loop.

`infer` itself infers `InferenceResult` (abstract in its parameters but not `Any`).

## JET (`jet_rxinfer.jl`; `results/rxinfer_jet_P12_*.txt`), target module RxInfer

`JET.report_opt` on `infer` and on `batch_inference` / `streaming_inference` directly (ssm1, iid,
the streaming filter) reports only 1–7 runtime dispatches. All of them are setup or teardown, O(#
variables) at most, and **none are on a per-iteration path**:
- `obtain_prediction(ref)` (`reactivemp_inference.jl:709`): `ref.variable::AbstractVariable |>
  skip_initial()`, once per predicted variable. Intended type erasure.
- `streaming.jl:690/694`: the `ntuple` over the datastream's names reads the `Any`-valued vardict,
  once per data variable at setup. Intended.
- `batch.jl:481`: the force-marginal teardown loop over factor nodes. Intended.

JET's optimisation analysis does not report boxed captures here. The lowered code does
(`Core.Box`): **8 boxed captured variables**:
- `batch_inference`: `data`, `postprocess`, `vardict`, `predictoption`;
- `infer`: `callbacks`;
- `streaming_inference`: `historyvars`, `postprocess`, `vardict`.

They make the setup closures dynamic but touch the iteration loop only as O(#data keys) dynamic
calls per iteration, which is negligible. (The earlier audit's claim that the loop dispatches
per data entry is right, but the dispatch is once per key, not per observation.)

## The fixes (X1), each measured

| diff | what | effect | keep? |
|---|---|---|---|
| `P6a-RxInfer.diff` | removes the 8 boxes: single assignments (`_postprocess`, `_callbacks = infer_callbacks(…)`, one `vardict`), distinct names per branch, a plain loop instead of a capturing `foreach` | 8 → 0 `Core.Box`; per-iteration effect below noise | yes, hygiene, non-breaking |
| `P6b-RxInfer.diff` | `GraphVariableRef{V}` typed field | **regression**: building the refs costs 3× (ssm1 n = 1000: 230 → 690 µs; tiny 0.6 → 2.5 µs). Building a parametric struct from an `AbstractVariable`-typed value computes `V` at run time, P2's phenomenon again. The gain is setup-only and negligible | **no** (reverted in X1) |
| `P6c-RxInfer.diff` | single-pass typed getters (`getrandomvars` & co.: one `variable_nodes` pass pushing into `Vector{NodeData}`, no `collect`/`filter`/`map`), `nodes_by_kind` used by the free energy's `score` (5 graph passes → 2), the typed `ReactiveMPExtraVariableKey` in `degree_fn` instead of the `Any` symbol key, and `gettoplevelvardict` for `infer` | ssm1 n = 1000: getters 0.16 → one pass; ≈ 0.5 ms of 120 ms. **Submodel-heavy models**: `gettoplevelvardict` 36 µs vs `getvardict` 1.69 ms (same process, 500-step submodel chain), ≈ 1–2% of that `infer` | yes, non-breaking (`getvardict` kept) |
| `P6c2-RxInfer.diff` | `gettoplevelvardict` maps with `similar` + `map!` (same keys, same order, no `return_type` call) | small (tiny model: 0.57 → 0.46 µs) | optional |
| `P6d-RxInfer.diff` | (1) `ReactiveMP.listens(::RxInferBenchmarkCallbacks, T)` = only the 8 events it records; (2) `RxInferBenchmarkCallbacks` a `mutable struct` with `const` fields: the engine stores its callbacks by value in every message mapping, `DeferredMessage`, product context and fold closure, and an immutable struct of eight references made each of those 64 bytes larger and boxed a copy per mapping; (3) `ReactiveMP.listens(::RxInferTraceCallbacks, T)` honours `trace = (names…)` (always `AfterModelCreationEvent`, which saves the trace). **Needs ReactiveMP P1** (`listens`, `@invoke_callback`) | `benchmark = true` on iid: **63.9 → 53.2 ms (−17%)**, 81.7 → 52.8 MB (base 51.9 ms, 51.7 MB): overhead +23% → +2.4%. (1) alone: 81.7 → 59.7 MB; (2) takes the rest. Filtered trace `trace = (:before_iteration, :after_iteration)`: 66.9 MB / 66 ms → 53.8 MB / 51.9 ms (= no trace), same 20 events | **yes**; the one user-visible win |
| `P6bug-RxInfer.diff` | `Base.iterate(::InferenceResult)` read the non-existent `:returnval` field: `a, b = infer(…)` threw `FieldError` (verified). Now iterates posteriors, predictions, free energy, model, error | bug fix | yes, separately |
| `P6-RxInfer.diff` | P6a + P6c + P6c2 + P6d combined (not P6b, not the bug fix) | | |

Allocation totals, P12 → X1: iid 51.92 → 51.72 MB; ssm1 84.11 → 83.06 MB (−1.2%); tiny model 2 156
→ 2 052 allocations, 111.7 → 105.5 KB (−5%).

## Timings, P12 vs X1 (final X1; `driver_rxinfer.sh`, 6 interleaved rounds, fresh process each, load average ≈ 3)

Paired ratio = the median over rounds of X1/P12 in the same round (`results/rxinfer_ab.tsv`,
`results/rxinfer_ab_summary.txt`):

| case | P12 ms | X1 ms | X1/P12 (range) | P12 MB | X1 MB |
|---|---|---|---|---|---|
| iid base | 51.93 | 51.40 | 0.994 (0.983–1.000) | 51.92 | 51.66 |
| iid session default | 53.34 | 52.80 | 0.991 (0.962–0.996) | 51.93 | 51.67 |
| iid `free_energy = true` | 53.41 | 52.91 | 0.988 (0.956–1.002) | 52.72 | 52.46 |
| iid `benchmark = true` | 63.85 | **53.16** | **0.832 (0.818–0.838)** | 81.74 | **52.82** |
| iid `trace = true` | 66.38 | 66.01 | 0.996 | 81.40 | 81.14 |
| iid all defaults | 54.14 | 53.37 | 0.993 | 52.73 | 52.47 |
| ssm1 base | 118.93 | 119.50 | 1.007 (0.992–1.014) | 84.11 | 82.93 |
| ssm1 all defaults | 121.42 | 120.03 | 0.990 | 84.76 | 83.58 |

Apart from `benchmark = true`, P6 is 0–1% (inside the noise, though the paired medians lean
below 1); ssm1 is neutral. `results/rxinfer_ab_discarded_midrun.tsv` is an earlier run
discarded because X1 changed during it.

## Not RxInfer's, found on the way (for the GraphPPL and engine owners)

- **GraphPPL `ConstraintStack`** (`variational_constraints_engine.jl:552`, reached from the
  constraints plugin's `apply_constraints!` even for `NoConstraints`) creates a DataStructures
  `Stack`, whose `Deque` allocates a 1024-element block on every model creation. That is 12–29% of
  the tiny-model `infer` profile (`GenericMemory` via `DequeBlock`). A `Vector` or a small block
  size removes it.
- **The free energy is 13–15% of `infer`** (iid, ssm1) and runs in the engine (`score`,
  re-resolving each average energy on every update). `Real` vs `Float64` is not what costs.
- Dictionaries' `map` over an `UnorderedDictionary` calls `Compiler.return_type` at run time. It
  showed up large in one profile, but direct measurement puts it at ≈ 0.5 µs a call, so it was a
  sampling artifact and is not a cost.

## `free_energy = true` → `Real` (task 4)

Measured cost against `Float64`: ≤ 2% on iid and ssm1, nothing on a tiny model, within the noise.
`Real` makes `ScoreActor` a `Matrix{Real}` and the per-node streams `CountingReal{Real}`, but there
are only O(N) additions per iteration, each boxed, next to O(N) rule calls that cost 10–100× more.

Options:
- **(a) keep the default `Real`** (recommended). It is non-breaking and costs nothing measurable.
- **(b) default `Float64`, `free_energy = Real` opt-in.** Breaking for automatic differentiation
  through the free energy (ForwardDiff duals), for no measured gain.
- **(c) take the element type from the first value.** Non-breaking, but complexity with no measured
  gain.

## Streaming

A streaming VMP filter (`prof_streaming.jl`: @autoupdates, 5 iterations per point, n = 500, history
kept) takes 47–49 µs per data point. RxInfer is 0.5–0.6% of the samples; the rest is the engine
(the Rocket and GraphPPL clones' paths contain the substring the classifier used for ReactiveMP,
so its "ReactiveMP" share includes them). Nothing to gain in RxInfer's executor.

## Tests

- **RxInfer's full suite on the final X1** (`Pkg.test()` in the worktree, ReactiveMP P1+P2 from
  the sibling): **14 713 pass, 0 errors, 0 failures** (`logs/rxinfer_tests_X1.log`).
- Baseline P12 in the same way: 14 699 pass and 3 errors, all `DiscreteTransitionMessagePassingRules
  failed to precompile` inside a test item, from two suites precompiling into one depot at once
  (`logs/rxinfer_tests_P12.log`). That is environmental.
- An earlier X1 run (before the P6b revert) ended at 1 631 pass and 147 errors,
  `Package Test not found in current path`. It was also environmental: it started right after a
  killed run, and only 1 778 tests were collected at all. The rerun above supersedes it.
- **Posteriors and free energies bit-identical** to P12 (`isequal` on serialized summaries, the
  `bench_models.jl` posteriors dump) on ssm1, iid, hmm, nl, gmm and the streaming filter.

## Recommendation

1. **Take P6d with P1** (release). `benchmark = true` then costs +2.4% instead of +23%: RxInfer's
   own benchmarking tool stops inflating what it measures. A filtered `trace` becomes free.
   Non-breaking; making `RxInferBenchmarkCallbacks` mutable changes nothing observable.
2. **Take the bug fix** (release).
3. **Take P6a and P6c** as hygiene (release, non-breaking). They remove every boxed capture and
   most redundant graph passes, but expect ≤ 1–2% on real models.
4. **Do not take P6b.** A typed `GraphVariableRef` regresses setup and gains nothing.
5. **Leave `free_energy = true` as `Real`.**
6. What would actually move the constant `infer` floor is in GraphPPL (the `ConstraintStack` deque
   block) and in the engine's activation, not in RxInfer.

## Files

- `bench_rxinfer_opts.jl`: the option costs. `driver_rxinfer.sh`: the P12/X1 interleaved A/B.
  `jet_rxinfer.jl`: JET. `prof_streaming.jl`: the streaming profile.
- `rxinfer_scripts/`:
  - `boxes.jl`: `Core.Box` in the lowered `infer`/`batch_inference`/`streaming_inference`;
  - `vd2.jl`, `sub.jl`: vardict construction, plain and submodel-heavy;
  - `bcb3.jl`, `bcb5.jl`, `cbmicro.jl`: where `benchmark = true` allocated;
  - `trc.jl`: the filtered trace;
  - `tiny.jl`, `tinyprof.jl`: the per-`infer` floor;
  - `passes.jl`: graph-pass costs;
  - `cmp.jl`: the posterior equality check.
- Variant X1 is ReactiveMP `d50fac9c` + `P1-ReactiveMP.diff` + `P2-ReactiveMP.diff`, and RxInfer
  `4ca9eb88` + `P6-RxInfer.diff` + `P6bug-RxInfer.diff`, with Rocket and GraphPPL clones at master.
