# Benchmark — the performance pass

The record of the end-of-refactor performance pass (2026-09-26 to 2026-09-28, user): ReactiveMP v7
with RxInfer's branch `refactor/reactivemp-v7`, measured against v6, RxInfer 5.5.2 over ReactiveMP
6.5.0. It was done in two rounds. The second audited the first, whose verdict was too optimistic,
and completed it. The changes are applied: ReactiveMP `b46046c83` and RxInfer `1a6bb502`. The
Rocket and GraphPPL fixes are pull requests of their own (ReactiveBayes/Rocket.jl#91 and
ReactiveBayes/GraphPPL.jl#333). The scripts, diffs, profiles and raw data are in the history, up
to `b46046c83` (`investigations/`).

## 1. Result

**v7 runs faster than v6 on every benchmarked model.** The final sweep ran on an idle machine
(Julia 1.13.0, M-series Mac, 14 cores). It compares the commit before the change,
the commit after it, and the commit after it with the Rocket and GraphPPL pull requests, all on
the same Rocket 1.10.0 and GraphPPL 4.8.0 unless noted, over 17 models and 5 paired rounds. Each
figure is the median over rounds of the variant's minimum time over v6's minimum in the same
round:

| model | before / v6 | after / v6 | after + PRs / v6 |
|---|---|---|---|
| Kalman smoother (ssm1, BP, n = 10³) | 2.04 | **0.93** | **0.89** |
| 2-D Kalman smoother (BP) | 1.79 | **0.91** | **0.88** |
| Beta–Bernoulli (BP, n = 5 000) | 2.24 | **0.82** | **0.81** |
| iid VMP (n = 10³, 20 iterations) | 3.21 | **0.95** | **0.93** |
| nonlinear SSM (Delta, 10 iterations) | 1.76 | **0.76** | **0.73** |
| HMM (structured VMP, 20 iterations) | 1.77 | **0.92** | **0.89** |
| Gaussian mixture (20 iterations) | 2.59 | **0.86** | **0.82** |
| streaming Kalman filter (10³ points) | 1.58 | **0.51** | **0.52** |
| ssm1 / iid with RxInfer's defaults | 2.04 / 3.16 | **0.93 / 0.95** | **0.88 / 0.92** |
| ssm1 at n = 10² / 10⁴ | 1.95 / 2.13 | **0.89 / 0.92** | **0.87 / 0.88** |
| iid at n = 10² / 10⁴ / 10⁵ | 2.81 / 3.50 / 1.96 | **0.93** / 0.99 / **0.95** | **0.89 / 0.86 / 0.67** |
| Beta–Bernoulli at n = 5·10⁴ | 1.88 | **0.86** | **0.85** |
| loopy linear regression (v6 fails) | 384 ms | 156 ms | 147 ms |

- **Every paired round is below 1**, except iid at n = 10⁴ without the pull requests (0.99–1.01,
  level) and one round of Beta–Bernoulli at 5·10⁴ with them (GC placement at that scale).
- **Without collections** (samples run with `GC.enable(false)`), the ratios are the same within
  one or two points. The one exception shows why the Rocket fix matters: without it, iid at 10⁵
  is 1.03× v6, and with it 0.69×.
- **Setup and per iteration both beat v6.** iid: 10.2 against 11.8 ms setup, 0.50 against 0.51 ms
  per iteration. Gaussian mixture: 1.62 against 2.25 ms per iteration. nl: 1.51 against 2.97 ms
  per iteration.
- **Correctness.** Every posterior parameter and free energy of the two "after" variants is
  bitwise that of "before", on every model and round, 170 comparisons, the streaming filter's
  history included. Against v6 it is bitwise identical except the mixture (9·10⁻¹³) and the HMM
  (3·10⁻¹²).
- **First inference is still slower than v6**: 1.1–1.4× without a precompile workload on either
  side, and the change itself adds 1–9%. The typed design compiles more specialisations; a
  PrecompileTools workload in RxInfer is the remedy.

## 2. Method

- **Paired rounds.** For each model, the variants run back to back in fresh processes, round after
  round, with the order rotating. A ratio is taken between samples minutes apart, and a
  difference is claimed only when its range over the rounds excludes 1.
- **Minima, of one quantity each.** The minimum is the achievable time, and load only adds to it.
  - Per iteration is the difference of the GC-excluded minima at 2I and at I iterations, divided
    by I; setup is the GC-excluded minimum at I, less I per-iteration costs.
  - Round 1 subtracted minima taken from different samples, so where a GC pause fell decided its
    n = 10⁴ result.
- **What each sample records.** Wall time, the thread's CPU time, GC time, bytes, GC counts, a
  calibration loop and the load average; 11 samples per iteration count, 5 for the largest
  models.
- **GC-off samples.** A full collection, then `GC.enable(false)` for the sample, then a full
  collection again. They skip any model whose samples allocate more than 6 GB.
- **No callbacks.** v6 draws a `uuid4()` span id for every rule call whenever any callback is set,
  so timing through callbacks would slow v6 alone.
- **The correctness gate** compares every parameter of every posterior, the type in full, and the
  free energies, and names the reference variant.

## 3. What made v7 slower, and what fixed it

Before the pass, v7 was 1.4–3.5× slower than v6 end to end, on the same Rocket and GraphPPL. The
design's checks were not the cause. Mechanical costs were, each fixed by an ordinary Julia
practice:

| cost | fix |
|---|---|
| `RuleSpec.body::Function` made every rule's result `Any`: an event built for nobody, a `Message` whose type was computed at run time, the spec boxed (873 ns per rule call) | a typed `RuleSpec{B, P, S, L, A}` (§4.1), events built only for handlers that listen, span ids from a salted counter: 27–40 ns |
| every node re-resolved its interfaces by name, checked its groups and converted its factorisation, in code that never specialised on the node type: 1.6–5 µs per node, against v6's 0.34 | a creation plan per node declaration, interface keys and factorisation; its dependencies resolved once per shape |
| every `DeferredMessage` copied its immutable `MessageMapping`, factor node included | `MessageMapping` a mutable struct of constant fields |
| a fixed ≈ 500 ns per product at a variable (also in v6): the product rebuilt from a value of unknown type | a barrier after the fold; the product kept when the form constraint returns it |
| the rule context built per mapping, `Val` from a run-time `Bool`, a `Vector{Any}` of interfaces for grouped nodes | built once per node, a branch to a static `Val`, a small `Union` vector |
| RxInfer's plugin: per-edge `applicable` checks, a `map` widened at run time, keyword options from `Any` values | hoisted, typed, positional options behind a function barrier |
| Rocket: `all(vstatus)` over every source on every event, O(N²) per round; wrappers copied into every boxed emission | pending counters; mutable wrappers (Rocket#91) |
| GraphPPL: `apply_meta!` and matrix-variable constraints quadratic in the model | the context's own nodes; cached prefix sums (GraphPPL#333) |

## 4. Findings kept for reference

### 4.1 The typed `RuleSpec`

With the rule body a type parameter, resolution from concrete input types returns one concrete
spec and the body call is static and inlines. `DISCUSSION.md` §3.14 had rejected a parametric spec
on a toy call site that could reach two rules, where it gives a union of spec types. An engine's
call site reaches one rule, so that case does not arise; where resolution cannot be inferred, the
call is dynamic in either design. Measured: a rule call went from 873 ns and 15 allocations to 27
ns and 2 allocations called directly, and 40 ns and 3 across the stream barrier; the body alone
takes 4 ns.

### 4.2 Scratch

A rule's scratch is kept between calls in an untyped slot, since its type depends on the input
types. That costs about 40 ns and 3 allocations per call, 5–7% of BIFM's rules, the only ones
with scratch. A declared `scratch_type` makes it typed (`PLAN.md` § Scratch), and BIFM's rules
declare it.

### 4.3 The garbage collector and latency

Julia 1.13's GC no longer marks objects loaded from package images (JuliaLang/julia#61474). What
the JIT compiles lives on the ordinary heap and is marked by every full collection: 55–57 ms per
full collection after the first inference, against 30 ms right after loading, on v6 and v7
alike. A precompile workload moves that into the package image. This is why round 1 saw steady
state 4–20% faster with one, and v6 would gain the same.

### 4.4 The stack

Subscribing along a chain recurses, about 8 KB of stack per link. Without RxInfer's
`limit_stack_depth`, v6 overflows an 8 MiB stack beyond 1 344 links of a state-space model, and
v7 beyond 2 304. A guard in Rocket that hops to a fresh task (round 1's P9) raises that to 3 776,
at 1–8% of setup; it is not applied.

## 5. Measured and rejected

- **The equality chain's two-message partial products (round 1's P4).** They applied the form
  constraint once per outbound message instead of per partial product, which let an unsupported
  `ProductOf` reach a `missing` boundary and broke one of RxInfer's tests. Only its typed
  `ChainOutboundMapping{C}` is applied.
- **Rocket's stack guard** (§4.4).
- **A function barrier per interface at activation.** It was slower: `NodeInterface.variable` is
  an `AbstractVariable`, so the stream types stay `Any` behind the barrier as well.
- **Dropping `Message{D}`'s type parameter.** 1.1–1.7× slower per `infer`.
- **Typed Rocket listeners** (FunctionWrappers, 10–25× slower), **a typed `GraphVariableRef` in
  RxInfer** (setup 3× slower), **a GraphPPL storage redesign** (≲ 2%), **a custom rule dispatch**
  (resolution is already static, 26 ns), **`free_energy = Float64` by default** (≤ 2%, and
  breaks automatic differentiation).

## 6. Left open

- **A precompile workload** for first inference (1.1–1.4× v6): RxInfer's, with `free_energy =
  true` and the session, and one per rule package with many nodes.
- **Log scale `nothing` for data and constants when log scales are not tracked.** Neither knows
  whether they are, so the flag would thread through `constvar` and the data variable's
  activation, and a caller tracking log scales that forgot it would lose them silently; the gain
  is 8 bytes per observation and the specialisations where observed and computed point masses
  meet. Not done; for the user to decide.
- **Scratch in more rules**: the Delta rules' sigma points and Jacobians, `*` with a matrix, AR and
  ContinuousTransition, now that a typed scratch costs nothing.
- **A product workspace per variable**, for the temporaries of multivariate products in
  BayesBase; to be profiled first.
- **A fast path for variables of degree 2 in the equality chain.** Every state-space model's chain
  variables have that degree, but the path changes the product events callbacks see.
- **The BayesBase ambiguity `prod(::GenericProd, ::ProductOf, ::Missing)`**, upstream.
