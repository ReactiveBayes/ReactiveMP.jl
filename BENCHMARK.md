# Benchmark — the performance pass

The record of the end-of-refactor performance pass (2026-09-26 to 2026-09-28, user): ReactiveMP v7
with RxInfer's branch `refactor/reactivemp-v7`, measured against v6, RxInfer 5.5.2 over ReactiveMP
6.5.0. It was done in two rounds. The second audited the first, whose verdict was too optimistic,
and completed it. The changes are applied: ReactiveMP `b46046c83`, then `83b77d179` (typed scratch,
annotations only where written, scratch slots on demand), and RxInfer `1a6bb502`, then `af1f082f`
(the precompile workload). The Rocket and GraphPPL fixes are pull requests of their own:
ReactiveBayes/Rocket.jl#91 is merged, for Rocket 1.10.1, and ReactiveBayes/GraphPPL.jl#333 is open.
The scripts, diffs, profiles and raw data are in the history, up to `b46046c83` (`investigations/`).

## 1. Result

**v7 runs faster than v6 on every benchmarked model.** The final sweep ran on an idle machine
(Julia 1.13.0, M-series Mac, 14 cores). It compares the commit before the change,
the commit after it (`b46046c83`), and that commit with the Rocket and GraphPPL pull requests'
changes applied, all on Rocket 1.10.0 and GraphPPL 4.8.0, over 17 models and 5 paired rounds. Each
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
- **Allocation** (bytes, exact, measured again at `cad05f6b6`): per iteration iid 0.98×, the
  mixture 0.73×, nl 0.54× and the HMM 1.12× v6's; a whole belief-propagation inference (ssm1,
  Beta–Bernoulli) allocates what v6's does, within 1%.
- **Correctness.** Every posterior parameter and free energy of the two "after" variants is
  bitwise that of "before", on every model and round, 170 comparisons, the streaming filter's
  history included. Against v6 it is bitwise identical except the mixture (9·10⁻¹³) and the HMM
  (3·10⁻¹²).
- **First inference**: without a precompile workload on either side, v7 is 1.1–1.4× v6, and the
  pass itself adds 1–9% to it: more of the code is specialised (the typed spec, the creation
  plans, the barriers), though the typed spec alone measured no compile cost. RxInfer's branch
  now has a workload (§4.5): a first inference on the common paths takes 0.27–0.44 s instead of
  6.7–7.4 s, and RxInfer's own precompile takes 12 s instead of 2.6.

## 2. Method

- **Paired rounds.** For each model, the variants run back to back in fresh processes, round after
  round, with the order rotating. A ratio is taken between samples minutes apart, and a
  difference is claimed only when its range over the rounds excludes 1.
- **Minima, of one quantity each.** The minimum is the achievable time, and load only adds to it.
  - Per iteration is the difference of the GC-excluded minima at 2I and at I iterations, divided
    by I; setup is the GC-excluded minimum at I, less I per-iteration costs.
  - The first round subtracted minima taken from different samples, so where a GC pause fell
    decided its n = 10⁴ result; the second took each from one quantity.
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

Before the pass, v7 was 1.6–3.5× slower than v6 end to end, on the same Rocket and GraphPPL. The
design's checks were not the cause. Mechanical costs were, each fixed by an ordinary Julia
practice:

| cost | fix |
|---|---|
| `RuleSpec.body::Function` made every rule's result `Any`: an event built for nobody, a `Message` whose type was computed at run time, the spec boxed (873 ns per rule call) | a typed `RuleSpec{B, P, S, L, A}` (§4.1), events built only for handlers that listen, span ids from a salted counter: 27–40 ns |
| every node re-resolved its interfaces by name, checked its groups and converted its factorisation, in code that never specialised on the node type: 1.6–5 µs per node, against v6's 0.34 | a creation plan per node declaration, interface keys and factorisation; its dependencies resolved once per shape |
| every `DeferredMessage` copied its immutable `MessageMapping`, factor node included | `MessageMapping` a mutable struct, every field constant but the scratch slot |
| a fresh `AnnotationDict` for every message and marginal, and a `ScratchSlot` for every mapping, used or not | one frozen, empty dict shared by those that carry none, a fresh one only where something may write; a slot at the first call of a rule with scratch (`83b77d179`): one allocation fewer per rule call |
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

A rule's scratch is kept between calls in a slot the mapping creates before the input types are
known, since its type depends on them. Kept untyped, it costs about 40 ns and 3 allocations per
call, 5–7% of BIFM's rules, the only ones with scratch. The engine now infers the type from the
inputs' types at the call and asserts the kept scratch to it (`PLAN.md` § Scratch): a toy rule's
call takes 8 ns and allocates nothing, against 26 ns and 32 bytes untyped, and BIFM's rules, whose
builders infer, run as fast as with the type declared by hand (a `scratch_type` keyword, tried
and dropped, user).

### 4.3 The garbage collector and latency

Julia 1.13's GC no longer marks objects loaded from package images (JuliaLang/julia#61474). What
the JIT compiles lives on the ordinary heap and is marked by every full collection: 55–57 ms per
full collection after the first inference, against 30 ms right after loading, on v6 and v7
alike. A precompile workload moves that into the package image. This is why the first round saw
steady state 4–20% faster with a workload, and v6 would gain the same.

### 4.4 The stack

Subscribing along a chain recurses, about 8 KB of stack per link. Without RxInfer's
`limit_stack_depth`, v6 overflows an 8 MiB stack beyond 1 344 links of a state-space model, and
v7 beyond 2 304. A guard in Rocket that continues a deep subscription on a fresh task raises that to 3 776,
at 1–8% of setup; it is not applied.

### 4.5 RxInfer's precompile workload

A PrecompileTools workload on RxInfer's branch (`af1f082f`) runs four small inferences while RxInfer
precompiles: a belief-propagation chain, an iid model under mean field, a Beta–Bernoulli pair and
the chain with `limit_stack_depth`. The limit is 2, so that the short chain reaches it and the
scheduler's path onto a new task compiles too. Measured on an idle machine, the telemetry off,
each figure the minimum of fresh processes:

- **The cost.** RxInfer's precompile goes from 2.6 to 12.0 s, once per installation or update;
  `using RxInfer` from 1.35 to 1.56 s. The user set 12 s as the limit. Each call's share: the
  chain 6.1 s (it compiles the whole `infer` path), the iid model 1.1, Beta–Bernoulli 0.7,
  `limit_stack_depth` 1.4.
- **The common calls** (defaults, n = 10³): the chain, with and without `limit_stack_depth`, the
  iid model and Beta–Bernoulli take 0.27–0.44 s to their first result, against 6.7–7.4 s without the
  workload.
- **Models outside it** still gain, since the engine, GraphPPL and the common rules are cached:
  the 2-D Kalman smoother 11.6 → 6.3 s, nl 11.5 → 4.7, the HMM 12.3 → 7.3, the mixture 11.2 → 5.6,
  linreg 9.9 → 4.7, the streaming filter 7.9 → 2.5. (These are with a larger version of the
  workload that also ran the three models with `free_energy = true`; the benchmark models use
  `free_energy = Float64`, which neither covers.)
- **Left out: `free_energy = true`.** What it compiles depends on the model: covering it for the
  iid model (1 s more) left the chain with it at 2.3 s. With the workload, a first inference with
  it takes 1.5–2.8 s on these models, and with `free_energy = Float64` about 2 s.
  `limit_stack_depth` was kept instead, since it helps every model that sets it: 2.2 s saved on
  the chain, 0.8–2 s on ssm2, nl and the HMM.
- **Telemetry.** RxInfer's usage ping at `using` compiles HTTP and JSON on a spawned task. In a
  process with one thread and no interactive one (`-t1`), that task takes the main thread at the
  first yield, which adds 2–4 s to a first inference with `limit_stack_depth`. A default launch
  has an interactive thread, and the benchmarks set `LOG_USING_RXINFER=false`.

## 5. Measured and rejected

- **Two-message partial products in the equality chain.** They applied the form
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
- **Log scale `nothing` for data and constants when log scales are not tracked** (decided against,
  user). Neither knows whether they are, so the flag would thread through `constvar` and the data
  variable's activation, and a caller tracking log scales that forgot it would lose them silently;
  the gain is 8 bytes per observation and the specialisations where observed and computed point
  masses meet.

## 6. Left open

- **Rocket 1.10.1 and GraphPPL#333.** Rocket#91 is merged and its release is to be tagged;
  GraphPPL#333 awaits review. Neither needs a compat bump, since both change performance only.
- **Precompile workloads in the rule packages** with many nodes (the multivariate Gaussians,
  Delta, DiscreteTransition), as extensions on ReactiveMP; RxInfer's is done (§4.5).
- **Scratch in more rules**: the Delta rules' sigma points and Jacobians, `*` with a matrix, AR and
  ContinuousTransition, now that a typed scratch costs nothing.
- **A product workspace per variable**, for the temporaries of multivariate products in
  BayesBase; to be profiled first.
- **A fast path for variables of degree 2 in the equality chain.** Every state-space model's chain
  variables have that degree, but the path changes the product events callbacks see.
- **The BayesBase ambiguity `prod(::GenericProd, ::ProductOf, ::Missing)`**, upstream.
- **The HMM allocates 1.12× v6's bytes per iteration** (§1), though it runs at 0.89–0.92× v6's
  time; not profiled yet.
