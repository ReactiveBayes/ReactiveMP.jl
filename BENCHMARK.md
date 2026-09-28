# Benchmark — the end-of-refactor performance pass

The record of the performance pass that `PHASES.md` scheduled before the v7 release
(2026-09-26 to 2026-09-28, user). It covers ReactiveMP v7 with RxInfer's `refactor/reactivemp-v7`
branch, Rocket and GraphPPL, measured against v6.

**Nothing here is applied yet.** Every change is a diff in
`investigations/performance-pass/diffs/`. The adopted design changes are in `PLAN.md` § Rule
surface (*The `RuleSpec` is the execution vehicle*, *Scratch*), and the reasoning is in
`DISCUSSION.md` §5.

The pieces:

- `investigations/performance-pass/`: the scripts, raw data and per-package notes.
- `investigations/performance-pass/README.md`: a shorter summary.
- The published report: a private artifact linked from the session.

Revisions measured:

| package | revision |
|---|---|
| ReactiveMP | `d50fac9c` |
| RxInfer | `4ca9eb88` |
| Rocket | master (1.10.0 plus CI-only commits) |
| GraphPPL | 4.8.0 (`4cc7c6c`) |
| v6 | ReactiveMP 6.5.0 with RxInfer 5.5.2, on the same Rocket 1.10.0 and GraphPPL 4.8.0 |

The machine: Julia 1.13.0, an M-series Mac, 10 cores.

---

## 1. Verdict

> **Corrected by round 2 (2026-09-28, in progress; `investigations/performance-pass/round2/`).**
> The tables below are transcribed correctly, but this verdict overstates them:
> - **"Faster than v6 on six of seven"** holds only for the marginal cost of one more iteration.
>   The n = 10⁴ figure is an artefact of where a GC pause fell: GC-free on both sides, FULLT is
>   1.29× *slower*.
>   - Two slower benchmarks are left out of the count: iid@100 and iid with RxInfer's defaults.
>   - End to end, at the iteration counts benchmarked, FULLT is slower than v6 on every model
>     except the streaming filter and nl: 1.06–1.61× in round 2's paired baseline. Setup is the
>     gap, at 1.4–2.0× v6.
> - **"Under 1 s"** holds for iid and Beta–Bernoulli only; ssm1 goes 13.6 → 6.6 s.
>   - Without a workload on either side, v7 is 1.13–1.30× v6 to first inference. v6 has no
>     workload and would gain from one too.
>   - The steady-state gain from a workload is most likely Julia 1.13's GC no longer marking
>     package-image objects (JuliaLang/julia#61474).
>   - The precompile times given have no data behind them.
> - **"Bit-identical"** compared `(type name, mean, cov)` summaries, not distributions, and run
>   2's reference is not recorded.

- **Where HEAD stood.** v7 was 1.5–4.6× slower than v6 in steady state, 2.2–3.2× in setup, and
  15–30% slower to first inference, on the same Rocket and GraphPPL. So every difference comes
  from the engine.
- **The rule system's design is not the cause.** Of an 873 ns rule call, the rule body is 49 ns
  and resolution 26 ns. The rest came from the rule's result being typed `Any`, because
  `RuleSpec.body::Function` erases it:
  - a callback event built with nobody listening: ≈ 300 ns;
  - `Message{D, L}` built from `Any`: ≈ 380 ns, through `Core._compute_sparams`;
  - `uuid4()` span ids whenever callbacks are set: ≈ 760 ns, because `uuid4()` reads the OS
    entropy source;
  - the 18-field `RuleSpec` boxed at every dynamic call.
- **Activation** cost ≈ 25 µs per node, a third of it in `input_names`.
- **With the recommended set and a typed `RuleSpec`** (variant FULLT):
  - v7 is faster than v6 in steady state on six of the seven iterative benchmarks, large models
    included;
  - the setup-bound belief-propagation models run at 1.25–1.43× v6;
  - a rule call takes 27 ns and 2 allocations.
- **With RxInfer's precompile workload**, the first inference drops from ≈ 9 s to under 1 s on
  models shaped like the workload, and by 29–60% on the others.
- **Correctness.** Posteriors and free energies are bit-identical to HEAD on every model: 204/204
  and 153/153 comparisons in the two final runs. Every package's suite passes on the variants
  recommended.
- **No breaking release of Rocket or GraphPPL is needed.** The breaking candidates were measured
  and don't pay (§ 6).
- **Left for the next round:** setup, still 1.4–2.2× v6, and the remaining allocations (§ 8).

## 2. Method

### Variants

A variant is a set of sibling git worktrees of ReactiveMP, RxInfer, Rocket and GraphPPL, with an
environment developing them (`mkvariant.sh`, `mkenv.jl`). Each is rebuilt from the recorded
revisions and diffs.

| variant | what |
|---|---|
| **v6** | RxInfer 5.5.2 over ReactiveMP 6.5.0 (registry) |
| **A** | HEAD |
| **P12** | A + P1 (lazy callback events) + P2 (constructor barrier) |
| **E4** | every engine change, with the P3 barrier (`ENGINE-ReactiveMP.diff`) |
| **FULL** | E4 + Rocket P7, P9 + GraphPPL C1, C2, C3, C6, C7 + RxInfer P6 and its bug fix |
| **FULLPC** | FULL + P11, RxInfer's precompile workload |
| **T** | P12 + P4 + P5 + a typed `RuleSpec` (`ENGINE-T-ReactiveMP.diff`) |
| **TS** | T + typed scratch (`TS-typed-scratch.diff`) |
| **FULLT, FULLTPC** | FULL and FULLPC with T's engine instead of E4's |

### Protocol

- **The final runs.** `driver.sh` ran each variant on 11 models and 6 scaling cases, plus the
  micro suite, over three rounds:
  - every run in a fresh process;
  - the variant order rotated each round;
  - nothing else running on the machine.

  Figures are the median over rounds of each round's minimum. There were two such runs:
  - run 1: v6, A, P12, E4, FULL, FULLPC;
  - run 2: v6, FULL, FULLT, FULLTPC.

  Run 2 was restarted once from scratch after an interruption; only complete runs are used.
- **Development measurements** (marked *indicative*) were taken while other benchmark jobs shared
  the machine. They always alternated variants in fresh processes.
- **Iterative models** are run at I and 2I iterations, so the cost per iteration is
  (T(2I) − T(I)) / I and setup is T(I) − I · (per iteration).
- **Options.** Runs use `session = nothing` and `free_energy = Float64`, except the `+defaults`
  rows, which use RxInfer's defaults.
- **Time to first inference** is measured from a fresh process after `using`, so package load
  (≈ 2 s) is not included.

### Benchmarks

| script | what it measures |
|---|---|
| `bench_micro.jl` | one `MessageMapping` call for six rules (direct, through a barrier, with an empty callbacks object), resolution alone, `execute_rule` alone, deferred materialisation, a joint marginal, products of 2–100 messages, Rocket primitives (subject fan-out, `combineLatest`, `collectLatest` at N = 10²–10⁴), the event and constructor costs, and engine graph loops without RxInfer (iid at n = 10²–10⁴, chain at n = 300 and 1 000, with and without free energy, build+activate separately) |
| `bench_models.jl` | end to end through RxInfer, v6 or v7: ssm1, ssm2, iid, betabern, nl, hmm, gmm, linreg, and the streaming `filter` with `@autoupdates`. Sizes via `@n`. Records load, first inference, steady state, bytes and GC share, and dumps posteriors |
| `bench_creation.jl` | model creation by stage (GraphPPL graph; plus plugins; plus ReactiveMP nodes and activation) and size |
| `bench_scratch.jl`, `bench_scratch_toy.jl` | rules with scratch, typed against untyped |
| `bench_rxinfer_opts.jl` | RxInfer's options: session, free energy element type, benchmark and trace |

### Analyses

- CPU profiles with `Profile`, 0.1–0.2 ms sampling, with a classifier that attributes the
  innermost Julia frame to a module and counts marker functions inclusively: `prof_model.jl`,
  `prof_graph.jl`, `prof_activate.jl`, `prof_creation.jl`, `prof_streaming.jl`.
- Allocation profiles with `Profile.Allocs`.
- JET's `report_opt` on a rule call, a product, a joint marginal, `batch_inference`,
  `create_model` and the BIFM mapping: `jet_engine.jl`, `jet_rxinfer.jl`, `jet_graphppl.jl`.
- SnoopCompile's `@snoop_inference` for first-inference time by module: `snoop_ttfx.jl`.
- Stack depth per chain link: `stack_depth.jl`, `stack_frames.jl`.
- Rocket's fast paths against simple alternatives: `rocket_fastpaths.jl`.

The guidance followed:

- the Julia manual's *Performance Tips* and *Profiling*;
- the devdocs on dispatch, the method cache and the specialisation heuristics (`Function`,
  `Type` and `Vararg` arguments; closures and keyword arguments);
- the Julia 1.13 release notes;
- JET's optimisation analysis;
- PrecompileTools and SnoopCompile on latency.

The user asked that no private compiler API (`Core.*`) and no new `@generated` function be used
in anything recommended. None is.

### Correctness gate

Every change counted only if its posteriors and free energies were bit-identical to HEAD
(`compare_variants.jl`, `compare_posteriors.jl`) and its packages' suites passed.

- ReactiveMP root: 15 339 pass, 6 known broken, with Aqua on for T and TS.
- MessagePassingRulesBase: 617, allocation gates included.
- StandardMessagePassingRules: 10 139, 1 known broken.
- MessagePassingRulesTestUtils: 147.
- Every node package on T: Delta 336, DiscreteTransition 2 128, Autoregressive 12 855,
  GaussianCoupling 293, Probit 398, GCV 463, SoftDot 928, ContinuousTransition 519, Pólya 108,
  BIFM 419, Flow 1 020, and Approximations.
- GraphPPL: 76 803, 1 known broken.
- Rocket: 12 295.
- RxInfer on P6: 14 713.

## 3. Results

### End to end

The median over three rounds of each round's minimum. The v6, FULL, FULLT and FULLTPC columns are
from run 2; A, P12 and E4 are from run 1. v6 agreed between the two runs within 6%, except the
GC-bound iid at n = 10⁴ (11.2 ms in run 1, 16.8 ms in run 2).

| | v6 | HEAD | P1+P2 | engine (P3) | FULL (P3) | FULLT | FULLT + workload |
|---|---|---|---|---|---|---|---|
| iid VMP, per iteration, n = 10³ | 647 µs | 3.1 ms | 1.1 ms | 826 µs | 829 µs | 705 µs | 566 µs |
| iid VMP, per iteration, n = 10⁴ | 16.8 ms | 48.3 ms | 25.0 ms | 19.2 ms | 23.2 ms | 14.5 ms | 11.8 ms |
| iid VMP, per iteration, n = 10⁵ | 217.8 ms | 784.8 ms | 506.6 ms | 295.6 ms | 309.9 ms | 204.4 ms | 175.7 ms |
| Gaussian mixture, per iteration | 3.0 ms | 7.4 ms | 3.3 ms | 3.0 ms | 2.7 ms | 2.4 ms | 2.3 ms |
| HMM, per iteration | 3.1 ms | 5.1 ms | 3.3 ms | 3.2 ms | 3.1 ms | 3.0 ms | 2.9 ms |
| nonlinear SSM (Delta), per iteration | 3.7 ms | 5.5 ms | 2.8 ms | 2.5 ms | 2.4 ms | 2.3 ms | 1.9 ms |
| loopy linear regression, per iteration | fails (`DomainError`) | 20.7 ms | 12.1 ms | 10.6 ms | 9.0 ms | 7.9 ms | 7.9 ms |
| streaming Kalman filter, 1 000 points | 9.0 ms | 13.6 ms | 7.2 ms | 6.1 ms | 5.9 ms | 5.3 ms | 4.4 ms |
| Kalman smoother (BP), n = 10³ | 59.0 ms | 119.8 ms | 114.8 ms | 78.1 ms | 78.8 ms | 78.0 ms | 74.7 ms |
| 2-D Kalman smoother (BP) | 106.3 ms | 190.8 ms | 184.2 ms | 134.2 ms | 131.0 ms | 132.9 ms | 128.1 ms |
| Beta–Bernoulli (BP), n = 5 000 | 71.4 ms | 158.5 ms | 155.1 ms | 103.5 ms | 102.5 ms | 102.0 ms | 95.8 ms |
| iid setup | 18.1 ms | 41.1 ms | 40.5 ms | 26.7 ms | 25.9 ms | 25.5 ms | 23.8 ms |
| mixture setup | 21.7 ms | 75.5 ms | 80.7 ms | 47.4 ms | 47.2 ms | 46.5 ms | 44.4 ms |
| HMM setup | 14.4 ms | 35.7 ms | 36.7 ms | 22.1 ms | 23.3 ms | 20.0 ms | 21.8 ms |
| first inference, iid | 7.51 s | 8.71 s | 8.68 s | 8.90 s | 8.74 s | 8.84 s | 0.69 s |
| first inference, Kalman smoother | 11.40 s | 13.58 s | 13.51 s | 13.78 s | 13.72 s | 13.61 s | 6.59 s |
| first inference, HMM | 14.07 s | 17.13 s | 16.95 s | 17.39 s | 17.50 s | 17.67 s | 12.52 s |
| first inference, mixture | 10.45 s | 13.66 s | 13.46 s | 13.83 s | 13.76 s | 13.65 s | 5.48 s |

**Allocations per iteration** (iid, n = 10⁴):

| v6 | FULL (P3) | FULLT |
|---|---|---|
| 7.2 MB | 13.3 MB | 9.4 MB |

GC share of the whole run at n = 10⁵: 20% on v6, 28% with P3, 21% with the typed spec.

**Every row is in** `investigations/performance-pass/results/final/tables.md` and
`results/final2/tables.md`. The raw data is in `models.tsv` and `micro.tsv`.

### One rule call

`MessageMapping`, NMV towards `out`, belief propagation, concretely typed inputs:

| | time | allocations |
|---|---|---|
| HEAD | 873 ns | 15 (752 B) |
| with any callbacks set, HEAD | 1 662 ns | 16 |
| P1 + P2 | 181 ns | 11 |
| + P3 barrier | 69 ns | 5 (192 B) |
| + typed `RuleSpec` (T) | **27 ns** | **2 (64 B)** |
| the rule body alone, untyped / typed spec | 49 / 3.8 ns | 4 / 0 |
| resolution alone (static dispatch, incl. a barrier) | 26 ns | 2 |
| Categorical towards `p` (HEAD → P3 → T) | 1 129 → 169 → 124 ns | 19 → 7 → 4 |
| joint marginal NMV `(out, μ)` (HEAD → P3 → T) | 236 → 148 → 95 ns | 15 → 9 → 7 |
| product of two Gaussian messages (HEAD → P1+P2) | 1 262 → 514 ns | 9 → 6 |

### Engine graphs without RxInfer

Per iteration, then build+activate:

| | HEAD | P1+P2 | E4 | FULL (P3) | FULLT |
|---|---|---|---|---|---|
| iid n = 10³, per iteration | 1.33 ms | 0.60 ms | 0.46 ms | 0.45 ms | 0.37 ms |
| iid n = 10⁴, per iteration | 20.3 ms | 12.3 ms | 6.8 ms | 5.7 ms | 4.6 ms |
| chain n = 10³, per iteration | 9.4 ms | 5.2 ms | stack overflow | 4.1 ms | 3.8 ms |
| chain n = 10³, build+activate | 74.8 ms | 74.9 ms | stack overflow | 40.0 ms | 39.0 ms |

E4 overflows the stack because it lacks P9; see § 5.2.

## 4. Findings in the engine

### 4.1 The untyped rule result (the largest cost)

**The first profile.** It was of iid VMP on HEAD:

- 19% of all samples were self time on one line, `message.jl:812`: the
  `AfterMessageRuleCallEvent`, and the `Message` built after it.
- `_compute_sparams` was 13.8% of samples, against 1.5% on v6.
- Dynamic dispatch took 11.6% and GC 9%.

The micro suite then split that one rule call into its parts (§ 1). All of it follows from
`body::Function`. v6 dispatched the rule itself, so its result was inferred. v7 stores the body
in the spec, and everything downstream (`unwrap_result`, the event, the `Message`) runs on
`Any`. Four fixes were measured, in this order:

- **P1: lazy callback events.** The `@invoke_callback` macro builds an event only when
  `listens(callbacks, EventType)` holds, which folds for a `NamedTuple`. Span ids come from a
  counter rather than `uuid4()`; a draw from the task RNG would shift the rules' random streams.
- **P2: a constructor barrier.** `@noinline new_message` and `new_marginal` at the two sites that
  build a message or marginal from an `Any` value, so the construction is a cache-hit dispatch
  instead of a runtime computation of type parameters.
- **P3: a function barrier on the body.** `execute_rule(then, spec, …)` calls the body behind one
  `@noinline` function specialised on `typeof(spec.body)`, and builds the message inside it
  (`MessageTail`, `MarginalTail`; P3c does the same for the node free energy). P3b passes the
  barrier the spec's fields rather than the spec, because the 18-field immutable `RuleSpec` was
  being copied into a fresh 288-byte box at every dynamic call. P3 keeps `RuleSpec` untyped.
- **T: a typed `RuleSpec{B, P, S, L, A}`.** The body, preallocation, scratch and log-scale
  functions, and the algorithm type, become type parameters. `RuleResult` gains the spec's type as
  a parameter, so the interactive calls still allocate nothing.
  - At the engine's call site, resolution runs on concrete input types and reaches one rule. So
    `find_message_rule` returns one concrete spec, the body call is static and inlines, and
    `rule_algorithm`'s `isa` folds.
  - No barrier, continuation or extra API is needed.
  - Against P3: 0.83–0.92× per iteration, 0.62–0.66× at n ≥ 10⁴ (fewer allocations, less GC),
    setup unchanged, time to first inference unchanged (0.99–1.01×).

**T supersedes P3.** It reverses `DISCUSSION.md` §3.14's decision. That decision measured a toy
call site that could reach two rules, where a parameterised spec gives a union of spec types.
Engine call sites don't have that shape. Where resolution genuinely can't be inferred, the call
is dynamic in either design.

### 4.2 Scratch (TS)

Under T, the one path left untyped was the scratch kept between calls. `ScratchSlot` holds it as
`Any`, because its type depends on the input types, which aren't known at activation.

**TS** adds an optional `scratch_type` keyword to the rule macros: a function over the same slots
as `scratch` that returns the scratch's type from the input types. It constant-folds.

- **How the engine uses it.** It asserts the kept scratch to the declared type after an `isa`
  guard, and rebuilds when the input types change.
- **A wrong declaration** is an `ArgumentError` naming the rule, not a silent rebuild on every
  call.

| | untyped scratch | typed scratch |
|---|---|---|
| a toy rule with a small scratch | 73.8 ns, 5 allocations | 33.5 ns, 2 allocations |
| BIFM towards `zprev`, dz = 2 / 4 / 8 | 910 / 1 238 / 1 800 ns | 855 / 1 167 / 1 721 ns |
| runtime dispatches on BIFM's path (JET) | 2 | 0 |

An untyped scratch costs the engine ≈ 40 ns and 3 allocations per call. BIFM, the only rule
package with scratch, gains 5–7%, because its own matrix algebra dominates.

In-place output needs no such declaration: under T, `prealloc` is a typed field, so the output is
inferred. Reusing the output buffer across calls is a separate question (`PLAN.md` open item
#10): the buffer escapes into the emitted `Message`, so reuse needs ownership rules, not types.
No rule package declares `inplace = true` today.

### 4.3 Activation (P5)

Build+activate cost ≈ 25 µs per node on HEAD, two thirds of it in the C runtime:
`lookup_type_setvalue`, subtype checks, `has_free_typevars`.

- **`input_names`** was ≈ 33% of build+activate. It built `Any[]` vectors and a
  `Val{Tuple(...)}` at run time, for every interface of every node and for every node's free
  energy.
- **`alias_interface`** was the other large cost. Its node argument is a `Type`, which Julia
  doesn't specialise on (the manual's heuristic for `Type` arguments), so every call was a dynamic
  `nodespec` followed by a linear scan.
- **`node_specification`** called `applicable` for every node.

P5a caches `input_names` and replaces `applicable` with a `MethodError`-guarded call. P5b caches
`alias_interface`.

| engine chain, n = 10³ | before | after |
|---|---|---|
| node activation | 37.5 ms | 13.4 ms |
| free-energy subscription | 13 ms | 4.5 ms |
| factor-node creation | 18 ms | 12 ms |
| build+activate overall | 75 ms | 40 ms |

### 4.4 The equality chain (P4)

Each partial product in the chain ran the full n-ary `compute_product_of_messages`: the events,
the fold, the form constraint (RxInfer's `EnsureSupportedFunctionalForm` check) and a second
`Message`. P4 changes three things:

- the steps use `compute_product_of_two_messages`;
- the form constraint and fold apply once per outbound message;
- `ChainOutboundMapping{C}` is typed.

Custom fold functions keep the old path. **This is a semantic change:** "check last" now means
the whole product. The results are identical under RxInfer's default constraint, a check that
returns its input.

Loopy linear regression goes from 21.8 to 13.6 ms per iteration (indicative), and from 12.1 to
7.9 ms in the final runs, with the rest of the changes.

### 4.5 Smaller engine observations

- **JET on the prototype stack** reports only cold warning and error paths on the hot path, plus
  one dynamic `as_message(::AbstractMessage)` per message in a product. The cold paths are
  `audit_rule`'s warning, which is inlined into every mapping (code size, not time), and
  `throw_missing_services`.
- **Rule resolution is static and cheap.** `find_message_rule` has 234 methods; a fresh
  resolution compiles in ≈ 2 ms; MessagePassingRulesBase is ≈ 1.2% of first-inference inference
  time.
- **Free-energy setup for deterministic nodes** costs ≈ 58 µs per `+`/`*` node. Loopy linear
  regression's setup is 15 ms without the free energy and 131 ms with it.
- **The equality chain** costs ≈ 10 µs per observation in loopy BP, against ≈ 1 µs for a VMP
  update. It is linear, not quadratic.
- **Observations and constants** carry log scale `0::Int`, while rules produce `Float64` or
  `nothing`: two `Message` specialisations per distribution.
- **When log scales are tracked,** `UndefinedLogScale(:no_declaration, spec)` boxes the whole spec
  for each message, and `product_logscale` calls `applicable` for each product.
- **The `MessageMapping` constructor** builds `Val(logscales)` from a runtime `Bool`, making the
  mapping type a runtime construction per interface.
- **The engine's existing `@generated` functions** (`rule_messages`/`rule_marginals`,
  `canonical_keys`, `Marginals`, `rule_inputs`, `input_value`) are candidates for `ntuple`/`map`
  rewrites, to be measured before any change.

## 5. Findings in the other packages

### 5.1 RxInfer (P6, P11)

**RxInfer's own code is a small share.** It is 1.0% of iid VMP, 3.0% of the Kalman smoother and
0.5% of a streaming filter (47 µs per data point). The per-iteration data feed is already
concretely typed, and no RxInfer code runs per message. There is a floor of ≈ 140 µs and 2 100
allocations per `infer` call, mostly GraphPPL.

**Option costs:**

| option | cost |
|---|---|
| `session` default | +1–3% |
| `free_energy = true` (`Real`) against `Float64` | ≤ 2%, within noise |
| the free energy itself | 13–15% |
| `benchmark = true` | +23%, 81.7 MB against 51.9 MB |
| `trace = true` | +26% |

**JET** finds only setup and teardown dispatches. The lowered code had 8 boxed captured variables,
none on a per-iteration path.

**Changes:**

- **P6d:** `listens` for `RxInferBenchmarkCallbacks`, and the trace honours its include filter.
  `benchmark = true` adds 2.4% instead of 23%. Needs P1.
- **P6a:** the 8 boxes removed; hygiene.
- **P6c:** single-pass typed graph getters, and a top-level vardict. Submodel-heavy models:
  1.69 ms → 36 µs.
- **Bug fix:** `iterate(::InferenceResult)` read a missing `:returnval` field, so `a, b = infer(…)`
  threw.
- **Rejected, P6b:** a typed `GraphVariableRef{V}` regresses setup 3×. It builds a parametric
  struct from an `AbstractVariable` value, P2's phenomenon again.
- **P11, a PrecompileTools workload:** BP state space, iid VMP and Beta–Bernoulli, each with and
  without free energy.
  - First inference: iid 8.8 → 0.69 s, Beta–Bernoulli 7.8 → 0.53 s, streaming filter
    9.5 → 2.6 s, mixture 13.7 → 5.5 s, Kalman smoother 13.6 → 6.6 s, HMM 17.7 → 12.5 s.
  - Steady state is also 4–20% faster than the JIT-compiled build. The cause wasn't investigated.
  - The cost: RxInfer precompiles in ≈ 21 s instead of 10 s, and loads 0.2–0.3 s slower.
- **Telemetry:** RxInfer's `__init__` sends a telemetry ping on a background task by default,
  compiling HTTP and JSON on each session's first use. That costs 0–2.5 s of time to first
  inference, and affects v6 and v7 alike. It is a policy choice, not proposed here.

### 5.2 Rocket (P7, P9)

- **P7, non-breaking:**
  - `collectLatest` and `GenericUpdatesStatus` keep pending counters instead of running
    `all(vstatus) && !all(cstatus)` over their BitArrays on every event, which was O(N²/64) per
    iteration for the free energy and for a variable's marginal.
  - The three combine and collect wrappers become `mutable struct`s, so an emission into an
    abstractly typed actor no longer boxes the whole wrapper (128 → 48 bytes per event).
  - Results: `collectLatest` round at N = 10⁴ 484 → 194 µs; the engine graph at n = 10⁴
    6.8 → 5.6 ms per iteration; allocations 7–9% lower; neutral at n ≈ 10³.
- **P9, new public API:** `stackguarded` and `set_stack_guard_limit!` (default 256). Lazy
  subscribe and unsubscribe continue on a fresh task past the limit.
  - Subscription costs ≈ 8 KB of stack per chain link: 22 native frames, four of them dynamic
    `LazyObservable` crossings.
  - The longest chain without RxInfer's `limit_stack_depth` goes from ≈ 1 030 to 3 264 links, at
    no measured steady-state cost.
  - The engine changes need P9: without it the 1 000-link chain graph overflows the stack on E4,
    whose links are slightly deeper.
  - The guard's counter is one process-wide object. It must become per task (or per thread)
    before release, and it needs a unit test.
- **Rejected:**
  - Guarding `Subject.on_next!`: at least 8 192 links, but 11–25% slower per iteration.
  - Typed listener storage: FunctionWrappers is 10–25× slower than today's dispatch and depends
    on `llvmcall`/`cfunction` internals. An actor type parameter helps only for homogeneous
    listeners, which engine subjects almost never have, and it is breaking.
  - `LazyObservable` swap-remove, typed `PendingScheduler` storage and `map`'s `::R`: they cost
    only at teardown or not at all, and `::R` would be mildly breaking.
- **The hand-written fast paths** (`UInt8UpdatesStatus`, `MStorage1…16`) are neither wrong nor
  faster. Against simple alternatives (`NTuple{N,Bool}` or `Vector{Bool}` with a counter), runtime
  is equal for 2–16 sources with zero allocations, because the engine's element types are
  abstract. `Vector{Any}` storage compiles 20–35% faster for new source types. Keep them.
- **The `snapshot`** costs 36 ns and 32 bytes per emission, and is the barrier that hands the
  rules concrete types.
- **`actor::Any`** costs ≈ 1.6 ns of dispatch per listener, and is the compile-time barrier.

### 5.3 GraphPPL (P8)

For ordinary models GraphPPL is 5–8% of model creation. For ssm1 at n = 10⁴, full creation splits
as:

| share | where |
|---|---|
| 30% | factor-node construction |
| 41% | activation |
| 17% | the free-energy plugin |
| 4.6% | GraphPPL's own graph |

Beyond ≈ 3·10⁴ nodes creation turns superlinear through GC, while bytes per node stay constant.

| change | effect | breaking |
|---|---|---|
| **C1** `apply_meta!` walks the context's own factor nodes | it was quadratic, 84% of creation: nl with `@algorithm` at n = 3 000, 4.51 → 0.47 s | no |
| **C6** prefix sums for `flattened_index` while constraints are applied | matrix-variable constraints were quadratic (19 s at 10⁵ elements): 3.07 → 0.81 s at 4·10⁴ | no |
| **C2** `NodeLabel ==` compares the counter first | 30 → 19 ns per lookup | no |
| **C3** constraint bitsets built directly | graph stage 10–13% faster on hmm and iid | no |
| **C7** `ConstraintStack` as a `Vector` | its `Deque` allocated a 1 024-element block per model: 12–29% of a tiny model's `infer` | no |
| **C4** per-node extras as a small vector of pairs | 29 → 7 ns per typed access; graph stage 3–5% faster; +16–20 ms on a session's first model | soft |
| **C5** `created_by` as an `Expr` | 70–87% fewer specialisations, no measurable time | soft |
| **P8b** storage redesign (spike only) | ≲ 1–2% of creation | yes |

- **P8b's scope:** `Vector{NodeData}` indexed by counter, typed extras, no `edge_data`. It would
  not break RxInfer, which never reads `model.graph`, MetaGraphsNext or `.extra`. It would break
  GraphPPL's own engine (25 sites), the plotting and GraphViz extensions, and
  `savegraph`/`loadgraph`/`prune!`.
- **`is_factorized` blowing up exponentially was ruled out:** named `:=` variables carry no
  links, and `all` short-circuits.

## 6. Options for the release

| change | package | call |
|---|---|---|
| P1 lazy callback events, counter span ids | ReactiveMP | release |
| P2 constructor barrier (still used on the fallback path) | ReactiveMP | release |
| **T typed `RuleSpec`** | MessagePassingRulesBase, ReactiveMP | **release**; supersedes P3 |
| **TS typed scratch (`scratch_type`)** | MessagePassingRulesBase, ReactiveMP, BIFM | **release** |
| P4 equality chain | ReactiveMP | release (semantic note) |
| P5a, P5b activation caches | ReactiveMP | release |
| P3, P3b, P3c the function barrier | MessagePassingRulesBase, ReactiveMP | superseded by T |
| P11 precompile workload | RxInfer | release |
| P6d benchmark callbacks (needs P1), P6a, P6c, the `iterate` fix | RxInfer | release |
| P6b typed `GraphVariableRef` | RxInfer | no |
| default `free_energy = Float64` | RxInfer | no: breaks AD, ≤ 2% |
| P7 counters and mutable wrappers | Rocket 1.11 | release |
| P9 stack guard | Rocket 1.11 | release after making the counter per task |
| P9b emission guard, P7b typed listeners | Rocket | no |
| C1, C2, C3, C6, C7 | GraphPPL 4.9 | release |
| C4 typed extras | GraphPPL | later, with a GraphPPL workload |
| C5 `created_by` as an `Expr` | GraphPPL | optional |
| P8b storage redesign | GraphPPL | no |
| P10 a restricted custom rule dispatch | MessagePassingRulesBase | no: resolution is static, 26 ns and ≈ 1% of compile time |

**No breaking release of Rocket or GraphPPL is needed for performance.** A Rocket 2.0 planned for
other reasons could aim at fewer frames per subscription link (typed or flattened message
streams, co-designed with the engine), which would raise the stack limit and cut dispatches.

## 7. Found on the way

- RxInfer's `iterate(::InferenceResult)` bug (§ 5.1).
- v6 fails on loopy linear regression with a `DomainError`, and v7 runs it: a line for the
  migration guide.
- The engine changes need Rocket's P9 (§ 5.2).
- The comparison is fair on packages: v6 resolves the same Rocket and GraphPPL versions.
- RxInfer's telemetry compiles HTTP and JSON on each session's first use (§ 5.1).

## 8. Next round (measured, not prototyped)

- **Setup, 1.4–2.2× v6.** This is also why the one-shot BP models trail v6. Candidates:
  - `Val(logscales)` from a runtime `Bool`;
  - a cached resolution plan per (node type, interface names, factorisation);
  - the deterministic free-energy setup (≈ 58 µs per node).
- **Allocations, 1.3× v6's bytes per iteration.**
  - Per rule call: the `Message` (mutable, by the §3.19 benchmark) and its `AnnotationDict`. A
    shared empty dictionary with copy-on-write would remove the second.
  - Per update: the `DeferredMessage`, and boxed `CountingReal`s on every free-energy subject.
- **Workloads per rule package.** HMM, Delta and 2-D Kalman still take 10–12 s to first
  inference.
- **Log scales when tracked** (§ 4.5).
- **The engine's `@generated` functions**, measured before any rewrite.

## 9. Reproducing

- **The worktrees are gone after the session.** Recreate a variant with
  `VROOT=<dir> investigations/performance-pass/mkvariant.sh <name>`, applying its diffs:
  - E4: `ENGINE-ReactiveMP.diff`;
  - T: `ENGINE-T-ReactiveMP.diff`;
  - TS: T + `TS-typed-scratch.diff`;
  - FULL(T): the engine diff + `P7-Rocket.diff` + `P9-Rocket.diff` + `P8rel-GraphPPL.diff` +
    `P6-RxInfer.diff` + `P6bug-RxInfer.diff`;
  - FULL(T)PC: that + `P11-RxInfer.diff`.
- **Run** `VROOT=<dir> investigations/performance-pass/driver.sh <outdir> "<variants>" 3`, then
  `python3 summarise.py <outdir>` for `tables.md` and `summary.json`, and `compare_variants.jl`
  for the posterior gate.
- **The report page** is `report/performance-pass.template.html`, with its data from
  `results/final2/pagedata.json`.
