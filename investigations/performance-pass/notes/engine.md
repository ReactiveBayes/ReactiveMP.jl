# Engine (ReactiveMP + MessagePassingRulesBase): findings and prototypes

Measured at ReactiveMP `d50fac9c`, RxInfer `4ca9eb88`, Rocket master (1.10.0), GraphPPL 4.8.0,
Julia 1.13.0, 10 cores. Numbers marked *indicative* were taken while other benchmark
processes ran; the final table comes from `driver.sh`, sequentially.

## Where v7 stood against v6 (baseline A, indicative)

| model | v7 (A) | v6 | ratio |
|---|---|---|---|
| iid VMP, per iteration (n = 1000) | 2.98 ms | 0.57 ms | 5.2× |
| hmm, per iteration (n = 300) | 5.3 ms | 3.4 ms | 1.6× |
| nl (Delta), per iteration | 5.6 ms | 3.7 ms | 1.5× |
| ssm1 (BP, one infer, n = 1000) | 128 ms | 63 ms | 2.0× |
| iid setup | 42 ms | 20 ms | 2.1× |
| time to first inference, ssm1 / hmm | 14.5 / 17.8 s | 12.0 / 14.6 s | 1.2× |

Same Rocket and GraphPPL on both sides: the gap is the engine.

## What the profiles showed (`prof_model.jl`, `prof_graph.jl`, `prof_activate.jl`)

- iid on A: `_compute_sparams` 13.8% of samples (v6: 1.5%), dynamic dispatch 11.6%, GC 9%; the
  hottest line `message.jl:812` (the `AfterMessageRuleCallEvent` and the `Message` built after it,
  from an `Any`-typed rule result), 19% self.
- micro, one rule call on A (`bench_micro.jl`): 890–1180 ns, of which the rule body is ~50 ns
  and resolution ~26 ns; an event with `result::Any` 300 ns; `Message` from `Any` 400 ns; with
  any callbacks set, `uuid4()` (the OS entropy source) 725 ns more per call.
- product of 2 Gaussian messages: 1.29 µs, almost all envelope.
- activation (`prof_activate.jl`, chain n = 1000): ~33% of build+activate in `input_names`
  (`rule_arguments.jl:60,67`, `Any[]` and a runtime `Val{Tuple(...)}` per interface), node
  creation in `alias_interface` (`node::Type` not specialised, `nodespec` dynamic plus a linear
  scan) and `clusterkey`; C-runtime type instantiation dominates the self time.
- superlinear graph scaling (iid n = 1000 → 10000: 1.36 → 13.5 ms per iteration): Rocket's
  `collectLatest` `all(vstatus)` per event (variable marginal and free energy), handed to the
  Rocket fork.
- JET (`jet_engine.jl`, `results/jet_engine*.txt`): on the prototype stack the hot path has only
  the intended barrier and one `as_message(::AbstractMessage)` per message in a product; the
  remaining reports are cold warning and error paths (`audit_rule`, `throw_missing_services`).
- Compile time (`snoop_ttfx.jl`): MessagePassingRulesBase ~1.2% of first-inference inference
  time; `find_message_rule` has 234 methods, a fresh resolution compiles in ~2 ms. No package has
  a precompile workload. RxInfer's `__init__` telemetry ping compiles HTTP/JSON in the background
  on every session's first use (0–2.5 s of time to first inference, model dependent).

## Prototypes (all bit-identical posteriors and free energies to A on ssm1, iid, hmm, nl, linreg, betabern; root suite passes: 15 339 pass, 6 known broken; base 617; standard 10 139 + 1 known broken)

| | what | diff | measured (indicative) |
|---|---|---|---|
| P1 | lazy callback events (`@invoke_callback`, `listens`) and a counter span id instead of `uuid4()` | `diffs/P1-ReactiveMP.diff` | callbacks=(;): 1.72 → 0.18 µs per rule call (with P2); iid 2.98 → 2.56 ms/iteration alone |
| P2 | `@noinline` constructor barrier for `Message`/`Marginal` from an `Any` value (2 sites) | `diffs/P2-ReactiveMP.diff` | with P1: rule call 0.89 → 0.18 µs; product ×2 1.29 → 0.53 µs; iid 2.98 → 1.26 ms/it; hmm 5.3 → 3.2 ms/it |
| P3 | `execute_rule(then, spec, …)`: one `@noinline` barrier specialised on `typeof(spec.body)`, the message built inside it (`MessageTail`, `MarginalTail`) | `diffs/P3-ReactiveMP.diff` | rule call 0.18 → 0.08 µs (allocs 11 → 6); iid 1.26 → 0.96 ms/it; no time-to-first-inference cost measured |
| P5a | `input_names` cache; `node_specification` without `applicable` | `diffs/P5a-ReactiveMP.diff` | node activation 37.5 → 13.4 ms, FE subscription 13 → 4.5 ms (chain n = 1000) |
| P5b | `alias_interface` cache in `given_key` | `diffs/P5b-ReactiveMP.diff` | factornode creation 18 → 12 ms (chain n = 1000) |
| P4 | equality chain: two-message product for partial products, the form constraint once per outbound message, typed `ChainOutboundMapping{C}` | `diffs/P4-ReactiveMP.diff` | linreg (loopy BP, n = 1000) 21.8 → 13.6 ms/it; others within noise |
| P11 | PrecompileTools workload in RxInfer (BP state space, iid VMP, Beta–Bernoulli, with and without FE) | `diffs/P11-RxInfer.diff` | time to first inference: iid 9.9 → 0.63 s, betabern 8.9 → 0.61 s, ssm1 15.6 → 6.7 s, nl 18.2 → 10.2 s, hmm 19.2 → 12.7 s; load +0.2–0.3 s; RxInfer precompile ~5 → ~70 s |
| P10 | restricted custom dispatch | not built | dispatch is static and ~free at run time (26 ns incl. a barrier) and ~1% of compile time: no target |

## Smaller items seen, not prototyped

- `MessageMapping` constructor: `Val(logscales)` from a runtime `Bool` makes the mapping type a
  runtime construction per interface (branch on the Bool).
- `AnnotationDict()` per rule call and per product (16 B each; a shared empty value needs
  copy-on-write, since rules annotate).
- `UndefinedLogScale(:no_declaration, spec)` boxes the whole `RuleSpec` per message when log
  scales are tracked; `product_logscale` calls `applicable` per product when tracked.
- observations and constants carry log scale `0::Int`, rules `Float64` or `nothing`: two
  `Message` specialisations per distribution.
- `ScratchSlot{spec::Any, scratch::Any}`: fine after P3 (the scratch passes through the barrier).
- `audit_rule`: the warning path is inlined into every mapping (code size, not time).
- Free-energy setup for deterministic nodes: linreg setup 15 ms without FE, 131 ms with (58 µs
  per `+`/`*` node).
- Subscription stack depth: the engine chain graph overflows the stack at n = 3000 without
  RxInfer's `limit_stack_depth` (Rocket fork, P9); linreg at n = 3000 likewise.
- Existing `@generated` functions (`rule_messages`/`rule_marginals`, `canonical_keys`,
  `Marginals`, `rule_inputs`, `input_value`): the user prefers to avoid them; candidates for
  `ntuple`/`map`-over-`Val` rewrites where inference allows, measured before any change.
- v6 fails on loopy linear regression with a `DomainError`; v7 runs it.

## T: a typed `RuleSpec` (the user's follow-up question), supersedes P3

`RuleSpec{B, P, S, L, A}`: the body, preallocation, scratch and log-scale functions and the
algorithm type become type parameters (`RuleResult` gains the spec's type as a parameter, so the
interactive calls still allocate nothing). At the engine's call site the resolution is inferable,
so `find_message_rule` returns one concrete `RuleSpec{…}` and the body call is static and inlines:
no barrier, no continuation, no `execute_rule(then, …)`. The engine is HEAD's code plus P1, P2,
P4, P5a, P5b (`diffs/T-typed-RuleSpec.diff` on top of them; all of it `diffs/ENGINE-T-ReactiveMP.diff`).

§3.14 rejected the parameterised spec on a toy: a call site that can reach two rules gets a
union of spec types. The engine's call sites resolve from concrete input types and reach one rule,
so the union does not arise there; where it would (an uninferable call site), the call is dynamic
in either design.

| | E4 (P3 barrier) | T (typed spec) |
|---|---|---|
| one rule call (NMV, BP) | 69 ns, 5 allocs, 192 B | **27.6 ns, 2 allocs, 64 B** |
| `execute_rule` alone | 49 ns, 4 allocs | **3.8 ns, 0 allocs** |
| Categorical rule call / joint marginal | 170 / 148 ns | 124 / 95 ns |
| per iteration: iid / hmm / nl / gmm (T/E4, two interleaved rounds, idle) | | 0.87 / 0.89 / 0.83 / 0.92 |
| setup, time to first inference | | 1.00 / 0.99–1.01 |

Posteriors bit-identical to HEAD on iid, hmm, nl, gmm, ssm1, betabern, filter. Suites on T: base
617 (allocation gates included), root 15 339 + 6 known broken with Aqua on, Standard 10 139 + 1,
TestUtils 147, and every node package (Delta 336, DiscreteTransition 2 128, Autoregressive
12 855, GaussianCoupling 293, Probit 398, GCV 463, SoftDot 928, ContinuousTransition 519, Pólya
108, BIFM 419, Flow 1 020, Approximations) pass.

Left dynamic: rules that declare scratch (only BIFM), whose scratch passes through the
`Any`-typed `ScratchSlot`; P2 still catches their result.

## TS: a typed scratch (the user's follow-up), on top of T

A rule may declare `scratch_type`, a function over the same slots as `scratch` that returns the
scratch's type from the inputs' types (`scratch_type = (args) -> @NamedTuple{acc::Vector{Float64}}`,
or computed from `eltype`s, as BIFM does). `RuleSpec` gains the parameter `ST`; the engine's
`scratch_for!` asserts the slot's value to the declared type after an `isa` guard (another type,
from other input types, is rebuilt), and a freshly built scratch that is not of the declared type
is an `ArgumentError` naming the rule, so a wrong declaration fails rather than rebuilding on every
call. Without `scratch_type` nothing changes. `diffs/TS-typed-scratch.diff` (on top of T).

| | untyped scratch (T) | typed scratch (TS) |
|---|---|---|
| toy rule with a small scratch (`bench_scratch_toy.jl`) | 73.8 ns, 5 allocs, 160 B | **33.5 ns, 2 allocs, 64 B** |
| BIFM towards `zprev`, dz = 2 / 4 / 8 (`bench_scratch.jl`, two rounds) | 910 / 1 238 / 1 800 ns, 34 allocs | 855 / 1 167 / 1 721 ns, 30 allocs |
| JET, BIFM mapping, runtime dispatches outside cold paths | 2 (the body call, `new_message`) | **0** |

An untyped scratch costs the engine ≈ 40 ns and 3 allocations per call; typed, a scratch rule
costs what a rule without scratch does. For BIFM the saving is 5–7%, its matrix algebra dominating.
Suites on TS: base 617, BIFM 419, TestUtils 147, root 15 339 + 6 known broken (Aqua on).

In-place output needs no such declaration: with the typed spec, `prealloc` is a typed field and
the output is inferred. Reusing the output buffer across calls is a separate question (PLAN open
item #10): the buffer escapes into the emitted `Message`, which the graph keeps, so reuse needs
ownership rules, not types. No rule package declares `inplace = true` today.
