# Does `Message{D}`'s type parameter help or hurt? (2026-09-25)

An investigation, not a decision: nothing here is applied to the engine. It was asked in Phase 7
(user): now that rules are resolved by `MessagePassingRulesBase.find_message_rule` over
`RuleArgs` rather than by v6's dispatch on `Message{Gaussian}`, would an untyped message
(`data::Any`) gain on the stream side, whose observables already carry `AbstractMessage`, more
than it loses on the rule-call side? And would Julia's function-barrier pattern
([kernel functions](https://docs.julialang.org/en/v1/manual/performance-tips/#kernel-functions))
recover the untyped version? It is the input to the end-of-refactor performance pass, together
with other ideas.

Measured at commit `851559a4` (branch `refactor/rule-node-system-rewrite`), Julia 1.13.0, 10
cores, nothing else running. End-to-end models through RxInfer's local branch
`refactor/reactivemp-v7`. **Every variant gives posteriors and free energies bit-identical to A's**
on all five models and all runs (`compare_posteriors_abc.jl`, `post*/`), and passes the root suite
(`*_make_test.log`).

## Variants (`variantB.diff` … `variantF.diff`, against `851559a4`)

| | what it is |
|---|---|
| **A** | HEAD: `Message{D}`, `Marginal{D}` |
| **B** | `data::Any`, no type parameter; the few `Message{<:T}`/`Marginal{<:T}` dispatches moved onto the data |
| **C** | B + function barriers: `MessageMapping`/`MarginalMapping` unpack the data and call a one-method kernel (`run_message_rule`, `run_marginal_rule`) that builds `RuleArgs`, resolves and runs the rule; `product_kernel` around each message product; `average_energy_kernel` in the node free energy |
| **D** | A + lazy callback events (`@invoke_callback` at 12 sites): an event is built only when callbacks are set |
| **E** | C + lazy callback events |
| **F** | D + a `@noinline` constructor barrier (`new_message`, `new_marginal`) at the 4 sites that build a message or marginal from an `Any`-typed value |

## Results

Figures are the median over rounds of each round's minimum; variants alternate in fresh processes.
Full tables: `tables.md` (A vs B), `tables_abc.md` (A–E, batch 1), `tables_f.md` (A, C, E, F,
batch 2). A, re-measured in both batches, agreed within 3%.

**Relative to A:**

| | B | C | E | F |
|---|---|---|---|---|
| one rule call (`MessageMapping`, five rules) | 1.5–2.1× | 0.48–0.66× | 0.18–0.31× | **0.14–0.25×** |
| rule resolution alone | 25–38× | 5–12× | 5–11× | 1.0× |
| product of 10 Gaussian messages | 2.7× | 0.98× | 0.65× | 0.24× |
| engine graph loops (iid n = 1000; chain n = 300; 10 iterations) | 3.6×, 2.7× | 1.29×, 1.35× | 0.99×, 0.73× | **0.48×, 0.52×** |
| per iteration: iid VMP / Delta chain / HMM | 2.4 / 3.2 / 2.0× | 0.87 / 1.41 / 1.17× | 0.68 / 0.91 / 0.78× | **0.37 / 0.63 / 0.75×** |
| one `infer`: Kalman 1-d, 2-d; iid; Delta; HMM | 1.13–1.65× | 0.96–1.12× | 0.85–0.98× | **0.76–0.96×** |
| time to first inference | ±2% | ±2% | ±2% | ±2% |
| ReactiveMP specialisations (5 models) | 4 384 | 4 487 | 3 749 | 4 396 (A: 5 849) |

**Measured costs behind them** (`bench_micro.jl`, `ctor_cost.jl`, `sparams.jl`, `prof_*.jl`):
- building `Message{D}` or `Marginal{D}` from a value inferred `Any`: **510 ns** (Julia computes
  `D` at run time, `Core._compute_sparams`); untyped, 4.5 ns. It falls on every rule output;
- building a callback event with nobody listening, `result::Any`: **~270 ns** per rule call, in
  every variant;
- a one-method kernel called with `Any`-typed arguments: 23–24 ns and one allocation (4 ns with
  known types); unpacking two untyped messages' data: 29 ns and one allocation;
- in A, rule resolution is static (27 ns); with `data::Any` it is a runtime dispatch over every
  rule package's methods (0.7–1 µs), and every pairwise product a runtime `prod` (≈390 ns).

## Reading (the investigation's; nothing decided)

- **The type parameter is not what costs time.** Dropping it (B) makes everything slower: rule
  resolution and products become runtime dispatches, and the streams, which already carry
  abstract types and allocate nothing, gain almost nothing.
- **Function barriers do recover the untyped version** (C), most of B's loss, but not A's
  performance in whole graphs: each barrier's dispatch and allocation fall on every call and every
  product. How the per-call costs add up at graph level is inferred, not profiled.
- **Two costs in today's engine are the real target** (F): building parametric messages from
  `Any`-typed values, and building callback events nobody listens to. Fixing both while keeping
  `Message{D}` makes inference 4–63% faster, more with more iterations, at no compile-time cost.
- **Typed streams are not needed** for these gains; today's one dispatch where a stream hands a
  message to its mapping is 20–30 ns of an 800 ns call.
- **Next suspects, inferred:** the abstract `RuleSpec` and `Union{Nothing, Function}` body behind
  `execute_rule`, and the `Any` tuple a message product returns.

## Files

- `variant{B..F}.diff`, `apply_lazy_events.py` — the variants.
- `bench_micro.jl`, `bench_models.jl`; `driver.sh` (A vs B), `driver_abc.sh` (batch 1),
  `driver_f.sh` (batch 2); `mkenv.jl` — the benchmarks and their environments.
- `typeinfo.jl` → `typeinfo_{A,B,C}.txt` (`@code_warntype`, JET); `spec_count.jl` →
  `spec_*.tsv`; `sparams.jl`, `ctor_cost.jl`, `prof_*.jl` — the analyses.
- `micro_*.tsv`, `model_*.tsv` — raw results; `summarise*.py` → `tables*.md`.
- `post*/`, `compare_posteriors*.jl` — the posteriors and their equality check.
- `logs*/`, `*_make_test.log` — the runs.
- `env{A..F,JET}/` — the environments. They point at worktrees in the job's scratch directory,
  since removed: to rerun a variant, create a worktree of `851559a4`, apply its diff, and point
  the environment's `[sources]` at it.
