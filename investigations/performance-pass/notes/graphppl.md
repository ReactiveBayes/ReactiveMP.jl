# GraphPPL: model creation (P8, P8b, `created_by`)

Measured on GraphPPL 4.8.0 (clone at `4cc7c6c`), Julia 1.13.0, M-series Mac with 10 cores, **with
other benchmark jobs sharing the machine**: every comparison is interleaved (fresh process per run,
variant order rotated each round) and reported as the median over rounds of each round's minimum,
so the numbers are **indicative**. Baseline `P12` = ReactiveMP `d50fac9c` + P1 + P2, RxInfer
`4ca9eb88`, Rocket 1.10.0, GraphPPL 4.8.0. The GraphPPL variants change GraphPPL only.

Scripts: `bench_creation.jl` (creation by stage and size), `prof_creation.jl` (profile),
`jet_graphppl.jl`, `bench_ttfx_creation.jl` (time to first model, specialisations),
`bench_graphppl_storage.jl` (P8b sizing), `dump_graph_extras.jl` + `compare_posteriors.jl`
(correctness), `driver_graphppl.sh`, `summarise_graphppl.py`. Raw data: `results/graphppl_*.tsv`.

## Stages

`bench_creation.jl` creates the same conditioned generator three ways:

- `graph0` — GraphPPL's graph only (RxInfer's backend, no plugins);
- `graph` — plus the constraints, meta (`@algorithm`) and initialization plugins: all of GraphPPL;
- `full` — plus RxInfer's ReactiveMP plugin and the free-energy plugin (`Float64`), i.e. what
  `batch_inference` builds: ReactiveMP variables, nodes, activation, free-energy streams.

**GraphPPL is a small share of creation for ordinary models**: ssm1 at n = 10 000, `graph` is 82 ms of
a 0.97 s `full` (8%); the profile of `full` puts GraphPPL's graph building at 4.6% and the rest in
RxInfer's plugins: ReactiveMP factor-node construction 30%, factor-node activation 41%, the
free-energy plugin 17%, variable activation 4% — of which ReactiveMP's `input_names` alone is ~20%
(outside this fork's scope; reported to the parent). The exceptions are the two quadratic paths below.

## Findings (confirmed / refuted)

| audit claim | verdict | evidence |
|---|---|---|
| `apply_meta!` O(N_model × N_ctx) per factor meta entry | **confirmed, the largest GraphPPL cost** | `nl` with an `@algorithm` block: `graph` 3.4 ms → 25 ms → 278 ms → 2.48 s for n = 100, 300, 1000, 3000 (×9 per ×3: quadratic); at n = 3000 it is 84% of `full` (2.95 s) |
| `flattened_index` O(rows) → matrix constraints O(n²) | **confirmed** | `mat` (x[i, j] with `q(x, τ) = q(x)q(τ)`): constraints (`graph` − `graph0`) 7.5 ms → 209 ms → 19 s for n = 10³, 10⁴, 10⁵; profile: 10.9k of the 16.2k constraint samples under `flattened_index` → `isassigned` |
| `NodeLabel ==` compares `name::Any` first | confirmed, small | hashed `model[label]` 30 → 19 ns per node with the counter compared first |
| double hashed lookups through MetaGraphsNext | confirmed, small | `model[label]` 19–30 ns vs 2–3 ns from a `Vector{NodeData}`; ~0.7% of `full`, ~4% of `graph` |
| `UnorderedDictionary{Symbol,Any}` extra per node | **confirmed** | constructing it is 28% of ssm1's `graph0`-heavy build (the `graph` profile, 736 of 2630 samples); typed `hasextra`+`getextra` 29 ns → 7.3 ns per node with a vector of pairs |
| constraint materialisation: a BitMatrix per factor via `ones` → `BitMatrix`, `unique(eachcol)` | confirmed | ~25% of ssm1's `graph` stage (`BoundedBitSetTuple` 12%, `materialize_constraints!` 13%) |
| `created_by` closure per `~` multiplies compile time | **specialisations confirmed, time refuted** | GraphPPL specialisations after the first model: 2025 (K = 40 statements) and 4905 (K = 120) → 643 for both with a constant expression; first-model time unchanged within noise (graph 12–30 ms, full 1.5–1.9 s, dominated by ReactiveMP) |
| `is_factorized` recursion, exponential on deterministic chains | **refuted for real models** | named `:=` variables carry no links; an anonymous variable whose links are all data/constant becomes data; `all(...)` short-circuits on the first non-factorised parent. Worst case O(depth²) on deeply nested anonymous expressions; a 26-deep Fibonacci-shaped `:=` DAG builds in 70 µs |
| Context construction (7 dicts, `Ref{Any}`) | not significant | `ssmsub` (one Context per step): `graph0` 5.2 ms vs ssm1's 4.7 ms at n = 1000 |
| constants named through `Symbol(string(name, "_", counter))` | new, small | `to_symbol` 13% of ssm1's `graph` stage; not changed (constants are registered in the context under that name) |
| no PrecompileTools workload in GraphPPL (nor RxInfer) | new | first `full` creation 5–7 s after `using` for the benchmark models (compile) |
| superlinear creation at n ≥ 3·10⁴ | new, GC | bytes per node constant (9 kB `graph0`, 57 kB `full`) but GC share rises: `graph0` at n = 10⁵ 1.79 s of which 0.87 s GC; `full` 22 s of which 7 s GC |

JET (`report_opt`, `target_modules = (GraphPPL,)`, on `create` for ssm1's `graph` stage): 11 reports,
all in the postprocess path — `apply_meta!`/`getspecificsubmodelmeta`/`getgeneralsubmodelmeta` on
`Any` meta objects, `apply_constraints!` on `Any` inline constraints, `all(is_factorized, ::Any)`,
`Tuple(::Union{Vector{Any},Vector{Vector{Int}}})` in materialisation, `Ref{Any}` returnval, and one
`(::Any == ::Any)` (the `NodeLabel` name comparison). All are intended type erasure over
user-supplied specifications except the last (fixed by C2). JET stops at the model body's dynamic
call, so it does not see `make_node!`.

## P8: the changes (one commit each, `diffs/P8-GraphPPL-C*.diff`, all in `diffs/P8-GraphPPL.diff`)

| | change | breaking? |
|---|---|---|
| C1 | `apply_meta!` walks the context's own factor nodes (`values(factor_nodes(context))`) filtered by the descriptor, not every node of the model filtered by membership in the context | no |
| C2 | `NodeLabel ==` compares `global_counter` first | no |
| C3 | a factor's constraint bitset is `trues(n, n)` directly; a node with no factorisation (all-true bitset) gets its single cluster `(collect(1:n),)` without `unique`/`is_valid_partition` | no (same stored value and type) |
| C4 | `NodeData.extra` is a `NodeExtras` (a `Vector{Pair{Symbol,Any}}`, linear scan) instead of an `UnorderedDictionary{Symbol,Any}`; same accessors, same `IndexError`s | soft: `getextra(node)` returns a `NodeExtras` (supports `haskey`/`getindex`/`get`/`insert!`/`keys`/`values`/`pairs`/iteration), not a Dictionaries.jl dictionary |
| C5 | `created_by = $(QuoteNode(expr))` instead of `created_by = () -> :(expr)` in the macro | soft: the `created_by` option is an `Expr`, not a closure (`CreatedBy` shows both; a user-written closure still works); macro-expansion tests updated (23 expectations) |
| C6 | `flattened_index` from prefix sums cached per array, bound with `task_local_storage` for the duration of `apply_constraints!` (the model is complete then) | no |

No private compiler API and no `@generated` function is used.

## Results

Two interleaved rounds over the ladder `P12` (baseline) → `Ga` (C1) → `Gb` (C1–C3) → `Gc` (C1–C4)
→ `G1` (C1–C6), median of per-round minima (`results/graphppl_compare.tsv`, full table in
`results/graphppl_compare_table.md`). ssm1 was re-measured P12 vs G1 over four alternating rounds on a
quieter machine (`results/graphppl_compare_ssm1.tsv`); its ladder rows were contended and are not used.

| model | n | stage | P12 | G1 (C1–C6) | G1/P12 | which change |
|---|---|---|---|---|---|---|
| nl (`@algorithm` block) | 300 | full | 116 ms | 47 ms | 0.41 | C1 |
| nl | 1000 | full | 604 ms | 156 ms | 0.26 | C1 |
| nl | 3000 | graph | 3.99 s | 26.9 ms | **0.007** | C1 |
| nl | 3000 | full | 4.51 s | 467 ms | **0.10** | C1 |
| mat (x[i,j], constraint) | 10000 | graph | 274 ms | 146 ms | 0.53 | C6 |
| mat | 40000 | graph | 3.07 s | 808 ms | 0.26 | C6 |
| mat | 40000 | full | 6.88 s | 4.78 s | 0.69 | C6 |
| hmm (structured) | 10000 | graph | 82.7 ms | 72.3 ms | 0.87 | C3 |
| hmm | 10000 | full | 1.14 s | 1.09 s | 0.96 | |
| iid (mean-field) | 10000 | graph | 22.3 ms | 19.4 ms | 0.87 | C3, C4 |
| iid | 10000 | full | 420 ms | 402 ms | 0.96 | |
| ssm1 | 10000 | graph | 70 ms | 66.8 ms | 0.95 | |
| ssm1 | 10000 | full | 966 ms | 938 ms | 0.97 | |
| ssmsub (submodels) | 10000 | graph | 113 ms | 102 ms | 0.90 | |
| ssmsub | 10000 | full | 1.21 s | 1.01 s | 0.83 (noisy) | |

Scaling of the baseline (`results/graphppl_scaling.tsv`, P12 only, n = 100 → 10⁵): ssm1 `full` 9.4 ms
→ 92 ms → 0.97 s → 15.3 s; `graph` 0.72 → 6.5 → 82 → 906 ms. Linear up to 10⁴ and superlinear
beyond it, from GC (bytes per node constant). `mat` `graph` reaches 20 s at 10⁵ (C6's target), and `nl`
`graph` 2.5–4 s at 3000 (C1's target).

Time to first model (`bench_ttfx_creation.jl`, K = 40 statements, rounds 4–5): first `graph` creation
11–12 ms for P12/Ga/Gb, **27–38 ms for Gc/G1** (+16–20 ms, C4's `NodeExtras` methods compiled on first
use, where Dictionaries' come precompiled; GraphPPL has no precompile workload); first `full` 1.5–1.6 s
for all (ReactiveMP's compilation). GraphPPL specialisations: 2044 → 624 with C5.


## Correctness

- GraphPPL's suite on C1–C6: **76 800 passed, 0 failed, 0 errored, 1 broken** (the broken one is
  pre-existing); C5 needs its 23 macro-expansion expectations updated (included in the diff).
- Posteriors of `bench_models.jl`'s ssm1, iid, hmm, nl, gmm: **bit-identical** P12 vs G1
  (`compare_posteriors.jl`).
- Per factor node, the resolved factorisation clusters and meta (algorithm) after the `graph`
  stage: **identical** P12 vs G1 for `nl` (300 `@algorithm` nodes), `mat` (the matrix constraint),
  `hmm` (structured), `iid`, `ssmsub` (`dump_graph_extras.jl`).

## P8b (breaking): storage redesign — spike only

Estimated from `bench_graphppl_storage.jl` on a built ssm1 model (n = 10 000) and the profiles:

- a `Vector{NodeData}` indexed by the node counter instead of MetaGraphsNext's two hashed lookups:
  19 ns → ~2.5 ns per `model[label]` (after C2); hashed graph lookups are ~0.7% of `full` creation and
  ~4% of `graph` → **≲ 1–2% of creation**;
- typed per-plugin extras (a layout from the `PluginsCollection` type) instead of `NodeExtras`: C4 already
  takes typed access from 29 to 7 ns; the remaining gain is a few ns per access, **< 1%**;
- dropping `edge_data` (the `neighbors` vector already holds the `EdgeLabel`s): one Dict insert
  per edge in `add_edge!` (`add_edge!` is ~7% of ssm1's `graph` stage) → **≲ 0.5% of creation**.

What it would touch: RxInfer does **not** read `model.graph`, MetaGraphsNext or `.extra` (checked:
no occurrence in `RxInfer.jl/src` or `ext`); it uses `labels`, `factor_nodes`/`variable_nodes`
callbacks, `model[label]`, `getproperties`, `neighbors` (as `(label, EdgeLabel, NodeData)`),
`getextra`/`setextra!`/`hasextra` with `NodeDataExtraKey`, `getcontext`, `VarDict`,
`ResizableArray`, `degree`. A redesign behind those accessors breaks GraphPPL's own engine (25
`model.graph`/MetaGraphsNext sites in `graph_engine.jl`, the constraints plugin), its plotting and
GraphViz extensions, `savegraph`/`loadgraph` (the JLD2 format) and `prune!`, not RxInfer.
**Not worth a breaking release for performance.**

## Recommendation

- **Release (non-breaking, GraphPPL patch/minor): C1, C6, C2, C3.** C1 removes a quadratic that makes
  any model with an `@algorithm`/`@meta` block on per-time-step nodes unusable beyond a few thousand
  nodes (10× faster creation at n = 3000, and growing with n); C6 removes the quadratic of constraints
  on matrix variables (graph stage 3.07 s → 0.81 s at n = 4·10⁴; the baseline's 20 s at 10⁵ was not
  re-measured with C6); C2
  and C3 are one-liners with a 5–13% gain on the graph stage. All four keep the stored values and
  pass GraphPPL's suite unchanged.
- **C4 (`NodeExtras`)**: 4× faster typed extras access, 3–5% on the graph stage, but +16–20 ms to the
  first model unless GraphPPL gets a PrecompileTools workload, and a soft API change (`getextra(node)`
  returns another container). Take it together with a workload, in a minor release.
- **C5 (`created_by` constant)**: 70–87% fewer GraphPPL specialisations, no measurable time gain;
  soft-breaking for anyone calling the stored closure. Optional, minor release.
- **Add a PrecompileTools workload** to GraphPPL (and RxInfer): not measured here, but first creation
  costs 5–7 s after `using` for the benchmark models, all compilation.
- **P8b (storage redesign): no.** ≲ 1–2% of creation for a breaking change; RxInfer would not break, but
  GraphPPL's engine, extensions and file format would.
- **Where creation time actually is**: for ordinary models GraphPPL is ~5–8% of `full` creation; the
  rest is RxInfer's ReactiveMP plugin (node construction 30%, activation 41%, free energy 17% of ssm1 at
  10⁴), and GC beyond 3·10⁴ nodes. The creation lever is on the ReactiveMP/RxInfer side (`input_names`
  ~20%), not in GraphPPL.

