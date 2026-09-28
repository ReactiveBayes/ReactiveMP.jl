# Rocket.jl: what was tried, measured and kept

These are the Rocket prototypes for the performance pass. The base is Rocket master, which is
1.10.0 plus CI-only commits. The engine runs at P1+P2 (lazy callbacks and the constructor barrier),
so engine overhead doesn't hide Rocket's own share. The baseline variant `P12` is P1+P2 on stock
Rocket. The variants below are `R1` (P7), `R9` (P7 + P9 with the emission guard) and `R10` (as R9,
with the guard written without `try`).

Measurement: variants alternated in fresh processes; three micro rounds and two model rounds; the
figure is the median of each round's minimum. **All numbers are indicative**: other forks were
benchmarking on the same machine. Raw data:

- `results/rocket_micro.tsv`, `results/rocket_models.tsv`: P12 against R1 against R9.
- `results/rocket_micro_guard.tsv`, `results/rocket_models_guard.tsv`: R1 against R10.
- `results/rocket_stack.tsv`: stack depth.
- `results/rocket_fastpaths.tsv`: the fast-path alternatives.

Scripts: `bench_micro.jl`, `bench_models.jl`, `stack_depth.jl`, `stack_frames.jl`,
`rocket_fastpaths.jl`.

## Correctness

- **Rocket's suite:**
  - `P7-Rocket.diff`: 12295/12295 pass.
  - P7 + `P9-Rocket.diff` (final, without the emission guard): 12295/12295 pass.
  - P7 + P9 with the emission guard: 12295/12295 pass.
- **ReactiveMP's `make test`:** passes on P1+P2 with P7 + P9, including the emission guard. The
  worktree's `[sources]` pointed Rocket at the patched clone, and the log shows
  ``Rocket v1.10.0 `../Rocket.jl` ``.
- **Posteriors:** bit-identical to P12 for R1 and for R9 on every run of iid, iid@10000, ssm1,
  ssm1@3000 and hmm.

## P7: pending counters and mutable wrappers (`diffs/P7-Rocket.diff`), non-breaking

What changed:

- **Counters instead of rescans:**
  - `collectLatest` keeps `ncompleted` and `nvalues` counts next to its BitArrays, so the check on
    each event is O(1) instead of `all(vstatus) && !all(cstatus)` over every source.
  - `GenericUpdatesStatus`, which `combineLatest` uses above 8 sources, does the same; it is now a
    mutable struct with `const` BitArrays.
  - Every write goes through `set_*status!`/`fill_*status!`. ReactiveMP only ever calls
    `Rocket.fill_vstatus!`, which keeps the counts right.
- **Mutable wrappers:** `CollectLatestObservableWrapper`, `CombineLatestActorWrapper` and
  `CombineLatestUpdatesActorWrapper` are now plain `mutable struct`s.
  - Why: an emission into an abstractly typed actor, `next!(wrapper.actor, snapshot)` with a tuple
    whose type is only known at run time, is a dynamic call, and every argument gets boxed. That
    includes the immutable actor chain, which carried the whole wrapper struct inline.
  - Measured with `Profile.Allocs`: the per-event `MapActor` box on the free-energy path dropped
    from 128 to 48 bytes (144 with the counters before this change).
  - I used plain `mutable struct` rather than `const` fields, since Rocket still declares
    `julia = "1"`.

Results (ratios are R1/P12):

| case | P12 | R1 | ratio |
|---|---|---|---|
| collectLatest N=100 round | 2.25 µs | 1.96 µs | 0.87 |
| collectLatest N=1000 round | 24.8 µs | 20.4 µs | 0.82 |
| collectLatest N=10000 round | 514 µs | 209 µs | **0.41** |
| combineLatest PushNew k=20 round | 447 ns | 400 ns | 0.89 |
| engine graph iid n=1000, per iteration | 622 µs | 573 µs | 0.92 |
| engine graph iid n=10000, per iteration | 12.4 ms | 12.8 ms | 1.04 (noise) |
| engine graph allocations per iteration (iid n=1000) | 1.10 MB | 1.02 MB | **−7%** |
| engine graph build+activate allocations (iid n=10000) | 191.6 MB | 175.1 MB | **−9%** |
| RxInfer iid@10000, one `infer` with 20 iterations | 1.10 s | 0.95 s | 0.86 |
| RxInfer iid@10000, one `infer` with 10 iterations | 0.746 s | 0.745 s | 1.00 |
| RxInfer iid, hmm, ssm1, ssm1@3000 | | | 0.93–1.01 (neutral) |

The O(N²/64) rescan only shows at N ≳ 10⁴ sources per `collectLatest`: the free energy over every
node, or a variable's marginal over every message. Below that P7 is neutral.

Not done, and why:

- **`PendingScheduler` typed storage and O(1) deregistration:** its listeners have different types
  by nature, so the dynamic `release!` cannot go away. Deregistration is O(n) only at teardown, and
  the engine uses it only for RxInfer's streaming tick and the postprocessor. Not worth it.
- **`LazyObservable` swap-remove:** `filter!` runs only when unsubscribing before the stream was
  set. Rare, not worth it.
- **`map` `::R` assertion:** no gain for ReactiveMP, whose `R` is `Marginal` or
  `AbstractMessage`, both abstract, so downstream dispatch stays dynamic. It is mildly breaking, since
  it rejects values not of type `R`. Not done.
- **The 16 or 32 bytes per event in `collectLatest` over `Float64` or `CountingReal`:** this is not
  `collectLatest`. `Subject` stores `actor::Any`, and a dynamic call with an isbits payload boxes the
  payload. This hits the free energy's `CountingReal` streams. See P7b.

## `snapshot` on `MStorageN` (the coordinator's question)

- **The cost itself:** in isolation, `snapshot(::MStorage3{Marginal, Marginal, Marginal})` costs
  36 ns and one 32-byte allocation. It builds the tuple of the values' concrete types at run time,
  because the fields are declared abstract.
- **Where the profile's samples come from:** most of the ~1000 samples in the n = 10000 graph
  profile are GC. With `C = false`, GC pauses are charged to the Julia frame that allocated.
- **Why it can't simply go:** that runtime tuple is also the function barrier that gives the
  downstream mapping concrete types, which is what makes rule resolution static. Removing it would
  move the dispatch, not save it.
- **What would reduce it:** fewer emissions, for example the free energy's per-node streams
  combined once per iteration rather than on every marginal update. That is an engine design
  question.

## Task 3: the hand-written fast paths (`rocket_fastpaths.jl`)

Setup: `combineLatest` with PushNew over k sources with abstract element types, as the engine
uses it. Two rounds, fresh process per alternative. Columns are the steady round and the first
round for never-seen source types, compile included.

| k | current (UInt8 + MStorage) | BitArray+counters + MStorage | UInt8 + `Vector{Any}` storage | both simple |
|---|---|---|---|---|
| 2 | 101–104 ns / 0.074–0.083 s | 104–113 ns / 0.087–0.102 s | 104–105 ns / 0.064 s | 105–106 ns / 0.066 s |
| 5 | 262–282 ns / 0.16–0.17 s | 250–264 ns / 0.16–0.22 s | 299–306 ns / 0.13 s | 285–300 ns / 0.14 s |
| 8 | 746–808 ns / 0.26–0.28 s | 780–802 ns / 0.29–0.31 s | 802–824 ns / 0.21 s | 831–838 ns / 0.21–0.23 s |
| 16 | 2.06–2.14 µs / 0.60–0.65 s | 2.13–2.18 µs / 0.62–0.64 s | 1.98–2.03 µs / 0.40–0.43 s | 1.79–1.82 µs / 0.42–0.43 s |

Zero allocations everywhere.

- **Runtime:** a wash. With abstract element types, `MStorageN`'s typed fields buy nothing, since
  the snapshot tuple is built at run time either way. `UInt8UpdatesStatus` and BitArray+counters
  are equal.
- **Compile time:** `Vector{Any}` storage compiles 20–35% faster for new source tuples, because
  there are no `MStorageN{V1…Vn}` specialisations.
- **Recommendation:**
  - Keep `UInt8UpdatesStatus`: harmless and small.
  - Consider sending sources with *abstract* element types to `Vector{Any}` storage, and keep
    `MStorageN` for concrete ones, where it does give a type-stable snapshot. Non-breaking.
  - Low priority: measured time to first inference is unaffected at the model level, and the
    `combineLatest` compile share is small.

## P9: subscription stack depth (`diffs/P9-Rocket.diff`), non-breaking, adds a public helper

Measured on an 8 MiB task with `stack_depth.jl` and `stack_frames.jl`:

- **Stock:** the engine chain graph and RxInfer's ssm1 without `limit_stack_depth` both overflow
  at about **1030 links**, roughly **8 KB of stack per link**.
- **Frames per link:** subscription costs **22 native frames**, emission plus materialisation 18.
- **One link's cycle:** four `LazyObservable` crossings (`stream::Any`, so a dynamic call each),
  plus the inlined proxy → ref_count → connect → multicast layers and the unrolled `combineLatest`
  fill.
- **Unsubscription:** teardown recurses along the chain the same way.

The change: `Rocket.stackguarded(f)` and `set_stack_guard_limit!` (default 256), in
`helpers/stackguard.jl`.

- It counts nested guarded calls. Past the limit it runs `f` on a fresh task and waits, with the
  result and exception type unchanged, the way RxInfer's `LimitStackScheduler` does.
- It is applied only where a `LazyObservable` subscribes or unsubscribes, which happens at
  activation and teardown.

| variant | longest chain (8 MiB) | cost |
|---|---|---|
| stock (P12) | 1024–1040 | — |
| guard on lazy subscribe only | 2976 | none measurable (activation only) |
| **+ lazy unsubscribe (P9, final)** | **3264** | none measurable; RxInfer ssm1 n=1000 0.096–0.100 s (P12 0.100) |
| + `Subject.on_next!` guard | ≥ 8192 (cap), ≥ 16384 with the next row | **+11–25% per iteration** on engine graphs, `combineLatest` round up to 1.7× (fan-out k=1 16 → 29 ns with `try`; 16 → 17 ns without, graphs still +11–25%) |
| + ReactiveMP `as_message` guard | ≥ 16384 | not measured separately |

**Recommendation:**

- Adopt P9. It roughly **triples** the longest chain RxInfer handles without `limit_stack_depth`,
  at no steady-state cost.
- Don't guard emissions in `Subject`: every `next!` pays for it. Keep `limit_stack_depth` for
  deeper models, at 7–10% on ssm1 n=1000 (0.100 → 0.106–0.114 s).
- A cheaper route to deeper chains is fewer frames per link. The four dynamic `LazyObservable`
  crossings and the proxy layers per link are the target. That is an engine and Rocket co-design
  (typed or flattened message streams), not attempted here.
- A Rocket unit test for `stackguarded` would still need adding; the diff has none.

## P7b: typed listener storage (not built)

Micro-benchmark (`p7b.jl`, scratch), ns per fan-out round:

| k listeners | today (`actor::Any`, dynamic) | FunctionWrappers | homogeneous typed `Vector{A}` |
|---|---|---|---|
| 1 | 3.6 | 40.0 | 2.7 |
| 10 | 18.0 | 355 | 5.7 |
| 100 | 165 | 4381 | 33.5 |

- **FunctionWrappers:** 10–25× *slower*. With an abstract argument type the wrapper boxes and
  converts on every call, and it relies on `llvmcall`/`cfunction` internals. Rejected.
- **An actor type parameter:** helps only when all listeners share one type. Engine subjects nearly
  always have one listener (multicast plus ref_count), or listeners of different types. It would be
  breaking, since ReactiveMP spells out `Subject{M, AsapScheduler, AsapScheduler}`, and it would
  multiply specialisations.
- **Recommendation:** leave `actor::Any` as it is. It is cheap (~1.6 ns of dispatch per listener)
  and it is the compile-time barrier.

## For the parent (engine side, out of Rocket's scope)

- **`RuleSpec` boxing:** `Profile.Allocs` shows a `RuleSpec` box of **288 bytes per rule call**
  (`execute_rule@rulespec.jl:254`, P1+P2). The 19-field immutable spec is copied into a box for a
  dynamic call. It is the largest allocation site per iteration in the iid graph: 288 KB of about
  1.1 MB for n = 1000.
- **Isbits payloads boxed at `Subject`s:** every `CountingReal` pushed through a `Subject` is boxed,
  16–32 bytes per node per iteration on the free-energy path.
