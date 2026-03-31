# Refactor Addon System to Rule Annotations System

## Context

The current "addon" system in ReactiveMP.jl is overly complex: it uses typed tuples of `AbstractAddon` subtypes, requires rules to return `(result, addons)` tuples, and has a nonsensical `multiply_addons` product concept. The goal is to replace it with a simpler "rule annotations" system where:
- Messages carry an optional `AnnotationDict` (lazy-allocated, defined in `src/annotations.jl`)
- Rules receive a mutable `AnnotationDict` as an argument and annotate it directly (no return tuple)
- Annotation merging during message products is handled by `AbstractAnnotations` subtypes (e.g. `LogScaleAnnotations`)
- `AddonDebug` is removed (use callbacks instead)

## Key Design Decisions
- **Lazy allocation**: `AnnotationDict` only allocates its inner `Dict{Symbol, Any}` on the first `annotate!` call
- **Clean break**: No deprecation shims for old names
- **Field naming**: `annotations` (not `addons`, not `rule_context`, not `rule_metadata`)

---

## Phase 1: New Core Infrastructure (new files, no old code changes)

### Step 1.1: `src/annotations.jl` — `AnnotationDict` [DONE]

`AnnotationDict` provides lazy-allocated annotation storage. Already implemented:
- `AnnotationDict()` — creates an empty instance
- `has_annotation(ann, key)` — check for key presence
- `annotate!(ann, key, value)` — write an annotation
- `get_annotation(ann, key)` — read an annotation (throws `KeyError` if absent)
- `get_annotation(ann, ::Type{T}, key)` — typed read

### Step 1.2: Extend `src/annotations.jl` with processor infrastructure [DONE]
- `abstract type AbstractAnnotations end`
- `AnnotationDict()` is always constructed (allocation is lazy inside `AnnotationDict` itself)
- `post_product_annotations!(processors, left_ann, right_ann, new_dist, left_dist, right_dist)` -> always returns an `AnnotationDict`; dispatches on `::Missing` dists, then iterates processors
- `post_product_annotations!(p::AbstractAnnotations, merged, left_ann, right_ann, ...)` -> per-processor callback; no default; unimplemented processors throw `MethodError`
- `post_rule_annotations!(p::AbstractAnnotations, ann, mapping, messages, marginals, result)` -> no default; unimplemented processors throw `MethodError`

### Step 1.3: Create `src/annotations/logscale.jl`
- `struct LogScaleAnnotations <: AbstractAnnotations end`
- `post_product_annotations!` for LogScaleAnnotations: reads `:logscale` from both annotation dicts, adds `compute_logscale(new_dist, ...)`
- `post_rule_annotations!` for LogScaleAnnotations: if `:logscale` not set and all inputs are PointMass, set to 0; otherwise error
- New `@logscale` macro: expands to `annotate!(_annotations, :logscale, value)`
- `getlogscale(ann::AnnotationDict)` reads `:logscale` key

### Step 1.4: Create `src/annotations/rule_input_arguments.jl`
- `struct PropagateRuleInputArguments <: AbstractAnnotations end`
- `RuleInputArgumentsRecord` struct (replaces `AddonMemoryMessageMapping`)
- `RuleInputArgumentsProd` struct (replaces `AddonMemoryProd`)
- `post_rule_annotations!`: stores `RuleInputArgumentsRecord` under `:rule_input_arguments`
- `post_product_annotations!`: merges records from left/right annotation dicts
- `getmemory(ann::AnnotationDict)` reads `:rule_input_arguments` key

### Files created:
- `src/annotations.jl` ✓
- `src/annotations/logscale.jl`
- `src/annotations/rule_input_arguments.jl`

---

## Phase 2: Refactor Message & Marginal Structs

### Step 2.1: `src/message.jl` — Message struct
- `Message{D, A}` -> `Message{D}`, remove type param `A`
- Field `addons::A` -> `annotations::AnnotationDict` (always present; zero-cost when unused due to lazy allocation)
- `getaddons` -> `getannotations`
- Update all `Message(data, clamped, initial, addons)` constructor calls in this file

### Step 2.2: `src/marginal.jl` — Marginal struct
- Same transformation: `Marginal{D, A}` -> `Marginal{D}`
- Field `addons::A` -> `annotations::AnnotationDict` (always present)
- `getaddons` -> `getannotations`

### Step 2.3: `src/ReactiveMP.jl` — Bridge functions (lines 47-59)
- `as_marginal`/`as_message`: use `getannotations` instead of `getaddons`
- `getlogscale(msg::Message)` -> `getlogscale(getannotations(msg))`
- `getmemory(msg::Message)` -> `getmemory(getannotations(msg))`
- Remove `getmemoryaddon` helpers

---

## Phase 3: Refactor Rule System

### Step 3.1: `src/rule.jl` — `rule_function_expression` (line 343)
- Rename `addonsvar` to `annotationsvar`
- The generated `rule()` function receives an `AnnotationDict` (or `nothing`) as 9th arg
- Expose `_annotations` variable (replaces `local getaddons = () -> $addonsvar`)

### Step 3.2: `src/rule.jl` — `@rule` macro (line 462)
- Replace `local _addons = getaddons()` with `local _annotations = $annotationsvar`
- Rule body no longer wrapped to return `(_message, _addons)` — just return `_message`
- The `_annotations` variable is in scope for `@logscale` and similar macros to write to

### Step 3.3: `src/rule.jl` — `@call_rule` macro (line 593)
- Replace `addons = ...` keyword with `annotations = ...`; defaults to a fresh `AnnotationDict()`
- Remove `[return_addons = true]` option entirely — caller reads back from the dict after the call:
  ```julia
  # old
  dist, addons = @call_rule [return_addons = true] Bernoulli(:out, Marginalisation) (m_p = Beta(1, 2), addons = (AddonLogScale(),))

  # new
  ann = AnnotationDict()
  dist = @call_rule Bernoulli(:out, Marginalisation) (m_p = Beta(1, 2), annotations = ann)
  logscale = get_annotation(ann, :logscale)
  ```
- `@call_marginalrule` does not use addons — no changes needed there
- Update `test/rule_tests.jl` lines 1295–1315 (the `return_addons` testset) to use the new pattern

### Step 3.4: Remove `@invokeaddon` from `src/addons.jl`
- The `@logscale` macro now directly calls `annotate!`, no need for `@invokeaddon`

---

## Phase 4: Refactor MessageMapping

### Step 4.1: `src/message.jl` — MessageMapping struct (line 499)
- Rename field `addons::X` to `annotations::X` (holds the processors tuple or nothing)

### Step 4.2: `src/message.jl` — MessageMapping callable (line 614)
- New flow:
  1. `ann = AnnotationDict()` — always created (allocation is lazy inside `AnnotationDict`)
  2. Call `rule(...)` passing `ann` as 9th arg. Rule returns only the distribution.
  3. Loop over `mapping.annotations` processors, call `post_rule_annotations!(p, ann, mapping, messages, marginals, result)` for each
  4. Construct `Message(result, ..., ann)`
- Remove `message_mapping_addons` and `message_mapping_addon` functions

### Step 4.3: MessageMapping constructors (lines 564-612)
- Rename `addons` param to `annotations` in both constructors

---

## Phase 5: Refactor Message Products (Annotation Merging)

### Step 5.1: `src/message.jl` — `MessageProductContext` (line 133)
- Add field `annotations::X = nothing` to hold processors tuple

### Step 5.2: `src/message.jl` — `compute_product_of_two_messages` (line 160)
- Replace lines 203-211:
  ```julia
  left_ann = getannotations(left)
  right_ann = getannotations(right)
  new_ann = post_product_annotations!(context.annotations, left_ann, right_ann, new_dist, left_dist, right_dist)
  result = Message(new_dist, is_prod_clamped, is_prod_initial, new_ann)
  ```
- Remove all `multiply_addons` calls

### Step 5.3: `src/variables/random.jl` — `RandomVariableActivationOptions`
- Add `annotations` field to `RandomVariableActivationOptions` to hold processors tuple
- Pass it into `MessageProductContext` construction
- **Design decision**: annotation processors are configured explicitly in both `FactorNodeActivationOptions`
  (for rule-time annotation) and `MessageProductContext` (for product-time merging). RxInfer sets both.
  There is no implicit inference of processors from neighbouring nodes.

### Step 5.4: Handle `Missing` distribution in `post_product_annotations!`
- The current `multiply_addons` uses `left_dist::Missing` / `right_dist::Missing` as a sentinel
  for a missing observation (unobserved variable on that edge)
- When one side's distribution is `missing`, the convention is to pass the other side's annotations
  through unchanged (there is nothing to merge from the missing side)
- Add a top-level fallback in `post_product_annotations!` (before calling processors) that handles the
  `left_dist isa Missing` / `right_dist isa Missing` cases, so individual processor implementations
  do not need to repeat this logic

---

## Phase 6: Refactor Node Infrastructure

### Step 6.1: `src/nodes/nodes.jl` — `FactorNodeActivationOptions` (line 277)
- Rename field `addons::A` to `annotations::A`
- `getaddons(options)` -> `getannotations(options)`

### Step 6.2: `src/nodes/dependencies.jl` — `activate!` (line 30)
- `addons = getaddons(options)` -> `annotations = getannotations(options)`
- Pass `annotations` to `MessageMapping` constructor

### Step 6.3: Delta node files
- `src/nodes/predefined/delta/delta.jl` — rename `addons` param in `rule()` redirect (line 82)
- `src/nodes/predefined/delta/layouts/default.jl` — rename all `addons` references
- `src/nodes/predefined/delta/layouts/cvi.jl` — rename all `addons` references

### Step 6.4: `src/rules/fallbacks.jl`
- All `rulefallback_nodefunction` methods: rename `addons` param to `annotations`
- Line 120: `return FallbackNodeFunctionUnnormalizedLogPdf(fn), addons` -> just return `FallbackNodeFunctionUnnormalizedLogPdf(fn)` (no tuple)

---

## Phase 7: Update Callbacks

### Step 7.1: `src/callbacks.jl`
- `AfterMessageRuleCallEvent`: rename `addons::A` field to `annotations::A`
- `AfterProductOfTwoMessagesEvent`: rename `addons::A` field to `annotations::A`
- Update docstrings

---

## Phase 8: Clean Up Old Addon Files

### Step 8.1: Delete old files
- Delete `src/addons.jl`
- Delete `src/addons/debug.jl`
- Delete `src/addons/logscale.jl`
- Delete `src/addons/memory.jl`

### Step 8.2: Update `src/ReactiveMP.jl` includes
- Remove old addon includes
- Add annotation subtype includes (after `src/annotations.jl`):
  ```julia
  include("annotations/logscale.jl")
  include("annotations/rule_input_arguments.jl")
  ```
- Update exports: remove `AddonLogScale`, `AddonMemory`, `AddonDebug`, `getmemoryaddon`, `multiply_addons`
- Add exports: `AbstractAnnotations`, `LogScaleAnnotations`, `PropagateRuleInputArguments`, `getannotations`

---

## Phase 9: Rule Files Using @logscale (17 files — no changes needed)

The `@logscale` macro is updated in Phase 1.3 to expand to `annotate!(_annotations, :logscale, value)`. Since `_annotations` is now exposed by the `@rule` macro (Phase 3.2), all existing `@logscale 0` calls in rule files will work without modification.

Verify compilation of these files:
- `src/rules/bernoulli/{out,p}.jl`
- `src/rules/categorical/{out,p}.jl`
- `src/rules/mixture/{out,inputs,switch}.jl`
- `src/rules/multiplication/{out,in}.jl`
- `src/rules/normal_mean_variance/{out,mean}.jl`
- `src/rules/normal_mean_precision/{out,mean}.jl`
- `src/rules/mv_normal_mean_precision/{out,mean}.jl`
- `src/rules/mv_normal_mean_covariance/{out,mean}.jl`

---

## Phase 10: Update Tests

### Step 10.1: Rewrite addon tests -> annotation tests
- `test/addons_tests.jl` -> `test/rule_annotations_tests.jl`
  - Test `annotate!`, `get_annotation`, `has_annotation` with `AnnotationDict`
  - Test `post_product_annotations!` with processors
  - Test `LogScaleAnnotations` merge logic
- `test/addons/logscale_tests.jl` -> `test/rule_annotations/logscale_tests.jl`
- `test/addons/memory_tests.jl` -> `test/rule_annotations/rule_input_arguments_tests.jl`
- Delete `test/addons/debug_tests.jl`
- `test/annotations_tests.jl` ✓

### Step 10.2: Update existing tests
- `test/message_tests.jl` — update `Message(...)` constructor calls (4th arg = nothing), `getaddons` -> `getannotations`
- `test/marginal_tests.jl` — same
- `test/rule_tests.jl` — rules no longer return tuples, `addons=` -> `annotations=`
- `test/variables/random_tests.jl` — update `MessageProductContext()` if signature changes

---

## Phase 11: Update Documentation

- Rewrite `docs/src/custom/custom-addons.md` -> `docs/src/custom/custom-rule-annotations.md`
- Document: `AbstractAnnotations`, `@logscale`, `annotate!`, `post_product_annotations!`, `post_rule_annotations!`
- Update any docstrings in source files

---

## Verification

1. Run `julia --project -e 'using ReactiveMP'` to verify compilation
2. Run full test suite: `julia --project -e 'using Pkg; Pkg.test()'`
3. Specifically verify:
   - `@logscale` works in all 17 rule files
   - Message products with `LogScaleAnnotations` correctly merge log scales
   - `PropagateRuleInputArguments` captures rule inputs
   - `AnnotationDict` doesn't allocate its inner dict when no annotations are written
