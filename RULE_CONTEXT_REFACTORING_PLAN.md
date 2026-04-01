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

### Step 1.3: Create `src/annotations/logscale.jl` [DONE]
- `struct LogScaleAnnotations <: AbstractAnnotations end`
- `post_product_annotations!` for LogScaleAnnotations: reads `:logscale` from both annotation dicts, adds `compute_logscale(new_dist, ...)`
- `post_rule_annotations!` for LogScaleAnnotations: if `:logscale` not set and all inputs are PointMass, set to 0; otherwise error
- New `@logscale` macro: expands to `annotate!(getannotations(), :logscale, value)`
- `getlogscale(ann::AnnotationDict)` reads `:logscale` key

### Step 1.4: Create `src/annotations/input_arguments.jl` [DONE]
- `struct InputArgumentsAnnotations <: AbstractAnnotations end` (replaces `AddonMemory` from `src/addons/memory.jl`)
- `RuleInputArgumentsRecord` struct (replaces `AddonMemoryMessageMapping`)
- `ProductInputArgumentsRecord` struct (replaces `AddonMemoryProd`)
- `post_rule_annotations!`: stores `RuleInputArgumentsRecord` under `:rule_input_arguments`
- `post_product_annotations!`: merges records from left/right annotation dicts
- `get_rule_input_arguments(ann::AnnotationDict)` reads `:rule_input_arguments` key
- Old name `getmemory` is noted for CHANGELOG / breaking-changes documentation

### Files created:
- `src/annotations.jl` ✓
- `src/annotations/logscale.jl` ✓
- `src/annotations/input_arguments.jl` ✓

---

## Phase 2: Refactor Message & Marginal Structs + MessageMapping + Message Products

### Step 2.1: `src/message.jl` — Message struct [DONE]
- `Message{D, A}` -> `Message{D}`, remove type param `A`
- Field `addons::A` -> `annotations::AnnotationDict` (always present; zero-cost when unused due to lazy allocation)
- Added 3-arg constructor `Message(data, is_clamped, is_initial)` that defaults to fresh `AnnotationDict()`
- `getaddons` -> `getannotations`
- `show` only prints annotations when non-empty via `isempty(ann)`
- Equality no longer compares annotations (metadata, not semantic content)
- Updated docstring and jldoctest example
- Added `Base.isempty(::AnnotationDict)` and `Base.show(::IO, ::AnnotationDict)` to `src/annotations.jl`
- Updated `test/message_tests.jl`: all `Message(..., nothing)` -> `Message(...)`, `getaddons` -> `getannotations`
- Added `isempty` and `show` tests to `test/annotations_tests.jl`
- Updated `docs/src/lib/message.md`: `getaddons` -> `getannotations`, references `AnnotationDict`

### Step 2.2: `src/message.jl` — MessageMapping struct [DONE]
- Field `addons::X` -> `annotations::X` (holds the processors tuple or nothing)
- Renamed `addons` param to `annotations` in both constructors
- Removed `message_mapping_addons` and `message_mapping_addon` functions
- Updated callable `(mapping::MessageMapping)(messages, marginals)`:
  1. Creates `ann = AnnotationDict()` upfront
  2. Passes `ann` as 9th arg to `rule()`. Rule returns only the distribution.
  3. Loops over `mapping.annotations` processors calling `post_rule_annotations!`
  4. Constructs `Message(result, ..., ann)`

### Step 2.3: `src/message.jl` — MessageProductContext + compute_product_of_two_messages [DONE]
- Added `annotations::N = nothing` field to `MessageProductContext`
- `compute_product_of_two_messages`: replaced `getaddons`/`multiply_addons` with `getannotations`/`post_product_annotations!` using `context.annotations`
- `compute_product_of_messages`: `getaddons(result)` -> `getannotations(result)` when re-wrapping after form constraint

### Step 2.4: `src/marginal.jl` — Marginal struct [DONE]
- `Marginal{D, A}` -> `Marginal{D}`, removed type param `A`
- Field `addons::A` -> `annotations::AnnotationDict` (always present)
- Added 3-arg constructor `Marginal(data, is_clamped, is_initial)` that defaults to fresh `AnnotationDict()`
- `getaddons` -> `getannotations`
- `show` only prints annotations when non-empty via `isempty(ann)`
- Equality no longer compares annotations (metadata, not semantic content)
- Fixed `setmarginal!` and `MarginalMapping` callable to use 3-arg constructor
- Updated docstring and jldoctest example
- Updated `test/marginal_tests.jl`: all `Marginal(..., nothing)` -> `Marginal(...)`, `getaddons` -> `getannotations`
- Updated `docs/src/lib/marginal.md`: `getaddons` -> `getannotations`, example uses 3-arg constructor

### Step 2.5: `src/ReactiveMP.jl` — Bridge functions (lines 45-57) [DONE]
- `as_marginal`/`as_message`: use `getannotations` instead of `getaddons`
- Removed `getlogscale(msg::Message)`, `getlogscale(marginal::Marginal)` convenience wrappers
- Removed `getmemoryaddon` and `getmemory` convenience wrappers

### Step 2.6: `src/variables/random.jl` — `RandomVariableActivationOptions` [DONE — no changes needed]
- `RandomVariableActivationOptions` already holds two `MessageProductContext` instances
- Since `MessageProductContext` gained the `annotations` field in Step 2.3, annotation processors
  flow through naturally — no separate `annotations` field needed on `RandomVariableActivationOptions`
- **Design decision**: annotation processors are configured explicitly in `FactorNodeActivationOptions`
  (for rule-time annotation) and in the `MessageProductContext` instances (for product-time merging).
  RxInfer sets both. There is no implicit inference of processors from neighbouring nodes.

---

## Phase 3: Refactor Rule System

### Step 3.1: `src/rule.jl` — `rule_function_expression` [DONE]
- Renamed `addonsvar` to `annotationsvar`
- The generated `rule()` function receives an `AnnotationDict` as 9th arg
- `local getaddons = () -> ...` → `local getannotations = () -> ...`

### Step 3.2: `src/rule.jl` — `@rule` macro [DONE]
- Removed `local _addons = getaddons()` and the `_messagebody` wrapper lambda
- Rule body no longer wrapped to return `(_message, _addons)` — just returns the result directly
- Removed the comment block about type-stability of addons

### Step 3.3: `src/rule.jl` — `@call_rule` macro [DONE]
- Replaced `addons = ...` keyword with `annotations = ...`; defaults to `ReactiveMP.AnnotationDict()`
- Removed `return_addons` option entirely — caller reads back from the dict after the call
- No tuple unpacking — result is returned directly
- Fallback also receives `annotations` as 9th arg
- Updated docstring
- Updated `test/rule_tests.jl`: replaced `return_addons` testset with `annotations` testset
- Also updated `call_rule_macro_parse_fn_args` tests and `RuleMethodError` constructor calls to use 3-arg `Message`/`Marginal`

### Step 3.4: `@call_marginalrule` — `call_rule_macro_parse_fn_args` [DONE]
- Line 258: `:($(proxy)($any, false, false, nothing))` → `:($(proxy)($any, false, false))`
- `@call_marginalrule` itself does not use addons/annotations — no other changes needed

### Step 3.5: Remove `@invokeaddon` from `src/addons.jl` [DONE — already deleted in Phase 1]
- `src/addons.jl` was deleted in Phase 1
- `@logscale` in `src/annotations/logscale.jl` already calls `annotate!` directly

---

## Phase 4: Refactor Node Infrastructure

### Step 4.1: `src/nodes/nodes.jl` — `FactorNodeActivationOptions` (line 277)
- Rename field `addons::A` to `annotations::A`
- `getaddons(options)` -> `getannotations(options)`

### Step 4.2: `src/nodes/dependencies.jl` — `activate!` (line 30)
- `addons = getaddons(options)` -> `annotations = getannotations(options)`
- Pass `annotations` to `MessageMapping` constructor

### Step 4.3: Delta node files
- `src/nodes/predefined/delta/delta.jl` — rename `addons` param in `rule()` redirect (line 82)
- `src/nodes/predefined/delta/layouts/default.jl` — rename all `addons` references
- `src/nodes/predefined/delta/layouts/cvi.jl` — rename all `addons` references

### Step 4.4: `src/rules/fallbacks.jl`
- All `rulefallback_nodefunction` methods: rename `addons` param to `annotations`
- Line 120: `return FallbackNodeFunctionUnnormalizedLogPdf(fn), addons` -> just return `FallbackNodeFunctionUnnormalizedLogPdf(fn)` (no tuple)

---

## Phase 5: Update Callbacks

### Step 5.1: `src/callbacks.jl`
- `AfterMessageRuleCallEvent`: rename `addons::A` field to `annotations::A`
- `AfterProductOfTwoMessagesEvent`: rename `addons::A` field to `annotations::A`
- Update docstrings

---

## Phase 6: Clean Up Old Addon Files

### Step 6.1: Delete old files
- Delete `src/addons.jl`
- Delete `src/addons/debug.jl`
- Delete `src/addons/logscale.jl`
- Delete `src/addons/memory.jl`

### Step 6.2: Update `src/ReactiveMP.jl` includes
- Remove old addon includes
- Add annotation subtype includes (after `src/annotations.jl`):
  ```julia
  include("annotations/logscale.jl")
  include("annotations/input_arguments.jl")
  ```
- Update exports: remove `AddonLogScale`, `AddonMemory`, `AddonDebug`, `getmemoryaddon`, `multiply_addons`
- Add exports: `AbstractAnnotations`, `LogScaleAnnotations`, `InputArgumentsAnnotations`, `getannotations`

---

## Phase 7: Rule Files Using @logscale (17 files — no changes needed)

The `@logscale` macro is updated in Phase 1.3 to expand to `annotate!(getannotations(), :logscale, value)`. Since `getannotations` is defined by the `@rule` macro (Phase 3.2) as `() -> $annotationsvar`, all existing `@logscale 0` calls in rule files will work without modification.

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

## Phase 8: Update Tests

### Step 8.1: Rewrite addon tests -> annotation tests
- `test/addons_tests.jl` -> `test/rule_annotations_tests.jl`
  - Test `annotate!`, `get_annotation`, `has_annotation` with `AnnotationDict`
  - Test `post_product_annotations!` with processors
  - Test `LogScaleAnnotations` merge logic
- `test/addons/logscale_tests.jl` -> `test/rule_annotations/logscale_tests.jl`
- `test/addons/memory_tests.jl` -> `test/annotations/rule_input_arguments_tests.jl`
- Delete `test/addons/debug_tests.jl`
- `test/annotations_tests.jl` ✓

### Step 8.2: Update existing tests
- `test/message_tests.jl` — update `Message(...)` constructor calls (4th arg = nothing), `getaddons` -> `getannotations` ✓
- `test/marginal_tests.jl` — same
- `test/rule_tests.jl` — rules no longer return tuples, `addons=` -> `annotations=`
- `test/variables/random_tests.jl` — update `MessageProductContext()` if signature changes

---

## Phase 9: Update Documentation

### New pages [DONE]
- `docs/src/lib/annotations.md` ✓ — overview of `AnnotationDict`, `AbstractAnnotations`, custom processor guide
- `docs/src/lib/annotations/logscale.md` ✓ — log-scale theory from van Erp et al. (2023), `LogScaleAnnotations`, `@logscale`
- `docs/src/lib/annotations/input_arguments.md` ✓ — `InputArgumentsAnnotations`, `RuleInputArgumentsRecord`, `ProductInputArgumentsRecord`
- All pages added to `docs/make.jl` under a new top-level "Annotations" section ✓

### Remaining
- Rewrite `docs/src/custom/custom-addons.md` -> `docs/src/custom/custom-annotations.md`
  - Update the step-by-step example to use `AbstractAnnotations`, `annotate!`, `post_rule_annotations!`, `post_product_annotations!`
  - Remove references to `AbstractAddon`, `multiply_addons`, `@invokeaddon`
- Update `docs/src/lib/message.md`: ✓
  - Replace `getaddons` with `getannotations` in the `Message` section
  - Update the `MessageProductContext` description to mention the `annotations` field
- Update any remaining docstrings that still reference addon terminology

---

## Verification

1. Run `julia --project -e 'using ReactiveMP'` to verify compilation
2. Run full test suite: `julia --project -e 'using Pkg; Pkg.test()'`
3. Specifically verify:
   - `@logscale` works in all 17 rule files
   - Message products with `LogScaleAnnotations` correctly merge log scales
   - `InputArgumentsAnnotations` captures rule inputs
   - `AnnotationDict` doesn't allocate its inner dict when no annotations are written

---

## Breaking Changes (for CHANGELOG)

### Removed exports / public API
- `getaddons(::Message)`, `getaddons(::Marginal)` → use `getannotations(...)` instead
- `getlogscale(::Message)`, `getlogscale(::Marginal)` → use `getlogscale(getannotations(...))` instead
- `getmemoryaddon(::Message)`, `getmemoryaddon(::Marginal)` → removed entirely
- `getmemory(::Message)`, `getmemory(::Marginal)` → use `get_rule_input_arguments(getannotations(...))` instead
- `AddonLogScale`, `AddonMemory`, `AddonDebug` structs → replaced by `LogScaleAnnotations`, `InputArgumentsAnnotations`
- `AbstractAddon` → replaced by `AbstractAnnotations`
- `multiply_addons` → replaced by `post_product_annotations!`
- `@invokeaddon` → removed (macros like `@logscale` call `annotate!` directly)

### Changed signatures
- `Message{D, A}` → `Message{D}` (type parameter `A` removed)
- `Marginal{D, A}` → `Marginal{D}` (type parameter `A` removed)
- `Message(data, is_clamped, is_initial, addons)` → `Message(data, is_clamped, is_initial[, annotations::AnnotationDict])`
- `Marginal(data, is_clamped, is_initial, addons)` → `Marginal(data, is_clamped, is_initial[, annotations::AnnotationDict])`
- `@logscale(expr)` → `@logscale value` (no longer wraps in parentheses, directly sets annotation)
- `@rule` body no longer returns `(result, addons)` tuple — just return the result
- `@call_rule` no longer supports `return_addons` option or `addons` keyword — use `annotations` keyword with `AnnotationDict`

### Internal / rule-author changes
- `getaddons()` lambda inside `@rule` body → `getannotations()` (returns the `AnnotationDict` passed as 9th arg)
- `MessageMapping.addons` field → `MessageMapping.annotations` (holds processors tuple or nothing)
- `MessageProductContext` gained `annotations` field for product-time annotation processors
- `message_mapping_addons`, `message_mapping_addon` helper functions removed
- `rule()` generated function: 9th parameter is now an `AnnotationDict` instead of an addons tuple
- `rulefallback` functions: receive `AnnotationDict` as 9th arg, return only the distribution (no tuple)
