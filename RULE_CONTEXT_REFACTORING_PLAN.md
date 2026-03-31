# Refactor Addon System to Rule Annotations System

## Context

The current "addon" system in ReactiveMP.jl is overly complex: it uses typed tuples of `AbstractAddon` subtypes, requires rules to return `(result, addons)` tuples, and has a nonsensical `multiply_addons` product concept. The goal is to replace it with a simpler "rule annotations" system where:
- Messages carry an optional `AnnotationDict` (lazy-allocated, defined in `src/annotations.jl`)
- Rules receive a mutable `AnnotationDict` as an argument and annotate it directly (no return tuple)
- Annotation merging during message products is handled by `AbstractAnnotationProcessor` subtypes
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

### Step 1.2: Create `src/rule_annotations.jl`
- `abstract type AbstractAnnotationProcessor end`
- `create_annotations(processors::Nothing)` -> `nothing`
- `create_annotations(processors::Tuple)` -> `AnnotationDict()`
- `merge_annotations(processors, left_ann, right_ann, new_dist, left_dist, right_dist)` -> iterates processors calling `merge_annotations!`
- `merge_annotations!(p::AbstractAnnotationProcessor, merged, left_ann, right_ann, ...)` -> default no-op
- `post_rule_annotations!(p::AbstractAnnotationProcessor, ann, mapping, messages, marginals, result)` -> default no-op

### Step 1.3: Create `src/rule_annotations/logscale.jl`
- `struct PropagateLogScale <: AbstractAnnotationProcessor end`
- `merge_annotations!` for PropagateLogScale: reads `:logscale` from both annotation dicts, adds `compute_logscale(new_dist, ...)`
- `post_rule_annotations!` for PropagateLogScale: if `:logscale` not set and all inputs are PointMass, set to 0; otherwise error
- New `@logscale` macro: expands to `annotate!(_annotations, :logscale, value)`
- `getlogscale(ann::AnnotationDict)` reads `:logscale` key
- `getlogscale(::Nothing)` -> error

### Step 1.4: Create `src/rule_annotations/rule_input_arguments.jl`
- `struct PropagateRuleInputArguments <: AbstractAnnotationProcessor end`
- `RuleInputArgumentsRecord` struct (replaces `AddonMemoryMessageMapping`)
- `RuleInputArgumentsProd` struct (replaces `AddonMemoryProd`)
- `post_rule_annotations!`: stores `RuleInputArgumentsRecord` under `:rule_input_arguments`
- `merge_annotations!`: merges records from left/right annotation dicts
- `getmemory(ann::AnnotationDict)` reads `:rule_input_arguments` key

### Files created:
- `src/annotations.jl` ✓
- `src/rule_annotations.jl`
- `src/rule_annotations/logscale.jl`
- `src/rule_annotations/rule_input_arguments.jl`

---

## Phase 2: Refactor Message & Marginal Structs

### Step 2.1: `src/message.jl` — Message struct
- `Message{D, A}` -> `Message{D}`, remove type param `A`
- Field `addons::A` -> `annotations::Union{Nothing, AnnotationDict}`
- `getaddons` -> `getannotations`
- Update all `Message(data, clamped, initial, addons)` constructor calls in this file

### Step 2.2: `src/marginal.jl` — Marginal struct
- Same transformation: `Marginal{D, A}` -> `Marginal{D}`
- Field `addons::A` -> `annotations::Union{Nothing, AnnotationDict}`
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
- Parse `annotations = ...` instead of `addons = ...` from args
- Rule returns only the distribution (no destructuring of tuples)
- `return_addons` option -> `return_annotations` (returns `(result, annotation_dict)`)
- Adjust `@call_marginalrule` similarly if it references addons

### Step 3.4: Remove `@invokeaddon` from `src/addons.jl`
- The `@logscale` macro now directly calls `annotate!`, no need for `@invokeaddon`

---

## Phase 4: Refactor MessageMapping

### Step 4.1: `src/message.jl` — MessageMapping struct (line 499)
- Rename field `addons::X` to `annotations::X` (holds the processors tuple or nothing)

### Step 4.2: `src/message.jl` — MessageMapping callable (line 614)
- New flow:
  1. `ann = create_annotations(mapping.annotations)` — creates `AnnotationDict` or `nothing`
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
  new_ann = merge_annotations(context.annotations, left_ann, right_ann, new_dist, left_dist, right_dist)
  result = Message(new_dist, is_prod_clamped, is_prod_initial, new_ann)
  ```
- Remove all `multiply_addons` calls

### Step 5.3: `src/variables/random.jl` — `RandomVariableActivationOptions`
- May need to pass `annotations` processors into `MessageProductContext` construction
- Check how RxInfer sets this up (likely needs corresponding RxInfer changes)

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
- Add new rule_annotations includes (after `src/annotations.jl`):
  ```julia
  include("rule_annotations.jl")
  include("rule_annotations/logscale.jl")
  include("rule_annotations/rule_input_arguments.jl")
  ```
- Update exports: remove `AddonLogScale`, `AddonMemory`, `AddonDebug`, `getmemoryaddon`, `multiply_addons`
- Add exports: `AbstractAnnotationProcessor`, `PropagateLogScale`, `PropagateRuleInputArguments`, `getannotations`

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
  - Test `merge_annotations` with processors
  - Test `PropagateLogScale` merge logic
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
- Document: `AbstractAnnotationProcessor`, `@logscale`, `annotate!`, `merge_annotations!`, `post_rule_annotations!`
- Update any docstrings in source files

---

## Verification

1. Run `julia --project -e 'using ReactiveMP'` to verify compilation
2. Run full test suite: `julia --project -e 'using Pkg; Pkg.test()'`
3. Specifically verify:
   - `@logscale` works in all 17 rule files
   - Message products with `PropagateLogScale` correctly merge log scales
   - `PropagateRuleInputArguments` captures rule inputs
   - Messages without processors have `annotations === nothing` (zero overhead)
   - `AnnotationDict` doesn't allocate when rule doesn't call `annotate!`
