# Refactor Addon System to Rule Context System

## Context

The current "addon" system in ReactiveMP.jl is overly complex: it uses typed tuples of `AbstractAddon` subtypes, requires rules to return `(result, addons)` tuples, and has a nonsensical `multiply_addons` product concept. The goal is to replace it with a simpler "rule context" system where:
- Messages carry an optional `Dict{Symbol, Any}` context (lazy-allocated)
- Rules receive a mutable context dict as an argument and write to it directly (no return tuple)
- Context merging during message products is handled by `AbstractRuleContextPropagator` subtypes
- `AddonDebug` is removed (use callbacks instead)

## Key Design Decisions
- **Lazy allocation**: Use `Ref{Union{Nothing, Dict{Symbol, Any}}}` wrapper so Dict is only allocated on first `set_rule_context!` call
- **Clean break**: No deprecation shims for old names
- **Field naming**: `rule_context` (not `rule_context_propagators`)

---

## Phase 1: New Core Infrastructure (new files, no old code changes)

### Step 1.1: Create `src/rule_context.jl`
- `abstract type AbstractRuleContextPropagator end`
- `LazyRuleContext` wrapper: holds `Ref{Union{Nothing, Dict{Symbol, Any}}}`, only allocates Dict on first write
- `set_rule_context!(ctx::Nothing, key, value)` -> no-op
- `set_rule_context!(ctx::LazyRuleContext, key, value)` -> allocates Dict on first call, sets key
- `get_rule_context(ctx, key, default)` -> read from context
- `materialize_context(ctx::LazyRuleContext)` -> returns `nothing` or the `Dict` (for storing in Message)
- `create_rule_context(propagators::Nothing)` -> `nothing`
- `create_rule_context(propagators::Tuple)` -> `LazyRuleContext()`
- `merge_context(propagators, left_ctx, right_ctx, new_dist, left_dist, right_dist)` -> iterates propagators calling `context_merge!`
- `context_merge!(p::AbstractRuleContextPropagator, merged, left_ctx, right_ctx, ...)` -> default no-op
- `post_rule_context!(p::AbstractRuleContextPropagator, context, mapping, messages, marginals, result)` -> default no-op

### Step 1.2: Create `src/rule_context/logscale.jl`
- `struct PropagateLogScale <: AbstractRuleContextPropagator end`
- `context_merge!` for PropagateLogScale: reads `:logscale` from both contexts, adds `compute_logscale(new_dist, ...)`
- `post_rule_context!` for PropagateLogScale: if `:logscale` not set and all inputs are PointMass, set to 0; otherwise error
- New `@logscale` macro: expands to `set_rule_context!(_rule_context, :logscale, value)`
- `getlogscale(ctx::Dict)` reads `:logscale` key
- `getlogscale(::Nothing)` -> error

### Step 1.3: Create `src/rule_context/rule_input_arguments.jl`
- `struct PropagateRuleInputArguments <: AbstractRuleContextPropagator end`
- `RuleInputArgumentsRecord` struct (replaces `AddonMemoryMessageMapping`)
- `RuleInputArgumentsProd` struct (replaces `AddonMemoryProd`)
- `post_rule_context!`: stores `RuleInputArgumentsRecord` under `:rule_input_arguments`
- `context_merge!`: merges records from left/right contexts
- `getmemory(ctx::Dict)` reads `:rule_input_arguments` key

### Files created:
- `src/rule_context.jl`
- `src/rule_context/logscale.jl`
- `src/rule_context/rule_input_arguments.jl`

---

## Phase 2: Refactor Message & Marginal Structs

### Step 2.1: `src/message.jl` — Message struct
- `Message{D, A}` -> `Message{D}`, remove type param `A`
- Field `addons::A` -> `context::Union{Nothing, Dict{Symbol, Any}}`
- `getaddons` -> `getcontext`
- Update all `Message(data, clamped, initial, addons)` constructor calls in this file

### Step 2.2: `src/marginal.jl` — Marginal struct
- Same transformation: `Marginal{D, A}` -> `Marginal{D}`
- Field `addons::A` -> `context::Union{Nothing, Dict{Symbol, Any}}`
- `getaddons` -> `getcontext`

### Step 2.3: `src/ReactiveMP.jl` — Bridge functions (lines 47-59)
- `as_marginal`/`as_message`: use `getcontext` instead of `getaddons`
- `getlogscale(msg::Message)` -> `getlogscale(getcontext(msg))`
- `getmemory(msg::Message)` -> `getmemory(getcontext(msg))`
- Remove `getmemoryaddon` helpers

---

## Phase 3: Refactor Rule System

### Step 3.1: `src/rule.jl` — `rule_function_expression` (line 343)
- Rename `addonsvar` to `contextvar`
- The generated `rule()` function receives context dict (not addons tuple) as 9th arg
- Replace `local getaddons = () -> $addonsvar` with exposing `_rule_context` variable

### Step 3.2: `src/rule.jl` — `@rule` macro (line 462)
- Replace `local _addons = getaddons()` with `local _rule_context = $contextvar`
- Rule body no longer wrapped to return `(_message, _addons)` — just return `_message`
- The `_rule_context` variable is in scope for `@logscale` and similar macros to write to

### Step 3.3: `src/rule.jl` — `@call_rule` macro (line 593)
- Parse `rule_context = ...` instead of `addons = ...` from args
- Rule returns only the distribution (no destructuring of tuples)
- `return_addons` option -> `return_context` (returns `(result, context_dict)`)
- Adjust `@call_marginalrule` similarly if it references addons (it doesn't seem to currently)

### Step 3.4: Remove `@invokeaddon` from `src/addons.jl`
- The `@logscale` macro now directly calls `set_rule_context!`, no need for `@invokeaddon`

---

## Phase 4: Refactor MessageMapping

### Step 4.1: `src/message.jl` — MessageMapping struct (line 499)
- Rename field `addons::X` to `rule_context::X` (this holds the propagators tuple or nothing)

### Step 4.2: `src/message.jl` — MessageMapping callable (line 614)
- New flow:
  1. `ctx = create_rule_context(mapping.rule_context)` — creates LazyRuleContext or nothing
  2. Call `rule(...)` passing `ctx` as 9th arg. Rule returns only the distribution.
  3. `materialized = materialize_context(ctx)` — get Dict or nothing
  4. Loop over `mapping.rule_context` propagators, call `post_rule_context!(p, materialized, mapping, messages, marginals, result)` for each
  5. Construct `Message(result, ..., materialized)`
- Remove `message_mapping_addons` and `message_mapping_addon` functions

### Step 4.3: MessageMapping constructors (lines 564-612)
- Rename `addons` param to `rule_context` in both constructors

---

## Phase 5: Refactor Message Products (Context Merging)

### Step 5.1: `src/message.jl` — `MessageProductContext` (line 133)
- Add field `rule_context::X = nothing` to hold propagators tuple

### Step 5.2: `src/message.jl` — `compute_product_of_two_messages` (line 160)
- Replace lines 203-211:
  ```julia
  left_ctx = getcontext(left)
  right_ctx = getcontext(right)
  new_ctx = merge_context(context.rule_context, left_ctx, right_ctx, new_dist, left_dist, right_dist)
  result = Message(new_dist, is_prod_clamped, is_prod_initial, new_ctx)
  ```
- Remove all `multiply_addons` calls

### Step 5.3: `src/variables/random.jl` — `RandomVariableActivationOptions`
- May need to pass `rule_context` propagators into `MessageProductContext` construction
- Check how RxInfer sets this up (likely needs corresponding RxInfer changes)

---

## Phase 6: Refactor Node Infrastructure

### Step 6.1: `src/nodes/nodes.jl` — `FactorNodeActivationOptions` (line 277)
- Rename field `addons::A` to `rule_context::A`
- `getaddons(options)` -> `getrulecontext(options)` (or similar accessor)

### Step 6.2: `src/nodes/dependencies.jl` — `activate!` (line 30)
- `addons = getaddons(options)` -> `rule_context = getrulecontext(options)`
- Pass `rule_context` to `MessageMapping` constructor

### Step 6.3: Delta node files
- `src/nodes/predefined/delta/delta.jl` — rename `addons` param in `rule()` redirect (line 82)
- `src/nodes/predefined/delta/layouts/default.jl` — rename all `addons` references
- `src/nodes/predefined/delta/layouts/cvi.jl` — rename all `addons` references

### Step 6.4: `src/rules/fallbacks.jl`
- All `rulefallback_nodefunction` methods: rename `addons` param to `rule_context`
- Line 120: `return FallbackNodeFunctionUnnormalizedLogPdf(fn), addons` -> just return `FallbackNodeFunctionUnnormalizedLogPdf(fn)` (no tuple)

---

## Phase 7: Update Callbacks

### Step 7.1: `src/callbacks.jl`
- `AfterMessageRuleCallEvent`: rename `addons::A` field to `context::A`
- `AfterProductOfTwoMessagesEvent`: rename `addons::A` field to `context::A`
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
- Add new rule_context includes (after `src/marginal.jl`):
  ```julia
  include("rule_context.jl")
  include("rule_context/logscale.jl")
  include("rule_context/rule_input_arguments.jl")
  ```
- Update exports: remove `AddonLogScale`, `AddonMemory`, `AddonDebug`, `getmemoryaddon`, `multiply_addons`
- Add exports: `AbstractRuleContextPropagator`, `PropagateLogScale`, `PropagateRuleInputArguments`, `getcontext`

---

## Phase 9: Rule Files Using @logscale (17 files — no changes needed)

The `@logscale` macro is updated in Phase 1.2 to expand to `set_rule_context!(_rule_context, :logscale, value)`. Since `_rule_context` is now exposed by the `@rule` macro (Phase 3.2), all existing `@logscale 0` calls in rule files will work without modification.

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

### Step 10.1: Rewrite addon tests -> rule_context tests
- `test/addons_tests.jl` -> `test/rule_context_tests.jl`
  - Test `set_rule_context!`, `get_rule_context` with nothing and LazyRuleContext
  - Test `merge_context` with propagators
  - Test PropagateLogScale merge logic
- `test/addons/logscale_tests.jl` -> `test/rule_context/logscale_tests.jl`
- `test/addons/memory_tests.jl` -> `test/rule_context/rule_input_arguments_tests.jl`
- Delete `test/addons/debug_tests.jl`

### Step 10.2: Update existing tests
- `test/message_tests.jl` — update `Message(...)` constructor calls (4th arg = nothing), `getaddons` -> `getcontext`
- `test/marginal_tests.jl` — same
- `test/rule_tests.jl` — rules no longer return tuples, `addons=` -> `rule_context=`
- `test/variables/random_tests.jl` — update `MessageProductContext()` if signature changes

---

## Phase 11: Update Documentation

- Rewrite `docs/src/custom/custom-addons.md` -> `docs/src/custom/custom-rule-context.md`
- Document: `AbstractRuleContextPropagator`, `@logscale`, `set_rule_context!`, `context_merge!`, `post_rule_context!`
- Update any docstrings in source files

---

## Verification

1. Run `julia --project -e 'using ReactiveMP'` to verify compilation
2. Run full test suite: `julia --project -e 'using Pkg; Pkg.test()'`
3. Specifically verify:
   - `@logscale` works in all 17 rule files
   - Message products with `PropagateLogScale` correctly merge log scales
   - `PropagateRuleInputArguments` captures rule inputs
   - Messages without propagators have `context === nothing` (zero overhead)
   - LazyRuleContext doesn't allocate Dict when rule doesn't call `set_rule_context!`
