# Changelog

All notable changes to ReactiveMP.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- The 2-interface `DiscreteTransition(:in)` rule for `PointMass` inputs now uses `softmax!` like the twenty sibling rules in the same file, instead of `exp.` followed by `normalize!`, and two lines of unreachable code after its `return` have been removed. Output is unchanged — `clamplog` bounds the log values from below and the `probvec`-weighted average pulls them toward zero, so both formulations agree on any reachable input; this is a dead-code and consistency cleanup, not a bug fix ([#627](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/627))
- Rewrote the stale `# calculate marginals of input WRONG` marker in `src/rules/bifm/in.jl` to `# calculate the marginal of the input`. The marker was appended by the 2021 commit that *fixed* that rule (`edf799f4`, "fix BIFM input update rule"), most likely as a note that the previous formulation had been wrong, and read as a standing warning that the current rule is defective. Comment only — no behaviour change, and no claim either way about the rule's correctness ([#638](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/638))
- Pinned the formatter toolchain so CI and local runs cannot disagree. `scripts/Project.toml` had no compat bound on `JuliaFormatter`, `scripts/Manifest.toml` was gitignored, `scripts_init` ran `Pkg.update()` on every `make format` / `make check-format`, and `FormatCheck.yml` specified no Julia version — so which formatter (and which Julia, which matters because `JuliaFormatter` parses via `JuliaSyntax`) you got depended only on when you last ran the command. Now: `JuliaFormatter = "~2.12"` compat, `scripts/Manifest.toml` tracked, `Pkg.update()` moved out of `scripts_init` into a new `make scripts_update` target, and `FormatCheck.yml` pinned to Julia 1.12. Verified reproducible by instantiating the two files alone under an isolated depot
### Fixed
- `ImportanceSamplingApproximation`'s effective sample size is now computed with the scale-invariant `(Σwᵢ)²/Σwᵢ²` instead of `1/Σwᵢ²`. The latter is only valid for weights that already sum to one, but at that point the buffer holds raw unnormalised `g(z)` values, so the `n_eff < N/10` resampling threshold depended on the absolute magnitude of `g` rather than on how uneven the weights were — rescaling `g` by a constant, which cannot change the estimate, switched resampling on or off, and perfectly uniform weights of value 1 gave `n_eff = 1/N` and triggered resampling ([#637](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/637))
- `ImportanceSamplingApproximation` no longer throws from inside resampling on degenerate weights. `sample!` rejects all-zero weights (`ArgumentError: all weights are zero`) and non-finite weights (`ArgumentError: weights cannot contain Inf or NaN values`); reaching it with either pre-empted the existing degenerate-estimate fallbacks and turned a handled case into a hard failure. Resampling is now skipped when the weights are not resamplable ([#637](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/637))
- `ImportanceSamplingApproximation`'s degenerate-estimate guard now also covers `isinf(v)` and `isnan(v)`; previously only the mean was checked for being non-finite, so an infinite variance escaped into the returned moments ([#637](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/637))

### Changed
- `ImportanceSamplingApproximation` now warns (once per session) when it falls back to the proposal distribution's own moments, instead of substituting them silently. The fallback returns a confident-looking answer that never saw the target, so it should be visible ([#631](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/631))
- Documented that `ImportanceSamplingApproximation` is **not reentrant**: its sample and weight buffers are preallocated and overwritten in place, so an integrand `g` that itself calls `approximate_meancov` with the same approximation object silently corrupts the outer result. This is a reentrancy constraint, not a thread-safety one — ReactiveMP's schedulers are `@async` coroutines on a single thread ([#637](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/637))
### Fixed
- `LaplaceApproximation`'s `approximate_meancov` returned a **negated covariance for every input**, not just pathological ones. It computed `-cholinv(H)` from the Hessian `H` of the log-density at the mode. `-(H⁻¹)` and `(-H)⁻¹` agree in exact arithmetic, but `cholinv` factorizes via Cholesky, which is only defined for positive-definite input — and at a maximum `H` is negative-definite. Passing it directly does not raise; it silently returns a matrix that is not `H⁻¹`, so the sign flip was not compensated and the result was negative-definite. It now negates before inverting, giving `(-H)⁻¹`. Verified against the closed-form Gaussian-product answer, for which the Laplace approximation is exact. `LaplaceApproximation` had no test coverage at all, which is why this went unnoticed ([#635](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/635))
- `LaplaceApproximation` now rejects a stationary point that is not a strict local maximum instead of returning an invalid Gaussian. `Optim.converged` only reports that the optimizer stopped moving, so for a non-log-concave integrand it could settle on a saddle point or flat region where the negated Hessian is not positive-definite. The error names the offending matrix and suggests approximation methods that do not assume local concavity ([#635](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/635))

### Removed
- Dead `d_logf` gradient closure in `LaplaceApproximation`'s `approximate_meancov`, which was constructed but never used ([#635](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/635))
### Fixed
- A zero-covariance inbound message no longer crashes the `DeltaFn(:ins)` joint marginal rules. `unscented_statistics`' zero-covariance fallback returned `nothing` for the cross-covariance — "not computed" rather than a degenerate value of the same kind as the zeros beside it — and the marginal rules feed that third return value straight into `smoothRTS`, failing with `MethodError: no method matching *(::Nothing, ::Float64)`. It now returns a genuine zero cross-covariance, which is both correct (a deterministic input has zero covariance with anything) and type-stable with the non-degenerate path ([#630](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/630))
- `smoothRTS` now short-circuits when the forward output covariance is singular or non-finite, returning the forward inbound statistics unchanged. A deterministic node output cannot be revised by the backward message, so that *is* the smoothed marginal. Previously `cholinv(V_tilde)` on a zero covariance returned `Inf` rather than raising, so `D_tilde = C_tilde * W_tilde` evaluated to `0 * Inf = NaN` and a silently corrupted marginal propagated. This affects both the `Unscented` and `Linearization` delta-node paths ([#630](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/630))
- Documented the buffer-reuse contract of the multivariate `getpoints` generators for `GaussHermiteCubature` and `SphericalRadialCubature`. Both write every cubature point into a single preallocated vector and yield that same object each iteration, so `collect(getpoints(...))` returns one buffer repeated — silently wrong cubature with no error — and `map(copy, getpoints(...))` is required to materialize independent points. The reuse is intentional and load-bearing: `approximate_meancov` mutates the yielded point in place rather than allocating, so the sharing runs in both directions. Behaviour is unchanged; the contract is now stated in docstrings and pinned by tests ([#633](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/633))
- Documented that `==` on `Message` and `Marginal` compares `data`, `is_clamped` and `is_initial` but deliberately **not** `annotations`, so two messages carrying the same distribution are equal even when their `:logscale` differs. Annotations describe how a message was computed rather than the belief it represents; callers needing annotation-sensitive equality should compare `getannotations` explicitly. Added to both docstrings with runnable doctests and to the message library page; behaviour is unchanged ([#632](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/632))

## [6.3.3] - 2026-07-14

### Fixed
- Structured VMP with 3+ mutually-dependent clusters no longer deadlocks when an initialized message caused the first outbound message computation to consume only provisional (`is_initial`) marginals: the marginal dependencies of a node now stay consumable while all of their recent values are `is_initial`, instead of `PushNew()` requiring every dependency to refresh before the next computation. The relaxation applies to marginal dependencies only — message dependencies keep strict `PushNew()` semantics, so message-update schedules (and hence free-energy trajectories) of unaffected models remain unchanged ([#344](https://github.com/ReactiveBayes/RxInfer.jl/issues/344), [#620](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/620))

## [6.3.2] - 2026-07-13

### Fixed
- Mean-field `@average_energy` for the `softdot` node no longer applies a spurious extra `mean(q_γ)` factor to the cross term `−2·m_y·m_θᵀm_x`, which biased the term by `mean(q_γ)²` instead of `mean(q_γ)`. Message rules were unaffected (they were and remain correct); only the Bethe free energy contribution of a `softdot` node under full mean-field factorization was wrong, and only when `mean(q_γ) ≠ 1`. The mean-field variant now agrees with the (already-correct) structured `q(y, x)` variant. This bug was present unchanged from at least v4.4.3 through v6.3.1, so any Bethe free energy values previously reported for mean-field `softdot` models were biased by `½·mean(q_γ)·(mean(q_γ)−1)·(−2·m_y·m_θᵀm_x)` ([#615](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/615))

### Changed
- CI: the `test` job now runs with a 60-minute timeout so a hung dependency fails fast instead of consuming GitHub's default 6-hour limit ([#618](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/618))

## [6.3.1] - 2026-07-06

### Fixed
- `InputArgumentsAnnotations`'s `post_product_annotations!` no longer throws `KeyError: key :rule_input_arguments not found` when one side of a message product is a clamped/constant message (e.g. a prior passed into a model as a `Distribution` object) that never went through a rule ([#600](https://github.com/ReactiveBayes/RxInfer.jl/issues/600), [#614](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/614))

## [6.3.0] - 2026-06-19

### Added

- `ConjugateAR` node for fully-conjugate variational inference of autoregressive models. It keeps the AR coefficients `θ` and the transition precision `γ` jointly on a single `MvNormalGamma`-distributed parameter edge `w` (rather than the mean-field `q(θ)q(γ)` split of `AR`). The `ConjugateAR(:w)` message implements the Normal-Gamma / Bayesian-linear-regression parameter update; the state messages (`:y`, `:x`), the `(y, x)` joint marginal, and the average energy delegate to the existing `AR` rules via the effective `(q_θ, q_γ)` moments ([#612](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/612)).
- `MvNormalGamma` registered as a factor node (`[out, μ, Λ, α, β]`) with its `:out` message rule and average energy, so a joint `(θ, γ)` prior can be placed on the `ConjugateAR` parameter edge ([#612](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/612)).

## [6.2.0] - 2026-06-18

### Added

- `MatrixNormal` node with belief-propagation and variational (mean-field) message rules for `:out`, `:M`, `:U`, `:V`, joint marginal rule, and mean-field `@average_energy` ([#610](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/610))
- `MatrixNormalWishart` conjugate-prior node with its `:out` construction rule and analytic `@average_energy` ([#610](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/610))

## [6.1.1] - 2026-05-22

### Changed
- Improved `AddonMemory` deprecation error message to include a before/after migration example and a direct link to the v5-to-v6 migration guide ([#600](https://github.com/ReactiveBayes/RxInfer.jl/issues/600), [#605](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/605))
- Improved `AddonLogScale` deprecation error message to include a before/after migration example and a direct link to the v5-to-v6 migration guide ([#605](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/605))

### Fixed
- Added missing `BayesBase` module prefix to `paramfloattype(::ManyOf)` method so the extension is correctly registered ([#606](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/606))

## [6.1.0] - 2026-05-04

### Added
- `Base.show` methods for every callback `Event` defined in `src/callbacks.jl` (`Before/AfterMessageRuleCallEvent`, `Before/AfterProductOfTwoMessagesEvent`, `Before/AfterProductOfMessagesEvent`, `Before/AfterFormConstraintAppliedEvent`, `Before/AfterMarginalComputationEvent`) so the RxInfer trace logger no longer dumps raw struct contents into TBLogger text summaries ([#599](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/599), [RxInfer.jl#638](https://github.com/ReactiveBayes/RxInfer.jl/issues/638)). The methods honor the `IOContext` `:compact` flag: trace loggers pass `:compact => true` to get a one-line `nmsgs=N` / 4-char span summary, while REPL/Pluto/Jupyter sees the full form with actual messages and the full UUID span id.
- `Base.show` methods following the same `:compact` convention for the supporting types those events reference: `MessageMapping`, `MessageProductContext`, `FormConstraintCheckEach`, `FormConstraintCheckLast`
- `_show_span` internal helper that emits no field at all when the span id is `nothing` (callbacks disabled), a 4-char prefix in `:compact` contexts, and the full UUID otherwise

### Changed
- `Base.show(io, ::AnnotationDict)` now switches between the count summary `AnnotationDict(n=K)` and the full key/value listing based on `get(io, :compact, false)` rather than `MIME"text/plain"` dispatch, matching the convention used by the new event show methods

## [6.0.0] - 2026-04-17

### Added
- `AbstractStreamPostprocessor` abstraction unifying the old pipeline stages and the per-node `scheduler` argument under a single concept that postprocesses outbound message streams, marginal streams, and score streams uniformly
- `postprocess_stream_of_outbound_messages`, `postprocess_stream_of_marginals`, `postprocess_stream_of_scores` entry points with `::Nothing` pass-through fallbacks
- `CompositeStreamPostprocessor` for chaining multiple postprocessors
- `ScheduleOnStreamPostprocessor` — direct successor of `ScheduleOnPipelineStage` plus the per-node scheduler, applies a Rocket.jl scheduler to all three stream kinds
- Marginal streams and score streams now go through stream postprocessors (previously only outbound message streams did)
- Documentation page for stream postprocessors
- Callback/event system for hooking into message passing steps (rule calls, message products, form constraints, marginal computation)
- `MessageProductContext` struct to bundle product computation settings and callbacks
- Labels for variables (`RandomVariable`, `ConstVariable`, `DataVariable`)
- Docstrings for variable types, form constraints, and related functions
- Documentation page for callbacks
- `MethodError` hint for mismatched `handle_event` signatures
- New annotations system: `AnnotationDict`, `AbstractAnnotations`, `LogScaleAnnotations`, `InputArgumentsAnnotations`
- `post_rule_annotations!` and `post_product_annotations!` callbacks for annotation processors
- `@logscale value` macro for setting log-scale annotations inside `@rule` bodies
- `getannotations` function for `Message` and `Marginal`
- Migration guide for v5 to v6
- `skip_initial()`, `skip_clamped()`, `skip_clamped_and_initial()` filter operators replacing the `MarginalSkipStrategy` type hierarchy
- `new_observation!(datavar, value)` for pushing observed values into a `DataVariable`
- `get_stream_of_inbound_messages`, `get_stream_of_outbound_messages` accessors on `NodeInterface` and `IndexedNodeInterface`
- `get_stream_of_marginals`, `set_stream_of_marginals!` accessors on variables
- `get_stream_of_predictions`, `set_stream_of_predictions!` accessors on variables
- `set_initial_marginal!`, `set_initial_message!` for seeding variables before inference
- `create_new_stream_of_inbound_messages!` for allocating per-connection message streams
- Docstrings for `MessageObservable`, `MarginalObservable`, `FunctionalDependencies`, `collect_functional_dependencies`, `RandomVariableActivationOptions`, `DataVariableActivationOptions`, `FactorNodeActivationOptions`, and `activate!` methods
- Expanded documentation for variables (stream creation lifecycle per variable type), nodes (interfaces, activation), messages, and marginals

### Changed
- `FactorNodeActivationOptions` lost its `pipeline` and `scheduler` positional fields and gained a single `postprocessor` field
- `RandomVariableActivationOptions` renamed its `scheduler` field to `stream_postprocessor`; the default is now `nothing` (no-op) instead of `AsapScheduler()`
- `getpipeline(options)` and `getscheduler(options)` replaced by `getpostprocessor(options)`
- `EqualityChain` renamed its `pipeline` field to `postprocessor`
- Switched from `ReTestItems` to `TestItemRunner` for tests ([#584](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/584))
- Made formatting checks stricter
- Removed `variables/generic.jl`; generic variable interface moved into `variable.jl`
- Replaced hardcoded `DefaultMessageProdFn`/`DefaultMarginalProdFn` with `MessageProductContext`
- `Message{D, A}` → `Message{D}` (type parameter `A` removed)
- `Marginal{D, A}` → `Marginal{D}` (type parameter `A` removed)
- `Message` and `Marginal` now carry an `AnnotationDict` instead of a typed addons tuple
- Rules no longer return `(result, addons)` tuples — just the result
- `@call_rule` no longer supports `return_addons` option; use `annotations` keyword with `AnnotationDict`
- `MessageMapping.addons` field → `MessageMapping.annotations`
- `MessageProductContext` gained `annotations` field for product-time annotation processors
- `messagein(interface)` → `get_stream_of_inbound_messages(interface)`
- `messageout(interface)` → `get_stream_of_outbound_messages(interface)`
- `getmarginal(variable)` / `getmarginals` → `get_stream_of_marginals(variable)`
- `getprediction(variable)` / `getpredictions` → `get_stream_of_predictions(variable)`
- `setmarginal!(variable, value)` → `set_initial_marginal!(variable, value)`
- `setmessage!(variable, value)` → `set_initial_message!(variable, value)`
- `update!(datavar, value)` → `new_observation!(datavar, value)`

### Removed
- `AbstractPipelineStage`, `EmptyPipelineStage`, `CompositePipelineStage`, `ScheduleOnPipelineStage`, `apply_pipeline_stage`, `collect_pipeline`, `+` composition — replaced by the `AbstractStreamPostprocessor` abstraction (see migration guide)
- `LoggerPipelineStage` — equivalent behaviour can be implemented via callbacks
- `AsyncPipelineStage` — use `ScheduleOnStreamPostprocessor(AsyncScheduler())` instead
- `DiscontinuePipelineStage` — was unused; implement a custom `AbstractStreamPostprocessor` if needed
- `schedule_updates(vars; pipeline_stage = ...)` — construct a `ScheduleOnStreamPostprocessor` and pass it through the activation options instead
- `getaddons` — use `getannotations` instead
- `getlogscale(::Message)`, `getlogscale(::Marginal)` — use `getlogscale(getannotations(...))` instead
- `getmemory`, `getmemoryaddon` — use `get_rule_input_arguments(getannotations(...))` instead
- `AddonLogScale` — replaced by `LogScaleAnnotations` (calling `AddonLogScale()` throws a descriptive error)
- `AddonMemory` — replaced by `InputArgumentsAnnotations` (calling `AddonMemory()` throws a descriptive error)
- `AddonDebug` — use callbacks instead
- `AbstractAddon`, `multiply_addons`, `@invokeaddon`
- `message_mapping_addons`, `message_mapping_addon` helper functions
- `MarginalSkipStrategy` abstract type and `SkipClamped`, `SkipInitial`, `SkipClampedAndInitial`, `IncludeAll` subtypes — use `skip_clamped()`, `skip_initial()`, `skip_clamped_and_initial()` filter operators instead
- `apply_skip_filter`, `as_marginal_observable` — no longer part of the public API
- `messagein`, `messageout` — use `get_stream_of_inbound_messages`, `get_stream_of_outbound_messages`
- `getmarginal`, `getmarginals`, `getprediction`, `getpredictions` — use `get_stream_of_marginals`, `get_stream_of_predictions`
- `setmarginal!`, `setmarginals!`, `setmessage!`, `setmessages!` — use `set_initial_marginal!`, `set_initial_message!`
- `update!` — use `new_observation!`
- `create_messagein!` — use `create_new_stream_of_inbound_messages!`

## [5.6.6] - 2026-03-13

### Fixed
- Implemented effective rules with specialized dispatch for `MvNormalMeanScalePrecision` ([#579](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/579))

### Tests
- Added performance test for structured rule specialized for `MvNormalMeanScalePrecision`

## [5.6.5] - 2026-02-02

### Added
- Implemented `MvNormalWishart` node and `out` rule ([#565](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/565))
- Issue templates ([#558](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/558))
- Rule interface name checking for `@rule`, `@marginalrule`, and `@average_energy` macros ([#545](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/545))

### Changed
- Removed `Requires` dependency (used for Julia <1.9, no longer supported) ([#564](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/564))
- Removed vibe coded required fields from issue template ([#562](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/562))
- Bumped compat for Optim to 2 ([#574](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/574))
- Removed `Zygote` extension requirement for compatibility

### Fixed
- Fixed documentation build ([#567](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/567))

### Performance
- Pre-computed double loops in CT model ([#571](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/571))

## [5.6.4] - 2025-11-18

### Fixed
- Fixed bug in average energy of `Uninformative` type ([#553](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/553))
- Improved robustness of inverse precision matrix computation in `MvNormalMeanPrecision` rule ([#540](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/540))

### Documentation
- Added `MessageMapping` documentation ([#550](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/550))

## [5.6.3] - 2025-11-04

### Added
- Added `IntegrationTest.yml` workflow ([#525](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/525))
- Implemented missing marginal rule for multiplication node ([#531](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/531))

### Changed
- Changed dispatch to `AbstractVector` to allow other vector implementations ([#536](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/536))

### Fixed
- Renamed softdot marginal rules test file to include it in test runs ([#535](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/535))
- Avoided `Vararg` deprecation warnings ([#537](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/537))
- Defined `AverageEnergy` for `Mixture` node with warning ([#546](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/546))
- Allowed different numeric types for `GammaShapeLikelihood` constructor ([#544](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/544))

### Tests
- Improved code coverage with tests for BIFM, Autoregressive, Mixture, GammaMixture, Wishart, InverseWishart, DotProduct, Multiplication, and Uniform nodes ([#539](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/539))

## [5.6.2] - 2025-10-21

### Fixed
- Fixed `isonehot` to use approximate comparison for categorical rules ([#527](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/527))

## [5.6.1] - 2025-10-21

### Added
- Categorical rules: check if probability vector of `q_out` is a one-hot encoded vector ([#510](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/510))
- Support for non-linear node (univariate -> multivariate) with Unscented transform ([#508](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/508))

### Changed
- Updated `ForwardDiff` to version 1 ([#521](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/521))
- Skip Aqua.jl checks during selective test runs ([#523](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/523))
- Makefile: fixed selective test runs with `test_args` argument ([#517](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/517))

### Fixed
- Updated documentation for `as_marginal` ([#516](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/516))
- Updated test cases for `GammaShapeScale`

## [5.6.0] - 2025-09-23

### Changed
- Use `MvNormalMeanScaleMatrixPrecision` from ExponentialFamily package ([#509](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/509))

## [5.5.12] - 2025-09-11

### Fixed
- Fixed `q_t1` dimensionality bug in delta node rules ([#504](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/504))

## [5.5.11] - 2025-09-10

### Added
- Implemented `MvNormalMeanScaleMatrixPrecision` rules ([#497](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/497))

## [5.5.10] - 2025-09-09

### Fixed
- Added new linearization method ([#500](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/500))

## [5.5.9] - 2025-08-14

### Changed
- Reverted "Don't check for proper in division of" ([#496](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/496))

## [5.5.8] - 2025-08-14

### Added
- Show meta suggestions in rule error printing ([#495](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/495))

### Changed
- Tightened dispatch for summation and other optimized rules ([#492](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/492), [#493](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/493))

### Fixed
- Don't check for proper in division of to accommodate inference ([#486](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/486))

## [5.5.7] - 2025-07-24

### Fixed
- Fixed infinite RxInfer documentation build by not using lazy string ([#490](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/490))

## [5.5.6] - 2025-07-23

### Fixed
- Fixed invalidations: removed bad `eltype` methods, `convert`, and `println` method ([#489](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/489))

## [5.5.5] - 2025-07-23

### Fixed
- Resolved Gaussian division with proper Multivariate vs Univariate handling ([#479](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/479))

---

[Unreleased]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v6.3.3...HEAD
[6.3.3]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v6.3.2...v6.3.3
[6.3.2]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v6.3.1...v6.3.2
[6.3.1]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v6.3.0...v6.3.1
[6.3.0]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v6.2.0...v6.3.0
[6.2.0]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v6.1.1...v6.2.0
[6.1.1]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v6.1.0...v6.1.1
[6.1.0]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v6.0.0...v6.1.0
[6.0.0]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.6.6...v6.0.0
[5.6.6]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.6.5...v5.6.6
[5.6.5]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.6.4...v5.6.5
[5.6.4]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.6.3...v5.6.4
[5.6.3]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.6.2...v5.6.3
[5.6.2]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.6.1...v5.6.2
[5.6.1]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.6.0...v5.6.1
[5.6.0]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.5.12...v5.6.0
[5.5.12]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.5.11...v5.5.12
[5.5.11]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.5.10...v5.5.11
[5.5.10]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.5.9...v5.5.10
[5.5.9]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.5.8...v5.5.9
[5.5.8]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.5.7...v5.5.8
[5.5.7]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.5.6...v5.5.7
[5.5.6]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.5.5...v5.5.6
[5.5.5]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.5.4...v5.5.5
