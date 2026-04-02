# Changelog

All notable changes to ReactiveMP.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Changed
- Switched from `ReTestItems` to `TestItemRunner` for tests ([#584](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/584))
- Made formatting checks stricter
- Renamed `variables/variable.jl` to `variables/generic.jl`
- Replaced hardcoded `DefaultMessageProdFn`/`DefaultMarginalProdFn` with `MessageProductContext`
- `Message{D, A}` → `Message{D}` (type parameter `A` removed)
- `Marginal{D, A}` → `Marginal{D}` (type parameter `A` removed)
- `Message` and `Marginal` now carry an `AnnotationDict` instead of a typed addons tuple
- Rules no longer return `(result, addons)` tuples — just the result
- `@call_rule` no longer supports `return_addons` option; use `annotations` keyword with `AnnotationDict`
- `MessageMapping.addons` field → `MessageMapping.annotations`
- `MessageProductContext` gained `annotations` field for product-time annotation processors

### Removed
- `getaddons` — use `getannotations` instead
- `getlogscale(::Message)`, `getlogscale(::Marginal)` — use `getlogscale(getannotations(...))` instead
- `getmemory`, `getmemoryaddon` — use `get_rule_input_arguments(getannotations(...))` instead
- `AddonLogScale` — replaced by `LogScaleAnnotations` (calling `AddonLogScale()` throws a descriptive error)
- `AddonMemory` — replaced by `InputArgumentsAnnotations` (calling `AddonMemory()` throws a descriptive error)
- `AddonDebug` — use callbacks instead
- `AbstractAddon`, `multiply_addons`, `@invokeaddon`
- `message_mapping_addons`, `message_mapping_addon` helper functions

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

[Unreleased]: https://github.com/ReactiveBayes/ReactiveMP.jl/compare/v5.6.6...HEAD
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
