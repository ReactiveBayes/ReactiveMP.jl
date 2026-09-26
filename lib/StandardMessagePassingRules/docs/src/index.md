# StandardMessagePassingRules

StandardMessagePassingRules holds the message passing rules of the standard factor nodes: the
distributions a model writes most, the arithmetic functions, Boolean logic and the mixtures.
Load it next to an engine, such as ReactiveMP, and every rule is available; a model uses the
nodes by their usual names, `x ~ NormalMeanVariance(μ, v)`, `y ~ x + z`. Nodes outside this
set, such as `Delta`, `AR` or `Probit`, have packages of their own, which build on this one.

```@docs
StandardMessagePassingRules
```

## A first rule call

Rules are ordinary Julia methods, and
[`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) runs
one by hand, as an engine would. The message towards `out` of a `NormalMeanVariance` node, from a
normal message on its mean and a known variance, adds the variances:

```jldoctest index
julia> using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> result = @call_message_update_rule(
           node = NormalMeanVariance, target = :out,
           m = (μ = NormalMeanVariance(1.0, 1.0), v = PointMass(2.0)),
       );

julia> message = getresult(result);

julia> message isa NormalMeanVariance && mean(message) ≈ 1.0 && var(message) ≈ 3.0
true
```

With marginals instead of messages, the same call selects the variational rule, which takes
`1/E[1/v]` of the variance's marginal:

```jldoctest index
julia> message = getresult(@call_message_update_rule(
           node = NormalMeanVariance, target = :out,
           q = (μ = PointMass(1.0), v = GammaShapeRate(3.0, 4.0)),
       ));

julia> mean(message) ≈ 1.0 && var(message) ≈ inv(mean(inv, GammaShapeRate(3.0, 4.0)))
true
```

In a model, the engine picks the rule from the factorisation; nothing names it:

```julia
@model function noisy_mean(y)
    μ ~ NormalMeanVariance(0.0, 100.0)
    τ ~ GammaShapeRate(1.0, 1.0)
    y .~ NormalMeanPrecision(μ, τ)
end
```

## Which rules a node has

A rule is selected by the node, the target interface and the types of its inputs, which can be
messages `m` or marginals `q`. Under the engine's default dependency scheme each rule receives
the messages of the interfaces in its own cluster and the marginals of the other clusters, so
one node serves belief propagation (one cluster, messages only), mean-field variational message
passing (marginals only) and structured factorisations (a mix), under the one
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm).

`MessagePassingRulesBase.rule_coverage(node)` tabulates a node's rules: a row per message target
(`→ μ`), per joint marginal (`q(out, μ)`) and for the average energy, a column per algorithm, and
in each cell how many rules there are. An empty row is a target no rule reaches. Every node page
shows this table, generated from the loaded rules:

```@example index
using MessagePassingRulesBase, StandardMessagePassingRules, ExponentialFamily # hide
MessagePassingRulesBase.rule_coverage(NormalMeanVariance)
```

[`@which_message_update_rule`](@extref MessagePassingRulesBase.@which_message_update_rule) shows
the rule a call would run, with its source, and `MessagePassingRulesBase.list_rules(node)` lists
them all.

## Conventions

- **Variational rules** take the expectations naive variational message passing gives: a
  variance contributes `1/E[1/v]` and a covariance `E[Σ⁻¹]⁻¹`, as the nodes' average energies
  do.
- **Matrix corrections.** `*`, `dot` and the rule towards `MvNormalMeanPrecision`'s `Λ` read the
  context service `matrix_correction`. `*` and `dot` build precision matrices that may be
  singular, and correct them with MatrixCorrectionTools' `ReplaceZeroDiagonalEntries(tiny)`
  unless the context sets another correction; `NoCorrection()` applies none.
  `MvNormalMeanPrecision` corrects nothing unless one is set.
- **Sampling.** The `*` rules between two general univariate distributions draw from the
  context's `rng`, as many values as [`MultiplicationSampling`](@ref) says.
- **Log scales.** A belief propagation rule declares the log scale of its message; a variational
  one mostly does not, since the free energy does not need it.

## The site

| page | contents |
|---|---|
| [Normal distributions](@ref) | `NormalMeanVariance`, `NormalMeanPrecision`, the five multivariate normals and [`HalfNormal`](@ref) |
| [Gamma, Beta and discrete distributions](@ref) | `GammaShapeRate`, `Gamma`, `GammaInverse`, `Beta`, `Bernoulli`, `Categorical`, `Dirichlet`, `DirichletCollection`, `Poisson`, `Uniform` |
| [Matrix and joint distributions](@ref) | `Wishart`, `InverseWishart`, `MatrixNormal`, `MatrixNormalWishart`, `MvNormalGamma`, `MvNormalWishart` |
| [Arithmetic](@ref) | `+`, `-`, `*`, `dot` and [`MultiplicationSampling`](@ref) |
| [Logic](@ref) | [`AND`](@ref), [`OR`](@ref), [`NOT`](@ref), [`IMPLY`](@ref) |
| [Mixtures](@ref) | [`NormalMixture`](@ref), [`GammaMixture`](@ref), [`Mixture`](@ref) and their algorithms |
| [Helper nodes and message types](@ref) | [`StandaloneDistribution`](@ref), [`Uninformative`](@ref), [`GammaShapeLikelihood`](@ref), [`diageye`](@ref) |
| [Internals](@ref) | the helpers the rules and the node packages share |
