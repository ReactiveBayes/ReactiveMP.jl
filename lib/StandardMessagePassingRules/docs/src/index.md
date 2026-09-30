# StandardMessagePassingRules

StandardMessagePassingRules holds the [rules](@extref MessagePassingRulesBase glossary-rule) of
the standard [factor nodes](@extref MessagePassingRulesBase glossary-factor-node): the
distributions a model uses most, the arithmetic functions, Boolean logic and the mixtures. Load
it next to an engine, such as ReactiveMP, and every rule is available. A model uses the nodes by
their usual names, `x ~ NormalMeanVariance(μ, v)` or `y ~ x + z`. Nodes outside this set, such
as `Delta`, `AR` or `Probit`, have packages of their own, which build on this one.

```@docs
StandardMessagePassingRules
```

!!! info "Where these rules run"
    [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) runs these rules on a factor
    graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a
    [GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl) model. The examples here call the
    rules directly, as a test does.

## A first rule call

A rule is an ordinary Julia method.
[`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) runs
one by hand, as an engine would. Here it computes the
[message](@extref MessagePassingRulesBase glossary-message) towards `out` of a
`NormalMeanVariance` node, from a normal message on the mean and a known variance, a
[point mass](@extref MessagePassingRulesBase glossary-point-mass):

```@example index
using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

@call_message_update_rule(
    node = NormalMeanVariance, target = :out,
    m = (μ = NormalMeanVariance(1.0, 1.0), v = PointMass(2.0)),
)
```

The call returns a [`RuleResult`](@extref MessagePassingRulesBase.RuleResult), drawn as a card:
the node, the inputs it read as arrows in, and the target as the arrow out. The variances add, so
the message is `NormalMeanVariance(1.0, 3.0)`.
[`getresult`](@extref MessagePassingRulesBase.getresult) returns the message itself:

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

With [marginals](@extref MessagePassingRulesBase glossary-marginal) instead of messages, the same
call selects the [variational](@extref MessagePassingRulesBase glossary-vmp) rule. That rule
replaces the variance with `1/E[1/v]` under the variance's marginal:

```jldoctest index
julia> message = getresult(@call_message_update_rule(
           node = NormalMeanVariance, target = :out,
           q = (μ = PointMass(1.0), v = GammaShapeRate(3.0, 4.0)),
       ));

julia> mean(message) ≈ 1.0 && var(message) ≈ inv(mean(inv, GammaShapeRate(3.0, 4.0)))
true
```

## Which rules a node has

The engine selects a rule by the node, the target interface and the types of the inputs. An
input is a message `m` or a marginal `q`. The
[factorisation](@extref MessagePassingRulesBase glossary-factorisation) of the approximate
posterior decides which one a rule receives. It splits a node's interfaces into
[clusters](@extref MessagePassingRulesBase glossary-cluster), and under the
[default scheme](@extref MessagePassingRulesBase glossary-default-scheme) a rule takes messages
from its target's own cluster and marginals from the other clusters. One set of rules therefore
serves three kinds of inference:

| factorisation | a rule takes | inference |
|---|---|---|
| one cluster for the whole node | messages only | [belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation) |
| a cluster per interface | marginals only | [mean-field](@extref MessagePassingRulesBase glossary-mean-field) variational message passing |
| anything in between | messages and marginals | [structured](@extref MessagePassingRulesBase glossary-structured-vmp) variational message passing |

All of them run under the one
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm).
[`rule_coverage`](@extref MessagePassingRulesBase.rule_coverage) tabulates a node's rules:

```@example index
MessagePassingRulesBase.rule_coverage(NormalMeanVariance)
```

The table has a row per message target (`→ μ`), per joint marginal (`q(out, μ)`) and one for the
[average energy](@extref MessagePassingRulesBase glossary-average-energy). It has a column per
algorithm, and each cell counts the rules. An empty row is a target that no rule reaches. Every
node section on this site shows this table, generated from the loaded rules.

[`@which_message_update_rule`](@extref MessagePassingRulesBase.@which_message_update_rule) shows
the rule a call would run, with its source.
[`list_rules`](@extref MessagePassingRulesBase.list_rules) lists all the rules of a node.

## Conventions

- **Variational expectations.** A variational rule uses the expectations that naive variational
  message passing gives. A variance contributes `1/E[1/v]` and a covariance `E[Σ⁻¹]⁻¹`, as in the
  nodes' average energies.
- **Matrix corrections.** `*`, `dot` and the rule towards `MvNormalMeanPrecision`'s `Λ` read the
  [service](@extref MessagePassingRulesBase glossary-service)
  [`matrix_correction`](@extref MessagePassingRulesBase.matrix_correction). The precision matrices
  that `*` and `dot` build may be singular. These rules correct them with MatrixCorrectionTools'
  `ReplaceZeroDiagonalEntries(tiny)` unless the context sets another correction, and
  `NoCorrection()` applies none. `MvNormalMeanPrecision` corrects nothing unless the context sets
  a correction.
- **Sampling.** The `*` rules between two general univariate distributions draw samples from the
  context's `rng`. [`MultiplicationSampling`](@ref) sets how many.
- **Log scales.** Most belief propagation rules declare the
  [log scale](@extref MessagePassingRulesBase glossary-log-scale) of their message. Most
  variational rules declare none, since the free energy does not need it. A rule's card shows
  the log scale as `undefined` when the rule declares none.

## The site

| page | contents |
|---|---|
| [Normal distributions](@ref) | `NormalMeanVariance`, `NormalMeanPrecision`, the five multivariate normals and [`HalfNormal`](@ref) |
| [Gamma, Beta and discrete distributions](@ref) | `GammaShapeRate`, `Gamma`, `GammaInverse`, `Beta`, `Bernoulli`, `Categorical`, `Dirichlet`, `DirichletCollection`, `Poisson`, `Uniform` |
| [Matrix and joint distributions](@ref) | `Wishart`, `InverseWishart`, `MatrixNormal`, `MatrixNormalWishart`, `MvNormalGamma`, `MvNormalWishart` |
| [Arithmetic](@ref) | `+`, `-`, `*`, `dot` and [`MultiplicationSampling`](@ref) |
| [Logic](@ref) | [`AND`](@ref), [`OR`](@ref), [`NOT`](@ref), [`IMPLY`](@ref) |
| [Mixtures](@ref) | [`NormalMixture`](@ref), [`GammaMixture`](@ref), [`Mixture`](@ref) and their algorithms |
| [Helper nodes and message types](@ref) | [`StandaloneDistribution`](@ref), [`Uninformative`](@ref), [`GammaShapeLikelihood`](@ref), [`diageye`](@extref MessagePassingRulesBase.diageye) |
| [Internals](@ref) | the helpers the rules and the node packages share |
