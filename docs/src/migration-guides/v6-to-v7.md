# [Migrating from v6 to v7](@id migration-v6-to-v7)

This guide is for you if you wrote nodes and rules against ReactiveMP v6, maintain a package
that does, or build graphs with the engine by hand. If you write models with RxInfer, read
RxInfer's [Migration from v5 to v6](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/migration/v5-to-v6/)
instead: it covers `@model`, `infer` and the standard nodes a model uses.

ReactiveMP v7 moves nodes and rules out of the engine into packages of their own. Most of a port
is mechanical, and this guide gives each mechanical translation as a before/after pair, in the
order a port happens. The v6 half of a pair starts with `# v6` and does not run. The v7 half runs
when these docs are built, on small nodes declared on this page. If you port code mechanically,
as a person or a tool, follow the [Checklist for a mechanical port](@ref migration-v6-to-v7-checklist)
at the end.

For the concepts behind the pairs, read
[Your first node](@extref MessagePassingRulesBase tutorial-first-node), the
[keyword reference](@extref MessagePassingRulesBase keyword-reference) and the
[glossary](@extref MessagePassingRulesBase glossary).

## What moved where

| v6, in `ReactiveMP` | v7 |
|---|---|
| variables, factor nodes, messages, marginals, the free energy | `ReactiveMP`, the engine |
| `@node`, `@rule`, `@marginalrule`, `@average_energy`, `@call_rule`, rule lookup | [`MessagePassingRulesBase`](@extref MessagePassingRulesBase MessagePassingRulesBase) |
| `@test_rules`, `@test_marginalrules` | [`MessagePassingRulesTestUtils`](@extref MessagePassingRulesTestUtils MessagePassingRulesTestUtils) |
| the standard nodes and their rules | [`StandardMessagePassingRules`](@extref StandardMessagePassingRules StandardMessagePassingRules) |
| the approximation methods | [`MessagePassingRulesApproximations`](@extref MessagePassingRulesApproximations MessagePassingRulesApproximations) |
| the Delta node | [`DeltaMessagePassingRules`](@extref DeltaMessagePassingRules DeltaMessagePassingRules) |
| every other node | a package of its own ([Node packages](@ref migration-v6-to-v7-node-packages)) |

Loading a package is enough for the engine to find its rules. These names moved with their
packages:

| v6 | v7 |
|---|---|
| `Unscented`, `UT`, `UnscentedTransform`, `Linearization`, `GaussHermiteCubature`, `ghcubature`, `AbstractApproximationMethod`, `approximation_name`, `approximation_short_name` | the same names, exported by `MessagePassingRulesApproximations`: [`Unscented`](@extref MessagePassingRulesApproximations.Unscented), [`Linearization`](@extref MessagePassingRulesApproximations.Linearization), [`GaussHermiteCubature`](@extref MessagePassingRulesApproximations.GaussHermiteCubature), [`ghcubature`](@extref MessagePassingRulesApproximations.ghcubature), [`AbstractApproximationMethod`](@extref MessagePassingRulesApproximations.AbstractApproximationMethod), [`approximation_name`](@extref MessagePassingRulesApproximations.approximation_name), [`approximation_short_name`](@extref MessagePassingRulesApproximations.approximation_short_name) |
| `DeltaFn`, `CVIProjection`, `CVISamplingStrategy`, `FullSampling`, `MeanBased` | the same names, exported by `DeltaMessagePassingRules`: [`DeltaFn`](@extref DeltaMessagePassingRules.DeltaFn), [`CVIProjection`](@extref DeltaMessagePassingRules.CVIProjection), [`CVISamplingStrategy`](@extref DeltaMessagePassingRules.CVISamplingStrategy), [`FullSampling`](@extref DeltaMessagePassingRules.FullSampling), [`MeanBased`](@extref DeltaMessagePassingRules.MeanBased); `CVIProjection`'s rules load with ExponentialFamilyProjection |
| `DeltaFnNode`, the Delta node's own node type | none: a Delta node is an ordinary `FactorNode` of `DeltaFn`, which folds its constant and data inputs into its function ([Static inputs](@ref lib-node-static-inputs)) |
| `Autoregressive`, an alias of `AR` | the same alias, exported by `AutoregressiveMessagePassingRules` |
| `GaussianMixture`, an alias of `NormalMixture` | the same alias, exported by `StandardMessagePassingRules` |
| Flow's `compile`, `nr_params`, `getlayers` | the same names, exported by `FlowMessagePassingRules` |
| Flow's `getmodel(meta)`, `getapproximation(meta)` | `FlowMessagePassingRules.getmodel(algo)` and `FlowMessagePassingRules.getmethod(algo)` on a [`FlowApproximation`](@extref FlowMessagePassingRules.FlowApproximation), internal |

## Declaring a node

A node is declared with [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node),
keyword by keyword. Aliases are written on the interface, and the first interface is the output,
as before.

```julia
# v6
@node MyGaussian Stochastic [out, (μ, aliases = [mean]), (τ, aliases = [precision])]
```

```@example v7
using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
import MessagePassingRulesBase: annotate!, getannotation

struct MyGaussian end

@define_factor_node(
    node = MyGaussian,
    type = Stochastic,
    interfaces = [:out, (:μ, aliases = [:mean]), (:τ, aliases = [:precision])],
)
```

A node with a variable number of edges of one kind, which v6 wrote by hand with `ManyOf` and a
node type of its own, declares an interface [group](@extref MessagePassingRulesBase glossary-group),
`:in...`; see [Groups](@ref migration-v6-to-v7-groups). The checks a node's constructor made
become keywords of the declaration: `matched_groups`, `min_group_length` and `factorisation`
([keyword reference](@extref MessagePassingRulesBase keyword-factor-node)).

## Message update rules

`@rule` becomes [`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule).
The node, the target and the inputs are keywords. The inputs are typed like the old arguments:
the [message](@extref MessagePassingRulesBase glossary-message) `m_x` becomes `m[:x]`, and the
[marginal](@extref MessagePassingRulesBase glossary-marginal) `q_x` becomes `q[:x]`. The body is a
lambda over the slots it needs. `Marginalisation` is gone: a rule belongs to an
[algorithm](@extref MessagePassingRulesBase glossary-algorithm), the node's default unless it says
otherwise.

```julia
# v6
@rule MyGaussian(:out, Marginalisation) (m_μ::PointMass, m_τ::PointMass) = begin
    return NormalMeanPrecision(mean(m_μ), mean(m_τ))
end

@rule MyGaussian(:out, Marginalisation) (q_μ::Any, q_τ::Any) = NormalMeanPrecision(mean(q_μ), mean(q_τ))
```

```@example v7
@define_message_update_rule(
    node = MyGaussian, target = :out,
    args = (m[:μ]::PointMass, m[:τ]::PointMass),
    body = (args) -> NormalMeanPrecision(mean(args.m[:μ]), mean(args.m[:τ])),
)

@define_message_update_rule(
    node = MyGaussian, target = :out,
    args = (q[:μ]::Any, q[:τ]::Any),
    body = (args) -> NormalMeanPrecision(mean(args.q[:μ]), mean(args.q[:τ])),
)

getresult(call_message_update_rule(MyGaussian, :out; q = (μ = NormalMeanVariance(1.0, 1.0), τ = GammaShapeRate(2.0, 1.0))))
```

A joint marginal, `q_out_μ` in v6, is `q[:out, :μ]`, its members in interface order:

```julia
# v6
@rule MyGaussian(:τ, Marginalisation) (q_out_μ::Any,) = begin
    m, V = mean_cov(q_out_μ)
    return GammaShapeRate(3 / 2, (V[1, 1] - V[1, 2] - V[2, 1] + V[2, 2] + abs2(m[1] - m[2])) / 2)
end
```

```@example v7
@define_message_update_rule(
    node = MyGaussian, target = :τ,
    args = (q[:out, :μ]::Any,),
    body = (args) -> begin
        m, V = mean_cov(args.q[:out, :μ])
        GammaShapeRate(3 / 2, (V[1, 1] - V[1, 2] - V[2, 1] + V[2, 2] + abs2(m[1] - m[2])) / 2)
    end,
)

getresult(call_message_update_rule(MyGaussian, :τ; clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.0], [1.0 0.0; 0.0 1.0]),)))
```

A rule that took `meta::MyMeta` names its algorithm instead, `algorithm = MyMeta`, and reads the
value from its `algo` slot. The algorithm is a subtype of
[`DefaultAlgorithmExtension`](@extref MessagePassingRulesBase.DefaultAlgorithmExtension), which
keeps the default's rules for every target it has none for:

```julia
# v6
struct MyScale
    factor::Float64
end

@rule MyGaussian(:μ, Marginalisation) (m_out::PointMass, m_τ::PointMass, meta::MyScale) = begin
    return NormalMeanPrecision(mean(m_out), meta.factor * mean(m_τ))
end
```

```@example v7
struct MyScale <: DefaultAlgorithmExtension
    factor::Float64
end

@define_message_update_rule(
    node = MyGaussian, target = :μ, algorithm = MyScale,
    args = (m[:out]::PointMass, m[:τ]::PointMass),
    body = (algo, args) -> NormalMeanPrecision(mean(args.m[:out]), algo.factor * mean(args.m[:τ])),
)

getresult(call_message_update_rule(MyGaussian, :μ; m = (out = PointMass(1.0), τ = PointMass(2.0)), algorithm = MyScale(0.5)))
```

What else `meta` held is covered in [`meta`: algorithm or context service](@ref migration-v6-to-v7-meta).

## Marginal rules

`@marginalrule` becomes [`@define_marginal_update_rule`](@extref MessagePassingRulesBase.@define_marginal_update_rule).
Its target is the cluster's members as a tuple rather than a joined name: `:out_μ` becomes
`(:out, :μ)`. A result that factorises into independent blocks, which v6 returned as a
NamedTuple, is a [`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster), each
block labelled with the members it covers.

```julia
# v6
@marginalrule MyGaussian(:out_μ) (m_out::PointMass, m_μ::NormalMeanPrecision, q_τ::Any) = begin
    return (out = m_out, μ = prod(ClosedProd(), NormalMeanPrecision(mean(m_out), mean(q_τ)), m_μ))
end
```

```@example v7
@define_marginal_update_rule(
    node = MyGaussian, target = (:out, :μ),
    args = (m[:out]::PointMass, m[:μ]::NormalMeanPrecision, q[:τ]::Any),
    body = (args) -> FactorizedCluster(
        (:out,) => args.m[:out],
        (:μ,) => prod(ClosedProd(), NormalMeanPrecision(mean(args.m[:out]), mean(args.q[:τ])), args.m[:μ]),
    ),
)

getresult(call_marginal_update_rule(MyGaussian, (:out, :μ); m = (out = PointMass(1.0), μ = NormalMeanPrecision(0.0, 1.0)), q = (τ = PointMass(1.0),)))
```

## Average energies

`@average_energy` becomes [`@define_average_energy`](@extref MessagePassingRulesBase.@define_average_energy),
with the same argument syntax as the rules.

```julia
# v6
@average_energy MyGaussian (q_out::Any, q_μ::Any, q_τ::Any) = begin
    return (log(2π) - mean(log, q_τ) + mean(q_τ) * (var(q_out) + var(q_μ) + abs2(mean(q_out) - mean(q_μ)))) / 2
end
```

```@example v7
@define_average_energy(
    node = MyGaussian,
    args = (q[:out]::Any, q[:μ]::Any, q[:τ]::Any),
    body = (args) -> (log(2π) - mean(log, args.q[:τ]) + mean(args.q[:τ]) * (var(args.q[:out]) + var(args.q[:μ]) + abs2(mean(args.q[:out]) - mean(args.q[:μ])))) / 2,
)

getresult(call_average_energy(MyGaussian; q = (out = PointMass(1.0), μ = PointMass(0.0), τ = PointMass(1.0))))
```

## Log scales

A message's [log scale](@extref MessagePassingRulesBase glossary-log-scale) is part of the
message, not an annotation: it is the scalar with `message = exp(logscale) · distribution` (see
[Log scales](@ref lib-logscale)). A rule declares it with the `logscale` keyword instead of
writing it in its body:

- `@logscale v` becomes `logscale = v` for a constant;
- `logscale = (args) -> …` computes it from the inputs;
- `logscale = from_body` takes it from the body, which returns
  [`with_logscale`](@extref MessagePassingRulesBase.with_logscale)`(result, v)`.

A rule that reads the log scales its inputs arrived with, which v6 read from the raw `messages`
tuple, declares `reads_logscale = true` and reads `args.logscale.m[:x]`
([keyword reference](@extref MessagePassingRulesBase keyword-message-logscale)).

```julia
# v6
@rule MyBernoulli(:p, Marginalisation) (m_out::PointMass,) = begin
    @logscale -log(2)
    return Beta(1 + mean(m_out), 2 - mean(m_out))
end
```

```@example v7
struct MyBernoulli end

@define_factor_node(node = MyBernoulli, type = Stochastic, interfaces = [:out, :p])

@define_message_update_rule(
    node = MyBernoulli, target = :p,
    args = (m[:out]::PointMass,),
    logscale = -log(2),
    body = (args) -> Beta(1 + mean(args.m[:out]), 2 - mean(args.m[:out])),
)

result = call_message_update_rule(MyBernoulli, :p; m = (out = PointMass(1.0),))
getresult(result), getlogscale(result)
```

The engine tracks log scales with the activation option `logscales = true`, which replaces
`LogScaleAnnotations`. RxInfer's `infer(...; logscales = true)` replaces
`annotations = LogScaleAnnotations()`, and `getlogscale(getannotations(q))` becomes
`getlogscale(q)`.

A rule that declares no log scale does not make inference fail. Its message's log scale is an
[`UndefinedLogScale`](@extref MessagePassingRulesBase.UndefinedLogScale) saying why, which
propagates through products. Only a rule or a user that needs the number gets an error. v6's
fallback, which set zero whenever every input was a point mass, is gone, since it was not right
for every rule; the rules it covered declare their log scale.

## [Groups](@id migration-v6-to-v7-groups)

A variable number of edges of one kind, `ManyOf` in v6, is an interface group. A rule towards a
member names it `(:in, k)`. Its inputs select members: `m[:in...]` takes all of them, `m[:in][k]`
the target's own, and `m[:in][!k]` all but it. A group arrives as a tuple in member order, with
`nothing` for a member the rule does not take. The node needs no node type, `factornode` or
`activate!` of its own, and no `{N}` parameter: the number of members is the group's length.
[Groups](@extref MessagePassingRulesBase tutorial-groups) builds such a node step by step.

```julia
# v6
@rule MySum{N}(:out, Marginalisation) (m_in::ManyOf{N, NormalMeanVariance},) where {N} = begin
    return NormalMeanVariance(sum(mean, m_in), sum(var, m_in))
end
```

```@example v7
struct MySum end

@define_factor_node(node = MySum, type = Deterministic, interfaces = [:out, :in...])

@define_message_update_rule(
    node = MySum, target = :out,
    args = (m[:in...]::NormalMeanVariance,),
    body = (args) -> NormalMeanVariance(sum(mean, args.m[:in]), sum(var, args.m[:in])),
)

@define_message_update_rule(
    node = MySum, target = (:in, k),
    args = (m[:out]::NormalMeanVariance, m[:in][!k]::NormalMeanVariance),
    body = (args) -> begin
        others = [m for m in args.m[:in] if m !== nothing]
        NormalMeanVariance(mean(args.m[:out]) - sum(mean, others), var(args.m[:out]) + sum(var, others))
    end,
)

getresult(call_message_update_rule(MySum, (:in, 1); m = (out = NormalMeanVariance(3.0, 1.0), in = (nothing, NormalMeanVariance(2.0, 1.0)))))
```

A group may be empty where the node declares `min_group_length = 0`. A joint may hold some of a
group's members with other interfaces. It is keyed with the members, `(:out, (:T, 1))`, and read
as `q[:out, (:T, 1)]`. A joint holding every member of a group names the group once, so with one
member, `T₁` alone, the joint over `out`, `in` and it is `(:out, :in, :T)`: a rule written for
`(:out, :in, (:T, 1))` is valid but never runs in that graph. v6 handed such joints to hand-written `rule` methods under mangled names,
`q_out_T1`, which the rule parsed. A rule that takes whatever inputs the factorisation delivers,
as those did, is written with `default` in its arguments. It walks its inputs by key with
[`MessagePassingRulesBase.rule_inputs`](@extref): an interface by its name, a member as `(:T, k)`,
a joint by its key.

```julia
# v6
function ReactiveMP.rule(::Type{<:MyTensor}, ::Val{:out}, ::Marginalisation, mnames, messages, qnames, marginals, meta, annotations, node)
    # split names such as `:in_T1` on `_` to find which edges each input covers
end
```

```@example v7
struct MyTensor end

@define_factor_node(node = MyTensor, type = Stochastic, interfaces = [:out, :in, :T...], min_group_length = 0)

@define_message_update_rule(
    node = MyTensor, target = :out, args = (default,),
    body = (args) -> sum(last, MessagePassingRulesBase.rule_inputs(MyTensor, args.q)),
)

# Under q(out) q(in, T1) q(T2): the joint by its key, and the second member of `T`.
getresult(call_message_update_rule(MyTensor, :out; clusters = ((:in, (:T, 1)) => 0.5,), q = (T = (nothing, 1.0),)))
```

A rule with typed inputs for the same node, target and algorithm is more specific than one with
`default`, and wins wherever its inputs fit; the `default` rule takes the rest. A model adds a rule
for a packaged node this way, for inputs the package's `default` rule does not handle as it needs:

```@example v7
@define_message_update_rule(node = MyTensor, target = :out, args = (q[:in]::Float64,), body = (args) -> -args.q[:in])

# The typed rule fits, so it wins; with a joint the `default` rule still answers.
getresult(call_message_update_rule(MyTensor, :out; q = (in = 2.0,))),
getresult(call_message_update_rule(MyTensor, :out; clusters = ((:in, (:T, 1)) => 0.5,), q = (T = (nothing, 1.0),)))
```

### [A node with a group and its own dependencies](@id migration-v6-to-v7-gate)

A node that v6 wrote as an engine node type, with its own `factornode`, functional dependencies
and `activate!`, such as a gate that passes on the input a switch selects, is a declaration in v7.
The group holds the inputs, the algorithm declares what each rule reads, and the joint over the
inputs, which a deterministic node's free energy takes, is a
[`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster) when its blocks are
independent:

```julia
# v6
struct GateNode{N} <: AbstractFactorNode
    out::NodeInterface
    switch::NodeInterface
    inputs::NTuple{N, IndexedNodeInterface}
end
# … a `factornode` method, functional dependencies, `collect_latest_messages`, `activate!` …
```

```@example v7
struct MyGate end                                     # out = inputs[switch]
struct MyGateMP <: MessagePassingRulesBase.AbstractAlgorithm end

@define_factor_node(
    node = MyGate, type = Deterministic,
    interfaces = [:out, :switch, :inputs...],
    algorithm = MyGateMP,
    dependencies = [
        :out => (m[:inputs...], q[:switch]),
        :switch => (m[:out], m[:inputs...]),
        (:inputs, k) => (m[:out], q[:switch]),
    ],
)

# Towards `out`: the inputs' messages weighted by the switch.
@define_message_update_rule(
    node = MyGate, target = :out,
    args = (m[:inputs...]::NormalMeanVariance, q[:switch]::Categorical),
    body = (args) -> BayesBase.MixtureDistribution(collect(args.m[:inputs]), probs(args.q[:switch])),
)

# Towards `switch`: how well each input explains `out`'s message, from the log scale of their product.
@define_message_update_rule(
    node = MyGate, target = :switch,
    args = (m[:out]::NormalMeanVariance, m[:inputs...]::NormalMeanVariance),
    body = (args) -> begin
        logweights = [BayesBase.compute_logscale(prod(ClosedProd(), args.m[:out], input), args.m[:out], input) for input in args.m[:inputs]]
        weights = exp.(logweights .- maximum(logweights))
        Categorical(weights ./ sum(weights))
    end,
)

# Towards input k: `out`'s message raised to the probability that the switch selects k.
@define_message_update_rule(
    node = MyGate, target = (:inputs, k),
    args = (m[:out]::NormalMeanVariance, q[:switch]::Categorical),
    body = (args) -> begin
        ξ, w = weightedmean_precision(args.m[:out])
        z = probs(args.q[:switch])[k]
        NormalWeightedMeanPrecision(z * ξ, z * w)
    end,
)

# The joint over the switch and the inputs: their messages, independent.
@define_marginal_update_rule(
    node = MyGate, target = (:switch, :inputs),
    args = (m[:out]::Any, m[:switch]::Categorical, m[:inputs...]::NormalMeanVariance),
    body = (args) -> FactorizedCluster((:switch,) => args.m[:switch], (:inputs,) => BayesBase.FactorizedJoint(args.m[:inputs])),
)

inputs = (NormalMeanVariance(0.0, 1.0), NormalMeanVariance(5.0, 1.0))
getresult(call_message_update_rule(MyGate, :switch; m = (out = NormalMeanVariance(4.5, 1.0), inputs = inputs)))
```

The joint's key names the group once, `(:switch, :inputs)`, since it holds every member. A rule
that needs v6's per-node schedule declares it this way; one that reads what the default scheme
delivers declares nothing, or `default` beside the inputs it adds.

## [`meta`: algorithm or context service](@id migration-v6-to-v7-meta)

`meta` served two purposes, and each has its own place.

- **What a rule computes**, such as an approximation method, is an **algorithm**: a type
  `struct MyMethod <: AbstractAlgorithm end`, possibly with fields, that the rule names with
  `algorithm = MyMethod` and receives in its `algo` slot. A direct subtype of
  [`AbstractAlgorithm`](@extref MessagePassingRulesBase.AbstractAlgorithm) stands alone; a
  subtype of `DefaultAlgorithmExtension` inherits the default's rules, as `MyScale` above does.
  The node's user chooses the algorithm per node, as they chose the meta.
  `DeltaMeta(method = Unscented())` is
  [`DeltaApproximation`](@extref DeltaMessagePassingRules.DeltaApproximation)`(method = Unscented())`,
  and `DeltaMeta(method = Linearization(), inverse = f⁻¹)` is
  `DeltaApproximation(method = Linearization(), inverse = f⁻¹)`.
  [A node with its own algorithm](@extref MessagePassingRulesBase tutorial-algorithm) builds
  one step by step.
- **How a rule computes it numerically**, such as a matrix correction or a random number
  generator, is a **context service** ([service](@extref MessagePassingRulesBase glossary-service)).
  The rule declares it with `ctx = (:matrix_correction,)` and reads it as
  [`matrix_correction`](@extref MessagePassingRulesBase.matrix_correction)`(ctx, default)` or
  `ctx.rng`. A generator that lived in a meta is given to the nodes as the activation option
  `context`, `(rng = …,)` ([Services: the context](@ref lib-activation-options-context)).

A `default_meta` becomes the rule's `default`: the node's `algorithm` keyword for a default
algorithm, and the `default` of `matrix_correction(ctx, default)` for a default correction.

```julia
# v6
struct MySampling
    rng::AbstractRNG
end

@rule MyBernoulli(:out, Marginalisation) (q_p::Beta, meta::MySampling) = begin
    return Bernoulli(mean(rand(meta.rng, q_p, 100)))
end
```

```@example v7
using Random

@define_message_update_rule(
    node = MyBernoulli, target = :out,
    args = (q[:p]::Beta,), ctx = (:rng,),
    body = (ctx, args) -> Bernoulli(mean(rand(ctx.rng, args.q[:p], 100))),
)

getresult(call_message_update_rule(MyBernoulli, :out; q = (p = Beta(2.0, 2.0),), ctx = MessagePassingRulesBase.RuleContext(rng = Xoshiro(1))))
```

The engine supplies `rng`, `node` and `matrix_correction`, and the activation option `context`
adds to them or overrides them (see [`RuleContext`](@extref MessagePassingRulesBase.RuleContext)).

## Functional dependencies

v6's functional dependencies become declared
[dependencies](@extref MessagePassingRulesBase glossary-dependencies). The
[default scheme](@extref MessagePassingRulesBase glossary-default-scheme), which gives a rule the
messages in its own cluster and the marginals of the others, needs no declaration.

| v6 | v7 |
|---|---|
| `DefaultFunctionalDependencies` | nothing: the default |
| a node with its own `functional_dependencies` | an algorithm of the node's own, with `dependencies = [...]` on [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node) |
| `RequireMessageFunctionalDependencies`, `RequireMarginalFunctionalDependencies`, `RequireEverythingFunctionalDependencies` | for one model, a [`DefaultAlgorithmExtension`](@extref MessagePassingRulesBase.DefaultAlgorithmExtension) with its own dependencies ([`@define_dependencies`](@extref MessagePassingRulesBase.@define_dependencies)); for a node, its own algorithm |
| `RequireMarginalFunctionalDependencies(a = nothing)`, keeping the default and adding `q(a)` | `:a => (default, q[:a])`, with `default` alone for the other targets ([`@define_dependencies`](@extref MessagePassingRulesBase.@define_dependencies)) |
| RxInfer's `where { dependencies = RequireMessageFunctionalDependencies(in = d) }`, a start for the message on this node's own edge | RxInfer's `where { initial_messages = (in = d,) }`, the activation option `initial_messages` ([Initial messages](@ref lib-activation-options-initial-messages)); `@initialization μ(x) = d` starts the messages on every edge of `x` instead, which is not the same |
| RxInfer's `where { dependencies = … }`, otherwise | choosing the node's algorithm |

A target's inputs are subscribed to in the order they are declared, which is the update
schedule in [variational message passing](@extref MessagePassingRulesBase glossary-vmp).

## Calling rules

`@call_rule` returned the message, or a tuple with its add-ons under an option. Every call
returns a [`RuleResult`](@extref MessagePassingRulesBase.RuleResult), the same shape whatever the
rule. [`getresult`](@extref MessagePassingRulesBase.getresult)`(result)` is the message and
`getlogscale(result)` its log scale. [`getrule`](@extref MessagePassingRulesBase.getrule),
`MessagePassingRulesBase.getalgorithm` and the other getters say what produced it.

| v6 | v7 |
|---|---|
| `@call_rule Node(:out, Marginalisation) (m_x = …,)` | [`getresult`](@extref MessagePassingRulesBase.getresult)`(call_message_update_rule(Node, :out; m = (x = …,)))`, or [`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) |
| `@call_rule … (…, meta = x)` | `call_message_update_rule(…; algorithm = x)` |
| `@call_rule typeof(f)(:out, Marginalisation) (…)`, for a function node | the node is the function itself, `call_message_update_rule(f, :out; …)`; `typeof(f)` is an `ArgumentError` saying so |
| `@call_marginalrule` | [`call_marginal_update_rule`](@extref MessagePassingRulesBase.call_marginal_update_rule), [`@call_marginal_update_rule`](@extref MessagePassingRulesBase.@call_marginal_update_rule) |
| `score(AverageEnergy(), Node, Val{…}(), marginals, meta)` | [`call_average_energy`](@extref MessagePassingRulesBase.call_average_energy)`(Node; q = …)` |
| a rule calling another rule | the same: the rule calls the other one, forwarding its `ctx` ([Delegating to another rule](@ref migration-v6-to-v7-delegate)) |

The calls for the standard nodes that a model author meets, such as
`@call_rule typeof(+)(:out, Marginalisation) (…)`, are in RxInfer's
[migration guide](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/migration/v5-to-v6/).

### [Delegating to another rule](@id migration-v6-to-v7-delegate)

A rule may compute its message with another node's rule, a packaged one included, as a v6 rule
did with `@call_rule`. It calls the other rule with the inputs it builds and forwards its own
context, so the other rule sees the engine's services and the model's:

```julia
# v6
@rule MyShiftedSum(:out, Marginalisation) (m_in::ManyOf{N, NormalMeanVariance},) where {N} = begin
    s = @call_rule MySum(:out, Marginalisation) (m_in = m_in,)
    return NormalMeanVariance(mean(s) + 1.0, var(s))
end
```

```@example v7
struct MyShiftedSum end

@define_factor_node(node = MyShiftedSum, type = Deterministic, interfaces = [:out, :in...])

@define_message_update_rule(
    node = MyShiftedSum, target = :out,
    args = (m[:in...]::NormalMeanVariance,),
    body = (ctx, args) -> begin
        s = getresult(call_message_update_rule(MySum, :out; m = (in = args.m[:in],), ctx))
        NormalMeanVariance(mean(s) + 1.0, var(s))
    end,
)

getresult(call_message_update_rule(MyShiftedSum, :out; m = (in = (NormalMeanVariance(1.0, 1.0), NormalMeanVariance(2.0, 3.0)),)))
```

The keyword call builds its arguments from named tuples, which allocates. Where that matters, the
positional form allocates nothing:
[`message_passing_rule`](@extref MessagePassingRulesBase.message_passing_rule)`(MySum, Target(:out), DefaultAlgorithm(), RuleArgs(m = (in = args.m[:in],)), ctx)`,
and `message_passing_marginalrule` and `message_passing_average_energy` for the other kinds. The
other rule runs under the algorithm the call names, the node's default unless `algorithm` says
otherwise.

These helpers and extension points have new names or homes:

| v6 | v7 |
|---|---|
| `to_marginal(d)` | [`public_equivalent`](@extref MessagePassingRulesBase.public_equivalent)`(d)`, a method a package adds for its working types |
| `getnodefn(node)`, `getnode()` in a rule | [`getnodefn`](@extref MessagePassingRulesBase.getnodefn)`(ctx.node, target)`, `ctx.node` |
| `nodefunction(node, meta, Val(:out))`; a known inverse as `nodefunction(node, meta, (Val(:in), k))` | `getnodefn(ctx.node, Target(:out))`; an inverse is the algorithm's, read from `algo` |
| `ReactiveMP.rank1update`, `mul_trace`, `negate_inplace!`, `mul_inplace!`, `v_a_vT` | MessagePassingRulesBase's [math helpers](@extref MessagePassingRulesBase math-helpers): [`add_outer`](@extref MessagePassingRulesBase.add_outer), [`trace_product`](@extref MessagePassingRulesBase.trace_product), [`negate!!`](@extref MessagePassingRulesBase.negate!!), [`scale!!`](@extref MessagePassingRulesBase.scale!!), [`scaled_outer`](@extref MessagePassingRulesBase.scaled_outer); public, not exported |
| `ReactiveMP.approximate(method, f, (d,))`, a distribution | [`approximate`](@extref MessagePassingRulesApproximations Moments-through-a-function)`(method, f, (mean(d),), (cov(d),))`, the mean and covariance, from `MessagePassingRulesApproximations` |
| `StandardBasisVector(n, k)`, as in `softdot(x, StandardBasisVector(n, k), γ)` | [`AutoregressiveMessagePassingRules.StandardBasisVector`](@extref)`(n, k)`, public in that package: the rules' products with it read one entry, so it gives a dense one-hot vector's messages faster |
| `ReactiveMP.MatrixCorrectionTools`, and a correction given as a node's meta, `*() -> ClampSingularValues(…)` | the registered package MatrixCorrectionTools; the correction is the service `matrix_correction` of the activation option `context`, `(matrix_correction = ClampSingularValues(…),)`, which reaches every rule of the node that reads it |
| `diageye` | the same, [`MessagePassingRulesBase.diageye`](@extref), exported by StandardMessagePassingRules |
| [`NodeFunctionRuleFallback`](@extref MessagePassingRulesBase.NodeFunctionRuleFallback)`()` as the engine's `rulefallback` | the same, from `MessagePassingRulesBase`, as the activation option `rulefallback` ([Rule fallbacks](@ref lib-activation-options-rulefallback)); its message is a [`NodeFunctionLogPdf`](@extref MessagePassingRulesBase.NodeFunctionLogPdf) |
| `RuleMethodError`, `MarginalRuleMethodError` | [`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError), for every kind of rule |

## Testing rules

Table tests move to `MessagePassingRulesTestUtils` and take keywords. A case is the inputs of
one call paired with the expected result.

| v6 | v7 |
|---|---|
| `@test_rules [opts] Node(:out, Marginalisation) [(input = (m_x = …,), output = d), …]` | [`@test_message_update_rule`](@extref MessagePassingRulesTestUtils.@test_message_update_rule)`(node = Node, target = :out, cases = [(m = (x = …,),) => d, …], opts…)` |
| `@test_marginalrules [opts] Node(:out_μ) [(input = …, output = d), …]` | [`@test_marginal_update_rule`](@extref MessagePassingRulesTestUtils.@test_marginal_update_rule)`(node = Node, target = (:out, :μ), cases = […], opts…)` |
| an average energy checked by hand | [`@test_average_energy`](@extref MessagePassingRulesTestUtils.@test_average_energy)`(node = Node, cases = [(q = …,) => value, …])` |
| the options `atol`, `rtol`, `check_type_promotion` (default `false`) | the same keywords; `check_type_promotion` defaults to `true` |
| the option `extra_float_types` | `float_types`, with the same default, `(Float32, Float64, BigFloat)` |
| nothing | [`check_rule_coverage`](@extref MessagePassingRulesTestUtils.check_rule_coverage), which fails a suite that leaves a rule untested |

## [What cannot be translated mechanically](@id migration-v6-to-v7-manual)

These need a person who knows what the rule means.

- **Rules reading the raw `messages` or `marginals` tuple**, often for their annotations or log
  scales. Each use has to be matched to a named input, to `ann.m`/`ann.q` or to
  `args.logscale.m`, which depends on what the rule meant by the index.
- **Rules building graph objects**, such as a `randomvar` for a product with a log scale. The log
  scale of a product of two distributions is `BayesBase.compute_logscale` of it, as `Mixture`'s
  rule towards its switch computes it; anything else has no counterpart.
- **`meta` used as mutable workspace**, such as a cache filled across calls. A rule is pure unless
  it says `pure = false`. State belongs to an algorithm that declares itself impure, and whether
  that is right depends on the model. Working memory that carries nothing between calls is the
  rule's `scratch`. State that several nodes share, as one meta given to them did, is one
  algorithm object given to each (below).

### [State shared by several nodes](@id migration-v6-to-v7-shared-state)

An algorithm is any value, so a mutable one holds state, and every node given the same object
reads and writes the same state: in RxInfer, `where { algorithm = shared }` on each node, or one
`@algorithm` block naming them. It declares itself impure, which the purity audit then reports
for every rule under it. Keeping the state consistent is the algorithm's job, as it was the
meta's: the engine computes each message when a variable needs it, in the order of the update
schedule, and promises no other order between the nodes.

```julia
# v6
mutable struct MyTally
    calls::Int
end

@rule MyTallyA(:out, Marginalisation) (q_in::Any, meta::MyTally) = (meta.calls += 1; q_in)
@rule MyTallyB(:out, Marginalisation) (q_in::Any, meta::MyTally) = (meta.calls += 1; q_in)
```

```@example v7
mutable struct MyTally <: MessagePassingRulesBase.AbstractAlgorithm
    calls::Int
end
MessagePassingRulesBase.ispure(::Type{<:MyTally}) = false

struct MyTallyA end
struct MyTallyB end
@define_factor_node(node = MyTallyA, type = Stochastic, interfaces = [:out, :in])
@define_factor_node(node = MyTallyB, type = Stochastic, interfaces = [:out, :in])

for node in (MyTallyA, MyTallyB)
    @eval @define_message_update_rule(
        node = $node, target = :out, algorithm = MyTally,
        args = (q[:in]::Any,),
        body = (algo, args) -> (algo.calls += 1; args.q[:in]),
    )
end

shared = MyTally(0)
getresult(call_message_update_rule(MyTallyA, :out; q = (in = 1.0,), algorithm = shared))
getresult(call_message_update_rule(MyTallyB, :out; q = (in = 2.0,), algorithm = shared))
shared.calls
```

## [Verifying a port](@id migration-v6-to-v7-verify)

Check a ported rule three ways:

1. A table of cases, [`@test_message_update_rule`](@extref MessagePassingRulesTestUtils.@test_message_update_rule),
   with values derived by hand or taken from the old tests.
2. Where the inputs allow, [`@verify_message_update_rule`](@extref MessagePassingRulesTestUtils.@verify_message_update_rule),
   which checks a message against the node's definition.
3. While the old implementation is at hand,
   [`compare_with_reference`](@extref MessagePassingRulesTestUtils.compare_with_reference) on the
   same inputs: every difference is either a bug in the port or a declared correction with its
   reason.

## Building a graph by hand

The engine's calls keep their shape. A factorisation names interfaces instead of positions, and
the activation options are keywords. [Getting started](@ref getting-started) builds a whole graph
this way.

```julia
# v6
node = factornode(NormalMeanVariance, [(:out, y), (:μ, x), (:v, v)], ((1,), (2,), (3,)))
activate!(node, FactorNodeActivationOptions(meta, DefaultFunctionalDependencies(), nothing, nothing, nothing, nothing))
```

```@example engine
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
import ReactiveMP: activate!, FactorNodeActivationOptions
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))   # declares the node `Gaussian`

x, y, v = randomvar(label = :x), datavar(label = :y), constvar(1.0)
node = factornode(Gaussian, [(:out, y), (:μ, x), (:v, v)], ((:out,), (:μ,), (:v,)))
```

```@example engine
activate!(x, RandomVariableActivationOptions())
activate!(y, DataVariableActivationOptions())
activate!(node, FactorNodeActivationOptions(; logscales = true))
```

- **[`factornode`](@ref)** takes its factorisation by interface name, `((:out, :μ), (:v,))`,
  where v6 took positions, `((1, 2), (3,))`. A group member is `(:in, k)`
  ([Creating a node](@ref lib-node-create)).
- **[`ReactiveMP.FactorNodeActivationOptions`](@ref)** was positional,
  `(metadata, dependencies, postprocessor, annotations, rulefallback, callbacks)`. It takes the
  keywords `algorithm`, `postprocessor`, `annotations`, `callbacks`, `diagnostics`, `context`,
  `rulefallback`, `logscales` and `initial_messages`, each with a default, and a positional form
  of the first four, `(algorithm, postprocessor, annotations, callbacks)`. `metadata` becomes
  `algorithm`, and `dependencies` becomes the algorithm's declaration
  ([Activation options](@ref lib-activation-options)).
- **[`bethe_free_energy`](@ref)**`(Float64, nodes, variables)` is the stream of the free energy of
  an activated graph. v6 left the sum to its caller, which combined each node's
  `score(T, FactorBoundFreeEnergy(), node, meta, postprocessors)` and each variable's
  `VariableBoundEntropy`, as RxInfer's `BetheFreeEnergy` did
  ([Computing the free energy](@ref lib-score-bethe-stream)). The node's score takes the
  node's algorithm in the place of `meta`.
- **`factorisation`, `localmarginals` and `localmarginalnames`**, which v6 exported without a
  method, are gone. [`ReactiveMP.getlocalclusters`](@ref)`(node)` and
  [`ReactiveMP.get_node_local_marginals`](@ref) give a node's clusters. `AverageEnergy` is gone
  too: [`call_average_energy`](@extref MessagePassingRulesBase.call_average_energy) computes an
  average energy.

## Features without a v6 form

These have no v6 name to translate. The left column says what served the purpose in v6, if
anything.

| v6 | v7 |
|---|---|
| nothing | [in-place rules](@extref MessagePassingRulesBase glossary-in-place-rule), `inplace = true` with `preallocate`, which write the result into a buffer ([keyword reference](@extref MessagePassingRulesBase keyword-message-inplace)) |
| `meta` as a cache | `scratch`, working memory the engine keeps per outbound stream and the rule writes before it reads ([scratch](@extref MessagePassingRulesBase glossary-scratch)) |
| nothing | `pure`, a rule's declared purity, which overrides its algorithm's [`ispure`](@extref MessagePassingRulesBase.ispure) ([keyword reference](@extref MessagePassingRulesBase keyword-message-pure)) |
| `RequireMessageFunctionalDependencies(in = d)` as a start | `initial_messages` on [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node), set where the graph sets none ([keyword reference](@extref MessagePassingRulesBase keyword-node-initial_messages)) |
| `DeltaFnNode`'s handling of constants and data | `static_inputs = :fold` on any deterministic node ([Static inputs](@ref lib-node-static-inputs)) |
| the factorisation alone | `free_energy_partition` on [`@define_dependencies`](@extref MessagePassingRulesBase.@define_dependencies), the clusters an algorithm's free energy splits into ([keyword reference](@extref MessagePassingRulesBase keyword-dependencies-free_energy_partition)) |
| nothing | [`which_message_update_rule`](@extref MessagePassingRulesBase.which_message_update_rule), [`which_marginal_update_rule`](@extref MessagePassingRulesBase.which_marginal_update_rule) and [`which_average_energy`](@extref MessagePassingRulesBase.which_average_energy), which resolve a rule without running it |
| nothing | [`rule_coverage`](@extref MessagePassingRulesBase.rule_coverage), a node's rules by target and algorithm, and [`check_rules`](@extref MessagePassingRulesBase.check_rules), which checks rules against their nodes' declarations |
| `RuleMethodError`'s "Possible fix, define:" | the not-found report of a [`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError): the closest rules, why each does not fit, and a `what to try` line |
| nothing | [`rule_not_found_hint`](@extref MessagePassingRulesBase.rule_not_found_hint), a sentence a node's package adds to that report |
| nothing | [`ReactiveMP.EngineDiagnostics`](@ref), the activation option `diagnostics`: audits of purity, in-place coverage and scratch reuse ([Diagnostics](@ref lib-activation-options-diagnostics)) |

## [Node packages](@id migration-v6-to-v7-node-packages)

The nodes that are not standard have a package each, loaded next to the engine: loading it is
enough for the engine to find its rules. Their v6 `meta` is the node's own algorithm:

| v6 | v7 |
|---|---|
| `GaussianCoupling` | [GaussianCouplingMessagePassingRules](https://reactivebayes.github.io/GaussianCouplingMessagePassingRules.jl/dev/), no algorithm of its own |
| `Probit`, `ProbitMeta(p)` | [ProbitMessagePassingRules](https://reactivebayes.github.io/ProbitMessagePassingRules.jl/dev/), [`ProbitEP`](@extref ProbitMessagePassingRules.ProbitEP)`(; p = 32)` |
| `GCV`, `GCVMetadata(GaussHermiteCubature(n))` | [GCVMessagePassingRules](https://reactivebayes.github.io/GCVMessagePassingRules.jl/dev/), [`GCVApproximation`](@extref GCVMessagePassingRules.GCVApproximation)`(; method = GaussHermiteCubature(n))` |
| `AR`, `ConjugateAR`, `ARMeta(form, order, stype)` | [AutoregressiveMessagePassingRules](https://reactivebayes.github.io/AutoregressiveMessagePassingRules.jl/dev/), [`ARVMP`](@extref AutoregressiveMessagePassingRules.ARVMP)`(form, order, stype)` with [`ARsafe`](@extref AutoregressiveMessagePassingRules.ARsafe)`()` or [`ARunsafe`](@extref AutoregressiveMessagePassingRules.ARunsafe)`()` |
| `SoftDot` (`softdot`) | [SoftDotMessagePassingRules](https://reactivebayes.github.io/SoftDotMessagePassingRules.jl/dev/), no algorithm of its own |
| `ContinuousTransition` (`CTransition`), `CTMeta(f)` or `ContinuousTransitionMeta(f)` | [ContinuousTransitionMessagePassingRules](https://reactivebayes.github.io/ContinuousTransitionMessagePassingRules.jl/dev/), [`CTVMP`](@extref ContinuousTransitionMessagePassingRules.CTVMP)`(f)` |
| `BinomialPolya`, `BinomialPolyaMeta(n, rng)` | [PolyaMessagePassingRules](https://reactivebayes.github.io/PolyaMessagePassingRules.jl/dev/), [`BinomialPolyaApproximation`](@extref PolyaMessagePassingRules.BinomialPolyaApproximation)`(; samples = n)`, drawing from the engine's generator |
| `MultinomialPolya`, `MultinomialPolyaMeta(points)` | [PolyaMessagePassingRules](https://reactivebayes.github.io/PolyaMessagePassingRules.jl/dev/), [`MultinomialPolyaApproximation`](@extref PolyaMessagePassingRules.MultinomialPolyaApproximation)`(; points)` |
| `BIFM`, `BIFMHelper`, `BIFMMeta(A, B, C)` | [BIFMMessagePassingRules](https://reactivebayes.github.io/BIFMMessagePassingRules.jl/dev/), [`BIFMSmoother`](@extref BIFMMessagePassingRules.BIFMSmoother)`(A, B, C)` |
| `Flow`, `FlowMeta(model, approximation)` | [FlowMessagePassingRules](https://reactivebayes.github.io/FlowMessagePassingRules.jl/dev/), [`FlowApproximation`](@extref FlowMessagePassingRules.FlowApproximation)`(model; method = approximation)` |
| `DiscreteTransition` | [DiscreteTransitionMessagePassingRules](https://reactivebayes.github.io/DiscreteTransitionMessagePassingRules.jl/dev/), no algorithm of its own |

- **Probit** declared `RequireMessageFunctionalDependencies(in = NormalMeanPrecision(0, 100))`. Its
  algorithm declares that the rule towards `in` reads the message on its own edge, and the
  node declares that message's start, set only where the model sets none. A model that chose
  another initial message keeps it. v6's rules without expectation propagation run under
  [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm)`()`.
- **GCV**'s [`ExponentialLinearQuadratic`](@extref GCVMessagePassingRules.ExponentialLinearQuadratic) is exported by its package, which also holds the rules
  that let `NormalMeanVariance` and `NormalMeanPrecision` take one on `out`.
- **AR** and **ConjugateAR** declare no algorithm, as v6 had no default `ARMeta`. A model gives
  each node [`ARVMP`](@extref AutoregressiveMessagePassingRules.ARVMP)`(...)`, and without one no rule is found. ConjugateAR's marginal over `w` alone
  is the engine's product of its messages, as for any single interface.
- **SoftDot** does not need the AR package; loading its own package is enough.
- **ContinuousTransition** declares no algorithm, as v6 had no default `CTMeta`: a model gives
  each node [`CTVMP`](@extref ContinuousTransitionMessagePassingRules.CTVMP)`(f)`. Its rule towards `a` still reads `q(a)`, as
  `RequireMarginalFunctionalDependencies(a = nothing)` made it, through its declaration
  ([`@define_dependencies`](@extref MessagePassingRulesBase.@define_dependencies)). As in v6, the node sets no
  initial `q(a)`; a model initialises it.
- **BinomialPolya** and **MultinomialPolya** read the message on their weights' own edge, which v6
  required each model to ask for with `where { dependencies = RequireMessageFunctionalDependencies(β
  = …) }`. The nodes declare it; drop the `dependencies` and initialise the message instead,
  `μ(β) = …` in RxInfer's `@initialization`. Their package is GPL-3 licensed, through
  PolyaGammaHybridSamplers.
- **BIFM** keeps nothing between calls. v6's `BIFMMeta` was a cache its rules shared, so they had
  to run in a set order and each node needed a meta of its own. [`BIFMSmoother`](@extref BIFMMessagePassingRules.BIFMSmoother) is an ordinary
  value that nodes may share, and the order the posteriors are subscribed in does not matter.
  `BIFMMeta(A, B, C, μu, Σu)` has no counterpart: the input's statistics come from its message.
- **Flow**'s models, layers and [`PermutationMatrix`](@extref FlowMessagePassingRules.PermutationMatrix) live in its package. `ReactiveMP.forward(model, x)`
  and its siblings are `FlowMessagePassingRules.forward`, public and unexported. Building a
  model that draws takes a generator first, `compile(rng, model)`, `PermutationMatrix(rng, dim)`,
  and without one draws from the task's as v6 did. [`Unscented`](@extref MessagePassingRulesApproximations.Unscented)`()` needs no dimension.
- **DiscreteTransition**'s interfaces are `out`, `in`, `a` and the group `T`, which may be empty.
  v6 took `DiscreteTransition(out, in, a, t1, t2)` positionally and aliased the extra arguments
  `T1`, `T2`; they are the members `(:T, 1)`, `(:T, 2)`, and a joint over `out` and `t1` alone is
  keyed `(:out, (:T, 1))`. Every factorisation v6 handled is handled, with one rule per target,
  none per factorisation. A marginal rule's observed members come as [`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster)
  blocks, `(:out,) => …, (:in, (:T, 2)) => …`, where v6 returned `(out = …, in_T2 = …)`; one inside
  a joint over the whole group `T` stays in the joint as a one-hot axis. v6 ignored the node's
  `meta`, and it has no algorithm. Its rules clamp probabilities into `[tiny, huge]`
  relative to `a`'s own values, where v6 did it after normalising the whole tensor, so entries
  near `1e-9` differ from v6's.
- **A distribution value as a prior**, `x ~ d` for `d = Beta(4.0, 8.0)` or `Truncated(…)`, was
  v6's `StandaloneDistributionNode`, an engine node type. It is Standard's
  [`StandaloneDistribution`](@extref StandardMessagePassingRules.StandaloneDistribution), an ordinary node `out ~ d` with `d` a constant: the message towards
  `out` is `d`, and the free-energy term `KL(q ‖ d)`, as before. It is stochastic, so a model's
  `ReactiveMP.sdtype(::StandaloneDistributionNode) = Stochastic()`, which some v6 models added, is
  deleted.

## Behaviour that changed

The rules of the standard nodes follow naive variational message passing where v6 did not, and
fix errors v6 had. A result that differs from v6's for these nodes is expected:

- **Variational rules** take a variance's contribution as `1/E[1/v]` and a covariance's as
  `E[Σ⁻¹]⁻¹` (NormalMeanVariance, MvNormalMeanCovariance), where v6 took `E[v]` and `E[Σ]`.
- **Average energies** of Gamma and GammaInverse compute `E[x]·E[1/θ]` and `θ·E[1/x]`, where v6
  divided expectations.
- **Wishart**'s variational rule towards `out` uses `E[S⁻¹]`; **MvNormalGamma**'s includes the
  covariance of `μ` in the rate; **MvNormalWeightedMeanPrecision**'s marginal labels its blocks
  correctly.
- **Arithmetic:** `+` and `-` fix sign errors for weighted-mean normals and for `-`'s marginal;
  `*`'s sampled messages towards a factor drop a spurious weight, its log scale towards `in` is
  `-d·log|a|`, and a product that would need `in * A` for a matrix operand has no rule. v6's
  `*` rules towards `in` with their arguments reversed, reachable only through `@call_rule`, are
  gone.
- **`+`** takes any number of terms, its interfaces `out` and the group `in`: v6's `in1` and `in2`
  are `(:in, 1)` and `(:in, 2)`, `call_message_update_rule(+, (:in, 2); m = (out = …, in = (…, nothing)))`
  and `((:in, 1), x)` in a hand-built graph. `a + b + c` is one node, where v6 had no rule for it,
  and so is 6.6's `ManyPlus`, which v7 does not have. The joint of the terms with `out` known has
  a rule, so the free energy of a model observing a sum is defined. `-` keeps `in1` and `in2`.
- **When a rule fires.** A rule runs once every input has refreshed since its last run. v6
  relaxed this for marginal inputs whenever every one of them was still initial, which started
  factorisations whose clusters wait on each other (RxInfer#344), but also let rules re-fire each
  other on initial values alone before any data arrived, without bound. v7 relaxes it, after a
  run on initial marginals alone, once for any update and every time a value computed from data
  arrives ([`ReactiveMP.MarginalRelaxation`](@ref)), which starts them as well. A model with every
  marginal initialised can therefore converge elsewhere than on v6, so far always to a free energy
  as low or lower, with far fewer rule calls; v6's mixtures and Delta wired their own inputs and
  were never relaxed, so models with them can differ the most.
- **Mixture** has no average energy: the free energy of a model with one is an error, not zero.
- **NormalMixture** and **GammaMixture** take no type parameter: v6's `NormalMixture{N}` and
  `GammaMixture{N}` are `NormalMixture` and `GammaMixture`, and the number of components is the
  length of their groups.
- **`*`** runs under [`MultiplicationSampling`](@extref StandardMessagePassingRules.MultiplicationSampling)`(; samples = 3000)`, its default algorithm: its three
  rules that sample draw `samples` values from the engine's generator, where v6 always drew 3000.
- **MatrixNormal's average energy** takes a point mass or a MatrixNormal for `q(out)` and `q(M)`,
  whose second moments it uses; v6 took any type and dropped the second moments of anything but
  a MatrixNormal. **MatrixNormalWishart's** is exact for known parameters, where v6 took
  `f(E[x])` for `E[f(x)]`.
- **MvNormalWeightedMeanPrecision's average energy** takes a Wishart `q(Λ)`, not only a point
  mass.
- **Probit's average energy** is finite for a wide `q(in)`, where v6's underflowed at far cubature
  points and returned Inf or NaN.
- **ContinuousTransition's average energies** are the closed form; v6's were wrong in three
  terms, so a model's free energy changes with them. Its rules towards `a` and `W` keep the offset
  of an affine or nonlinear `f`, such as a rotation, which v6 dropped; for `reshape` nothing
  changes.
- **The Pólya nodes' average energies** are corrected: BinomialPolya's is the expectation of
  `softplus(xᵀβ)`, where v6 took it at the mean, and MultinomialPolya's is right for a Multinomial
  `q(x)` with more than one trial. A binomial regression's free energy is higher than v6's.
  BinomialPolya's sampling takes a univariate `β` too, each draw one sample, where v6 took all the
  draws as one.
- **DiscreteTransition** normalises its five-interface belief-propagation message towards `T3`
  with a DirichletCollection `q(a)` over the whole tensor, as every other rule of the node, where
  v6 normalised over `out` only. It also takes what v6 failed on: a Bernoulli `in` or `out`, the
  energy of a joint over three or more axes with a DirichletCollection `q(a)`, and that of a
  non-square `q(out, in)` with a point-mass `q(a)`. A point-mass `A` is used as given; v6's
  belief-propagation rules clamped it to at most one, which changed nothing for a probability
  tensor. v6's fifty or so rules specialised to a number of interfaces are gone; they computed
  the generic contraction, and four of its five-interface rules towards `T2`, which read their own
  edge's message, never matched.
- **The free energy of a model with BIFM** raises an error naming the node, where v6's failed with
  an infinite node bound.
- **[`ARunsafe`](@extref AutoregressiveMessagePassingRules.ARunsafe)'s joint `q(y, x)`** is correct: v6's disagreed with [`ARsafe`](@extref AutoregressiveMessagePassingRules.ARsafe) even for an AR(1), and
  threw for a multivariate AR. `ARsafe` is unchanged.
- **AR's mean-field message towards `γ`** includes `tr(Vθ Vx)` in the expected squared residual
  `E[(y₁ - θᵀx)²]`, as its structured message and its average energy do. v6's left it out, so its
  `q(γ)` had a smaller rate, a larger expected precision, whenever `θ` and `x` were both uncertain.
  A model under `MeanField()` with an AR node gives a different `q(γ)`, and a different `q(θ)`
  and `q(x)` through it (ReactiveMP.jl#681).
- **The Delta node takes three methods**: [`Unscented`](@extref MessagePassingRulesApproximations.Unscented)`()`, [`Linearization`](@extref MessagePassingRulesApproximations.Linearization)`()` and, once
  `using ExponentialFamilyProjection` loads its rules, [`CVIProjection`](@extref DeltaMessagePassingRules.CVIProjection)`()`. v6's other methods are
  gone, with no replacement: `CVI` and `ProdCVI`, `LaplaceApproximation`,
  `ImportanceSamplingApproximation`, `GaussLaguerreQuadrature` and the spherical-radial cubature.
  [`DeltaApproximation`](@extref DeltaMessagePassingRules.DeltaApproximation) refuses any other method with an error that lists these three and, for
  `CVIProjection` without its package, says which package to load. `CVIProjection` has no `rng`
  field: it samples from the generator the engine gives the rule, and its joint rule, which
  keeps its result as the next proposal, is impure. With several inputs, that rule projects them
  in turn, each against the others' latest projections, where v6 used the previous proposal for
  all: its results differ from v6's, and on a posterior with several modes it settles on one
  where v6's alternated.

A random variable's outbound messages change under a form constraint on messages:

- **[`FormConstraintCheckLast`](@ref) applies once to each outbound message**, as documented,
  where v6 applied it to every partial product the variable's equality chain caches as well: for
  inbound messages `μ₁ … μ₄`, the message to the second connection is `f(μ₁ μ₃ μ₄)`, where v6's
  was `f(f(μ₁) f(μ₃ f(μ₄)))`. A constraint that changes the distribution, such as
  `μ(x) :: PointMassFormConstraint()` in RxInfer, gives a different result; one that only
  checks it, such as RxInfer's check that the form is supported, runs once per message. The
  callbacks see one [`ReactiveMP.BeforeProductOfMessagesEvent`](@ref) and one
  [`ReactiveMP.AfterProductOfMessagesEvent`](@ref) per outbound message, and one pair of
  two-message product events fewer per recomputation, since a cached partial product is not
  computed again. [`FormConstraintCheckEach`](@ref) is unchanged.

Two changes in the engine concern code that reads annotations or traces rule calls:

- **A message or marginal that nothing may annotate shares one frozen, empty
  [`ReactiveMP.AnnotationDict`](@ref)**, where v6 gave each its own: one from a mapping without
  annotation processors whose rule does not take the `ann` slot, say. Writing to it,
  `annotate!(getannotations(message), …)`, or to the `annotations` of a callback event, is an
  `ArgumentError`. An annotation processor and a rule taking the `ann` slot get a fresh one, as
  before.
- **Span ids** still pair a "before" event with its "after" event, but come from a counter salted
  once per session, not from `uuid4()`.

## Removed

These v6 names have no counterpart: `Marginalisation`, `MomentMatching`, the functional
dependency types, the per-node node types (`NormalMixtureNode` and its alias `GaussianMixtureNode`,
`GammaMixtureNode`, `MixtureNode`; the checks their constructors made, at least two components, as many of each
kind, a mean-field factorisation, are the `matched_groups`, `min_group_length` and
`factorisation` of [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node)), and the approximation methods
with no remaining consumer (`CVI`, `ProdCVI`, `Adam` and its `update!`, `ForwardDiffGrad`,
`LaplaceApproximation` and `laplace`, `ImportanceSamplingApproximation`, `GaussLaguerreQuadrature`,
`srcubature`), with the Optimisers extension that served `ProdCVI`. MvNormalMeanPrecision's two
marginal rules for BIFM's `TerminalProdArgument` messages are gone too: only the free energy of a
BIFM model reached them, and it is not supported.

The engine does not define these internal helpers, none of which it used: `skip_clamped` and
`skip_clamped_and_initial` (`skip_initial` stays), `KLDivergence` and its `score` method
(`BayesBase.kldivergence` computes the divergence directly), `dropproxytype`, `other_clusters`,
`getinboundinterfaces`, `interfaceindices`, `ReactiveMP.hasfield` (which shadowed
`Base.hasfield`), `split_underscored_symbol`, `fields`, `swapped`, the interface's `tag`, and the
macro helpers other than `@proxy_methods`. The exported `skipindex` and its `SkipIndexIterator`
are gone as well: nothing used them; `(x for (i, x) in enumerate(xs) if i != k)` or a `deleteat!`
copy does the same. The v5 stubs `AddonLogScale` and `AddonMemory`, which only raised an error
pointing to their replacements, are gone too: use the `logscales = true` option and
`InputArgumentsAnnotations`.

## [Checklist for a mechanical port](@id migration-v6-to-v7-checklist)

Follow this checklist when you port code by rote, or have a tool port it for you. A rule that
compiles and returns a plausible distribution can still be wrong, and nothing downstream
catches it.

- **Read first:** the macros of [MessagePassingRulesBase](https://reactivebayes.github.io/MessagePassingRulesBase.jl/dev/),
  [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node),
  [`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule) and
  [`@define_dependencies`](@extref MessagePassingRulesBase.@define_dependencies), then the pairs
  of this guide.
- **Translate only what a pair covers.** Each construct in the code you port must match a pair
  in this guide. Anything else, and everything in [What cannot be translated
  mechanically](@ref migration-v6-to-v7-manual), is a question for the person who owns the code.
  Stop and ask; do not guess.
- **Never guess** which inputs a rule consumes, whether an input is a message or a marginal, the
  order of interfaces, or what `meta` held: all of these change the result.
- **Verify every rule you port** as [Verifying a port](@ref migration-v6-to-v7-verify) describes:
  a table of cases, and a comparison with the old rule while it is available.
- **Stop** when a rule reads raw message tuples, builds graph objects or keeps state in `meta`,
  or when a verification fails and you cannot explain the difference.
