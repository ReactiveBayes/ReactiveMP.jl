# DeltaMessagePassingRules

```@meta
DocTestSetup = :(using DeltaMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesApproximations, ExponentialFamily, BayesBase)
```

The Delta node, `out = f(in₁, …, inₙ)` for any deterministic function `f`, and the rules that
pass [messages](@extref MessagePassingRulesBase glossary-message) through it. Use it for a
nonlinear function in a model, `z := f(x, y)`, when no node with exact rules exists for `f`.
The rules approximate the [pushforward](@extref MessagePassingRulesBase glossary-pushforward)
through `f`, the distribution of `f(x, y)`, with the method the model names. The unscented
transform and linearisation handle normal messages. Projection onto an exponential family
handles messages of other families.

```@docs
DeltaMessagePassingRules
```

!!! info "Where these rules run"
    The [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) engine runs these rules on
    a factor graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a model.
    The examples here call the rules directly.

## A first example

The message towards `out` of `out = 2in + 1`, by the unscented transform, which is exact for an
affine function. A rule reaches the node's function through its node object, which an engine
provides. Here a small stand-in holds it, and a
[`RuleContext`](@extref MessagePassingRulesBase.RuleContext) hands it to the rule.

```jldoctest index
julia> struct WithFunction{F}
           f::F
       end

julia> MessagePassingRulesBase.getnodefn(node::WithFunction, ::MessagePassingRulesBase.Target{:out}) = node.f

julia> ctx = MessagePassingRulesBase.RuleContext(node = WithFunction(x -> 2x + 1));

julia> result = @call_message_update_rule(
           node = DeltaFn, target = :out, algorithm = DeltaApproximation(method = Unscented()),
           m = (in = (NormalMeanVariance(1.0, 0.5),),), ctx = ctx,
       );

julia> all(mean_var(getresult(result)) .≈ (3.0, 2.0))
true
```

## The site

- [The Delta node](@ref delta-node): the node, its algorithm, its rules and their limitations;
- [Projection](@ref delta-projection): [`CVIProjection`](@ref), the method for messages that are
  not normal, and its sampling strategies;
- [Internals](@ref delta-internals): the helpers the Gaussian rules share, for contributors.
