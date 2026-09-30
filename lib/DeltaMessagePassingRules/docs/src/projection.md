# [Projection](@id delta-projection)

[`CVIProjection`](@ref) handles the functions and message families that the Gaussian methods
cannot. It draws samples of the inputs, pushes the samples through `f`, and projects the result
onto an exponential family with
[ExponentialFamilyProjection](https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl).
Projection here means finding the member of a chosen family, such as the normals or the gamma
distributions, that is closest to the result.

## Loading the rules

The rules of `CVIProjection` are in a package extension. Until ExponentialFamilyProjection is
loaded, [`DeltaApproximation`](@ref) refuses the method, and its error says which package to
load:

```@example projection
using DeltaMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

try
    DeltaApproximation(method = CVIProjection())
catch err
    showerror(stdout, err)
end
```

Loading the package loads the rules:

```@example projection
using ExponentialFamilyProjection

algorithm = DeltaApproximation(method = CVIProjection())
nothing # hide
```

The method reads different inputs from the Gaussian methods. Towards `out` it reads the message
on `out` itself, the [marginal](@extref MessagePassingRulesBase glossary-marginal) `q(out)` and
the joint over the inputs:

```@example projection
MessagePassingRulesBase.dependencies_spec(DeltaFn, algorithm)
```

## Example

Take `out = exp(in)`, a standard normal message on `in`, and a gamma message on `out`, a
positive quantity. The joint over the single input is the input's message reweighted by the
likelihood of `out`, projected back onto the normals:

```@example projection
using Random: Xoshiro

struct WithFunction{F}
    f::F
end

MessagePassingRulesBase.getnodefn(node::WithFunction, ::MessagePassingRulesBase.Target{:out}) = node.f

ctx = MessagePassingRulesBase.RuleContext(node = WithFunction(exp), rng = Xoshiro(42))

joint = @call_marginal_update_rule(
    node = DeltaFn, target = (:in,), algorithm = algorithm, ctx = ctx,
    m = (out = Gamma(2.0, 1.0), in = (NormalMeanVariance(0.0, 1.0),)),
)
```

The rules draw their samples from the rule context's generator, `ctx.rng`, which an engine
owns and a caller sets here. The node object is the same stand-in as on the
[overview](index.md). The message towards `out` samples that joint, pushes the samples through
`exp`, and projects them onto the family of `q(out)`, here the gamma distributions:

```@example projection
q_in = getresult(joint)

@call_message_update_rule(
    node = DeltaFn, target = :out, algorithm = algorithm, ctx = ctx,
    m = (out = Gamma(2.0, 1.0),), q = (out = Gamma(3.0, 1.0),), clusters = ((:in,) => q_in,),
)
```

The result is the projection divided by the message that arrived on `out`, left unevaluated:
its product with that message, at the variable, is the projection.

For a node with several inputs, the rule for the joint projects each input in turn, against
samples of the others. A [`CVISamplingStrategy`](@ref) decides how those samples are drawn:

```@example projection
DeltaApproximation(method = CVIProjection(outsamples = 200, sampling_strategy = MeanBased()))
nothing # hide
```

The rule for the joint keeps its result as the next call's proposal, in a
[`ProposalDistributionContainer`](@ref), so the method carries state and that rule is impure.

## Limitations

- The rules need ExponentialFamilyProjection loaded.
- Sampling makes the messages random: two runs agree only with the same generator.
- The message towards `out` reads `q(out)` and the joint over the inputs, which depend on the
  node's own messages; a model whose `out` is not observed needs initial marginals for them.
- One `CVIProjection` value given to several nodes shares one proposal among them.
- A known inverse is not used, and is ignored with a warning.

## API

The method:

```@docs
CVIProjection
```

How the other inputs are sampled while one input of a node with several is projected:

```@docs
CVISamplingStrategy
FullSampling
MeanBased
```

The proposal the inputs are sampled from:

```@docs
ProposalDistributionContainer
```
