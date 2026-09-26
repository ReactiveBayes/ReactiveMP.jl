# [Projection](@id delta-projection)

[`CVIProjection`](@ref) handles functions and message families the Gaussian methods cannot. It
samples the inputs, pushes the samples through `f`, and projects the result onto an exponential
family with
[ExponentialFamilyProjection](https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl).
Its rules are in a package extension, loaded with `using ExponentialFamilyProjection`; until then
[`DeltaApproximation`](@ref) does not accept the method, and its error says which package to
load.

```julia
using DeltaMessagePassingRules, ExponentialFamilyProjection

algorithm = DeltaApproximation(method = CVIProjection(outsamples = 200, sampling_strategy = MeanBased()))
```

Its rules draw their samples from the rule context's generator, `ctx.rng`, which the engine owns.
The rule for the joint over the inputs keeps its result as the next call's proposal, so the
method carries state and that rule is impure.

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
