"""
    DeltaMessagePassingRules

The Delta node, `out = f(in₁, …, inₙ)` for a deterministic function `f`, and its rules, written
with MessagePassingRulesBase. The function is arbitrary, so the rules approximate: they push the
messages through `f` by an approximation method, which the node's algorithm names.

- [`DeltaFn`](@ref): the node, with interfaces `out` and the group `in`;
- [`DeltaApproximation`](@ref): its algorithm, which a model must give, holding the method and
  an optional known inverse of `f`;
- the Gaussian methods [`Unscented`](@extref MessagePassingRulesApproximations.Unscented) and
  [`Linearization`](@extref MessagePassingRulesApproximations.Linearization), from
  MessagePassingRulesApproximations, for normal messages;
- [`CVIProjection`](@ref), sampling and projection onto an exponential family, for messages of
  other families. Its rules are in a package extension, loaded with
  `using ExponentialFamilyProjection`.

The node's function and its static inputs belong to the engine's node object: a rule reaches the
function as [`getnodefn`](@extref MessagePassingRulesBase.getnodefn)`(ctx.node, Target(:out))`,
with any input connected to a constant or to data already folded in. A known inverse belongs to
the algorithm, so a rule towards an input with one needs no node.

# Examples

The message towards the input of `out = 2in + 1` through its known inverse, by linearization:

```jldoctest
julia> algorithm = DeltaApproximation(method = Linearization(), inverse = y -> (y - 1) / 2);

julia> result = @call_message_update_rule(
           node = DeltaFn, target = (:in, 1), algorithm = algorithm,
           m = (out = NormalMeanVariance(3.0, 2.0), in = (nothing,)),
       );

julia> all(mean_var(getresult(result)) .≈ (1.0, 0.5))
true
```
"""
module DeltaMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm, Target, getnodefn
using MessagePassingRulesApproximations: MessagePassingRulesApproximations, Unscented, Linearization, unscented_statistics, smoothRTS

export DeltaFn, DeltaApproximation, is_delta_node_compatible
export CVIProjection, CVISamplingStrategy, FullSampling, MeanBased, ProposalDistributionContainer

include("node.jl")
include("approximate.jl")
include("rules/gaussian.jl")
include("cvi_projection.jl")

end
