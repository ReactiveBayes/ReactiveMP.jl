"""
    DeltaMessagePassingRules

The Delta node, `z = f(x₁, …, xₙ)` for a deterministic function `f`, written with
`MessagePassingRulesBase`: [`DeltaFn`](@ref), its own algorithm [`DeltaApproximation`](@ref)
(the approximation method and an optional known inverse), the dependencies
each form of the algorithm declares, and the rules for the `Unscented` and `Linearization`
methods.

The node's function and its static inputs belong to the engine's node object: a rule reaches
the function as `getnodefn(ctx.node, Target(:out))`, with any input connected to a constant or
to data already folded in.
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
