export score, DifferentialEntropy

"""
    score(::Type{T}, ::FactorBoundFreeEnergy, node::FactorNode, algorithm, stream_postprocessor) where {T <: CountingReal}
    score(::Type{T}, ::VariableBoundEntropy, variable::RandomVariable, stream_postprocessor) where {T <: CountingReal}
    score(::DifferentialEntropy, marginal::Marginal) -> Real

A contribution to the Bethe free energy, which [`bethe_free_energy`](@ref) sums:

- with [`FactorBoundFreeEnergy`](@ref), the stream of a factor node's contribution, one value of
  type `T`, a `BayesBase.CountingReal`, per update of its local marginals;
- with [`VariableBoundEntropy`](@ref), the stream of a random variable's contribution, one value
  of type `T` per update of its marginal;
- with [`DifferentialEntropy`](@ref), the differential entropy of `marginal`, a number: `-∞` for a
  point mass, in its point's float type, the parameters' for a point that is itself a distribution
  and `Float64` for a point with no number type, such as an observed text.

The streams skip initial marginals, and `stream_postprocessor` is applied to them (see
[`ReactiveMP.postprocess_stream_of_scores`](@ref)), `nothing` for none. `algorithm` is the one
the node runs under, `nothing` for its default.

# Throws

A factor node's stream fails when it computes a value: with a
[`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError) naming the node when no
average energy matches its marginals, and with an error naming the rule and the service when the
average energy declares a context service the engine does not supply.
"""
function score end

##

"""
    DifferentialEntropy()

Selects the differential entropy of a marginal, `-∫ q log q`, in [`score`](@ref). A
[`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster)'s is the sum of its
blocks'.
"""
struct DifferentialEntropy end

## Differential entropy function helpers

# A `FactorizedCluster`'s entropy is the sum over its blocks, which BayesBase's
# `FactorizedJoint` gives.
score(::DifferentialEntropy, marginal::Marginal) = entropy(marginal)
# A point mass is −∞, in the float type of its point. A point that is a distribution, the
# constant of a prior `x ~ d`, has its parameters' type, which BayesBase would not take; a point
# with no number type at all, an observed text or any other object a custom node reads, is in
# Float64. A number or an array keeps BayesBase's.
score(::DifferentialEntropy, marginal::Marginal{<:PointMass}) = point_entropy(BayesBase.getpointmass(getdata(marginal)), marginal)
point_entropy(point::Distribution, marginal) = BayesBase.MinusInfinity(paramfloattype(point))
point_entropy(point::Union{Number, AbstractArray, UniformScaling}, marginal) = entropy(marginal)
point_entropy(point, marginal) = BayesBase.MinusInfinity(Float64)
