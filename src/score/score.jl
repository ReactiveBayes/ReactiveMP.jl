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
- with [`DifferentialEntropy`](@ref), the differential entropy of `marginal`, a number: `-∞`, in
  the distribution's float type, for a point mass whose point is itself a distribution.

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
# A point mass whose point is a distribution, the constant of a prior `x ~ d`, is −∞ like any
# other, in the distribution's float type; BayesBase would take the distribution's type for it.
score(::DifferentialEntropy, marginal::Marginal{<:PointMass{<:Distribution}}) =
    BayesBase.MinusInfinity(paramfloattype(BayesBase.getpointmass(getdata(marginal))))
