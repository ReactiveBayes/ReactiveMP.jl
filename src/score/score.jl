export score, DifferentialEntropy

"""
    score(::Type{T}, kind, args...)

A stream of the quantity `kind` selects, of float type `T`: a factor node's contribution to the
free energy ([`FactorBoundFreeEnergy`](@ref)), a variable's ([`VariableBoundEntropy`](@ref)), or
a marginal's differential entropy ([`DifferentialEntropy`](@ref)). [`bethe_free_energy`](@ref)
combines them.
"""
function score end

##

"""
    DifferentialEntropy()

Selects the differential entropy of a marginal in [`score`](@ref).
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

