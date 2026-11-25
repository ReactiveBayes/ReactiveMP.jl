module ReactiveMPExponentialFamilyExt

using ReactiveMP, ExponentialFamily
using ExponentialFamily: ExponentialFamilyDistribution, exponential_family_typetag, isbasemeasureconstant,
    getnaturalparameters, getconditioner, isproper, NaturalParametersSpace, UnivariateNormalDistributionsFamily,
    MultivariateNormalDistributionsFamily, NormalWeightedMeanPrecision, MvNormalWeightedMeanPrecision
using BayesBase: weightedmean_precision
using Distributions: Distribution

# Damping in natural parameters, for two members of one exponential family with one conditioner.
# A family registers its base measure for its type tag, which tells its members from other
# distributions.
function ReactiveMP.damp(β, previous::Distribution, new::Distribution)
    family = exponential_family_typetag(new)
    (exponential_family_typetag(previous) === family && applicable(isbasemeasureconstant, family)) || return new
    left, right = convert(ExponentialFamilyDistribution, previous), convert(ExponentialFamilyDistribution, new)
    getconditioner(left) == getconditioner(right) || return new
    η = β .* getnaturalparameters(left) .+ (1 - β) .* getnaturalparameters(right)
    isproper(NaturalParametersSpace(), family, η, getconditioner(right)) || return new
    return convert(Distribution, ExponentialFamilyDistribution(family, η, getconditioner(right)))
end

# Gaussians mix in weighted-mean form, which needs no inversion. A site or a message pulled back
# through a wide matrix has a singular or indefinite precision, and a detour through the
# covariance would change it, so that `damp(β, m, m)` would not be `m`.
function ReactiveMP.damp(β, previous::UnivariateNormalDistributionsFamily, new::UnivariateNormalDistributionsFamily)
    ξ_previous, w_previous = weightedmean_precision(previous)
    ξ_new, w_new = weightedmean_precision(new)
    return NormalWeightedMeanPrecision(β * ξ_previous + (1 - β) * ξ_new, β * w_previous + (1 - β) * w_new)
end

function ReactiveMP.damp(β, previous::MultivariateNormalDistributionsFamily, new::MultivariateNormalDistributionsFamily)
    length(previous) == length(new) || return new
    ξ_previous, W_previous = weightedmean_precision(previous)
    ξ_new, W_new = weightedmean_precision(new)
    return MvNormalWeightedMeanPrecision(β .* ξ_previous .+ (1 - β) .* ξ_new, β .* W_previous .+ (1 - β) .* W_new)
end

end
